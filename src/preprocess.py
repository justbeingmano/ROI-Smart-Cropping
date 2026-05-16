import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

VOC_ROOT = ROOT / "project-image" / "VOC2012_train_val" / "VOC2012_train_val"
OUT_DIR = ROOT / "output" / "dataset"

# 5-class instance segmentation setup
TARGET_CLASSES = ["person", "chair", "car", "dog", "bottle"]

# False = true multi-class training
# person=0, chair=1, car=2, dog=3, bottle=4
SINGLE_CLASS = False
SINGLE_CLASS_NAME = "object"

SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_STATE = 42

# Mask/polygon processing
AREA_THRESHOLD = 10
EPSILON_FACTOR = 0.0005

# Rebuild YOLO dataset from scratch each run
CLEAN_OUT_DIR = True

# ============================================================
# BALANCING / UPSAMPLING
# ============================================================

# Do NOT downsample person now because segmentation data is already limited.
BALANCE_PERSON_ONLY_TRAIN = False
PERSON_ONLY_KEEP_RATIO = 0.4

# Upsample rare classes in TRAIN ONLY.
# This creates duplicate image/label files with _dupN suffixes.
UPSAMPLE_RARE_CLASSES = True

CLASS_UPSAMPLE_FACTORS = {
    "person": 1,
    "chair": 2,
    "car": 2,
    "dog": 2,
    "bottle": 3,
}


# ============================================================
# VOC MASK READING
# ============================================================

def voc_colormap(n: int = 256) -> np.ndarray:
    """Generate standard Pascal VOC color map."""
    cmap = np.zeros((n, 3), dtype=np.uint8)

    for i in range(n):
        r = g = b = 0
        cid = i

        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3

        cmap[i] = [r, g, b]

    return cmap


VOC_COLOR_TO_INDEX = {
    tuple(color.tolist()): idx for idx, color in enumerate(voc_colormap(256))
}


def read_voc_index_mask(mask_path: Path) -> np.ndarray | None:
    """
    Read VOC SegmentationObject mask as a 2D indexed array.

    Original VOC masks are palette PNGs.
    PIL preserves indexed values correctly.
    RGB fallback is included for dataset mirrors that saved masks as color images.
    """
    try:
        with Image.open(mask_path) as img:
            arr = np.array(img)
    except Exception:
        arr = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            return None

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        arr = arr[:, :, :3]

        # If RGB channels are equal, it is effectively grayscale.
        if (
            np.array_equal(arr[:, :, 0], arr[:, :, 1])
            and np.array_equal(arr[:, :, 0], arr[:, :, 2])
        ):
            return arr[:, :, 0]

        # RGB VOC-color fallback.
        h, w, _ = arr.shape
        flat = arr.reshape(-1, 3)
        out = np.full((flat.shape[0],), 255, dtype=np.uint8)

        unique_colors = np.unique(flat, axis=0)

        for color in unique_colors:
            idx = VOC_COLOR_TO_INDEX.get(tuple(color.tolist()), 255)
            matches = np.all(flat == color, axis=1)
            out[matches] = idx

        return out.reshape(h, w)

    return None


# ============================================================
# XML / CLASS HELPERS
# ============================================================

def get_image_objects(xml_path: Path) -> list[str]:
    """
    Parse VOC XML annotation and return object class names in XML order.

    For VOC SegmentationObject masks:
    instance_id=1 generally maps to the first XML object,
    instance_id=2 to the second object, and so on.
    """
    objects = []

    if not xml_path.exists():
        return objects

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            cls_obj = obj.find("name")

            if cls_obj is not None and cls_obj.text:
                objects.append(cls_obj.text.strip())

    except Exception:
        return []

    return objects


def get_target_set(objects: list[str]) -> set[str]:
    if not TARGET_CLASSES:
        return set(objects)

    return {c for c in objects if c in TARGET_CLASSES}


def class_id_for_name(class_name: str) -> int | None:
    if SINGLE_CLASS:
        return 0

    if class_name not in TARGET_CLASSES:
        return None

    return TARGET_CLASSES.index(class_name)


def names_for_yaml() -> dict[int, str]:
    if SINGLE_CLASS:
        return {0: SINGLE_CLASS_NAME}

    return {i: name for i, name in enumerate(TARGET_CLASSES)}


# ============================================================
# POLYGON CONVERSION
# ============================================================

def mask_to_polygons(
    mask_path: Path,
    img_w: int,
    img_h: int,
    object_names: list[str],
):
    """
    Convert VOC SegmentationObject mask to YOLO polygon labels.

    Rules:
    - use SegmentationObject, not SegmentationClass
    - use original-resolution mask
    - do not resize before contour extraction
    - export target instances only
    - keep all valid external contours
    - simplify polygons safely
    - normalize by original image width/height
    """
    mask = read_voc_index_mask(mask_path)

    if mask is None:
        return [], "read_error"

    unique_ids = np.unique(mask)
    unique_ids = unique_ids[(unique_ids != 0) & (unique_ids != 255)]

    if len(unique_ids) == 0:
        return [], "empty_mask"

    yolo_lines = []
    skipped_non_target = 0
    skipped_unmapped = 0

    for inst_id in unique_ids:
        inst_int = int(inst_id)
        xml_index = inst_int - 1

        if xml_index < 0 or xml_index >= len(object_names):
            skipped_unmapped += 1
            continue

        class_name = object_names[xml_index]

        if TARGET_CLASSES and class_name not in TARGET_CLASSES:
            skipped_non_target += 1
            continue

        class_id = class_id_for_name(class_name)

        if class_id is None:
            skipped_non_target += 1
            continue

        binary = np.where(mask == inst_id, 255, 0).astype(np.uint8)

        if binary.ndim != 2:
            return [], f"binary_not_single_channel_{binary.shape}"

        if binary.dtype != np.uint8:
            return [], f"binary_wrong_dtype_{binary.dtype}"

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < AREA_THRESHOLD:
                continue

            epsilon = EPSILON_FACTOR * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            pts = approx.reshape(-1, 2)

            # Must have at least 3 unique points for a valid polygon.
            if len(np.unique(pts, axis=0)) < 3:
                continue

            pts = pts.astype(np.float32)

            pts[:, 0] /= img_w
            pts[:, 1] /= img_h

            pts = np.clip(pts, 0.0, 1.0)

            coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts)

            yolo_lines.append(f"{class_id} {coords}")

    if not yolo_lines:
        if skipped_non_target > 0:
            return [], "no_target_polygons"

        if skipped_unmapped > 0:
            return [], "unmapped_instances"

        return [], "no_valid_polygons"

    return yolo_lines, None


# ============================================================
# TRAIN BALANCING / UPSAMPLING
# ============================================================

def apply_train_balancing(
    train_ids: list[str],
    id_to_targets: dict[str, set[str]],
) -> list[str]:
    """
    Optional downsampling for person-only train images.

    Currently disabled by default:
    BALANCE_PERSON_ONLY_TRAIN = False
    """
    if not BALANCE_PERSON_ONLY_TRAIN:
        print("ℹ️ Train person-downsampling disabled. Keeping all matching train images.")
        return train_ids

    if "person" not in TARGET_CLASSES or len(TARGET_CLASSES) <= 1:
        print("ℹ️ Train balancing skipped: use it only for multi-class experiments with person.")
        return train_ids

    rng = random.Random(RANDOM_STATE)

    balanced = []
    dropped = 0

    for img_id in train_ids:
        targets = id_to_targets.get(img_id, set())
        is_person_only = targets == {"person"}

        if is_person_only and rng.random() > PERSON_ONLY_KEEP_RATIO:
            dropped += 1
            continue

        balanced.append(img_id)

    print(
        f"⚖️ Train balancing enabled: dropped {dropped} person-only images. "
        f"Kept {len(balanced)}/{len(train_ids)} train images."
    )

    return balanced


def get_upsample_factor(img_id: str, id_to_targets: dict[str, set[str]]) -> int:
    """
    Return duplication factor for a training image.

    If image has multiple target classes, use the maximum factor.
    Example:
    targets = {"person", "bottle"} -> factor 3
    """
    if not UPSAMPLE_RARE_CLASSES:
        return 1

    targets = id_to_targets.get(img_id, set())

    if not targets:
        return 1

    factor = 1

    for cls in targets:
        factor = max(factor, CLASS_UPSAMPLE_FACTORS.get(cls, 1))

    return max(1, factor)


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"📂 Project root: {ROOT.resolve()}")
    print(f"📂 Scanning VOC Dataset at: {VOC_ROOT.resolve()}")

    mask_dir = VOC_ROOT / "SegmentationObject"
    img_dir = VOC_ROOT / "JPEGImages"
    ann_dir = VOC_ROOT / "Annotations"

    if not VOC_ROOT.exists():
        raise FileNotFoundError(f"VOC_ROOT does not exist: {VOC_ROOT}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Could not find SegmentationObject at: {mask_dir}")

    if not img_dir.exists():
        raise FileNotFoundError(f"Could not find JPEGImages at: {img_dir}")

    if not ann_dir.exists():
        raise FileNotFoundError(f"Could not find Annotations at: {ann_dir}")

    if CLEAN_OUT_DIR and OUT_DIR.exists():
        print(f"🧹 Cleaning existing output dataset: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    for sub in [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    # Prefer VOC official segmentation split.
    id_list_path = VOC_ROOT / "ImageSets" / "Segmentation" / "trainval.txt"

    if id_list_path.exists():
        print(f"✅ Using official segmentation split: {id_list_path}")
        with open(id_list_path, "r", encoding="utf-8") as f:
            all_ids = [line.strip() for line in f if line.strip()]
    else:
        print("⚠️ ImageSets/Segmentation/trainval.txt not found. Scanning SegmentationObject masks.")
        all_ids = sorted(f.stem for f in mask_dir.glob("*.png"))

    stats = Counter()
    stats["total_scanned"] = len(all_ids)

    error_samples = defaultdict(list)

    id_to_objects: dict[str, list[str]] = {}
    id_to_targets: dict[str, set[str]] = {}

    valid_ids = []

    for img_id in tqdm(all_ids, desc="Filtering target classes"):
        xml_path = ann_dir / f"{img_id}.xml"

        objects = get_image_objects(xml_path)
        targets = get_target_set(objects)

        id_to_objects[img_id] = objects
        id_to_targets[img_id] = targets

        if targets:
            valid_ids.append(img_id)

    stats["target_filtered"] = len(valid_ids)

    print(f"\n✅ Target classes: {TARGET_CLASSES}")
    print(f"✅ Total scanned segmentation IDs: {len(all_ids)}")
    print(f"✅ Matching images after class filter: {len(valid_ids)}")
    print(f"🏷️ YAML names: {names_for_yaml()}")

    if not valid_ids:
        print("❌ No images found for selected classes.")
        return

    temp_ratio = SPLIT_RATIOS[1] + SPLIT_RATIOS[2]
    test_fraction_of_temp = SPLIT_RATIOS[2] / temp_ratio

    train_ids, temp_ids = train_test_split(
        valid_ids,
        test_size=temp_ratio,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_fraction_of_temp,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_ids = apply_train_balancing(train_ids, id_to_targets)

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    print("\n📊 Target image counts by split:")

    for split_name, ids in splits.items():
        class_counter = Counter()

        for img_id in ids:
            for cls in id_to_targets.get(img_id, set()):
                class_counter[cls] += 1

        print(f"   {split_name:5s}: {dict(class_counter)}")

    print("\n📊 Upsampling config:")
    print(f"   UPSAMPLE_RARE_CLASSES: {UPSAMPLE_RARE_CLASSES}")
    print(f"   CLASS_UPSAMPLE_FACTORS: {CLASS_UPSAMPLE_FACTORS}")
    print("   Applies to train split only.")

    # Process splits
    for split_name, ids in splits.items():
        print(f"\n⏳ Processing {split_name} split ({len(ids)} base images)...")

        split_img_dir = OUT_DIR / "images" / split_name
        split_lbl_dir = OUT_DIR / "labels" / split_name

        for img_id in tqdm(ids, leave=False):
            src_img = img_dir / f"{img_id}.jpg"
            src_mask = mask_dir / f"{img_id}.png"

            if not src_img.exists() or not src_mask.exists():
                stats["skipped_read_error"] += 1

                if len(error_samples["read_error"]) < 5:
                    error_samples["read_error"].append(img_id)

                continue

            img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)

            if img is None:
                stats["skipped_read_error"] += 1

                if len(error_samples["read_error"]) < 5:
                    error_samples["read_error"].append(img_id)

                continue

            h, w = img.shape[:2]

            polygons, error = mask_to_polygons(
                src_mask,
                w,
                h,
                id_to_objects.get(img_id, []),
            )

            if not polygons:
                error = error or "unknown"
                stats[f"skipped_{error}"] += 1

                if len(error_samples[error]) < 5:
                    error_samples[error].append(img_id)

                continue

            # Upsampling is train-only.
            factor = get_upsample_factor(img_id, id_to_targets) if split_name == "train" else 1

            for dup_idx in range(factor):
                if dup_idx == 0:
                    out_id = img_id
                else:
                    out_id = f"{img_id}_dup{dup_idx}"

                shutil.copy2(src_img, split_img_dir / f"{out_id}.jpg")

                with open(split_lbl_dir / f"{out_id}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(polygons))

                stats["processed"] += 1

                if split_name == "train" and dup_idx > 0:
                    stats["train_upsampled_duplicates"] += 1

    yaml_path = OUT_DIR / "data.yaml"

    yaml_content = {
        "path": str(OUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names_for_yaml(),
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            yaml_content,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    assert yaml_path.exists(), f"data.yaml was not created at {yaml_path}"

    # Count final generated files
    final_train_images = len(list((OUT_DIR / "images" / "train").glob("*.jpg")))
    final_val_images = len(list((OUT_DIR / "images" / "val").glob("*.jpg")))
    final_test_images = len(list((OUT_DIR / "images" / "test").glob("*.jpg")))

    print("\n" + "=" * 60)
    print("📊 PREPROCESSING STATISTICS")
    print("=" * 60)
    print(f"Total scanned segmentation IDs : {stats['total_scanned']}")
    print(f"Filtered target images         : {stats['target_filtered']}")
    print(f"Successfully written samples   : {stats['processed']}")
    print(f"Train upsample duplicates      : {stats['train_upsampled_duplicates']}")
    print("-" * 60)
    print(f"Base split train IDs           : {len(train_ids)}")
    print(f"Base split val IDs             : {len(val_ids)}")
    print(f"Base split test IDs            : {len(test_ids)}")
    print("-" * 60)
    print(f"Final train images after upsample: {final_train_images}")
    print(f"Final val images                 : {final_val_images}")
    print(f"Final test images                : {final_test_images}")
    print("-" * 60)

    skipped_keys = sorted(k for k in stats if k.startswith("skipped_"))

    if skipped_keys:
        print("Skipped summary:")

        for key in skipped_keys:
            print(f"  {key.replace('skipped_', ''):24s}: {stats[key]}")
    else:
        print("Skipped summary                : none")

    if error_samples:
        print("\nSample problematic IDs:")

        for err, ids in error_samples.items():
            print(f"  {err:24s}: {', '.join(ids)}")

    print("-" * 60)
    print(f"📦 Dataset root                : {OUT_DIR.resolve()}")
    print(f"✅ data.yaml saved at           : {yaml_path}")
    print(f"🏷️ YAML names                  : {names_for_yaml()}")
    print("\n🚀 5-class preprocessing with rare-class upsampling complete.")
    print("Run training with: python train.py")


if __name__ == "__main__":
    main()