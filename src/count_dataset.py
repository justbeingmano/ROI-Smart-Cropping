from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
VOC_ROOT = ROOT / "project-image" / "VOC2012_train_val" / "VOC2012_train_val"

IMG_DIR = VOC_ROOT / "JPEGImages"
ANN_DIR = VOC_ROOT / "Annotations"
MASK_DIR = VOC_ROOT / "SegmentationObject"
SEG_SPLIT = VOC_ROOT / "ImageSets" / "Segmentation" / "trainval.txt"

TARGET_CLASSES = ["person", "chair", "car", "dog", "bottle"]


def read_objects(xml_path):
    objects = []
    if not xml_path.exists():
        return objects

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name = obj.find("name")
        if name is not None and name.text:
            objects.append(name.text.strip())

    return objects


jpg_ids = {p.stem for p in IMG_DIR.glob("*.jpg")}
xml_ids = {p.stem for p in ANN_DIR.glob("*.xml")}
mask_ids = {p.stem for p in MASK_DIR.glob("*.png")}

if SEG_SPLIT.exists():
    seg_ids = {line.strip() for line in SEG_SPLIT.read_text().splitlines() if line.strip()}
else:
    seg_ids = mask_ids

all_xml_class_instances = Counter()
all_xml_class_images = Counter()

seg_class_instances = Counter()
seg_class_images = Counter()

seg_with_mask_class_instances = Counter()
seg_with_mask_class_images = Counter()

for img_id in xml_ids:
    objects = read_objects(ANN_DIR / f"{img_id}.xml")
    unique_classes = set(objects)

    all_xml_class_instances.update(objects)
    all_xml_class_images.update(unique_classes)

    if img_id in seg_ids:
        seg_class_instances.update(objects)
        seg_class_images.update(unique_classes)

    if img_id in seg_ids and img_id in mask_ids:
        seg_with_mask_class_instances.update(objects)
        seg_with_mask_class_images.update(unique_classes)


print("=" * 60)
print("DATASET COUNTS")
print("=" * 60)
print(f"JPEGImages count               : {len(jpg_ids)}")
print(f"Annotations XML count          : {len(xml_ids)}")
print(f"SegmentationObject masks count : {len(mask_ids)}")
print(f"Segmentation trainval IDs      : {len(seg_ids)}")
print("=" * 60)

print("\nALL XML / DETECTION-LIKE COUNTS")
for cls in TARGET_CLASSES:
    print(
        f"{cls:10s} | images: {all_xml_class_images[cls]:5d} | "
        f"instances: {all_xml_class_instances[cls]:5d}"
    )

print("\nSEGMENTATION SPLIT COUNTS")
for cls in TARGET_CLASSES:
    print(
        f"{cls:10s} | images: {seg_class_images[cls]:5d} | "
        f"instances: {seg_class_instances[cls]:5d}"
    )

print("\nSEGMENTATION SPLIT + MASK EXISTS COUNTS")
for cls in TARGET_CLASSES:
    print(
        f"{cls:10s} | images: {seg_with_mask_class_images[cls]:5d} | "
        f"instances: {seg_with_mask_class_instances[cls]:5d}"
    )

print("\n5-CLASS IMAGE FILTER COUNTS")
five_class_all_xml = 0
five_class_seg = 0
five_class_seg_with_mask = 0

for img_id in xml_ids:
    objects = set(read_objects(ANN_DIR / f"{img_id}.xml"))

    if objects.intersection(TARGET_CLASSES):
        five_class_all_xml += 1

    if img_id in seg_ids and objects.intersection(TARGET_CLASSES):
        five_class_seg += 1

    if img_id in seg_ids and img_id in mask_ids and objects.intersection(TARGET_CLASSES):
        five_class_seg_with_mask += 1

print(f"5-class images in all XMLs              : {five_class_all_xml}")
print(f"5-class images in segmentation split    : {five_class_seg}")
print(f"5-class images in segmentation + masks  : {five_class_seg_with_mask}")