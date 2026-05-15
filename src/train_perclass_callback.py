import csv
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "output" / "dataset" / "data.yaml"

MODEL_VARIANT = "yolov8s-seg.pt"

EPOCHS = 150
PATIENCE = 40

IMG_SIZE = 640
BATCH_SIZE = 4
WORKERS = 2
DEVICE = 0

# "ram", "disk", or False
CACHE_MODE = "ram"

RUN_NAME = "voc2012_person_singleclass_v3"

# Quality watcher thresholds.
# These are intentionally conservative. For the single-class person baseline,
# the callback will warn and save recommendations if learning is clearly poor.
QUALITY_MIN_EPOCH = 30
QUALITY_PATIENCE = 12
MIN_MASK_MAP5095 = 0.08
MIN_MASK_RECALL = 0.20


# ============================================================
# METRIC HELPERS
# ============================================================

def get_metric(results_dict: dict, candidates: list[str]) -> float:
    """Handle small naming differences across Ultralytics versions."""
    for key in candidates:
        if key in results_dict:
            return float(results_dict[key])
    return 0.0


def safe_array(value):
    if value is None:
        return np.array([])
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return np.array([])


def extract_per_class_mask_metrics(metrics) -> list[dict]:
    """
    Extract per-class segmentation metrics from an Ultralytics validation result.

    This is defensive because Ultralytics metric attribute names can vary slightly
    across versions. It targets the common SegmentMetrics structure:
    metrics.seg.p, metrics.seg.r, metrics.seg.ap50, metrics.seg.ap, metrics.seg.maps.
    """
    names = getattr(metrics, "names", {}) or {}
    seg = getattr(metrics, "seg", None)

    if seg is None:
        return []

    p = safe_array(getattr(seg, "p", None))
    r = safe_array(getattr(seg, "r", None))
    ap50 = safe_array(getattr(seg, "ap50", None))
    ap = safe_array(getattr(seg, "ap", None))
    maps = safe_array(getattr(seg, "maps", None))
    ap_class_index = getattr(seg, "ap_class_index", None)

    if ap_class_index is not None:
        class_ids = [int(x) for x in np.asarray(ap_class_index).tolist()]
    else:
        max_len = max(len(p), len(r), len(ap50), len(ap), len(maps), len(names))
        class_ids = list(range(max_len))

    rows = []
    for pos, cls_id in enumerate(class_ids):
        class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

        row = {
            "class_id": cls_id,
            "class_name": class_name,
            "precision_M": float(p[pos]) if pos < len(p) else 0.0,
            "recall_M": float(r[pos]) if pos < len(r) else 0.0,
            "mAP50_M": float(ap50[pos]) if pos < len(ap50) else 0.0,
            "mAP50_95_M": float(maps[cls_id]) if cls_id < len(maps) else (float(ap[pos]) if pos < len(ap) else 0.0),
        }
        rows.append(row)

    return rows


def save_per_class_metrics(rows: list[dict], save_dir: Path, filename: str = "per_class_mask_metrics.csv"):
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / filename

    fieldnames = ["class_id", "class_name", "precision_M", "recall_M", "mAP50_M", "mAP50_95_M"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


def print_per_class_metrics(rows: list[dict]):
    if not rows:
        print("⚠️ Per-class mask metrics were not available from this Ultralytics version/result object.")
        return

    print("\n📊 PER-CLASS MASK PERFORMANCE")
    print("-" * 86)
    print(f"{'ID':>3} | {'Class':<15} | {'Prec(M)':>8} | {'Recall(M)':>9} | {'mAP50(M)':>9} | {'mAP50-95(M)':>12}")
    print("-" * 86)

    for row in rows:
        print(
            f"{row['class_id']:>3} | "
            f"{row['class_name']:<15} | "
            f"{row['precision_M']:>8.4f} | "
            f"{row['recall_M']:>9.4f} | "
            f"{row['mAP50_M']:>9.4f} | "
            f"{row['mAP50_95_M']:>12.4f}"
        )

    print("-" * 86)


# ============================================================
# QUALITY CALLBACK
# ============================================================

class QualityWatchCallback:
    """
    Training callback that watches validation mask metrics and writes a tuning plan
    if the run appears weak or plateaued.

    It does NOT mutate training parameters mid-run. That is intentional:
    changing augmentations/LR/image size during training can make experiments hard
    to interpret. Instead, it saves a concrete next-run recommendation file.
    """

    def __init__(
        self,
        min_epoch: int,
        patience: int,
        min_mask_map5095: float,
        min_mask_recall: float,
    ):
        self.min_epoch = min_epoch
        self.patience = patience
        self.min_mask_map5095 = min_mask_map5095
        self.min_mask_recall = min_mask_recall
        self.best_map = -1.0
        self.bad_epochs = 0
        self.already_written = False

    def __call__(self, trainer):
        epoch = int(getattr(trainer, "epoch", -1)) + 1

        metrics = getattr(trainer, "metrics", {}) or {}
        if not isinstance(metrics, dict):
            return

        map5095 = get_metric(
            metrics,
            ["metrics/mAP50-95(M)", "metrics/mAP50-95_mask", "metrics/mAP50-95(B)"],
        )
        recall = get_metric(
            metrics,
            ["metrics/recall(M)", "metrics/recall_mask", "metrics/recall(B)"],
        )

        if epoch < self.min_epoch:
            return

        if map5095 > self.best_map + 1e-4:
            self.best_map = map5095
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        clearly_weak = map5095 < self.min_mask_map5095 and recall < self.min_mask_recall
        plateaued = self.bad_epochs >= self.patience

        if not (clearly_weak or plateaued) or self.already_written:
            return

        save_dir = Path(getattr(trainer, "save_dir", ROOT / "output" / "runs" / RUN_NAME))
        save_dir.mkdir(parents=True, exist_ok=True)

        recommendations = {
            "trigger_epoch": epoch,
            "current_mask_mAP50_95": map5095,
            "current_mask_recall": recall,
            "best_mask_mAP50_95_seen": self.best_map,
            "reason": "weak_metrics" if clearly_weak else "plateau",
            "recommended_next_edits": [
                "Use EPOCHS=200 and PATIENCE=50 if the curve is still improving slowly.",
                "Try IMG_SIZE=768 with BATCH_SIZE=2 if VRAM allows; person masks benefit from higher resolution.",
                "Try mosaic=0.0 for one run to test whether mosaic is hurting boundary quality.",
                "Keep mixup=0.0 for segmentation.",
                "Keep copy_paste=0.2 unless it creates unrealistic masks.",
                "If single-class person remains weak, expand to 5 classes with SINGLE_CLASS=False to restore semantic priors.",
                "For 5 classes, set TARGET_CLASSES=['person', 'chair', 'car', 'dog', 'bottle'] and consider BALANCE_PERSON_ONLY_TRAIN=True.",
            ],
        }

        json_path = save_dir / "auto_tuning_recommendations.json"
        md_path = save_dir / "auto_tuning_recommendations.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(recommendations, f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Auto Tuning Recommendations\n\n")
            f.write(f"- Trigger epoch: {epoch}\n")
            f.write(f"- Current mask mAP50-95: {map5095:.4f}\n")
            f.write(f"- Current mask recall: {recall:.4f}\n")
            f.write(f"- Best mask mAP50-95 seen: {self.best_map:.4f}\n")
            f.write(f"- Reason: {recommendations['reason']}\n\n")
            f.write("## Recommended next-run edits\n\n")
            for item in recommendations["recommended_next_edits"]:
                f.write(f"- {item}\n")

        print("\n" + "!" * 72)
        print("⚠️ QUALITY WATCH CALLBACK TRIGGERED")
        print(f"Mask mAP50-95={map5095:.4f}, Recall={recall:.4f}, Best={self.best_map:.4f}")
        print(f"Saved next-run recommendations to: {json_path}")
        print("!" * 72 + "\n")

        self.already_written = True


# ============================================================
# MAIN
# ============================================================

def main():
    if not DATA_YAML.exists():
        print("❌ Error: data.yaml not found.")
        print(f"   Expected location: {DATA_YAML.resolve()}")
        print("   Please run preprocess_final.py first.")
        return

    print("=" * 56)
    print("🚀 YOLOv8-Seg Training")
    print("=" * 56)
    print(f"Data YAML    : {DATA_YAML.resolve()}")
    print(f"Model        : {MODEL_VARIANT}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Image size   : {IMG_SIZE}")
    print(f"Batch        : {BATCH_SIZE}")
    print(f"Cache        : {CACHE_MODE}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print("=" * 56)

    model = YOLO(MODEL_VARIANT)

    # Ultralytics supports callbacks through model.add_callback().
    model.add_callback(
        "on_fit_epoch_end",
        QualityWatchCallback(
            min_epoch=QUALITY_MIN_EPOCH,
            patience=QUALITY_PATIENCE,
            min_mask_map5095=MIN_MASK_MAP5095,
            min_mask_recall=MIN_MASK_RECALL,
        ),
    )

    results = model.train(
        data=str(DATA_YAML.resolve()),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        amp=True,
        cache=CACHE_MODE,

        lr0=5e-4,
        cos_lr=True,
        seed=42,
        deterministic=True,

        degrees=5.0,
        scale=0.3,
        fliplr=0.5,

        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.2,
        close_mosaic=20,

        project=str(ROOT / "output" / "runs"),
        name=RUN_NAME,
        exist_ok=True,
        save=True,
        plots=True,
    )

    print("\n" + "=" * 56)
    print("🏆 TRAINING COMPLETE")
    print("=" * 56)

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    if best_weights.exists():
        print(f"✅ Loading best checkpoint for test evaluation: {best_weights}")
        eval_model = YOLO(str(best_weights))
    else:
        print("⚠️ best.pt not found; evaluating current model state.")
        eval_model = model

    print("\n📊 Evaluating on Test Set...")
    metrics = eval_model.val(
        data=str(DATA_YAML.resolve()),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
    )

    results_dict = metrics.results_dict

    precision_m = get_metric(
        results_dict,
        ["metrics/precision(M)", "metrics/precision_mask", "metrics/precision(B)"],
    )
    recall_m = get_metric(
        results_dict,
        ["metrics/recall(M)", "metrics/recall_mask", "metrics/recall(B)"],
    )
    map50_m = get_metric(
        results_dict,
        ["metrics/mAP50(M)", "metrics/mAP50_mask", "metrics/mAP50(B)"],
    )
    map5095_m = get_metric(
        results_dict,
        ["metrics/mAP50-95(M)", "metrics/mAP50-95_mask", "metrics/mAP50-95(B)"],
    )

    print("\n📈 FINAL TEST SEGMENTATION PERFORMANCE:")
    print(f"   • Precision (M): {precision_m:.4f}")
    print(f"   • Recall (M)   : {recall_m:.4f}")
    print(f"   • mAP50 (M)    : {map50_m:.4f}")
    print(f"   • mAP50-95 (M) : {map5095_m:.4f}")

    per_class_rows = extract_per_class_mask_metrics(metrics)
    print_per_class_metrics(per_class_rows)

    csv_path = save_per_class_metrics(per_class_rows, save_dir)
    print(f"\n✅ Per-class mask metrics saved to: {csv_path}")

    # Final post-training recommendation if the completed run is weak.
    if map5095_m < MIN_MASK_MAP5095 or recall_m < MIN_MASK_RECALL:
        final_reco_path = save_dir / "final_low_score_recommendations.md"
        with open(final_reco_path, "w", encoding="utf-8") as f:
            f.write("# Final Low-Score Recommendations\n\n")
            f.write(f"- Final test mAP50-95(M): {map5095_m:.4f}\n")
            f.write(f"- Final test recall(M): {recall_m:.4f}\n\n")
            f.write("## Suggested next edits\n\n")
            f.write("1. Train longer: `EPOCHS=200`, `PATIENCE=50`.\n")
            f.write("2. Try higher resolution: `IMG_SIZE=768`, `BATCH_SIZE=2`.\n")
            f.write("3. Test no mosaic: `mosaic=0.0`.\n")
            f.write("4. Expand to 5 classes with `SINGLE_CLASS=False` after the person baseline is stable.\n")
            f.write("5. For 5 classes, enable person-only train downsampling only if person dominates the train split.\n")

        print(f"⚠️ Low final score detected. Recommendations saved to: {final_reco_path}")

    print(f"\n📁 Results saved in: {save_dir}")


if __name__ == "__main__":
    main()
