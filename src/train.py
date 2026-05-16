import csv
import gc
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

# YOLO26 segmentation experiment.
# Requires a recent Ultralytics version:
# pip install -U ultralytics
MODEL_VARIANT = "yolo26m-seg.pt"

EPOCHS = 250
PATIENCE = 60

IMG_SIZE = 896

BATCH_CANDIDATES = [16, 12, 8, 4]

WORKERS = 8
DEVICE = 0

# For 16GB RAM, disk cache is safer.
# If RAM is high and GPU utilization drops, try "ram".
CACHE_MODE = "disk"

RUN_NAME = "voc2012_5class_yolo26m_seg_img896_upsample"

QUALITY_MIN_EPOCH = 40
QUALITY_PATIENCE = 15
MIN_MASK_MAP5095 = 0.20
MIN_MASK_RECALL = 0.35


# ============================================================
# METRIC HELPERS
# ============================================================

def get_metric(results_dict: dict, candidates: list[str]) -> float:
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
        if isinstance(names, dict):
            class_name = names.get(cls_id, str(cls_id))
        else:
            class_name = str(cls_id)

        row = {
            "class_id": cls_id,
            "class_name": class_name,
            "precision_M": float(p[pos]) if pos < len(p) else 0.0,
            "recall_M": float(r[pos]) if pos < len(r) else 0.0,
            "mAP50_M": float(ap50[pos]) if pos < len(ap50) else 0.0,
            "mAP50_95_M": (
                float(maps[cls_id])
                if cls_id < len(maps)
                else float(ap[pos]) if pos < len(ap)
                else 0.0
            ),
        }

        rows.append(row)

    return rows


def save_per_class_metrics(rows: list[dict], save_dir: Path, filename: str = "per_class_mask_metrics.csv"):
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / filename

    fieldnames = [
        "class_id",
        "class_name",
        "precision_M",
        "recall_M",
        "mAP50_M",
        "mAP50_95_M",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


def print_per_class_metrics(rows: list[dict]):
    if not rows:
        print("⚠️ Per-class mask metrics were not available.")
        return

    print("\n📊 PER-CLASS MASK PERFORMANCE")
    print("-" * 86)
    print(
        f"{'ID':>3} | "
        f"{'Class':<15} | "
        f"{'Prec(M)':>8} | "
        f"{'Recall(M)':>9} | "
        f"{'mAP50(M)':>9} | "
        f"{'mAP50-95(M)':>12}"
    )
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
            [
                "metrics/mAP50-95(M)",
                "metrics/mAP50-95_mask",
                "metrics/mAP50-95(B)",
            ],
        )

        recall = get_metric(
            metrics,
            [
                "metrics/recall(M)",
                "metrics/recall_mask",
                "metrics/recall(B)",
            ],
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
                "If YOLO26 gives an error, update Ultralytics with: pip install -U ultralytics.",
                "If YOLO26m-seg is too heavy, try yolo26s-seg.pt.",
                "If RTX 3080 OOM happens often, reduce IMG_SIZE to 768.",
                "If recall is weak, test mosaic=0.0 for one run.",
                "If bottle/chair still underperform, increase their upsample factors in preprocessing.",
                "If YOLO26 underperforms YOLOv8m, keep YOLOv8m as the baseline.",
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
        print(f"Saved recommendations to: {json_path}")
        print("!" * 72 + "\n")

        self.already_written = True


# ============================================================
# CUDA / OOM HELPERS
# ============================================================

def is_cuda_oom(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "cuda out of memory" in message
        or "outofmemoryerror" in message
        or ("cublas" in message and "alloc" in message)
    )


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_gpu_status():
    if not torch.cuda.is_available():
        print("CUDA: False")
        return

    device_name = torch.cuda.get_device_name(0)
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

    print(f"GPU             : {device_name}")
    print(f"VRAM allocated  : {allocated:.2f} GB")
    print(f"VRAM reserved   : {reserved:.2f} GB")


# ============================================================
# TRAINING
# ============================================================

def train_once(batch_size: int):
    print("\n" + "=" * 64)
    print(f"🚀 Starting YOLO26 training attempt with batch={batch_size}, imgsz={IMG_SIZE}")
    print("=" * 64)

    model = YOLO(MODEL_VARIANT)

    model.add_callback(
        "on_fit_epoch_end",
        QualityWatchCallback(
            min_epoch=QUALITY_MIN_EPOCH,
            patience=QUALITY_PATIENCE,
            min_mask_map5095=MIN_MASK_MAP5095,
            min_mask_recall=MIN_MASK_RECALL,
        ),
    )

    run_name = f"{RUN_NAME}_bs{batch_size}"

    results = model.train(
        data=str(DATA_YAML.resolve()),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMG_SIZE,
        batch=batch_size,
        workers=WORKERS,
        device=DEVICE,
        amp=True,
        cache=CACHE_MODE,

        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        cos_lr=True,
        weight_decay=5e-4,
        seed=42,
        deterministic=True,

        degrees=5.0,
        scale=0.4,
        fliplr=0.5,

        mosaic=0.1,
        mixup=0.0,
        copy_paste=0.25,
        close_mosaic=40,

        project=str(ROOT / "output" / "runs"),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
    )

    return model, results, batch_size


def train_with_dynamic_batch():
    last_error = None

    for batch_size in BATCH_CANDIDATES:
        try:
            cleanup_cuda()
            model, results, used_batch = train_once(batch_size)
            return model, results, used_batch

        except RuntimeError as e:
            last_error = e

            if is_cuda_oom(e):
                print("\n" + "!" * 72)
                print(f"⚠️ CUDA OOM with batch={batch_size}. Retrying with smaller batch...")
                print("!" * 72 + "\n")
                cleanup_cuda()
                continue

            raise

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            print("\n" + "!" * 72)
            print(f"⚠️ CUDA OOM with batch={batch_size}. Retrying with smaller batch...")
            print("!" * 72 + "\n")
            cleanup_cuda()
            continue

    raise RuntimeError(
        f"All batch candidates failed: {BATCH_CANDIDATES}. "
        f"Last error: {last_error}"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_best_checkpoint(model, results, used_batch: int):
    print("\n" + "=" * 64)
    print("🏆 TRAINING COMPLETE")
    print("=" * 64)

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
        batch=used_batch,
        device=DEVICE,
        plots=True,
    )

    results_dict = metrics.results_dict

    precision_m = get_metric(
        results_dict,
        [
            "metrics/precision(M)",
            "metrics/precision_mask",
            "metrics/precision(B)",
        ],
    )

    recall_m = get_metric(
        results_dict,
        [
            "metrics/recall(M)",
            "metrics/recall_mask",
            "metrics/recall(B)",
        ],
    )

    map50_m = get_metric(
        results_dict,
        [
            "metrics/mAP50(M)",
            "metrics/mAP50_mask",
            "metrics/mAP50(B)",
        ],
    )

    map5095_m = get_metric(
        results_dict,
        [
            "metrics/mAP50-95(M)",
            "metrics/mAP50-95_mask",
            "metrics/mAP50-95(B)",
        ],
    )

    print("\n📈 FINAL TEST SEGMENTATION PERFORMANCE:")
    print(f"   • Model         : {MODEL_VARIANT}")
    print(f"   • Used batch    : {used_batch}")
    print(f"   • Image size    : {IMG_SIZE}")
    print(f"   • Precision (M): {precision_m:.4f}")
    print(f"   • Recall (M)   : {recall_m:.4f}")
    print(f"   • mAP50 (M)    : {map50_m:.4f}")
    print(f"   • mAP50-95 (M) : {map5095_m:.4f}")

    per_class_rows = extract_per_class_mask_metrics(metrics)
    print_per_class_metrics(per_class_rows)

    csv_path = save_per_class_metrics(per_class_rows, save_dir)
    print(f"\n✅ Per-class mask metrics saved to: {csv_path}")

    print(f"\n📁 Results saved in: {save_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not DATA_YAML.exists():
        print("❌ Error: data.yaml not found.")
        print(f"   Expected location: {DATA_YAML.resolve()}")
        print("   Please run preprocess_final.py first.")
        return

    print("=" * 64)
    print("🚀 YOLO26-Seg 5-Class Dynamic Batch Training")
    print("=" * 64)
    print(f"Data YAML        : {DATA_YAML.resolve()}")
    print(f"Model            : {MODEL_VARIANT}")
    print(f"Epochs           : {EPOCHS}")
    print(f"Patience         : {PATIENCE}")
    print(f"Image size       : {IMG_SIZE}")
    print(f"Batch candidates : {BATCH_CANDIDATES}")
    print(f"Cache mode       : {CACHE_MODE}")
    print(f"Workers          : {WORKERS}")
    print(f"Device           : {DEVICE}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    print_gpu_status()
    print("=" * 64)

    model, results, used_batch = train_with_dynamic_batch()

    evaluate_best_checkpoint(
        model=model,
        results=results,
        used_batch=used_batch,
    )


if __name__ == "__main__":
    main()