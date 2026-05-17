import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "output" / "dataset" / "data.yaml"

PREVIOUS_RUN_DIR = (
    ROOT
    / "output"
    / "runs"
    / "voc2012_5class_yolo26m_seg_img896_upsample_bs16"
)

BEST_WEIGHTS = PREVIOUS_RUN_DIR / "weights" / "best.pt"

# ============================================================
# RESUME / FINE-TUNE CONFIG
# ============================================================

IMG_SIZE = 896
BATCH_CANDIDATES = [16, 12, 8, 4]

EPOCHS = 120
PATIENCE = 80

WORKERS = 8
DEVICE = 0
CACHE_MODE = "disk"

RUN_NAME = "voc2012_5class_yolo26m_finetune_from_best"

QUALITY_MIN_EPOCH = 30
QUALITY_PATIENCE = 20
MIN_MASK_MAP5095 = 0.20
MIN_MASK_RECALL = 0.35


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def check_environment():
    print("=" * 70)
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 70)

    print(f"Torch       : {torch.__version__}")
    print(f"Torchvision : {torchvision.__version__}")
    print(f"CUDA        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    try:
        _ = torchvision.ops.nms
        print("✅ torchvision.ops.nms is available.")
    except AttributeError:
        print("❌ torchvision.ops.nms is missing.")
        print("Run:")
        print("pip uninstall torch torchvision torchaudio -y")
        print(
            "pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"❌ data.yaml not found: {DATA_YAML}")
        sys.exit(1)

    if not BEST_WEIGHTS.exists():
        print(f"❌ best.pt not found: {BEST_WEIGHTS}")
        sys.exit(1)

    print(f"✅ data.yaml   : {DATA_YAML}")
    print(f"✅ best.pt     : {BEST_WEIGHTS}")
    print("=" * 70)


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
        class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

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


def save_per_class_metrics(rows: list[dict], save_dir: Path):
    csv_path = save_dir / "per_class_mask_metrics.csv"

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
# CUDA HELPERS
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

    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM allocated  : {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"VRAM reserved   : {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")


# ============================================================
# TRAINING
# ============================================================

def train_once(batch_size: int):
    print("\n" + "=" * 70)
    print(f"🚀 Fine-tuning FROM best.pt")
    print(f"📌 Weights: {BEST_WEIGHTS}")
    print(f"📌 batch={batch_size}, imgsz={IMG_SIZE}")
    print("=" * 70)

    # IMPORTANT:
    # This line loads the old trained best.pt, NOT yolo26m-seg.pt
    model = YOLO(str(BEST_WEIGHTS))

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

        # Lower LR because this is fine-tuning from trained checkpoint.
        optimizer="AdamW",
        lr0=5e-5,
        lrf=0.01,
        cos_lr=True,
        weight_decay=5e-4,
        seed=42,
        deterministic=True,

        # Gentler augmentation than original training.
        degrees=5.0,
        scale=0.35,
        fliplr=0.5,

        mosaic=0.05,
        mixup=0.0,
        copy_paste=0.20,
        close_mosaic=50,

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
                print(f"⚠️ CUDA OOM with batch={batch_size}. Trying smaller batch...")
                cleanup_cuda()
                continue

            raise

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            print(f"⚠️ CUDA OOM with batch={batch_size}. Trying smaller batch...")
            cleanup_cuda()
            continue

    raise RuntimeError(
        f"All batch candidates failed: {BATCH_CANDIDATES}. Last error: {last_error}"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_best_checkpoint(model, results, used_batch: int):
    print("\n" + "=" * 70)
    print("🏆 FINE-TUNING COMPLETE")
    print("=" * 70)

    save_dir = Path(results.save_dir)
    new_best_weights = save_dir / "weights" / "best.pt"

    if new_best_weights.exists():
        print(f"✅ Loading new best checkpoint: {new_best_weights}")
        eval_model = YOLO(str(new_best_weights))
    else:
        print("⚠️ New best.pt not found. Evaluating current model.")
        eval_model = model

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
    print(f"   • Source weights : {BEST_WEIGHTS}")
    print(f"   • Used batch     : {used_batch}")
    print(f"   • Image size     : {IMG_SIZE}")
    print(f"   • Precision (M) : {precision_m:.4f}")
    print(f"   • Recall (M)    : {recall_m:.4f}")
    print(f"   • mAP50 (M)     : {map50_m:.4f}")
    print(f"   • mAP50-95 (M)  : {map5095_m:.4f}")

    per_class_rows = extract_per_class_mask_metrics(metrics)
    print_per_class_metrics(per_class_rows)

    csv_path = save_per_class_metrics(per_class_rows, save_dir)
    print(f"\n✅ Per-class mask metrics saved to: {csv_path}")
    print(f"📁 Results saved in: {save_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    check_environment()
    print_gpu_status()

    print("\n" + "=" * 70)
    print("🚀 YOLO26 Fine-tune from previous best.pt")
    print("=" * 70)
    print(f"Data YAML        : {DATA_YAML.resolve()}")
    print(f"Previous run dir : {PREVIOUS_RUN_DIR}")
    print(f"Best weights     : {BEST_WEIGHTS}")
    print(f"Epochs           : {EPOCHS}")
    print(f"Patience         : {PATIENCE}")
    print(f"Image size       : {IMG_SIZE}")
    print(f"Batch candidates : {BATCH_CANDIDATES}")
    print(f"Cache mode       : {CACHE_MODE}")
    print(f"Workers          : {WORKERS}")
    print("=" * 70)

    model, results, used_batch = train_with_dynamic_batch()
    evaluate_best_checkpoint(model, results, used_batch)


if __name__ == "__main__":
    main()