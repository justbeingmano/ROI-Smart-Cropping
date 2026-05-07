# src/train.py
from ultralytics import YOLO
from pathlib import Path

def main():
    data_yaml = "output/data.yaml"
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"❌ {data_yaml} not found. Run preprocess.py first.")

    print("🤖 Loading YOLOv8n-seg model...")
    model = YOLO("yolov8n-seg.pt")

    print("🚀 Starting full dataset training (50 epochs)...")
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project="output/runs",
        name="voc2012_full",
        patience=15,
        save=True,
        plots=True,
        verbose=True,
        lr0=0.001,
        mosaic=1.0,
        close_mosaic=10,
        cache=False
    )

    print("\n✅ Training finished. Running validation...")
    val_metrics = model.val(data=data_yaml, plots=True)
    metrics = val_metrics.results_dict

    print("\n📊 FINAL METRICS:")
    print(f"   • Precision (B): {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"   • Recall (B)   : {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"   • F1-Score (B) : {metrics.get('metrics/F1(B)', 0):.4f}")
    print(f"   • mAP50-95 (M) : {metrics.get('metrics/mAP50-95(M)', 0):.4f}")
    print(f"\n📁 Results: output/runs/voc2012_full/")

if __name__ == "__main__":
    main()