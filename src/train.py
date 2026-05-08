# src/train.py
from ultralytics import YOLO
from pathlib import Path

def main():
    data_yaml = "output/data.yaml"
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"❌ {data_yaml} not found.")

    print("🤖 Loading YOLOv8s-seg model (Small version)...")
    model = YOLO("yolov8s-seg.pt") # ✅ Reverted to Small for 4GB VRAM stability

    print(" Starting training (100 epochs)...")
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=4,           # ✅ Lower batch for 4GB VRAM
        workers=2,         # ✅ Low workers to prevent HDD thrashing
        device=0,
        project="output/runs",
        name="voc2012_stable",
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        lr0=0.001,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,
        cache='ram'        # ✅ Cache images in RAM to bypass HDD bottleneck
    )

    print("\n✅ Training finished.")
    val_metrics = model.val(data=data_yaml, plots=True)
    metrics = val_metrics.results_dict
    print(f"   • mAP50-95 (M): {metrics.get('metrics/mAP50-95(M)', 0):.4f}")

if __name__ == "__main__":
    main()