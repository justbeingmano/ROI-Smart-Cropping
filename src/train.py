# src/train.py
from ultralytics import YOLO
from pathlib import Path

def main():
    data_yaml = "output/data.yaml"
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"❌ {data_yaml} not found.")

    print("🤖 Loading YOLOv8m-seg model (Medium version)...")
    model = YOLO("yolov8m-seg.pt") # ✅ Upgraded to Medium for better capacity

    print(" Starting training (100 epochs)...")
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        project="output/runs",
        name="voc2012_optimized",
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        lr0=0.001,
        mosaic=0.8,        # ✅ Reduced from 1.0 for better boundary focus
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,
        cache=False
    )

    print("\n✅ Training finished.")
    val_metrics = model.val(data=data_yaml, plots=True)
    metrics = val_metrics.results_dict
    print(f"   • mAP50-95 (M): {metrics.get('metrics/mAP50-95(M)', 0):.4f}")

if __name__ == "__main__":
    main()