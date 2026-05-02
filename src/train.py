# src/train.py
# Training with Mosaic augmentation + custom learning rate

from ultralytics import YOLO
from pathlib import Path

def main():
    data_yaml = "output/data.yaml"
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"❌ {data_yaml} not found. Run preprocess.py first.")

    print("🔄 Loading YOLOv8n-seg model...")
    model = YOLO("yolov8n-seg.pt")

    print("📉 Starting training with Mosaic augmentation (50 epochs)...")
    
    # Custom training arguments
    results = model.train(
        data=data_yaml,
        epochs=50,              # Increased from 10 to 50 epochs
        imgsz=640,
        batch=8,
        device=0,               # "cpu" if no GPU
        project="output/runs",
        name="500_images_mosaic",
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        
        # 🎯 Custom Hyperparameters
        lr0=0.001,              # Initial learning rate (changed from default 0.01)
        lrf=0.01,               # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 🎨 Augmentation Settings
        hsv_h=0.015,            # Hue augmentation
        hsv_s=0.7,              # Saturation augmentation
        hsv_v=0.4,              # Value augmentation
        degrees=0.0,            # Rotation degrees
        translate=0.1,          # Translation
        scale=0.5,              # Scale augmentation
        shear=0.0,              # Shear
        perspective=0.0,        # Perspective
        flipud=0.5,             # Vertical flip probability
        fliplr=0.5,             # Horizontal flip probability
        mosaic=1.0,             # 🌟 Mosaic augmentation probability (1.0 = always)
        mixup=0.0,              # Mixup probability
        copy_paste=0.0,         # Copy-paste probability
    )

    print("\n✅ Training finished. Running validation...")
    val_metrics = model.val(data=data_yaml, save_json=False, plots=True)

    # Display metrics
    metrics = val_metrics.results_dict
    print("\n📊 FINAL EVALUATION METRICS:")
    print(f"   • Precision (Box) : {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"   • Recall (Box)    : {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"   • F1-Score (Box)  : {metrics.get('metrics/F1(B)', 0):.4f}")
    print(f"   • mAP50-95 (Mask) : {metrics.get('metrics/mAP50-95(M)', 0):.4f}")
    print(f"   • mAP50 (Mask)    : {metrics.get('metrics/mAP50(M)', 0):.4f}")
    print(f"\n📁 Results saved in: output/runs/500_images_mosaic/")
    print("🔍 Check: confusion_matrix.png, results.png, val_batch0_pred.jpg")

if __name__ == "__main__":
    main()