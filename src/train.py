# src/train.py
from ultralytics import YOLO
from pathlib import Path
import torch
import gc

def train_with_auto_batch(data_yaml: str, max_batch: int = 8):
    """
    Attempts to train with the highest possible batch size for the available VRAM.
    Starts at max_batch and reduces it if CUDA OOM occurs.
    """
    batch_sizes = [max_batch, 6, 4] # Sequence of batches to try
    
    for batch in batch_sizes:
        print(f"\n🚀 Attempting training with Batch Size: {batch}...")
        
        # Clear GPU memory cache before each attempt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        try:
            model = YOLO("yolov8s-seg.pt")
            
            results = model.train(
                data=data_yaml,
                epochs=100,
                imgsz=640,          # YOLO handles letterboxing internally
                batch=batch,        # Dynamic batch size
                workers=4,          # Balanced workers for SSD
                device=0,
                project="output/runs",
                name=f"voc2012_seg_best",
                patience=20,
                save=True,
                plots=True,
                verbose=True,
                lr0=0.001,          # Stable learning rate
                mosaic=0.8,         # Good balance for segmentation
                mixup=0.1,
                copy_paste=0.1,
                close_mosaic=10,    # Refine masks in last 10 epochs
                cache='ram'         # Critical for speed on NVMe SSD
            )
            
            # If we get here, training started successfully!
            print(f"\n✅ SUCCESS! Training started with Batch Size: {batch}")
            
            # Run final validation on the best model
            print("\n📊 Running final validation...")
            val_metrics = model.val(data=data_yaml, plots=True)
            metrics = val_metrics.results_dict
            
            print("\n🏆 FINAL SEGMENTATION METRICS:")
            print(f"   • Precision (B): {metrics.get('metrics/precision(B)', 0):.4f}")
            print(f"   • Recall (B)   : {metrics.get('metrics/recall(B)', 0):.4f}")
            print(f"   • mAP50-95 (M) : {metrics.get('metrics/mAP50-95(M)', 0):.4f}") # Most important for Segmentation
            print(f"\n📁 Results saved in: output/runs/voc2012_seg_best/")
            
            return # Exit function after success

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"❌ FAILED: CUDA Out of Memory with Batch={batch}. Trying smaller batch...")
                continue # Try the next smaller batch
            else:
                raise e # If it's a different error, stop and show it

    print("❌ CRITICAL ERROR: Could not start training even with Batch=2. Check your GPU drivers or hardware.")

def main():
    data_yaml = "output/data.yaml"
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"❌ {data_yaml} not found. Run preprocess.py first.")

    print("🤖 Loading YOLOv8s-seg model...")
    # Start with batch=8, let the function handle the rest
    train_with_auto_batch(data_yaml, max_batch=8)

if __name__ == "__main__":
    main()