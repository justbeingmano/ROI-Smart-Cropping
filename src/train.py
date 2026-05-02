# src/train.py
# Training script for YOLOv8-Seg on preprocessed VOC2012 sample

from ultralytics import YOLO
from pathlib import Path


def train_model(data_yaml_path: str, epochs: int = 10, imgsz: int = 640):
    """
    Train YOLOv8-Seg model on preprocessed dataset.
    
    Args:
        data_yaml_path: Path to data.yaml configuration file
        epochs: Number of training epochs
        imgsz: Image size for training
    """
    # Load pre-trained YOLOv8 segmentation model (nano version for speed)
    model = YOLO('yolov8n-seg.pt')
    
    # Train the model
    results = model.train(data=data_yaml_path, epochs=10, device='cpu',         # Use GPU 0; set to 'cpu' if no GPU
        save=True,
        project='output/runs',
        name='voc2012_sample'
    )
    
    print(f"✅ Training complete. Results saved to: output/runs/voc2012_sample")
    return results


if __name__ == "__main__":
    # Adjust path to your data.yaml
    data_yaml = Path("output/data.yaml").resolve()
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")
    
    train_model(str(data_yaml), epochs=10)