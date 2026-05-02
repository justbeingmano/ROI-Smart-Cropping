# src/preprocess.py
# Preprocessing pipeline for PASCAL VOC 2012 -> YOLOv8-Seg format
# Processes a sample of 50 images for quick testing

import os
import cv2
import numpy as np
import random
import yaml
from pathlib import Path
from tqdm import tqdm


def load_image_paths(base_path: Path, split_file: str, sample_size: int = 50) -> list[str]:
    """
    Load image names from a split file (e.g., val.txt) and return a random sample.
    
    Args:
        base_path: Path to VOC2012 root directory
        split_file: Name of split file (e.g., 'val.txt')
        sample_size: Number of images to sample
    
    Returns:
        List of image names (without extension)
    """
    split_path = base_path / 'ImageSets' / 'Segmentation' / split_file
    
    with open(split_path, 'r') as f:
        all_names = [line.strip() for line in f if line.strip()]
    
    return random.sample(all_names, min(sample_size, len(all_names)))


def preprocess_image(img: np.ndarray, target_size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Resize image and apply CLAHE histogram normalization.
    
    Args:
        img: Input image in BGR format
        target_size: Target dimensions (width, height)
    
    Returns:
        Preprocessed image in BGR format
    """
    # Resize with linear interpolation
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to LAB color space for CLAHE
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge back and convert to BGR
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def mask_to_yolo_polygons(mask: np.ndarray, img_w: int, img_h: int) -> list[str]:
    """
    Convert instance mask to YOLO segmentation format (polygon coordinates).
    
    Args:
        mask: Instance mask (single channel, integer labels)
        img_w: Image width for normalization
        img_h: Image height for normalization
    
    Returns:
        List of strings, each representing one instance in YOLO format
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    
    lines = []
    for label in unique_labels:
        # Create binary mask for this instance
        binary = (mask == label).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            continue
        
        # Take the largest contour (filter noise)
        cnt = max(contours, key=cv2.contourArea)
        
        # Normalize coordinates to [0, 1]
        pts = cnt.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        
        # Format: class_id x1 y1 x2 y2 ...
        class_id = 0  # Use 0 for single-class testing; update for multi-class
        coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts)
        lines.append(f"{class_id} {coords}")
    
    return lines


def run_preprocessing(
    base_path: Path,
    output_path: Path,
    sample_names: list[str],
    target_size: tuple[int, int] = (640, 640)
):
    """
    Main preprocessing function: resize, normalize, augment, convert masks.
    
    Args:
        base_path: Path to VOC2012 root
        output_path: Path to save preprocessed data
        sample_names: List of image names to process
        target_size: Target image dimensions
    """
    # Create output directories
    for subdir in ['images', 'masks', 'labels']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    img_dir = base_path / 'JPEGImages'
    mask_dir = base_path / 'SegmentationObject'
    
    for name in tqdm(sample_names, desc="Preprocessing"):
        # Load image and mask
        img_path = img_dir / f"{name}.jpg"
        mask_path = mask_dir / f"{name}.png"
        
        if not img_path.exists() or not mask_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        # Preprocess image
        img_processed = preprocess_image(img, target_size)
        
        # Resize mask with nearest-neighbor to preserve label values
        mask_resized = cv2.resize(
            mask, 
            target_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Save processed image and mask
        cv2.imwrite(str(output_path / 'images' / f"{name}.jpg"), img_processed)
        cv2.imwrite(str(output_path / 'masks' / f"{name}.png"), mask_resized)
        
        # Convert mask to YOLO polygons and save label
        polygons = mask_to_yolo_polygons(mask_resized, target_size[0], target_size[1])
        if polygons:
            with open(output_path / 'labels' / f"{name}.txt", 'w') as f:
                f.write('\n'.join(polygons))


def create_data_yaml(output_path: Path, class_names: dict[int, str]):
    """
    Create YOLO data.yaml configuration file.
    
    Args:
        output_path: Path to preprocessed data root
        class_names: Dictionary mapping class_id to class_name
    """
    config = {
        'path': str(output_path.absolute()),
        'train': 'images',
        'val': 'images',
        'names': class_names
    }
    
    yaml_path = output_path.parent / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[OK] Created data.yaml at: {yaml_path}")


def main():
    # Configuration
    BASE_PATH = Path("project-image/VOC2012_train_val/VOC2012_train_val")
    OUTPUT_PATH = Path("output/preprocessed_50")
    SAMPLE_SIZE = 500
    SPLIT_FILE = "val.txt"  # Use 'val.txt' for small test; 'train.txt' for full
    
    print(f"[INFO] Loading data from: {BASE_PATH}")
    
    # Load sample image names
    sample_names = load_image_paths(BASE_PATH, SPLIT_FILE, SAMPLE_SIZE)
    print(f"[INFO] Selected {len(sample_names)} images for preprocessing")
    
    # Run preprocessing
    run_preprocessing(BASE_PATH, OUTPUT_PATH, sample_names)
    
    # Create data.yaml (single-class for testing)
    class_names = {0: "object"}  # Update for multi-class: {0: "person", 1: "car", ...}
    create_data_yaml(OUTPUT_PATH, class_names)
    
    print(f"[OK] Preprocessing complete. Output: {OUTPUT_PATH}")
    print(f"[Summary]")
    print(f"   - Images: {len(list((OUTPUT_PATH / 'images').glob('*.jpg')))}")
    print(f"   - Masks: {len(list((OUTPUT_PATH / 'masks').glob('*.png')))}")
    print(f"   - Labels: {len(list((OUTPUT_PATH / 'labels').glob('*.txt')))}")


if __name__ == "__main__":
    main()            