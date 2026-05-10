# src/preprocess.py
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

def mask_to_yolo_polygons(mask: np.ndarray, img_w: int, img_h: int) -> list[str]:
    """
    Convert mask to YOLO polygon format. 
    Keeps ALL contours for occluded objects to maximize segmentation recall.
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    unique_labels = np.unique(mask)
    # Exclude background (0) and void (255)
    unique_labels = unique_labels[(unique_labels != 0) & (unique_labels != 255)]
    
    lines = []
    for label in unique_labels:
        binary = ((mask == label) * 255).astype(np.uint8)
        try:
            # RETR_EXTERNAL is best for getting the outer boundary of the object
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            continue
            
        for cnt in contours:
            # Filter very small noise (less than 10 pixels area)
            if cv2.contourArea(cnt) < 10:
                continue
                
            pts = cnt.reshape(-1, 2).astype(np.float32)
            # Normalize coordinates to [0, 1] based on ORIGINAL image size
            pts[:, 0] /= img_w
            pts[:, 1] /= img_h
            
            class_id = 0  # Single-class for stable ROI extraction
            coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts)
            lines.append(f"{class_id} {coords}")
        
    return lines

def process_split(split_name: str, base_path: Path, output_path: Path):
    split_file = base_path / 'ImageSets' / 'Segmentation' / f"{split_name}.txt"
    if not split_file.exists(): 
        print(f"⚠️ {split_name}.txt not found. Skipping.")
        return 0
    
    with open(split_file, 'r') as f:
        img_names = [line.strip() for line in f if line.strip()]
    
    img_dir = base_path / 'JPEGImages'
    mask_dir = base_path / 'SegmentationObject'
    
    out_img = output_path / 'images' / split_name
    out_mask = output_path / 'masks' / split_name
    out_lbl = output_path / 'labels' / split_name
    for d in [out_img, out_mask, out_lbl]: 
        d.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for name in tqdm(img_names, desc=f"Processing {split_name}"):
        img_path = img_dir / f"{name}.jpg"
        mask_path = mask_dir / f"{name}.png"
        
        if not img_path.exists() or not mask_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        
        if img is None or mask is None: 
            continue
        
        # ✅ SAVE ORIGINAL IMAGES & MASKS: Let YOLO handle letterboxing/resizing during training
        # This preserves sub-pixel precision for better segmentation masks
        cv2.imwrite(str(out_img / f"{name}.jpg"), img)
        cv2.imwrite(str(out_mask / f"{name}.png"), mask)
        
        # Extract polygons based on original dimensions
        h, w = img.shape[:2]
        try:
            polygons = mask_to_yolo_polygons(mask, w, h)
            if polygons:
                with open(out_lbl / f"{name}.txt", 'w') as f:
                    f.write('\n'.join(polygons))
            count += 1
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            continue
            
    return count

def create_data_yaml(output_path: Path):
    config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: "object"}
    }
    yaml_path = output_path.parent / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Created data.yaml at: {yaml_path}")

def main():
    # Adjust this path to match your actual folder structure
    BASE_PATH = Path("project-image/VOC2012_train_val/VOC2012_train_val")
    OUTPUT_PATH = Path("output/preprocessed_full")
    
    print(f"📂 Base path: {BASE_PATH}")
    print("⏳ Processing train split...")
    train_count = process_split('train', BASE_PATH, OUTPUT_PATH)
    
    print("⏳ Processing val split...")
    val_count = process_split('val', BASE_PATH, OUTPUT_PATH)
    
    create_data_yaml(OUTPUT_PATH)
    
    print(f"\n✅ Done! Train: {train_count} | Val: {val_count}")

if __name__ == "__main__":
    main()