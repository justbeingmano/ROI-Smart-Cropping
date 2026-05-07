# src/preprocess.py
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# VOC2012 Classes Mapping (Index -> Name)
# Note: Index 0 is background in VOC masks, so we shift indices by -1 for YOLO (0-19)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def preprocess_image(img: np.ndarray, target_size=(640, 640)) -> np.ndarray:
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def mask_to_yolo_polygons(mask: np.ndarray, img_w: int, img_h: int) -> list[str]:
    # Ensure mask is single channel uint8
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    unique_labels = np.unique(mask)
    # Exclude background (0) and void (255)
    unique_labels = unique_labels[(unique_labels != 0) & (unique_labels != 255)]
    
    lines = []
    for label in unique_labels:
        # Create binary mask for this specific instance/class
        binary = ((mask == label) * 255).astype(np.uint8)
        
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            continue
            
        if not contours: 
            continue
        
        # Take the largest contour to filter noise
        cnt = max(contours, key=cv2.contourArea)
        
        # Normalize coordinates to [0, 1]
        pts = cnt.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        
        # IMPORTANT: VOC labels start from 1. YOLO expects 0-indexed classes.
        # So we subtract 1. If label is 1 (aeroplane), class_id becomes 0.
        class_id = int(label) - 1
        
        # Safety check: ensure class_id is within 0-19 range
        if 0 <= class_id < 20:
            coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts)
            lines.append(f"{class_id} {coords}")
        
    return lines

def process_split(split_name: str, base_path: Path, output_path: Path, target_size=(640, 640)):
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
        
        img_out = preprocess_image(img, target_size)
        mask_out = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(str(out_img / f"{name}.jpg"), img_out)
        cv2.imwrite(str(out_mask / f"{name}.png"), mask_out)
        
        try:
            polygons = mask_to_yolo_polygons(mask_out, target_size[0], target_size[1])
            if polygons:
                with open(out_lbl / f"{name}.txt", 'w') as f:
                    f.write('\n'.join(polygons))
            count += 1
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            continue
            
    return count

def create_data_yaml(output_path: Path):
    # Create names dictionary for all 20 classes
    class_names = {i: name for i, name in enumerate(VOC_CLASSES)}
    
    config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names
    }
    yaml_path = output_path.parent / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Created data.yaml with {len(class_names)} classes at: {yaml_path}")

def main():
    BASE_PATH = Path("project-image/VOC2012_train_val/VOC2012_train_val")
    OUTPUT_PATH = Path("output/preprocessed_full_multiclass")
    
    print(f"📂 Base path: {BASE_PATH}")
    print("⏳ Processing train split...")
    train_count = process_split('train', BASE_PATH, OUTPUT_PATH)
    
    print("⏳ Processing val split...")
    val_count = process_split('val', BASE_PATH, OUTPUT_PATH)
    
    create_data_yaml(OUTPUT_PATH)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   🖼️ Train images: {train_count}")
    print(f"   🖼️ Val images:   {val_count}")
    print(f"   📁 Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()