# src/preprocess.py
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

def preprocess_image(img: np.ndarray, max_dim=640) -> np.ndarray:
    """Downscale large images to max_dim while preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def mask_to_yolo_polygons(mask: np.ndarray, img_w: int, img_h: int) -> list[str]:
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[(unique_labels != 0) & (unique_labels != 255)]
    
    lines = []
    for label in unique_labels:
        binary = ((mask == label) * 255).astype(np.uint8)
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception: continue
            
        if not contours: continue
        
        # Process ALL contours to handle occluded/split objects
        for cnt in contours:
            if cv2.contourArea(cnt) < 10: continue # Filter small noise
            pts = cnt.reshape(-1, 2).astype(np.float32)
            pts[:, 0] /= img_w
            pts[:, 1] /= img_h
            
            # ✅ FIX: Assign all objects to class 0 for stable training
            class_id = 0
            coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts)
            lines.append(f"{class_id} {coords}")
        
    return lines

def process_split(split_name: str, base_path: Path, output_path: Path):
    split_file = base_path / 'ImageSets' / 'Segmentation' / f"{split_name}.txt"
    if not split_file.exists(): return 0
    
    with open(split_file, 'r') as f:
        img_names = [line.strip() for line in f if line.strip()]
    
    img_dir = base_path / 'JPEGImages'
    mask_dir = base_path / 'SegmentationObject'
    
    out_img = output_path / 'images' / split_name
    out_mask = output_path / 'masks' / split_name
    out_lbl = output_path / 'labels' / split_name
    for d in [out_img, out_mask, out_lbl]: d.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for name in tqdm(img_names, desc=f"Processing {split_name}"):
        img_path = img_dir / f"{name}.jpg"
        mask_path = mask_dir / f"{name}.png"
        if not img_path.exists() or not mask_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None: continue
        
        img_out = preprocess_image(img)
        new_h, new_w = img_out.shape[:2]
        mask_out = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(str(out_img / f"{name}.jpg"), img_out)
        cv2.imwrite(str(out_mask / f"{name}.png"), mask_out)
        
        try:
            polygons = mask_to_yolo_polygons(mask_out, new_w, new_h)
            if polygons:
                with open(out_lbl / f"{name}.txt", 'w') as f:
                    f.write('\n'.join(polygons))
            count += 1
        except Exception: continue
            
    return count

def create_data_yaml(output_path: Path):
    config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: "object"} # ✅ Single class config
    }
    yaml_path = output_path.parent / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Created data.yaml at: {yaml_path}")

def main():
    BASE_PATH = Path("project-image/VOC2012_train_val/VOC2012_train_val")
    OUTPUT_PATH = Path("output/preprocessed_full")
    
    print("⏳ Processing train split...")
    train_count = process_split('train', BASE_PATH, OUTPUT_PATH)
    print("⏳ Processing val split...")
    val_count = process_split('val', BASE_PATH, OUTPUT_PATH)
    
    create_data_yaml(OUTPUT_PATH)
    print(f"\n✅ Done! Train: {train_count} | Val: {val_count}")

if __name__ == "__main__":
    main()