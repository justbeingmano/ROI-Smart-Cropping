# src/visualize.py
# Visualizes preprocessed VOC2012 data: images, masks, and YOLO polygon overlays

import os
import cv2
import numpy as np
import base64
import random
from pathlib import Path


OUTPUT_PATH = Path("output/preprocessed_50")
HTML_OUT   = Path("output/visualize.html")
NUM_SAMPLES = 12  # how many images to show in the viewer


def img_to_b64(img: np.ndarray, ext: str = ".jpg") -> str:
    """Encode a numpy image array to a base64 data-URI string."""
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return ""
    b64 = base64.b64encode(buf).decode("utf-8")
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def draw_yolo_overlay(img: np.ndarray, label_path: Path) -> np.ndarray:
    """Draw YOLO segmentation polygons on top of the image."""
    overlay = img.copy()
    h, w = img.shape[:2]

    if not label_path.exists():
        return overlay

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    colors = [
        (255, 80,  80),
        (80,  200, 80),
        (80,  80,  255),
        (255, 200, 0),
        (0,   200, 255),
        (200, 0,   255),
    ]

    for i, line in enumerate(lines):
        parts = line.split()
        coords = list(map(float, parts[1:]))
        pts = np.array(
            [[int(coords[j] * w), int(coords[j + 1] * h)]
             for j in range(0, len(coords) - 1, 2)],
            dtype=np.int32,
        )
        if len(pts) < 3:
            continue
        color = colors[i % len(colors)]
        # Semi-transparent fill
        mask_layer = overlay.copy()
        cv2.fillPoly(mask_layer, [pts], color)
        cv2.addWeighted(mask_layer, 0.35, overlay, 0.65, 0, overlay)
        # Border
        cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)

    return overlay


def colorize_mask(mask_gray: np.ndarray) -> np.ndarray:
    """Apply a color map to a grayscale segmentation mask."""
    norm = cv2.normalize(mask_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    # Black out true background (pixel value 0)
    bg = mask_gray == 0
    colored[bg] = [20, 20, 20]
    return colored


def build_html(samples: list[dict]) -> str:
    cards = ""
    for s in samples:
        poly_count = s["poly_count"]
        name = s["name"]
        cards += f"""
        <div class="card">
          <div class="card-header">
            <span class="img-name">{name}</span>
            <span class="badge">{poly_count} object{'s' if poly_count != 1 else ''}</span>
          </div>
          <div class="panels">
            <div class="panel">
              <img src="{s['orig']}" alt="Preprocessed Image">
              <div class="label">Image (640x640, CLAHE)</div>
            </div>
            <div class="panel">
              <img src="{s['mask']}" alt="Mask">
              <div class="label">Segmentation Mask</div>
            </div>
            <div class="panel">
              <img src="{s['overlay']}" alt="YOLO Overlay">
              <div class="label">YOLO Polygon Overlay</div>
            </div>
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VOC2012 Preprocessed Data Viewer</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Inter', sans-serif;
      background: #0d0f14;
      color: #e2e8f0;
      min-height: 100vh;
    }}

    header {{
      background: linear-gradient(135deg, #1a1d2e 0%, #12151f 100%);
      border-bottom: 1px solid #2d3748;
      padding: 28px 40px;
      display: flex;
      align-items: center;
      gap: 20px;
    }}

    .logo {{
      width: 44px; height: 44px;
      background: linear-gradient(135deg, #6366f1, #a855f7);
      border-radius: 12px;
      display: grid; place-items: center;
      font-size: 22px;
      flex-shrink: 0;
    }}

    header h1 {{
      font-size: 1.5rem; font-weight: 700;
      background: linear-gradient(90deg, #e2e8f0, #a5b4fc);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}

    header p {{
      font-size: 0.82rem; color: #718096; margin-top: 2px;
    }}

    .stats-bar {{
      background: #12151f;
      border-bottom: 1px solid #1e2535;
      padding: 14px 40px;
      display: flex; gap: 32px; flex-wrap: wrap;
    }}

    .stat {{
      display: flex; align-items: center; gap: 8px;
      font-size: 0.82rem; color: #94a3b8;
    }}

    .stat strong {{ color: #e2e8f0; font-weight: 600; }}
    .stat-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
    .dot-purple {{ background: #a855f7; }}
    .dot-blue   {{ background: #6366f1; }}
    .dot-green  {{ background: #34d399; }}

    main {{
      padding: 32px 40px;
      max-width: 1600px;
      margin: 0 auto;
    }}

    .section-title {{
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #6366f1;
      margin-bottom: 20px;
    }}

    .grid {{
      display: grid;
      gap: 24px;
      grid-template-columns: 1fr;
    }}

    .card {{
      background: #12151f;
      border: 1px solid #1e2535;
      border-radius: 16px;
      overflow: hidden;
      transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
    }}

    .card:hover {{
      border-color: #4f4fa8;
      transform: translateY(-2px);
      box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }}

    .card-header {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 20px;
      background: #0d0f14;
      border-bottom: 1px solid #1e2535;
    }}

    .img-name {{
      font-size: 0.85rem; font-weight: 600;
      color: #a5b4fc; font-family: monospace;
    }}

    .badge {{
      font-size: 0.72rem; font-weight: 600;
      background: rgba(168, 85, 247, 0.15);
      color: #c084fc;
      border: 1px solid rgba(168, 85, 247, 0.3);
      padding: 2px 10px; border-radius: 999px;
    }}

    .panels {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0;
    }}

    .panel {{
      position: relative;
      border-right: 1px solid #1e2535;
    }}

    .panel:last-child {{ border-right: none; }}

    .panel img {{
      width: 100%;
      display: block;
      aspect-ratio: 1;
      object-fit: cover;
      transition: opacity 0.2s;
    }}

    .panel img:hover {{ opacity: 0.9; cursor: zoom-in; }}

    .label {{
      position: absolute; bottom: 0; left: 0; right: 0;
      background: linear-gradient(0deg, rgba(13,15,20,0.9) 0%, transparent 100%);
      padding: 20px 12px 10px;
      font-size: 0.72rem; color: #94a3b8; font-weight: 500;
      letter-spacing: 0.04em;
    }}

    /* Lightbox */
    #lightbox {{
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,0.92);
      z-index: 1000;
      place-items: center;
      cursor: zoom-out;
    }}

    #lightbox.active {{ display: grid; }}

    #lightbox img {{
      max-width: 90vw; max-height: 90vh;
      border-radius: 12px;
      box-shadow: 0 0 60px rgba(0,0,0,0.8);
    }}

    @media (max-width: 700px) {{
      .panels {{ grid-template-columns: 1fr; }}
      .panel {{ border-right: none; border-bottom: 1px solid #1e2535; }}
      header, .stats-bar, main {{ padding-left: 16px; padding-right: 16px; }}
    }}
  </style>
</head>
<body>

<header>
  <div class="logo">🔍</div>
  <div>
    <h1>VOC2012 Preprocessed Data Viewer</h1>
    <p>Showing {len(samples)} randomly sampled images &mdash; 640&times;640, CLAHE normalised, YOLO-Seg labels</p>
  </div>
</header>

<div class="stats-bar">
  <div class="stat"><span class="stat-dot dot-purple"></span>Dataset: <strong>PASCAL VOC 2012</strong></div>
  <div class="stat"><span class="stat-dot dot-blue"></span>Resolution: <strong>640 × 640</strong></div>
  <div class="stat"><span class="stat-dot dot-green"></span>Normalisation: <strong>CLAHE (L-channel)</strong></div>
  <div class="stat"><span class="stat-dot dot-purple"></span>Format: <strong>YOLO-Segmentation</strong></div>
  <div class="stat"><span class="stat-dot dot-blue"></span>Sample shown: <strong>{len(samples)} / total</strong></div>
</div>

<main>
  <div class="section-title">Preprocessed Samples</div>
  <div class="grid">
    {cards}
  </div>
</main>

<div id="lightbox">
  <img id="lb-img" src="" alt="zoomed">
</div>

<script>
  const lb = document.getElementById('lightbox');
  const lbImg = document.getElementById('lb-img');

  document.querySelectorAll('.panel img').forEach(img => {{
    img.addEventListener('click', () => {{
      lbImg.src = img.src;
      lb.classList.add('active');
    }});
  }});

  lb.addEventListener('click', () => lb.classList.remove('active'));

  document.addEventListener('keydown', e => {{
    if (e.key === 'Escape') lb.classList.remove('active');
  }});
</script>
</body>
</html>"""


def main():
    img_dir   = OUTPUT_PATH / "images"
    mask_dir  = OUTPUT_PATH / "masks"
    label_dir = OUTPUT_PATH / "labels"

    all_imgs = sorted(img_dir.glob("*.jpg"))
    if not all_imgs:
        print("[ERROR] No images found in", img_dir)
        return

    picked = random.sample(all_imgs, min(NUM_SAMPLES, len(all_imgs)))
    print(f"[INFO] Building viewer for {len(picked)} images...")

    samples = []
    for img_path in picked:
        name = img_path.stem
        mask_path  = mask_dir  / f"{name}.png"
        label_path = label_dir / f"{name}.txt"

        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        overlay = draw_yolo_overlay(img, label_path)
        colored_mask = colorize_mask(mask) if mask is not None else np.zeros_like(img)

        poly_count = 0
        if label_path.exists():
            with open(label_path) as f:
                poly_count = sum(1 for l in f if l.strip())

        samples.append({
            "name":       name,
            "orig":       img_to_b64(img, ".jpg"),
            "mask":       img_to_b64(colored_mask, ".png"),
            "overlay":    img_to_b64(overlay, ".jpg"),
            "poly_count": poly_count,
        })

    html = build_html(samples)
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"[OK] Viewer saved to: {HTML_OUT.resolve()}")
    print("[INFO] Open it in your browser to explore the data.")

    # Auto-open in default browser
    import webbrowser
    webbrowser.open(HTML_OUT.resolve().as_uri())


if __name__ == "__main__":
    main()
