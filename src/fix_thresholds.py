import cv2
import numpy as np
from pathlib import Path

CLEAN_DIR = Path("data/clean")
files = list(CLEAN_DIR.glob("*.jpg")) + list(CLEAN_DIR.glob("*.jpeg"))

blur_scores, brightness_scores, crop_scores = [], [], []

print(f"Analyzing {len(files)} images...\n")

for f in files[:93]:
    img = cv2.imread(str(f))
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_scores.append(lap)

    # Brightness
    brightness_scores.append(float(np.mean(gray)))

    # Crop edge density
    h, w = gray.shape
    border = int(min(h,w) * 0.06)
    edges = cv2.Canny(gray, 50, 150)
    density = np.mean([
        edges[:border,:].sum(), edges[-border:,:].sum(),
        edges[:,:border].sum(), edges[:,-border:].sum()
    ]) / (border * max(h,w) * 255)
    crop_scores.append(density)

print("── Blur scores (Laplacian variance) ────────────────")
print(f"  Min    : {min(blur_scores):.1f}")
print(f"  Max    : {max(blur_scores):.1f}")
print(f"  Mean   : {np.mean(blur_scores):.1f}")
print(f"  Median : {np.median(blur_scores):.1f}")
print(f"  Suggest BLUR_THRESHOLD = {np.percentile(blur_scores, 25):.0f}")

print("\n── Brightness scores ───────────────────────────────")
print(f"  Min    : {min(brightness_scores):.1f}")
print(f"  Max    : {max(brightness_scores):.1f}")
print(f"  Mean   : {np.mean(brightness_scores):.1f}")
print(f"  Suggest DARK_THRESHOLD   = {np.percentile(brightness_scores, 10):.0f}")
print(f"  Suggest BRIGHT_THRESHOLD = {np.percentile(brightness_scores, 90):.0f}")

print("\n── Crop edge density scores ────────────────────────")
print(f"  Min    : {min(crop_scores):.4f}")
print(f"  Max    : {max(crop_scores):.4f}")
print(f"  Mean   : {np.mean(crop_scores):.4f}")
print(f"  Median : {np.median(crop_scores):.4f}")
print(f"  Suggest CROP_EDGE_THRESHOLD = {np.percentile(crop_scores, 85):.4f}")

print("\n── What this means ─────────────────────────────────")
blur_pct = sum(1 for s in blur_scores if s < np.percentile(blur_scores,25))
crop_pct = sum(1 for s in crop_scores if s > np.percentile(crop_scores,85))
print(f"  With suggested thresholds:")
print(f"  ~{blur_pct} images will be labeled blur")
print(f"  ~{crop_pct} images will be labeled crop_error")
print(f"  Rest will fall through to metadata_fail or pass")
