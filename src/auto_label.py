## src/auto_label.py
import cv2
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

CLEAN_DIR   = Path("data/clean")
LABELED_DIR = Path("data/labeled")
MASTER_CSV  = Path("data/master.csv")

# ── Thresholds — tuned for iPhone photos ──────────────────
BLUR_THRESHOLD       = 48      # was 80
DARK_THRESHOLD       = 82      # was 60
BRIGHT_THRESHOLD     = 142     # was 200
CROP_EDGE_THRESHOLD  = 0.1756  # was 0.08

def score_blur(gray):
    """Low variance = blurry"""
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def score_brightness(gray):
    """Mean pixel value 0-255"""
    return float(np.mean(gray))

def score_crop(img):
    """High edge density at image borders = subject cut off"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    border = int(min(h, w) * 0.06)

    edges = cv2.Canny(gray, 50, 150)
    top    = edges[:border, :]
    bottom = edges[-border:, :]
    left   = edges[:, :border]
    right  = edges[:, -border:]

    density = np.mean([
        top.sum(),  bottom.sum(),
        left.sum(), right.sum()
    ]) / (border * max(h, w) * 255)

    return density

def auto_label(img_path: Path, has_gps: bool, has_timestamp: bool) -> str:
    img = cv2.imread(str(img_path))
    if img is None:
        return "pass"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_score  = score_blur(gray)
    brightness  = score_brightness(gray)
    crop_score  = score_crop(img)

    # Priority order — first match wins
    if blur_score < BLUR_THRESHOLD:
        return "blur"

    if brightness < DARK_THRESHOLD or brightness > BRIGHT_THRESHOLD:
        return "exposure"

    if crop_score > CROP_EDGE_THRESHOLD:
        return "crop_error"

    # Missing BOTH gps and timestamp = metadata fail
    # Missing only one = still pass (all your images lack GPS)
    if not has_gps and not has_timestamp:
        return "metadata_fail"

    return "pass"

def run():
    df = pd.read_csv(MASTER_CSV)
    total = len(df)

    # Create labeled folders
    classes = ["pass", "blur", "exposure", "crop_error", "metadata_fail"]
    for cls in classes:
        (LABELED_DIR / cls).mkdir(parents=True, exist_ok=True)

    counts  = {c: 0 for c in classes}
    labeled = 0

    print(f"\nAuto-labeling {total} images...\n")

    for _, row in df.iterrows():
        img_path = CLEAN_DIR / row["clean_filename"]
        if not img_path.exists():
            print(f"  Missing: {row['clean_filename']}")
            continue

        label = auto_label(
            img_path,
            has_gps       = bool(row["has_gps"]),
            has_timestamp = bool(row["has_timestamp"])
        )

        # Copy to labeled folder
        dest = LABELED_DIR / label / img_path.name
        shutil.copy2(img_path, dest)

        # Update master.csv
        df.loc[df["clean_filename"] == row["clean_filename"], "label"] = label

        counts[label] += 1
        labeled += 1
        print(f"  {row['clean_filename']:25s} → {label}")

    # Save updated master.csv
    df.to_csv(MASTER_CSV, index=False)

    print(f"\n── Auto-label results ──────────────────────────────")
    for cls in classes:
        bar = "█" * counts[cls]
        print(f"  {cls:20s} {counts[cls]:4d}  {bar}")
    print(f"\n  Total labeled : {labeled}/{total}")
    print(f"\nNext: run python src/check_dataset.py")
    print("Then: run python src/augment.py")

if __name__ == "__main__":
    run()
