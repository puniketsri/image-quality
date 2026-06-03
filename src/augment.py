## src/augment.py
import cv2, shutil
import albumentations as A
from pathlib import Path

LABELED_DIR   = Path("data/labeled")
AUGMENTED_DIR = Path("data/augmented")
TARGET        = 400
classes       = ["pass","blur","exposure","crop_error","metadata_fail"]

# These transforms create realistic variation
# probabilities are low on purpose — not every aug applied every time
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Rotate(limit=10, p=0.4),
    A.GaussianBlur(blur_limit=(3, 3), p=0.15),   # mild only
    A.HueSaturationValue(
        hue_shift_limit=8, sat_shift_limit=15,
        val_shift_limit=10, p=0.3),
    A.RandomResizedCrop(
        height=224, width=224,
        scale=(0.88, 1.0), p=0.3),
])

print(f"\nAugmenting to {TARGET} images per class...\n")

for cls in classes:
    src_dir  = LABELED_DIR   / cls
    dest_dir = AUGMENTED_DIR / cls
    dest_dir.mkdir(parents=True, exist_ok=True)

    originals = list(src_dir.glob("*.jpg")) if src_dir.exists() else []
    if not originals:
        print(f"  {cls:20s}: 0 originals — skip")
        continue

    # Copy originals first
    for f in originals:
        shutil.copy2(f, dest_dir / f.name)

    needed = TARGET - len(originals)
    if needed <= 0:
        print(f"  {cls:20s}: {len(originals)} originals, no aug needed")
        continue

    print(f"  {cls:20s}: {len(originals):3d} originals → generating {needed} augmented")

    for i in range(needed):
        src     = originals[i % len(originals)]
        img_bgr = cv2.imread(str(src))
        if img_bgr is None: continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_rgb = transform(image=img_rgb)["image"]
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        out_name = f"aug_{i:05d}_{src.stem}.jpg"
        cv2.imwrite(str(dest_dir / out_name), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    final = len(list(dest_dir.glob("*.jpg")))
    print(f"  {cls:20s}: done → {final} total in augmented/")

print("\n✓ Augmentation complete. Check data/augmented/")
