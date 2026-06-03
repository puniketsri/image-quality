## src/labeler.py
import cv2
import os
import shutil
import json
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
RAW_DIR     = Path("data/raw")
LABELED_DIR = Path("data/labeled")
PROGRESS_FILE = Path("data/labeling_progress.json")

# Key → folder name mapping
KEYMAP = {
    ord('p'): 'pass',
    ord('b'): 'blur',
    ord('e'): 'exposure',
    ord('c'): 'crop_error',
    ord('m'): 'metadata_fail',
    ord('s'): 'skip',    # genuinely unsure — skip it
    ord('q'): 'quit',
}

def load_progress():
    """Resume from where you left off"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f)["labeled"])
    return set()

def save_progress(labeled_set):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"labeled": list(labeled_set)}, f)

def count_labels():
    counts = {}
    for cls_dir in LABELED_DIR.iterdir():
        if cls_dir.is_dir():
            counts[cls_dir.name] = len(list(cls_dir.glob("*.jpg"))) + \
                                   len(list(cls_dir.glob("*.png")))
    return counts

def label_images():
    # Get all images in raw/
    all_images = sorted(
        list(RAW_DIR.glob("*.jpg")) +
        list(RAW_DIR.glob("*.jpeg")) +
        list(RAW_DIR.glob("*.png"))
    )

    if not all_images:
        print("No images found in data/raw/ — add some images first")
        return

    already_labeled = load_progress()
    remaining = [p for p in all_images if p.name not in already_labeled]

    print(f"\nTotal images : {len(all_images)}")
    print(f"Already done : {len(already_labeled)}")
    print(f"Remaining    : {len(remaining)}")
    print(f"\nKeys: p=pass  b=blur  e=exposure  c=crop_error  m=metadata_fail  s=skip  q=quit")
    print("─" * 60)

    for i, img_path in enumerate(remaining):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping (can't read): {img_path.name}")
            continue

        # Resize for display — keeps large images manageable
        # Original file is NOT modified
        h, w = img.shape[:2]
        max_dim = 900
        scale = min(max_dim/w, max_dim/h, 1.0)
        display = cv2.resize(img, (int(w*scale), int(h*scale)))

        # Progress counter in window title
        title = f"[{i+1}/{len(remaining)}] {img_path.name} | p/b/e/c/m/s/q"
        cv2.imshow(title, display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("\nStopped. Progress saved.")
            break

        if key not in KEYMAP:
            print(f"  Unknown key — use p/b/e/c/m/s/q")
            continue

        action = KEYMAP[key]

        if action == 'skip':
            print(f"  Skipped: {img_path.name}")
            already_labeled.add(img_path.name)
            save_progress(already_labeled)
            cv2.destroyAllWindows()
            continue

        # Copy to labeled folder
        dest_dir = LABELED_DIR / action
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)

        already_labeled.add(img_path.name)
        save_progress(already_labeled)

        counts = count_labels()
        count_str = "  ".join(f"{k}:{v}" for k,v in sorted(counts.items()))
        print(f"  [{action:15s}] {img_path.name:30s} | {count_str}")
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    print("\n── Final counts ──")
    for cls, cnt in sorted(count_labels().items()):
        bar = "" * (cnt // 5)
        print(f"  {cls:20s} {cnt:4d}  {bar}")

if __name__ == "__main__":
    label_images()

