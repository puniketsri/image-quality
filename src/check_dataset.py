## src/check_dataset.py
from pathlib import Path

LABELED_DIR = Path("data/labeled")
TARGET      = 150   # minimum per class to start training
classes     = ["pass", "blur", "exposure", "crop_error", "metadata_fail"]

print("\n── Dataset status ──────────────────────────────────")
total = 0
for cls in classes:
    d = LABELED_DIR / cls
    n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.jpeg"))) if d.exists() else 0
    total += n
    pct   = min(n / TARGET, 1.0)
    bar   = ("█" * int(pct * 20)).ljust(20)
    if n == 0:
        status = "not started"
    elif n < TARGET:
        status = f"need {TARGET - n} more"
    else:
        status = "✓ ready to train"
    print(f"  {cls:20s} [{bar}] {n:4d}  {status}")

print(f"\n  Total labeled  : {total}")
print(f"  Target total   : {TARGET * len(classes)}")
ready = sum(1 for c in classes
            if (LABELED_DIR/c).exists() and
            len(list((LABELED_DIR/c).glob("*.jpg"))) >= TARGET)
print(f"  Classes ready  : {ready}/{len(classes)}")
if ready == len(classes):
    print(f"\n  ✓ All classes ready — move to augmentation!")
else:
    print(f"\n  Keep labeling — need more images in weak classes")
print("────────────────────────────────────────────────────\n")
