## src/inspect_data.py
from pathlib import Path
from PIL import Image
import piexif
from collections import defaultdict

RAW_DIR = Path("data/raw")

def inspect():
    files = (list(RAW_DIR.glob("*.jpg")) +
             list(RAW_DIR.glob("*.jpeg")) +
             list(RAW_DIR.glob("*.png")))

    print(f"\n── Inspect results ─────────────────────────────────")
    print(f"  Images found : {len(files)}")

    corrupt, sizes, no_exif = [], [], 0
    has_gps = has_ts = has_make = 0

    for f in files:
        # Check if file opens correctly
        try:
            with Image.open(f) as img:
                img.verify()
            sizes.append(f.stat().st_size / 1024)  # size in KB
        except Exception as e:
            corrupt.append(f.name)
            continue

        # Check EXIF metadata inside the file
        try:
            exif   = piexif.load(str(f))
            zeroth = exif.get("0th", {})
            gps    = exif.get("GPS", {})
            if gps.get(piexif.GPSIFD.GPSLatitude):  has_gps  += 1
            if zeroth.get(piexif.ImageIFD.DateTime): has_ts   += 1
            if zeroth.get(piexif.ImageIFD.Make):     has_make += 1
        except Exception:
            no_exif += 1

    # Print file size summary
    if sizes:
        print(f"  Smallest     : {min(sizes):.0f} KB")
        print(f"  Largest      : {max(sizes):.0f} KB")
        print(f"  Average      : {sum(sizes)/len(sizes):.0f} KB")

    # Print EXIF summary
    total = len(files)
    print(f"\n  EXIF results (out of {total} images):")
    print(f"  Has GPS         : {has_gps}")
    print(f"  Has timestamp   : {has_ts}")
    print(f"  Has camera make : {has_make}")
    print(f"  No EXIF at all  : {no_exif}")
    print(f"  Missing GPS     : {total - has_gps}  ← metadata_fail candidates")

    # Print any corrupt files
    if corrupt:
        print(f"\n  ⚠ Corrupt files ({len(corrupt)}):")
        for c in corrupt: print(f"    {c}")
    else:
        print(f"\n  ✓ No corrupt files")

    # Print filename problems
    spaces  = [f.name for f in files if ' ' in f.name]
    if spaces:
        print(f"\n  ⚠ {len(spaces)} filenames have spaces — will be fixed in next step")
        for s in spaces[:5]: print(f"    {s}")
    else:
        print(f"  ✓ All filenames are clean")
    print("────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    inspect()
