## src/prepare_data.py
import re, shutil, csv
from pathlib import Path
from PIL import Image
import piexif

RAW_DIR    = Path("data/raw")
CLEAN_DIR  = Path("data/clean")
MASTER_CSV = Path("data/master.csv")

def dms_to_dd(dms):
    """Convert GPS degrees/minutes/seconds tuple → decimal degrees"""
    d, m, s = dms
    return d[0]/d[1] + (m[0]/m[1])/60 + (s[0]/s[1])/3600

def get_exif(path):
    """Extract key EXIF fields from one image"""
    r = {"has_gps": False, "gps_lat": "", "gps_lon": "",
         "has_timestamp": False, "timestamp": "",
         "camera_make": "", "camera_model": "",
         "width": "", "height": ""}
    try:
        with Image.open(path) as img:
            r["width"], r["height"] = img.size
        exif   = piexif.load(str(path))
        zeroth = exif.get("0th", {})
        gps    = exif.get("GPS", {})
        lat = gps.get(piexif.GPSIFD.GPSLatitude)
        lon = gps.get(piexif.GPSIFD.GPSLongitude)
        if lat and lon:
            r["has_gps"] = True
            r["gps_lat"] = round(dms_to_dd(lat), 6)
            r["gps_lon"] = round(dms_to_dd(lon), 6)
        ts = zeroth.get(piexif.ImageIFD.DateTime)
        if ts:
            r["has_timestamp"] = True
            r["timestamp"] = ts.decode() if isinstance(ts, bytes) else ts
        make  = zeroth.get(piexif.ImageIFD.Make)
        model = zeroth.get(piexif.ImageIFD.Model)
        if make:  r["camera_make"]  = make.decode().strip()  if isinstance(make,  bytes) else make
        if model: r["camera_model"] = model.decode().strip() if isinstance(model, bytes) else model
    except Exception:
        pass
    return r

def prepare():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    files = (list(RAW_DIR.glob("*.jpg")) +
             list(RAW_DIR.glob("*.jpeg")) +
             list(RAW_DIR.glob("*.png")))
    print(f"\nPreparing {len(files)} images...")

    rows, skipped = [], 0
    for i, src in enumerate(sorted(files)):
        # Verify the file is readable
        try:
            with Image.open(src) as img: img.verify()
        except Exception:
            print(f"  ✗ Corrupt, skipping: {src.name}")
            skipped += 1
            continue

        # New clean filename: img_000001.jpg
        new_name  = f"img_{i:06d}{src.suffix.lower()}"
        dest      = CLEAN_DIR / new_name
        shutil.copy2(src, dest)

        exif = get_exif(dest)
        rows.append({
            "original_filename" : src.name,
            "clean_filename"    : new_name,
            "has_gps"           : exif["has_gps"],
            "gps_lat"           : exif["gps_lat"],
            "gps_lon"           : exif["gps_lon"],
            "has_timestamp"     : exif["has_timestamp"],
            "timestamp"         : exif["timestamp"],
            "camera_make"       : exif["camera_make"],
            "camera_model"      : exif["camera_model"],
            "width"             : exif["width"],
            "height"            : exif["height"],
            "label"             : "",   # you fill this during labeling
            "labeled_at"        : "",
        })

    # Write master CSV
    with open(MASTER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    no_gps = sum(1 for r in rows if not r["has_gps"])
    print(f"\n── Preparation complete ────────────────────────────")
    print(f"  Copied to data/clean/ : {len(rows)} images")
    print(f"  Skipped (corrupt)     : {skipped}")
    print(f"  master.csv created    : {len(rows)} rows")
    print(f"  Missing GPS           : {no_gps} → metadata_fail candidates")
    print(f"  Has GPS               : {len(rows)-no_gps}")
    print("────────────────────────────────────────────────────\n")
    print("Next: open data/master.csv in Excel to see your data")
    print("Then run: python src/labeler.py")

if __name__ == "__main__":
    prepare()
