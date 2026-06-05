import cv2, numpy as np, piexif
from dataclasses import dataclass

@dataclass
class QualitySignals:
    blur_score:      float  # 1.0=sharp,   0.0=blurry
    exposure_score:  float  # 1.0=perfect, 0.0=dark/washed
    crop_score:      float  # 1.0=clean,   0.0=cut off
    metadata_score:  float  # 1.0=complete,0.0=missing
    raw_laplacian:   float
    mean_brightness: float

def score_blur(gray):
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(variance / 1000.0, 0.0, 1.0)), variance

def score_exposure(gray):
    mean = float(np.mean(gray))
    if mean < 82.0:   score = mean / 82.0
    elif mean > 142.0: score = 1.0 - (mean-142.0)/(255.0-142.0)
    else:              score = 1.0 - abs(mean-112.0)/30.0*0.3
    return float(np.clip(score, 0.0, 1.0)), mean

def score_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    b = int(min(h,w)*0.06)
    e = cv2.Canny(gray, 50, 150)
    d = np.mean([e[:b,:].sum(),e[-b:,:].sum(),
                 e[:,:b].sum(),e[:,-b:].sum()]) / (b*max(h,w)*255)
    return float(np.clip(1.0-(d/0.1756), 0.0, 1.0))

def score_metadata(img_path):
    found = 0
    try:
        exif = piexif.load(img_path)
        z, g = exif.get("0th",{}), exif.get("GPS",{})
        if g.get(piexif.GPSIFD.GPSLatitude):    found += 1
        if z.get(piexif.ImageIFD.DateTime):     found += 1
        if z.get(piexif.ImageIFD.Make):         found += 1
    except: found = 0
    return found / 3.0

def extract_signals(img_path: str) -> QualitySignals:
    img = cv2.imread(img_path)
    if img is None: raise ValueError(f"Cannot read: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, rv = score_blur(gray)
    e, mb = score_exposure(gray)
    return QualitySignals(b, e, score_crop(img),
                          score_metadata(img_path), rv, mb)
