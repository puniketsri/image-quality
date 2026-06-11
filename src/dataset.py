## src/dataset.py
## This file defines HOW to load one image from your labeled dataset.
## The DataLoader uses this to build batches for training.

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, "src")
from scorer import extract_signals

# Maps class folder name → integer label
# The model outputs numbers, not strings
CLASS_MAP = {
    "pass":         0,
    "blur":         1,
    "exposure":     2,
    "crop_error":   3,
    "metadata_fail":4,
}

# Target quality score for each class
# pass = perfect score (1.0), all failures = low score (0.1–0.3)
SCORE_MAP = {
    0: 1.0,   # pass
    1: 0.1,   # blur
    2: 0.2,   # exposure
    3: 0.2,   # crop_error
    4: 0.3,   # metadata_fail
}

# Image preprocessing pipeline
# These EXACT mean/std values must be used for pretrained EfficientNet
# They are the statistics of the ImageNet dataset EfficientNet was trained on
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])


class QualityDataset(Dataset):
    """Loads images from data/augmented/ for training."""

    def __init__(self, data_dir: str, transform=None):
        self.transform = transform or TRANSFORM
        self.samples   = []  # will hold (image_path, class_index)

        # Walk every class folder and collect image paths
        for class_name, class_idx in CLASS_MAP.items():
            cls_dir = Path(data_dir) / class_name
            if not cls_dir.exists():
                print(f"  Warning: {cls_dir} not found, skipping")
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in cls_dir.glob(ext):
                    self.samples.append((str(img_path), class_idx))

        print(f"QualityDataset loaded: {len(self.samples)} images")

    def __len__(self):
        # DataLoader calls this to know how many items exist
        return len(self.samples)

    def __getitem__(self, idx):
        # DataLoader calls this to get one item by index
        img_path, label = self.samples[idx]

        # 1. Load image and apply transforms
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)   # shape: [3, 224, 224]

        # 2. Extract the 4 OpenCV signals (blur, exposure, crop, metadata)
        try:
            s = extract_signals(img_path)
            signals = torch.tensor([
                s.blur_score,
                s.exposure_score,
                s.crop_score,
                s.metadata_score,
            ], dtype=torch.float32)
        except Exception:
            # If scorer fails, use zeros — training still works
            signals = torch.zeros(4, dtype=torch.float32)

        # 3. Quality score target (what the score head should predict)
        quality_score = torch.tensor(
            SCORE_MAP[label], dtype=torch.float32
        )

        # Return all four things the training loop needs
        return img, signals, quality_score, label
        # img:           [3, 224, 224] tensor
        # signals:       [4] tensor
        # quality_score: scalar float tensor
        # label:         integer (0–4)
