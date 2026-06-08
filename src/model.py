## src/model.py
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 5  # pass, blur, exposure, crop_error, metadata_fail

class QualityGateModel(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Backbone: EfficientNet-B0 pretrained on ImageNet ──────────
        # We KEEP: features + avgpool (the learned visual knowledge)
        # We REPLACE: classifier (ImageNet has 1000 classes, we have 5)
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.features = backbone.features   # 9 MBConv blocks
        self.avgpool  = backbone.avgpool    # global average pooling
        feature_dim   = 1280               # EfficientNet-B0 output size

        # ── Signal dimensions ─────────────────────────────────────────
        signal_dim = 4    # blur, exposure, crop, metadata
        fused_dim  = feature_dim + signal_dim  # 1284

        # ── Head 1: quality score (regression) ───────────────────────
        # Outputs a single number 0-1
        # Sigmoid ensures it stays in valid range
        self.score_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # ── Head 2: failure class (classification) ────────────────────
        # Outputs 5 raw scores (logits), one per class
        # No softmax — CrossEntropyLoss includes it internally
        self.class_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image: torch.Tensor, signals: torch.Tensor):
        """
        image:   [B, 3, 224, 224]  — batch of images
        signals: [B, 4]            — batch of scorer outputs
        Returns: (quality_score [B], class_logits [B, 5])
        """
        # 1. Extract CNN features from image
        x = self.features(image)     # [B, 3, 224, 224] → [B, 1280, 7, 7]
        x = self.avgpool(x)          # [B, 1280, 7, 7]  → [B, 1280, 1, 1]
        x = torch.flatten(x, 1)     # [B, 1280, 1, 1]  → [B, 1280]

        # 2. Fuse CNN features with classical CV signals
        x = torch.cat([x, signals], dim=1)  # [B, 1284]

        # 3. Two heads in parallel on same fused representation
        quality_score = self.score_head(x)   # [B, 1]
        class_logits  = self.class_head(x)   # [B, 5]

        return quality_score.squeeze(1), class_logits
        # squeeze(1) removes the trailing 1: [B, 1] → [B]
