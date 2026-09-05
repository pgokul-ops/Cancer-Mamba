"""
models/tiny_cnn3d.py

Deliberately minimal 3D CNN baseline for Stage 2 pipeline & plumbing verification.
Architecture:
  4 Conv3D blocks: Conv3d(3x3x3) -> BatchNorm3d -> ReLU -> MaxPool3d(2x2x2)
  Channels: 1 -> 8 -> 16 -> 32 -> 64
  Head: AdaptiveAvgPool3d(1) -> Dropout(p) -> Linear(64, 1)
Parameter count: ~73K params (<1M).
"""

import torch
import torch.nn as nn


class TinyCNN3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        c1 = base_channels       # 8
        c2 = base_channels * 2   # 16
        c3 = base_channels * 4   # 32
        c4 = base_channels * 8   # 64

        self.features = nn.Sequential(
            # Block 1: 80 -> 40
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 2: 40 -> 20
            nn.Conv3d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 3: 20 -> 10
            nn.Conv3d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 4: 10 -> 5
            nn.Conv3d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: 3D input tensor of shape (B, C, D, H, W) e.g. (B, 1, 80, 80, 80).
        Returns:
            logits: Tensor of shape (B, 1).
        """
        feats = self.features(x)
        pooled = self.global_pool(feats)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits


def get_model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
