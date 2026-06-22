from __future__ import annotations

import torch
from torch import nn


class DFUFeatureClassifierHead(nn.Module):
    """Binary DFU classifier on top of a DINOv3 spatial feature map."""

    def __init__(
        self,
        feature_dim: int = 384,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
        head_type: str = "linear",
    ) -> None:
        super().__init__()
        self.head_type = head_type
        if head_type == "linear":
            self.net = nn.Linear(feature_dim, num_classes)
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected feature map [B, C, H, W], got shape={tuple(features.shape)}")
        pooled = features.mean(dim=(2, 3))
        return self.net(pooled)
