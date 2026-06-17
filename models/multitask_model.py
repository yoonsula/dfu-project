from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3Backbone
from .foot_head import FastInstFootHead
from .ulcer_head import FastInstUlcerHead


class MultiTaskSegModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module | None = None,
        feature_dim: int = 384,
        hidden_dim: int = 256,
        foot_num_queries: int = 8,
        ulcer_num_queries: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = backbone or DINOv3Backbone(feature_dim=feature_dim)
        self.foot_head = FastInstFootHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_queries=foot_num_queries,
        )
        self.ulcer_head = FastInstUlcerHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_queries=ulcer_num_queries,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_foot_logits(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        foot_logits = self.foot_head(features)
        if output_size is not None:
            foot_logits = F.interpolate(
                foot_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return foot_logits

    def predict_ulcer_logits(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        ulcer_logits = self.ulcer_head(features)
        if output_size is not None:
            ulcer_logits = F.interpolate(
                ulcer_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return ulcer_logits

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_size = x.shape[-2:]
        features = self.encode(x)
        return {
            "foot": self.predict_foot_logits(features, output_size),
            "ulcer": self.predict_ulcer_logits(features, output_size),
        }
