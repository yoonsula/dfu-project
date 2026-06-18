from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3Backbone
from .dfu_feature_head import DFUFeatureClassifierHead
from .foot_head import FastInstFootHead
from .wound_head import FastInstWoundHead


class DFUPipelineModel(nn.Module):
    """Inference-time assembly: one shared backbone with independently trained heads."""

    def __init__(
        self,
        backbone: nn.Module | None = None,
        *,
        feature_dim: int = 384,
        hidden_dim: int = 256,
        foot_num_queries: int = 8,
        wound_num_queries: int = 16,
        with_foot_head: bool = True,
        with_wound_head: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone or DINOv3Backbone(feature_dim=feature_dim)
        self.foot_head = (
            FastInstFootHead(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_queries=foot_num_queries,
            )
            if with_foot_head
            else None
        )
        self.wound_head = (
            FastInstWoundHead(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_queries=wound_num_queries,
            )
            if with_wound_head
            else None
        )
        self.dfu_head: DFUFeatureClassifierHead | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_foot_logits(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if self.foot_head is None:
            raise RuntimeError("Foot head is not attached to the pipeline model.")
        foot_logits = self.foot_head(features)
        if output_size is not None:
            foot_logits = F.interpolate(
                foot_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return foot_logits

    def predict_wound_logits(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if self.wound_head is None:
            raise RuntimeError("Wound head is not attached to the pipeline model.")
        wound_logits = self.wound_head(features)
        if output_size is not None:
            wound_logits = F.interpolate(
                wound_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return wound_logits

    def predict_dfu_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.dfu_head is None:
            raise RuntimeError("DFU classification head is not attached to the pipeline model.")
        return self.dfu_head(features)
