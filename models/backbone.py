from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import DINOv3ViTBackbone

from paths import DINOV3_MODEL_PATH as DEFAULT_DINOV3_MODEL_PATH


class DINOv3Backbone(nn.Module):
    """DINOv3 ViT-S/16 feature extractor used by foot, wound, and DFU heads."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_DINOV3_MODEL_PATH,
        feature_dim: int = 384,
        freeze: bool = True,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path).resolve()
        self.feature_dim = feature_dim
        self.freeze = freeze
        self.local_files_only = local_files_only

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"DINOv3 model directory not found: {self.model_path}. "
                "Download a Hugging Face snapshot into this folder or set DINOV3_MODEL_PATH."
            )

        self.encoder = DINOv3ViTBackbone.from_pretrained(
            str(self.model_path),
            local_files_only=local_files_only,
        )
        self.set_trainable(not freeze)

    def set_trainable(self, trainable: bool) -> None:
        self.freeze = not trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                outputs = self.encoder(pixel_values=x)
        else:
            outputs = self.encoder(pixel_values=x)
        feature_maps = outputs.feature_maps
        if not feature_maps:
            raise RuntimeError("DINOv3ViTBackbone returned no feature_maps.")
        return feature_maps[-1]
