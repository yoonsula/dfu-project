from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import nn

from paths import DINOV3_CHECKPOINT as DEFAULT_DINOV3_CHECKPOINT
from paths import DINOV3_REPO as DEFAULT_DINOV3_REPO


class DINOv3Backbone(nn.Module):
    """DINOv3 ViT-S/16 feature extractor used by both segmentation heads."""

    def __init__(
        self,
        repo_dir: str | Path = DEFAULT_DINOV3_REPO,
        checkpoint_path: str | Path = DEFAULT_DINOV3_CHECKPOINT,
        model_name: str = "dinov3_vits16",
        feature_dim: int = 384,
        n_layers: int = 12,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.repo_dir = Path(repo_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.freeze = freeze

        if not self.repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repo not found: {self.repo_dir}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {self.checkpoint_path}")
        if sys.version_info < (3, 10):
            raise RuntimeError("DINOv3 requires Python 3.10+. Use the `dfu-venv` conda environment.")

        self.encoder = self._load_encoder()
        self.set_trainable(not freeze)

    def _load_encoder(self) -> nn.Module:
        repo = str(self.repo_dir)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        backbones = importlib.import_module("dinov3.hub.backbones")
        build_model = getattr(backbones, self.model_name)
        return build_model(weights=str(self.checkpoint_path))

    def set_trainable(self, trainable: bool) -> None:
        self.freeze = not trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                features = self.encoder.get_intermediate_layers(
                    x,
                    n=range(self.n_layers),
                    reshape=True,
                    norm=True,
                )
        else:
            features = self.encoder.get_intermediate_layers(
                x,
                n=range(self.n_layers),
                reshape=True,
                norm=True,
            )
        return features[-1]
