from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel


class DinoV3LinearClassifier(nn.Module):
    """Frozen DINOv3 backbone + trainable linear classification head."""

    def __init__(
        self,
        backbone: PreTrainedModel,
        num_classes: int,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "backbone.config.hidden_size not found. "
                "Check that the DINOv3 config.json is valid."
            )

        self.head = nn.Linear(hidden_size, num_classes)

    def train(self, mode: bool = True) -> DinoV3LinearClassifier:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.head(cls_token)
