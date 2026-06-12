from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastInstUlcerHead(nn.Module):
    """
    Lightweight FastInst-style ulcer segmentation head.

    It predicts a fixed set of instance masks with dynamic 1x1 kernels and
    merges them into a single ulcer logit map.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        hidden_dim: int = 256,
        num_queries: int = 16,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)

        self.pixel_proj = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.kernel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_score = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] DINOv3 feature map.
        Returns:
            ulcer_logits: [B, 1, H, W]
        """
        pixel_features = self.pixel_proj(features)  # [B, D, H, W]
        batch_size, _, height, width = pixel_features.shape

        query = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Q, D]
        kernels = self.kernel_mlp(query)  # [B, Q, D]

        instance_logits = torch.einsum("bqk,bkhw->bqhw", kernels, pixel_features)
        instance_weight = torch.sigmoid(self.query_score(query)).view(batch_size, self.num_queries, 1, 1)
        weighted_instance_logits = instance_logits * instance_weight

        ulcer_logits = weighted_instance_logits.max(dim=1, keepdim=True).values
        return F.interpolate(
            ulcer_logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
