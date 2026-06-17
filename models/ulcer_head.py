from __future__ import annotations

from .fastinst_head import FastInstSegHead


class FastInstUlcerHead(FastInstSegHead):
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
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
        )
