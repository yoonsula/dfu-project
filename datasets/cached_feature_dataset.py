from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Reads feature shards produced by cache_features.py."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Feature cache manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        self.shards = list(self.manifest.get("shards", []))
        self.shard_sizes = [int(size) for size in self.manifest.get("shard_sizes", [])]
        if not self.shards or len(self.shards) != len(self.shard_sizes):
            raise ValueError(f"Invalid feature cache manifest: {manifest_path}")

        self.index: list[tuple[int, int]] = []
        for shard_index, shard_size in enumerate(self.shard_sizes):
            self.index.extend((shard_index, sample_index) for sample_index in range(shard_size))
        self._loaded_shard_index: int | None = None
        self._loaded_shard: dict | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, shard_index: int) -> dict:
        if self._loaded_shard_index != shard_index:
            path = self.cache_dir / self.shards[shard_index]
            self._loaded_shard = torch.load(path, map_location="cpu", weights_only=False)
            self._loaded_shard_index = shard_index
        assert self._loaded_shard is not None
        return self._loaded_shard

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        shard_index, sample_index = self.index[index]
        shard = self._load_shard(shard_index)
        item: dict[str, torch.Tensor | str] = {
            "features": shard["features"][sample_index].float(),
            "image_path": shard["image_paths"][sample_index],
        }
        if "masks" in shard:
            item["mask"] = shard["masks"][sample_index].float()
        if "loss_weights" in shard:
            item["loss_weight"] = shard["loss_weights"][sample_index].float()
        if "labels" in shard:
            item["label"] = shard["labels"][sample_index].long()
        return item
