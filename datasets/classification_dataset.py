from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.diabetic_foot_dataset import IMAGENET_MEAN, IMAGENET_STD


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ClassificationSample:
    image_path: Path
    label: int


def _resampling(name: str) -> int:
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def _iter_image_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


class ClassificationImageDataset(Dataset):
    """ImageFolder/CSV dataset for DFU classification."""

    def __init__(
        self,
        root: str | Path | None = None,
        csv_path: str | Path | None = None,
        split: str = "train",
        image_size: int = 768,
        val_ratio: float = 0.1,
        seed: int = 42,
        classes: Sequence[str] | None = None,
    ) -> None:
        if split not in {"train", "val", "validation", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if root is None and csv_path is None:
            raise ValueError("Provide --dfu-root or --dfu-csv for classification data.")

        self.root = Path(root) if root is not None else None
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.split = "val" if split == "validation" else split
        self.image_size = int(image_size)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)

        all_classes, samples = self._load_all(classes)
        self.classes = tuple(all_classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.id2label = {index: name for index, name in enumerate(self.classes)}
        self.samples = self._split_samples(samples)
        if not self.samples:
            raise RuntimeError(f"No classification samples found for split={self.split!r}")

    def _load_all(
        self,
        classes: Sequence[str] | None,
    ) -> tuple[list[str], list[ClassificationSample]]:
        if self.csv_path is not None:
            return self._load_csv(classes)
        if self.root is None:
            raise ValueError("root is required when csv_path is not provided.")
        return self._load_image_folder(classes)

    def _load_image_folder(
        self,
        classes: Sequence[str] | None,
    ) -> tuple[list[str], list[ClassificationSample]]:
        assert self.root is not None
        if not self.root.exists():
            raise FileNotFoundError(f"Classification root not found: {self.root}")

        split_candidates = [self.root / self.split]
        if self.split == "val":
            split_candidates.append(self.root / "validation")
        split_root = next((candidate for candidate in split_candidates if candidate.exists()), None)
        root = split_root if split_root is not None else self.root
        class_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
        class_names = list(classes) if classes is not None else [path.name for path in class_dirs]
        class_to_idx = {name: index for index, name in enumerate(class_names)}

        samples: list[ClassificationSample] = []
        for class_name in class_names:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for image_path in _iter_image_files(class_dir):
                samples.append(ClassificationSample(image_path=image_path, label=class_to_idx[class_name]))
        return class_names, samples

    def _load_csv(
        self,
        classes: Sequence[str] | None,
    ) -> tuple[list[str], list[ClassificationSample]]:
        assert self.csv_path is not None
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Classification CSV not found: {self.csv_path}")

        rows: list[dict[str, str]] = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {self.csv_path}")
            for row in reader:
                row_split = (row.get("split") or "").strip().lower()
                if row_split and row_split not in {self.split, "validation" if self.split == "val" else self.split}:
                    continue
                rows.append(row)

        label_values = [row.get("label") or row.get("class") or row.get("class_name") for row in rows]
        if any(value is None for value in label_values):
            raise ValueError("CSV must contain a label, class, or class_name column.")
        class_names = list(classes) if classes is not None else sorted({str(value) for value in label_values})
        class_to_idx = {name: index for index, name in enumerate(class_names)}

        base_dir = self.csv_path.parent
        samples: list[ClassificationSample] = []
        for row, label_value in zip(rows, label_values, strict=False):
            image_value = row.get("image_path") or row.get("path") or row.get("image")
            if not image_value:
                raise ValueError("CSV must contain an image_path, path, or image column.")
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = base_dir / image_path
            label_name = str(label_value)
            if label_name not in class_to_idx:
                continue
            samples.append(ClassificationSample(image_path=image_path, label=class_to_idx[label_name]))
        return class_names, samples

    def _split_samples(self, samples: list[ClassificationSample]) -> list[ClassificationSample]:
        if self.csv_path is not None:
            return samples
        assert self.root is not None
        if (
            (self.root / "train").exists()
            or (self.root / "val").exists()
            or (self.root / "validation").exists()
            or (self.root / "test").exists()
        ):
            return samples

        rng = random.Random(self.seed)
        by_label: dict[int, list[ClassificationSample]] = {}
        for sample in samples:
            by_label.setdefault(sample.label, []).append(sample)

        selected: list[ClassificationSample] = []
        for label_samples in by_label.values():
            shuffled = list(label_samples)
            rng.shuffle(shuffled)
            val_count = max(1, int(round(len(shuffled) * self.val_ratio)))
            val_count = min(val_count, len(shuffled))
            if self.split == "val":
                selected.extend(shuffled[:val_count])
            else:
                selected.extend(shuffled[val_count:])
        rng.shuffle(selected)
        return selected

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as raw_image:
            image = raw_image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), _resampling("BILINEAR"))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        return {
            "image": self._load_image(sample.image_path),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "task": "dfu",
            "image_path": str(sample.image_path),
        }
