from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from datasets.catalog import default_foot_roots
from datasets.catalog import foot_extra_coco_sources
from datasets.catalog import foot_primary_sources
from datasets.catalog import ulcer_sources
from datasets.samples import SegmentationSample
from datasets.source_loaders import load_coco_samples
from datasets.source_loaders import load_fuseg_samples
from datasets.source_loaders import load_wound_image_samples
from paths import DEFAULT_BODY_ROOT
from paths import DEFAULT_HUMANBODY_ROOT
from paths import DEFAULT_ULCER_ROOT
from paths import DEFAULT_WOUND_IMAGE_ROOT
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resampling(name: str) -> int:
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


class DiabeticFootDataset(Dataset):
    """Binary segmentation dataset for foot or ulcer masks."""

    def __init__(
        self,
        task: str,
        split: str,
        foot_roots: str | Path | Sequence[str | Path] | None = None,
        body_root: str | Path | None = DEFAULT_BODY_ROOT,
        humanbody_root: str | Path | None = DEFAULT_HUMANBODY_ROOT,
        ulcer_root: str | Path = DEFAULT_ULCER_ROOT,
        wound_image_root: str | Path | None = DEFAULT_WOUND_IMAGE_ROOT,
        image_size: int = 768,
        val_ratio: float = 0.1,
        val_negative_ratio: float = 0.25,
        seed: int = 42,
        augment: bool = False,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        hflip_prob: float = 0.0,
        negative_oversample: int = 1,
        neg_sample_weight: float = 1.0,
    ) -> None:
        if task not in {"foot", "ulcer"}:
            raise ValueError(f"Unsupported task: {task}")
        if split not in {"train", "val", "validation"}:
            raise ValueError(f"Unsupported split: {split}")

        self.task = task
        self.split = "val" if split == "validation" else split
        self.foot_roots = default_foot_roots() if foot_roots is None else self._normalize_foot_roots(foot_roots)
        self.body_root = Path(body_root) if body_root is not None else None
        self.humanbody_root = Path(humanbody_root) if humanbody_root is not None else None
        self.ulcer_root = Path(ulcer_root)
        self.wound_image_root = Path(wound_image_root) if wound_image_root is not None else None
        self.image_size = int(image_size)
        self.val_ratio = float(val_ratio)
        self.val_negative_ratio = max(0.0, min(float(val_negative_ratio), 0.9))
        self.seed = int(seed)
        self.augment = bool(augment and self.split == "train")
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.hflip_prob = float(hflip_prob)
        self.negative_oversample = max(1, int(negative_oversample))
        self.neg_sample_weight = float(neg_sample_weight)
        self.samples = self._load_samples()

        if not self.samples:
            raise RuntimeError(f"No samples found for task={self.task!r}, split={self.split!r}")

    @staticmethod
    def _normalize_foot_roots(foot_roots: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
        if isinstance(foot_roots, (str, Path)):
            return (Path(foot_roots),)
        roots = tuple(Path(root) for root in foot_roots)
        if not roots:
            raise ValueError("foot_roots must contain at least one path")
        return roots

    def _load_samples(self) -> list[SegmentationSample]:
        if self.task == "foot":
            return self._load_foot_samples()
        return self._load_ulcer_samples()

    def _load_foot_samples(self) -> list[SegmentationSample]:
        train_samples, val_samples = self._split_foot_train_val()
        if self.split == "val":
            return val_samples
        return self._oversample_negatives(train_samples)

    def _split_foot_train_val(self) -> tuple[list[SegmentationSample], list[SegmentationSample]]:
        rng = random.Random(self.seed)
        primary_profile = "positive" if self.split == "train" and self.augment else "natural"
        primary_samples = self._load_primary_foot_samples(primary_profile)

        train_extras: list[SegmentationSample] = []
        val_negative_pool: list[SegmentationSample] = []

        for source in foot_extra_coco_sources(
            body_root=self.body_root,
            humanbody_root=self.humanbody_root,
        ):
            source_samples = load_coco_samples(
                source.root,
                category_ids=source.category_ids,
                positive_profile=source.positive_profile,
                negative_profile=source.negative_profile,
                missing_ok=source.missing_ok,
            )
            train_extras.extend(sample for sample in source_samples if not sample.is_negative)
            source_negatives = [sample for sample in source_samples if sample.is_negative]
            if self.val_negative_ratio > 0:
                val_negative_pool.extend(source_negatives)
            else:
                train_extras.extend(source_negatives)

        primary_positives = [sample for sample in primary_samples if not sample.is_negative]
        primary_negatives = [sample for sample in primary_samples if sample.is_negative]
        rng.shuffle(primary_positives)
        rng.shuffle(primary_negatives)

        positive_val_count = max(1, int(round(len(primary_positives) * self.val_ratio)))
        positive_val_count = min(positive_val_count, len(primary_positives))
        val_positives = primary_positives[:positive_val_count]
        train_positives = primary_positives[positive_val_count:]

        if self.val_negative_ratio > 0:
            val_negative_pool.extend(primary_negatives)
            rng.shuffle(val_negative_pool)
            negative_val_count = self._foot_negative_val_count(
                len(val_positives),
                len(val_negative_pool),
            )
            val_negatives = val_negative_pool[:negative_val_count]
            train_negatives = val_negative_pool[negative_val_count:]
        else:
            negative_val_count = max(1, int(round(len(primary_negatives) * self.val_ratio)))
            negative_val_count = min(negative_val_count, len(primary_negatives))
            val_negatives = primary_negatives[:negative_val_count]
            train_negatives = primary_negatives[negative_val_count:] + val_negative_pool

        train_samples = train_positives + train_extras + train_negatives
        val_samples = val_positives + val_negatives
        return train_samples, val_samples

    def _foot_negative_val_count(self, positive_val_count: int, negative_count: int) -> int:
        if negative_count <= 0:
            return 0
        if self.val_negative_ratio <= 0:
            return max(1, int(round(negative_count * self.val_ratio)))
        if positive_val_count <= 0:
            return max(1, int(round(negative_count * self.val_ratio)))
        target = int(
            round(positive_val_count * self.val_negative_ratio / (1.0 - self.val_negative_ratio))
        )
        return max(1, min(target, negative_count))

    def _load_primary_foot_samples(self, positive_profile: str) -> list[SegmentationSample]:
        samples: list[SegmentationSample] = []
        missing: list[Path] = []
        for source in foot_primary_sources(self.foot_roots, positive_profile=positive_profile):
            annotation_path = source.root / "_annotations.coco.json"
            if not annotation_path.exists():
                missing.append(annotation_path)
                continue
            samples.extend(
                load_coco_samples(
                    source.root,
                    category_ids=source.category_ids,
                    positive_profile=source.positive_profile,
                    negative_profile=source.negative_profile,
                    missing_ok=source.missing_ok,
                )
            )
        if not samples:
            if missing and len(missing) == len(self.foot_roots):
                raise FileNotFoundError(
                    "COCO annotation file not found in any foot root: "
                    + ", ".join(str(path) for path in missing)
                )
            roots = ", ".join(str(root) for root in self.foot_roots)
            raise RuntimeError(f"No foot COCO samples found under: {roots}")
        return samples

    def _oversample_negatives(self, samples: list[SegmentationSample]) -> list[SegmentationSample]:
        if self.negative_oversample <= 1:
            return samples

        positives = [sample for sample in samples if not sample.is_negative]
        negatives = [sample for sample in samples if sample.is_negative]
        return positives + negatives * self.negative_oversample

    def _load_ulcer_samples(self) -> list[SegmentationSample]:
        fuseg_source, wound_source = ulcer_sources(
            ulcer_root=self.ulcer_root,
            wound_image_root=self.wound_image_root,
        )
        samples = load_fuseg_samples(fuseg_source.root, self.split)
        if wound_source is not None:
            wound_samples = load_wound_image_samples(wound_source.root)
            samples.extend(self._split_samples_by_val_ratio(wound_samples))
        return samples

    def _split_samples_by_val_ratio(self, samples: list[SegmentationSample]) -> list[SegmentationSample]:
        if not samples:
            return []

        rng = random.Random(self.seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * self.val_ratio)))
        val_count = min(val_count, len(shuffled))
        if self.split == "val":
            return shuffled[:val_count]
        return shuffled[val_count:]

    def _load_image_pil(self, path: Path) -> Image.Image:
        with Image.open(path) as raw_image:
            image = raw_image.convert("RGB")
        return image.resize((self.image_size, self.image_size), _resampling("BILINEAR"))

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def _load_mask_pil(self, sample: SegmentationSample) -> Image.Image:
        if self.task == "foot":
            mask = self._rasterize_foot_mask(sample)
        else:
            if sample.mask_path is None:
                with Image.open(sample.image_path) as raw_image:
                    mask = Image.new("L", raw_image.size, 0)
            else:
                with Image.open(sample.mask_path) as raw_mask:
                    mask = raw_mask.convert("L")

        return mask.resize((self.image_size, self.image_size), _resampling("NEAREST"))

    @staticmethod
    def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
        array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def _apply_augment_for_sample(
        self,
        image: Image.Image,
        mask: Image.Image,
        profile: str,
    ) -> tuple[Image.Image, Image.Image]:
        if self.split != "train":
            return image, mask

        if profile == "positive" and self.augment:
            return self._apply_scale_hflip(
                image,
                mask,
                self.scale_min,
                self.scale_max,
                self.hflip_prob,
            )
        return image, mask

    def _apply_scale_hflip(
        self,
        image: Image.Image,
        mask: Image.Image,
        scale_min: float,
        scale_max: float,
        hflip_prob: float,
    ) -> tuple[Image.Image, Image.Image]:
        scale = random.uniform(scale_min, scale_max)
        if scale != 1.0:
            image, mask = self._scale_and_crop_or_pad(image, mask, scale)

        if random.random() < hflip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image, mask

    def _scale_and_crop_or_pad(
        self,
        image: Image.Image,
        mask: Image.Image,
        scale: float,
    ) -> tuple[Image.Image, Image.Image]:
        size = self.image_size
        scaled_size = max(1, int(round(size * scale)))
        image_scaled = image.resize((scaled_size, scaled_size), _resampling("BILINEAR"))
        mask_scaled = mask.resize((scaled_size, scaled_size), _resampling("NEAREST"))

        if scaled_size >= size:
            max_offset = scaled_size - size
            left = random.randint(0, max_offset) if max_offset > 0 else 0
            top = random.randint(0, max_offset) if max_offset > 0 else 0
            box = (left, top, left + size, top + size)
            return image_scaled.crop(box), mask_scaled.crop(box)

        image_canvas = Image.new("RGB", (size, size), (0, 0, 0))
        mask_canvas = Image.new("L", (size, size), 0)
        max_offset = size - scaled_size
        left = random.randint(0, max_offset) if max_offset > 0 else 0
        top = random.randint(0, max_offset) if max_offset > 0 else 0
        image_canvas.paste(image_scaled, (left, top))
        mask_canvas.paste(mask_scaled, (left, top))
        return image_canvas, mask_canvas

    def _rasterize_foot_mask(self, sample: SegmentationSample) -> Image.Image:
        with Image.open(sample.image_path) as image:
            image_size = image.size
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)

        for annotation in sample.annotations:
            segmentation = annotation.get("segmentation", [])
            if isinstance(segmentation, list):
                for polygon in segmentation:
                    if len(polygon) < 6:
                        continue
                    points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                    draw.polygon(points, outline=1, fill=1)
            elif isinstance(segmentation, dict):
                self._draw_uncompressed_rle(draw, segmentation, mask.size)
        return mask

    @staticmethod
    def _draw_uncompressed_rle(draw: ImageDraw.ImageDraw, segmentation: dict[str, Any], size: tuple[int, int]) -> None:
        counts = segmentation.get("counts")
        if not isinstance(counts, list):
            return
        width, height = size
        values = []
        value = 0
        for run_length in counts:
            values.extend([value] * int(run_length))
            value = 1 - value
        total = width * height
        if len(values) < total:
            values.extend([0] * (total - len(values)))
        array = np.asarray(values[:total], dtype=np.uint8).reshape((height, width), order="F")
        ys, xs = np.nonzero(array)
        for x, y in zip(xs.tolist(), ys.tolist(), strict=False):
            draw.point((x, y), fill=1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = self._load_image_pil(sample.image_path)
        mask = self._load_mask_pil(sample)
        image, mask = self._apply_augment_for_sample(image, mask, sample.augment_profile)

        item: dict[str, torch.Tensor | str] = {
            "image": self._image_to_tensor(image),
            "mask": self._mask_to_tensor(mask),
            "task": self.task,
            "image_path": str(sample.image_path),
        }
        if self.task == "foot":
            weight = self.neg_sample_weight if sample.is_negative else 1.0
            item["loss_weight"] = torch.tensor(weight, dtype=torch.float32)
        return item
