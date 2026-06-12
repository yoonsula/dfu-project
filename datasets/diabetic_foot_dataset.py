from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from paths import DEFAULT_BODY_ROOT
from paths import DEFAULT_CLOSEUP_NEGATIVE_ROOT
from paths import DEFAULT_FOOT_ROOT
from paths import DEFAULT_HUMANBODY_ROOT
from paths import DEFAULT_ULCER_ROOT
BODY_FOOT_CATEGORY_IDS = {1}
HUMANBODY_FOOT_CATEGORY_IDS = {5, 10}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path | None = None
    image_id: int | None = None
    annotations: tuple[dict[str, Any], ...] = ()
    augment_profile: str = "none"

    @property
    def is_negative(self) -> bool:
        return len(self.annotations) == 0


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
        foot_root: str | Path = DEFAULT_FOOT_ROOT,
        body_root: str | Path | None = DEFAULT_BODY_ROOT,
        humanbody_root: str | Path | None = DEFAULT_HUMANBODY_ROOT,
        closeup_negative_root: str | Path | None = DEFAULT_CLOSEUP_NEGATIVE_ROOT,
        ulcer_root: str | Path = DEFAULT_ULCER_ROOT,
        image_size: int = 768,
        val_ratio: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        hflip_prob: float = 0.0,
        negative_oversample: int = 1,
        neg_sample_weight: float = 1.0,
        negative_fullbody_scale_min: float = 1.2,
        negative_fullbody_scale_max: float = 1.8,
        negative_closeup_scale_min: float = 2.0,
        negative_closeup_scale_max: float = 3.5,
        synthetic_closeup_from_humanbody: bool = False,
    ) -> None:
        if task not in {"foot", "ulcer"}:
            raise ValueError(f"Unsupported task: {task}")
        if split not in {"train", "val", "validation"}:
            raise ValueError(f"Unsupported split: {split}")

        self.task = task
        self.split = "val" if split == "validation" else split
        self.foot_root = Path(foot_root)
        self.body_root = Path(body_root) if body_root is not None else None
        self.humanbody_root = Path(humanbody_root) if humanbody_root is not None else None
        self.closeup_negative_root = (
            Path(closeup_negative_root) if closeup_negative_root is not None else None
        )
        self.ulcer_root = Path(ulcer_root)
        self.image_size = int(image_size)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.augment = bool(augment and self.split == "train")
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.hflip_prob = float(hflip_prob)
        self.negative_oversample = max(1, int(negative_oversample))
        self.neg_sample_weight = float(neg_sample_weight)
        self.negative_fullbody_scale_min = float(negative_fullbody_scale_min)
        self.negative_fullbody_scale_max = float(negative_fullbody_scale_max)
        self.negative_closeup_scale_min = float(negative_closeup_scale_min)
        self.negative_closeup_scale_max = float(negative_closeup_scale_max)
        self.synthetic_closeup_from_humanbody = bool(synthetic_closeup_from_humanbody)
        self.samples = self._load_samples()

        if not self.samples:
            raise RuntimeError(f"No samples found for task={self.task!r}, split={self.split!r}")

    def _load_samples(self) -> list[SegmentationSample]:
        if self.task == "foot":
            return self._load_foot_samples()
        return self._load_ulcer_samples()

    def _load_foot_samples(self) -> list[SegmentationSample]:
        samples = self._load_roboflow_foot_split()
        if self.split != "train":
            return samples

        extras: list[SegmentationSample] = []
        if self.body_root is not None:
            extras.extend(
                self._load_coco_foot_samples(
                    self.body_root,
                    BODY_FOOT_CATEGORY_IDS,
                    positive_profile="natural",
                    negative_profile="negative_fullbody",
                )
            )
        humanbody_samples: list[SegmentationSample] = []
        if self.humanbody_root is not None:
            humanbody_samples = self._load_coco_foot_samples(
                self.humanbody_root,
                HUMANBODY_FOOT_CATEGORY_IDS,
                positive_profile="natural",
                negative_profile="negative_fullbody",
            )
            extras.extend(humanbody_samples)
        if self.closeup_negative_root is not None:
            extras.extend(self._load_closeup_negative_images(self.closeup_negative_root))
        if self.synthetic_closeup_from_humanbody and humanbody_samples:
            extras.extend(
                replace(sample, augment_profile="negative_closeup")
                for sample in humanbody_samples
                if sample.is_negative
            )

        samples.extend(extras)
        return self._oversample_negatives(samples)

    def _load_roboflow_foot_split(self) -> list[SegmentationSample]:
        annotation_path = self.foot_root / "_annotations.coco.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Roboflow COCO annotation file not found: {annotation_path}")

        with annotation_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

        samples: list[SegmentationSample] = []
        for image_info in coco.get("images", []):
            image_path = self.foot_root / image_info["file_name"]
            if not image_path.exists():
                continue
            image_id = int(image_info["id"])
            annotations = tuple(annotations_by_image.get(image_id, []))
            if self.split == "train" and self.augment:
                profile = "positive"
            else:
                profile = "natural" if annotations else "none"
            samples.append(
                SegmentationSample(
                    image_path=image_path,
                    image_id=image_id,
                    annotations=annotations,
                    augment_profile=profile,
                )
            )

        rng = random.Random(self.seed)
        rng.shuffle(samples)
        val_count = max(1, int(len(samples) * self.val_ratio))
        if self.split == "val":
            return samples[:val_count]
        return samples[val_count:]

    def _load_coco_foot_samples(
        self,
        root: Path,
        foot_category_ids: set[int],
        positive_profile: str,
        negative_profile: str,
    ) -> list[SegmentationSample]:
        annotation_path = root / "_annotations.coco.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Roboflow COCO annotation file not found: {annotation_path}")

        with annotation_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco.get("annotations", []):
            if int(annotation["category_id"]) not in foot_category_ids:
                continue
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

        samples: list[SegmentationSample] = []
        for image_info in coco.get("images", []):
            image_path = root / image_info["file_name"]
            if not image_path.exists():
                continue
            image_id = int(image_info["id"])
            annotations = tuple(annotations_by_image.get(image_id, []))
            profile = positive_profile if annotations else negative_profile
            samples.append(
                SegmentationSample(
                    image_path=image_path,
                    image_id=image_id,
                    annotations=annotations,
                    augment_profile=profile,
                )
            )
        return samples

    @staticmethod
    def _load_closeup_negative_images(root: Path) -> list[SegmentationSample]:
        if not root.exists():
            return []

        samples: list[SegmentationSample] = []
        for image_path in sorted(root.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            samples.append(
                SegmentationSample(
                    image_path=image_path,
                    augment_profile="negative_closeup",
                )
            )
        return samples

    def _oversample_negatives(self, samples: list[SegmentationSample]) -> list[SegmentationSample]:
        if self.negative_oversample <= 1:
            return samples

        positives = [sample for sample in samples if not sample.is_negative]
        negatives = [sample for sample in samples if sample.is_negative]
        return positives + negatives * self.negative_oversample

    def _load_ulcer_samples(self) -> list[SegmentationSample]:
        split_dir = "validation" if self.split == "val" else "train"
        image_dir = self.ulcer_root / split_dir / "images"
        label_dir = self.ulcer_root / split_dir / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"FUSeg split directories not found under: {self.ulcer_root / split_dir}")

        label_by_stem = {path.stem: path for path in sorted(label_dir.iterdir()) if path.is_file()}
        samples = []
        for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
            mask_path = label_by_stem.get(image_path.stem)
            if mask_path is not None:
                samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path))
        return samples

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
                raise RuntimeError(f"Missing ulcer mask for: {sample.image_path}")
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
        if profile == "negative_fullbody":
            return self._apply_scale_hflip(
                image,
                mask,
                self.negative_fullbody_scale_min,
                self.negative_fullbody_scale_max,
                self.hflip_prob,
            )
        if profile == "negative_closeup":
            return self._apply_scale_hflip(
                image,
                mask,
                self.negative_closeup_scale_min,
                self.negative_closeup_scale_max,
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
