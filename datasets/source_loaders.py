from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets.samples import SegmentationSample
from utils.image_io import IMAGE_EXTENSIONS


def load_coco_samples(
    root: Path,
    *,
    category_ids: set[int] | None = None,
    positive_profile: str = "natural",
    negative_profile: str = "none",
    missing_ok: bool = False,
) -> list[SegmentationSample]:
    annotation_path = root / "_annotations.coco.json"
    if not annotation_path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as handle:
        coco: dict[str, Any] = json.load(handle)

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        if category_ids is not None and int(annotation["category_id"]) not in category_ids:
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


def load_image_mask_pairs(image_dir: Path, mask_dir: Path) -> list[SegmentationSample]:
    """Load paired image/mask files matched by filename stem."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_by_stem = {path.stem: path for path in sorted(mask_dir.iterdir()) if path.is_file()}
    samples: list[SegmentationSample] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is not None:
            samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path))
    return samples


def load_fuseg_samples(root: Path, split: str) -> list[SegmentationSample]:
    if split in ("val", "validation"):
        split_dir = "validation"
    elif split == "test":
        split_dir = "test"
    else:
        split_dir = "train"
    image_dir = root / split_dir / "images"
    label_dir = root / split_dir / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"FUSeg split directories not found under: {root / split_dir}")

    label_by_stem = {path.stem: path for path in sorted(label_dir.iterdir()) if path.is_file()}
    samples: list[SegmentationSample] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        mask_path = label_by_stem.get(image_path.stem)
        if mask_path is not None:
            samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path))
    return samples


def load_wound_image_samples(root: Path) -> list[SegmentationSample]:
    if not root.exists():
        return []

    samples: list[SegmentationSample] = []
    main_dir = root / "wound_main"
    mask_dir = root / "wound_mask"
    if main_dir.is_dir() and mask_dir.is_dir():
        for image_path in sorted(path for path in main_dir.iterdir() if path.is_file()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            suffix = image_path.stem.removeprefix("wound_main-")
            mask_path = mask_dir / f"wound_mask-{suffix}{image_path.suffix}"
            if mask_path.is_file():
                samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path))

    for normal_dir_name in ("Nomal", "Normal"):
        normal_dir = root / normal_dir_name
        if not normal_dir.is_dir():
            continue
        for image_path in sorted(path for path in normal_dir.iterdir() if path.is_file()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(SegmentationSample(image_path=image_path))
    return samples
