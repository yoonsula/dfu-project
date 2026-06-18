from __future__ import annotations

from pathlib import Path

from datasets.samples import SegmentationSample
from datasets.source_loaders import load_coco_samples, load_image_mask_pairs


def load_foot_test_samples(data_root: Path) -> list[SegmentationSample]:
    """Load all COCO foot samples from a single root (test/eval set)."""
    return load_coco_samples(
        data_root,
        positive_profile="natural",
        negative_profile="negative_fullbody",
    )


def load_wound_test_samples(image_dir: Path, mask_dir: Path) -> list[SegmentationSample]:
    return load_image_mask_pairs(image_dir, mask_dir)
