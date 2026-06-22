#!/usr/bin/env python3
"""Detect feet with the foot segmentation head, crop, and export ImageFolder crops.

When foot detection/cropping fails, the original image is saved unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.checkpoints import load_pipeline_model
from inference.checkpoints import resolve_image_size_from_checkpoint
from inference.pipeline import SegmentationConfig
from inference.pipeline import bbox_from_mask
from inference.pipeline import run_gated_segmentation
from paths import DINOV3_MODEL_PATH as DEFAULT_DINOV3_MODEL_PATH
from utils.image_io import IMAGE_EXTENSIONS
from utils.image_io import iter_images
from utils.runtime import autocast_context
from utils.runtime import resolve_device

DEFAULT_SPLITS = ("train", "val", "test", "validation")


@dataclass(frozen=True)
class SourceImage:
    image_path: Path
    class_name: str
    split: str | None


@dataclass(frozen=True)
class CropRecord:
    source_path: Path
    class_name: str
    split: str | None
    output_path: Path
    status: str
    foot_area_ratio: float | None
    bbox: tuple[int, int, int, int] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the foot segmentation head on images, crop the detected foot region, "
            "and write ImageFolder-style classification crops."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "ImageFolder root. Supports class folders directly (dfu/other/...) or "
            "split folders (train/dfu, val/other, ...)."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--foot-head-checkpoint",
        type=Path,
        required=True,
        help="Trained foot head checkpoint (best.pt).",
    )
    parser.add_argument("--dinov3-model", type=Path, default=DEFAULT_DINOV3_MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--foot-threshold", type=float, default=0.5)
    parser.add_argument("--min-foot-ratio", type=float, default=0.01)
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.10,
        help="Expand the foot mask bbox by this ratio on each side.",
    )
    parser.add_argument(
        "--square-crop",
        action="store_true",
        help="Expand the bbox to a square centered on the foot mask.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=None,
        help="Optional final resize (square) after cropping.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Optional CSV log of source/output paths and crop status.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary with counts per class/split/status.",
    )
    return parser.parse_args()


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _class_dirs(path: Path) -> list[Path]:
    return [child for child in sorted(path.iterdir()) if child.is_dir()]


def discover_source_images(input_root: Path) -> list[SourceImage]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    split_dirs = [child for child in _class_dirs(input_root) if child.name in DEFAULT_SPLITS]
    if split_dirs:
        samples: list[SourceImage] = []
        for split_dir in split_dirs:
            split_name = "val" if split_dir.name == "validation" else split_dir.name
            for class_dir in _class_dirs(split_dir):
                for image_path in iter_images(class_dir):
                    samples.append(
                        SourceImage(
                            image_path=image_path,
                            class_name=class_dir.name,
                            split=split_name,
                        )
                    )
        if samples:
            return samples

    class_dirs = _class_dirs(input_root)
    if not class_dirs:
        return [
            SourceImage(image_path=image_path, class_name="unknown", split=None)
            for image_path in iter_images(input_root)
        ]

    samples = []
    for class_dir in class_dirs:
        for image_path in iter_images(class_dir):
            samples.append(
                SourceImage(
                    image_path=image_path,
                    class_name=class_dir.name,
                    split=None,
                )
            )
    return samples


def square_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    x_min, y_min, x_max, y_max = bbox
    box_width = x_max - x_min
    box_height = y_max - y_min
    side = max(box_width, box_height)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    half = side / 2.0
    new_x_min = int(round(center_x - half))
    new_y_min = int(round(center_y - half))
    new_x_max = int(round(center_x + half))
    new_y_max = int(round(center_y + half))
    return (
        max(0, new_x_min),
        max(0, new_y_min),
        min(width, new_x_max),
        min(height, new_y_max),
    )


def output_path_for(
    sample: SourceImage,
    output_root: Path,
    source_path: Path,
) -> Path:
    digest = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:10]
    stem = f"{source_path.stem}_{digest}{source_path.suffix.lower()}"
    if sample.split:
        return output_root / sample.split / sample.class_name / stem
    return output_root / sample.class_name / stem


def crop_image(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    square: bool,
    output_size: int | None,
) -> Image.Image:
    if square:
        bbox = square_bbox(bbox, image.size)
    cropped = image.crop(bbox)
    if output_size is not None:
        cropped = cropped.resize((output_size, output_size), Image.Resampling.BILINEAR)
    return cropped


@torch.inference_mode()
def process_image(
    model,
    image_path: Path,
    *,
    image_size: int,
    foot_threshold: float,
    min_foot_ratio: float,
    crop_margin: float,
    square_crop: bool,
    output_size: int | None,
    device: torch.device,
    use_amp: bool,
) -> tuple[str, float | None, tuple[int, int, int, int] | None, Image.Image | None]:
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    segmentation = run_gated_segmentation(
        model,
        image,
        SegmentationConfig(
            image_size=image_size,
            foot_threshold=foot_threshold,
            wound_threshold=0.5,
            guide_enabled=False,
            min_foot_ratio=min_foot_ratio,
            max_foot_ratio=1.0,
            center_tolerance=1.0,
            min_wound_ratio=1.0,
            wound_feature_crop=False,
        ),
        device,
        output_size=image.size,
        autocast_context=lambda: autocast_context(device, use_amp),
    )

    if not segmentation.foot_detected:
        return "no_foot", segmentation.foot_area_ratio, None, image

    if segmentation.foot_area_ratio < min_foot_ratio:
        return "foot_too_small", segmentation.foot_area_ratio, None, image

    bbox = bbox_from_mask(segmentation.foot_mask, crop_margin)
    if bbox is None:
        return "empty_bbox", segmentation.foot_area_ratio, None, image

    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        return "invalid_bbox", segmentation.foot_area_ratio, bbox, image

    cropped = crop_image(
        image,
        bbox,
        square=square_crop,
        output_size=output_size,
    )
    return "ok", segmentation.foot_area_ratio, bbox, cropped


def write_manifest(records: list[CropRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "class_name",
                "split",
                "output_path",
                "status",
                "foot_area_ratio",
                "bbox",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_path": str(record.source_path),
                    "class_name": record.class_name,
                    "split": record.split or "",
                    "output_path": str(record.output_path),
                    "status": record.status,
                    "foot_area_ratio": (
                        "" if record.foot_area_ratio is None else f"{record.foot_area_ratio:.6f}"
                    ),
                    "bbox": "" if record.bbox is None else json.dumps(record.bbox),
                }
            )


def summarize(records: list[CropRecord]) -> dict[str, object]:
    summary: dict[str, object] = {
        "total": len(records),
        "by_status": {},
        "by_class": {},
    }
    by_status: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for record in records:
        by_status[record.status] = by_status.get(record.status, 0) + 1
        if record.status != "skipped_exists":
            key = record.class_name if record.split is None else f"{record.split}/{record.class_name}"
            by_class[key] = by_class.get(key, 0) + 1
    summary["by_status"] = by_status
    summary["by_class"] = by_class
    return summary


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    image_size = resolve_image_size_from_checkpoint(args.foot_head_checkpoint, args.image_size)

    samples = discover_source_images(args.input_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError(f"No images found under: {args.input_root}")

    print(f"Found {len(samples)} source images under {args.input_root}")
    print(f"Foot head: {args.foot_head_checkpoint}")
    print(f"image_size={image_size} device={device} amp={use_amp}")

    if args.dry_run:
        for sample in samples[:10]:
            print(f"  [{sample.split or '-'}] {sample.class_name}: {sample.image_path}")
        if len(samples) > 10:
            print(f"  ... and {len(samples) - 10} more")
        return

    model = load_pipeline_model(
        foot_head_checkpoint=args.foot_head_checkpoint,
        wound_head_checkpoint=None,
        dinov3_model=args.dinov3_model,
        device=device,
    )

    records: list[CropRecord] = []
    started = perf_counter()
    for index, sample in enumerate(samples, start=1):
        output_path = output_path_for(sample, args.output_root, sample.image_path)
        if output_path.exists() and not args.overwrite:
            records.append(
                CropRecord(
                    source_path=sample.image_path,
                    class_name=sample.class_name,
                    split=sample.split,
                    output_path=output_path,
                    status="skipped_exists",
                    foot_area_ratio=None,
                    bbox=None,
                )
            )
            continue

        status, foot_area_ratio, bbox, output_image = process_image(
            model,
            sample.image_path,
            image_size=image_size,
            foot_threshold=args.foot_threshold,
            min_foot_ratio=args.min_foot_ratio,
            crop_margin=args.crop_margin,
            square_crop=args.square_crop,
            output_size=args.output_size,
            device=device,
            use_amp=use_amp,
        )
        if output_image is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_image.save(output_path)

        records.append(
            CropRecord(
                source_path=sample.image_path,
                class_name=sample.class_name,
                split=sample.split,
                output_path=output_path,
                status=status,
                foot_area_ratio=foot_area_ratio,
                bbox=bbox,
            )
        )
        if index % 25 == 0 or index == len(samples):
            saved_count = sum(1 for record in records if record.status != "skipped_exists")
            ok_count = sum(1 for record in records if record.status == "ok")
            print(f"processed {index}/{len(samples)} | saved={saved_count} (cropped={ok_count})")

    summary = summarize(records)
    elapsed = perf_counter() - started
    print(f"done in {elapsed:.1f}s")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    manifest_path = args.manifest_csv or (args.output_root / "crop_manifest.csv")
    write_manifest(records, manifest_path)
    print(f"manifest: {manifest_path}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
