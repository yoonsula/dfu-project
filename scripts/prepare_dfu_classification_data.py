from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import (
    DEFAULT_DFU_CLASSIFICATION_DATA_ROOT,
    DEFAULT_DFU_CLASSIFICATION_SOURCE_ROOT,
    DEFAULT_DFU_PARTA_ROOT,
)

CLASSES = ("dfu", "other")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PreparedSample:
    source_path: Path
    source_dataset: str
    source_label: str
    label: str
    split: str
    group_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ImageFolder data for binary DFU classification: dfu vs other."
    )
    parser.add_argument(
        "--dfu-dataset-root",
        type=Path,
        default=DEFAULT_DFU_CLASSIFICATION_SOURCE_ROOT,
        help="Existing 3-class DFU Dataset root with train/val/test splits.",
    )
    parser.add_argument(
        "--dfu-parta-root",
        type=Path,
        default=DEFAULT_DFU_PARTA_ROOT,
        help="New dfu_partA root with dfu/ and others/ folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DFU_CLASSIFICATION_DATA_ROOT,
        help="Output ImageFolder root to create.",
    )
    parser.add_argument("--parta-val-ratio", type=float, default=0.10)
    parser.add_argument("--parta-test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize files in the prepared ImageFolder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate --output-root when it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without creating files.",
    )
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def group_id_from_filename(path: Path) -> str:
    match = re.match(r"^([^_]+)_", path.name)
    if match is not None:
        return match.group(1)
    return path.stem


def safe_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._") or "item"


def unique_filename(sample: PreparedSample) -> str:
    digest = hashlib.sha1(str(sample.source_path).encode("utf-8")).hexdigest()[:10]
    source = safe_name(sample.source_dataset)
    source_label = safe_name(sample.source_label)
    stem = safe_name(sample.source_path.stem)
    suffix = sample.source_path.suffix.lower()
    return f"{source}_{source_label}_{stem}_{digest}{suffix}"


def load_existing_dfu_dataset(root: Path) -> list[PreparedSample]:
    label_map = {
        "Diabetic Foot Ulcer": "dfu",
        "Healthy": "other",
        "Wound": "other",
    }
    samples: list[PreparedSample] = []
    for split in SPLITS:
        for source_label, label in label_map.items():
            class_root = root / split / source_label
            for image_path in iter_images(class_root):
                samples.append(
                    PreparedSample(
                        source_path=image_path,
                        source_dataset="dfu_dataset",
                        source_label=source_label,
                        label=label,
                        split=split,
                        group_id=group_id_from_filename(image_path),
                    )
                )
    return samples


def split_groups(
    group_ids: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("--parta-val-ratio + --parta-test-ratio must be in [0, 1).")

    shuffled = sorted(set(group_ids))
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_count = int(round(total * test_ratio))
    val_count = int(round(total * val_ratio))
    test_ids = set(shuffled[:test_count])
    val_ids = set(shuffled[test_count : test_count + val_count])
    return {
        group_id: "test" if group_id in test_ids else "val" if group_id in val_ids else "train"
        for group_id in shuffled
    }


def load_part_a_dataset(
    root: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[PreparedSample]:
    label_map = {
        "dfu": "dfu",
        "others": "other",
    }
    raw_samples: list[tuple[Path, str, str, str]] = []
    for source_label, label in label_map.items():
        for image_path in iter_images(root / source_label):
            group_id = group_id_from_filename(image_path)
            raw_samples.append((image_path, source_label, label, group_id))

    split_by_group = split_groups(
        [group_id for _, _, _, group_id in raw_samples],
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return [
        PreparedSample(
            source_path=image_path,
            source_dataset="partA",
            source_label=source_label,
            label=label,
            split=split_by_group[group_id],
            group_id=group_id,
        )
        for image_path, source_label, label, group_id in raw_samples
    ]


def validate_roots(args: argparse.Namespace) -> None:
    source_roots = [args.dfu_dataset_root.resolve(), args.dfu_parta_root.resolve()]
    output_root = args.output_root.resolve()
    for source_root in source_roots:
        if output_root == source_root or output_root.is_relative_to(source_root):
            raise ValueError(f"--output-root must not be inside a source root: {source_root}")


def materialize_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        target.hardlink_to(source)
    elif mode == "symlink":
        target.symlink_to(source)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def summarize(samples: list[PreparedSample]) -> dict[str, dict[str, int]]:
    counts = {split: {label: 0 for label in CLASSES} for split in SPLITS}
    for sample in samples:
        counts[sample.split][sample.label] += 1
    return counts


def write_manifest(samples: list[PreparedSample], output_root: Path) -> None:
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "label",
                "source_dataset",
                "source_label",
                "group_id",
                "source_path",
                "target_path",
            ],
        )
        writer.writeheader()
        for sample in samples:
            target_path = output_root / sample.split / sample.label / unique_filename(sample)
            writer.writerow(
                {
                    "split": sample.split,
                    "label": sample.label,
                    "source_dataset": sample.source_dataset,
                    "source_label": sample.source_label,
                    "group_id": sample.group_id,
                    "source_path": str(sample.source_path),
                    "target_path": str(target_path),
                }
            )


def prepare_output(samples: list[PreparedSample], args: argparse.Namespace) -> None:
    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists. Pass --overwrite to recreate: {output_root}")
        shutil.rmtree(output_root)

    for split in SPLITS:
        for label in CLASSES:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    for sample in samples:
        target_path = output_root / sample.split / sample.label / unique_filename(sample)
        materialize_file(sample.source_path, target_path, args.mode)
    write_manifest(samples, output_root)


def print_summary(samples: list[PreparedSample]) -> None:
    counts = summarize(samples)
    for split in SPLITS:
        total = sum(counts[split].values())
        print(
            f"{split}: total={total} "
            f"dfu={counts[split]['dfu']} other={counts[split]['other']}"
        )


def main() -> None:
    args = parse_args()
    validate_roots(args)

    samples = [
        *load_existing_dfu_dataset(args.dfu_dataset_root),
        *load_part_a_dataset(
            args.dfu_parta_root,
            val_ratio=args.parta_val_ratio,
            test_ratio=args.parta_test_ratio,
            seed=args.seed,
        ),
    ]
    if not samples:
        raise RuntimeError("No images found for DFU classification data preparation.")

    print_summary(samples)
    if args.dry_run:
        print(f"Dry run only. Output root would be: {args.output_root}")
        return

    prepare_output(samples, args)
    print(f"Prepared DFU classification data: {args.output_root}")
    print(f"Manifest: {args.output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
