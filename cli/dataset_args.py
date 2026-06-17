from __future__ import annotations

import argparse
from pathlib import Path

from datasets.catalog import default_foot_roots
from paths import DEFAULT_BODY_ROOT
from paths import DEFAULT_HUMANBODY_ROOT
from paths import DEFAULT_WOUND_ROOT
from paths import DEFAULT_WOUND_IMAGE_ROOT


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--foot-root",
        type=Path,
        action="append",
        help="COCO foot dataset root (repeatable). Multiple roots are merged automatically.",
    )
    parser.add_argument("--body-root", type=Path, default=DEFAULT_BODY_ROOT)
    parser.add_argument("--humanbody-root", type=Path, default=DEFAULT_HUMANBODY_ROOT)
    parser.add_argument(
        "--negative-oversample",
        type=int,
        default=4,
        help="Repeat foot-negative training samples this many times (train only).",
    )
    parser.add_argument(
        "--neg-loss-weight",
        type=float,
        default=3.0,
        help="Per-sample loss multiplier for foot images with empty masks.",
    )
    parser.add_argument("--wound-root", type=Path, default=DEFAULT_WOUND_ROOT)
    parser.add_argument("--wound-image-root", type=Path, default=DEFAULT_WOUND_IMAGE_ROOT)
    parser.add_argument(
        "--no-wound-image",
        action="store_true",
        help="Exclude Wound Image Dataset (wound_main/wound_mask + Nomal negatives).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--foot-augment", action="store_true")
    parser.add_argument("--foot-scale-min", type=float, default=1.5)
    parser.add_argument("--foot-scale-max", type=float, default=2.5)
    parser.add_argument("--foot-hflip-prob", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Foot roboflow positive val ratio.")
    parser.add_argument(
        "--val-negative-ratio",
        type=float,
        default=0.25,
        help="Target fraction of negatives in foot val (body/humanbody/closeup negatives included).",
    )


def foot_roots_for_args(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.foot_root:
        return tuple(args.foot_root)
    return default_foot_roots()
