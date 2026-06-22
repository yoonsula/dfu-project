from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch

from paths import DINOV3_MODEL_PATH as DEFAULT_DINOV3_MODEL_PATH
from paths import TRAIN_OUTPUT_DIR as DEFAULT_TRAIN_OUTPUT_DIR
from utils.runtime import resolve_device
from utils.runtime import seed_everything


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_image_size: int = 384,
    default_epochs: int = 30,
    default_batch_size: int = 32,
    default_lr: float = 5.0e-4,
) -> None:
    parser.add_argument(
        "--dinov3-model",
        type=Path,
        default=DEFAULT_DINOV3_MODEL_PATH,
        help="Local Hugging Face snapshot directory for the frozen DINOv3 ViT-S/16 backbone.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subdirectory under --output-dir. Defaults to timestamp when using the default output dir.",
    )
    parser.add_argument("--image-size", type=int, default=default_image_size)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=("none", "cosine"),
        default="cosine",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=7)


def prepare_run(args: argparse.Namespace) -> tuple[torch.device, bool]:
    args.output_dir = resolve_output_dir(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    return device, use_amp


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if args.run_name:
        return output_dir / args.run_name
    if output_dir.resolve() == DEFAULT_TRAIN_OUTPUT_DIR.resolve():
        return output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def limited_batches(loader: Iterable, limit: int | None) -> Iterable:
    if limit is None:
        yield from loader
        return
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        yield batch
