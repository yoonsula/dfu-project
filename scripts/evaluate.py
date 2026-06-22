#!/usr/bin/env python3
"""Evaluate a trained DFU classification head on a held-out ImageFolder split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from models import DINOv3Backbone, DFUFeatureClassifierHead
from paths import DINOV3_MODEL_PATH as DEFAULT_DINOV3_MODEL_PATH
from trainers.common import format_metrics
from trainers.dfu_trainer import DFU_CLASSES, validate_dfu
from datasets import ClassificationImageDataset
from utils.runtime import resolve_device


def load_head(checkpoint: Path, device: torch.device) -> DFUFeatureClassifierHead:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a dict: {checkpoint}")

    classes = tuple(payload.get("classes", DFU_CLASSES))
    head = DFUFeatureClassifierHead(
        feature_dim=int(payload.get("feature_dim", 384)),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        num_classes=len(classes),
        dropout=float(payload.get("dropout", 0.2)),
        head_type=str(payload.get("head_type", "linear")),
    ).to(device)

    state_dict = payload.get("head_state_dict", payload)
    head.load_state_dict(state_dict, strict=False)
    head.eval()
    return head


def resolve_image_size(checkpoint: Path, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        args = payload.get("args")
        if isinstance(args, dict) and args.get("image_size") is not None:
            return int(args["image_size"])
    return 384


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DFU classification checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="ImageFolder root (e.g. .../test).")
    parser.add_argument("--dinov3-model", type=Path, default=DEFAULT_DINOV3_MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    image_size = resolve_image_size(args.checkpoint, args.image_size)
    dataset = ClassificationImageDataset(
        root=args.data_root,
        split="train",
        image_size=image_size,
        classes=DFU_CLASSES,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    backbone = DINOv3Backbone(model_path=args.dinov3_model, freeze=True).to(device).eval()
    head = load_head(args.checkpoint, device)
    use_amp = bool(args.amp and device.type == "cuda")
    metrics = validate_dfu(backbone, head, loader, device, use_amp, None, args.limit_batches)

    report = {
        "task": "dfu",
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "image_size": image_size,
        "num_samples": len(dataset),
        "metrics": metrics,
    }
    print(f"samples={len(dataset)} image_size={image_size}")
    print(format_metrics(metrics))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
