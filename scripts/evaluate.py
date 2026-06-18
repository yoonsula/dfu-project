#!/usr/bin/env python3
"""Evaluate a trained head checkpoint on a held-out dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.runners import evaluate_dfu, evaluate_foot, evaluate_wound, save_eval_report
from paths import DINOV3_MODEL_PATH as DEFAULT_DINOV3_MODEL_PATH
from trainers.common import format_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate foot/wound segmentation or DFU classification on a held-out dataset.",
    )
    parser.add_argument("--task", choices=("foot", "wound", "dfu"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Head checkpoint (best.pt).")
    parser.add_argument(
        "--dinov3-model",
        type=Path,
        default=DEFAULT_DINOV3_MODEL_PATH,
        help="Local Hugging Face snapshot directory for the frozen DINOv3 ViT-S/16 backbone.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input resolution. Defaults to checkpoint args.image_size, then 384.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full evaluation report.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Foot: COCO root. DFU: ImageFolder root with dfu/other class subfolders.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Wound images (paired with --mask-dir by filename stem).",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Wound masks (paired with --image-dir by filename stem).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 32 if args.task == "dfu" else 8

    common = dict(
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        dinov3_model=args.dinov3_model,
        device_name=args.device,
        amp=args.amp,
        limit_batches=args.limit_batches,
    )

    if args.task == "foot":
        if args.data_root is None:
            raise SystemExit("--task foot requires --data-root (COCO folder with _annotations.coco.json).")
        report = evaluate_foot(data_root=args.data_root, **common)
        metric_keys = ("foot_val_dice", "foot_val_iou", "foot_val_accuracy")
    elif args.task == "wound":
        if args.image_dir is None or args.mask_dir is None:
            raise SystemExit("--task wound requires --image-dir and --mask-dir.")
        report = evaluate_wound(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            **common,
        )
        metric_keys = ("wound_val_dice", "wound_val_iou", "wound_val_accuracy")
    else:
        if args.data_root is None:
            raise SystemExit(
                "--task dfu requires --data-root (ImageFolder root with dfu/ and other/ subfolders)."
            )
        report = evaluate_dfu(data_root=args.data_root, **common)
        metric_keys = ("dfu_val_accuracy", "dfu_val_f1", "dfu_val_precision", "dfu_val_recall")

    metrics = report["metrics"]
    summary = {key: metrics[key] for key in metric_keys if key in metrics}
    print(f"task={report['task']} samples={report['num_samples']} image_size={report['image_size']}")
    print(format_metrics(summary))
    if args.output_json is not None:
        save_eval_report(report, args.output_json)
        print(f"Saved report: {args.output_json}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
