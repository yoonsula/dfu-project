from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cli.dataset_args import add_dataset_args
from data.loaders import make_loader
from datasets import ClassificationImageDataset
from models import DINOv3Backbone
from paths import DINOV3_CHECKPOINT as DEFAULT_DINOV3_CHECKPOINT
from paths import DINOV3_REPO as DEFAULT_DINOV3_REPO
from paths import FEATURE_CACHE_DIR as DEFAULT_FEATURE_CACHE_DIR
from utils.runtime import autocast_context
from utils.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and save frozen DINOv3 backbone features for head-only training.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("foot", "ulcer", "dfu", "both", "all"),
        default="both",
        help="Which task splits to cache.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "both"),
        default="both",
        help="Which dataset splits to cache.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subdirectory under --output-dir. Defaults to a timestamp.",
    )
    parser.add_argument("--shard-size", type=int, default=64, help="Samples per saved .pt shard.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("fp16", "fp32"),
        default="fp16",
        help="Storage dtype for backbone features.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional multitask checkpoint; loads backbone weights from it.",
    )
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument(
        "--dfu-root",
        type=Path,
        default=None,
        help="DFU classification ImageFolder root. Expected root/class_name/*.jpg or root/train/class_name/*.jpg.",
    )
    parser.add_argument(
        "--dfu-csv",
        type=Path,
        default=None,
        help="DFU classification CSV with image_path/path/image and label/class/class_name columns.",
    )
    parser.add_argument(
        "--dfu-class",
        action="append",
        default=None,
        help="Classification class name in fixed order. Repeat for deterministic label mapping.",
    )

    add_dataset_args(parser)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if args.run_name:
        return output_dir / args.run_name
    return output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")


def load_backbone(args: argparse.Namespace, device: torch.device) -> DINOv3Backbone:
    backbone = DINOv3Backbone(
        repo_dir=args.dinov3_repo,
        checkpoint_path=args.dinov3_checkpoint,
        freeze=True,
    )
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model_state = payload.get("model", payload)
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in model_state.items()
            if key.startswith("backbone.")
        }
        if not backbone_state:
            raise ValueError(f"No backbone.* weights found in checkpoint: {args.checkpoint}")
        backbone.load_state_dict(backbone_state, strict=True)
    return backbone.to(device).eval()


def storage_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.float32


def flush_shard(
    out_dir: Path,
    shard_index: int,
    features: list[torch.Tensor],
    masks: list[torch.Tensor] | None,
    loss_weights: list[torch.Tensor] | None,
    labels: list[torch.Tensor] | None,
    image_paths: list[str],
    dtype: torch.dtype,
) -> Path:
    shard_path = out_dir / f"shard_{shard_index:05d}.pt"
    payload = {
        "features": torch.stack(features).to(dtype=dtype),
        "image_paths": image_paths,
    }
    if masks is not None:
        payload["masks"] = torch.stack(masks)
    if loss_weights is not None:
        payload["loss_weights"] = torch.stack(loss_weights)
    if labels is not None:
        payload["labels"] = torch.stack(labels)
    torch.save(payload, shard_path)
    return shard_path


@torch.no_grad()
def cache_split(
    backbone: DINOv3Backbone,
    loader: DataLoader,
    out_dir: Path,
    dtype: torch.dtype,
    shard_size: int,
    device: torch.device,
    use_amp: bool,
    limit_batches: int | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pending_features: list[torch.Tensor] = []
    pending_masks: list[torch.Tensor] | None = None
    pending_weights: list[torch.Tensor] | None = None
    pending_labels: list[torch.Tensor] | None = None
    pending_paths: list[str] = []
    shard_names: list[str] = []
    shard_sizes: list[int] = []
    shard_index = 0
    total = 0
    feature_shape: list[int] | None = None
    mask_shape: list[int] | None = None

    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            features = backbone(images)

        features = features.detach().cpu()
        batch_size = features.shape[0]
        if feature_shape is None:
            feature_shape = list(features.shape[1:])

        masks = batch.get("mask")
        if masks is not None:
            masks = masks.detach().cpu()
            if pending_masks is None:
                pending_masks = []
            if mask_shape is None:
                mask_shape = list(masks.shape[1:])

        weights = batch.get("loss_weight")
        if weights is not None:
            weights = weights.detach().cpu()
            if pending_weights is None:
                pending_weights = []
        elif masks is not None:
            weights = torch.ones(batch_size, dtype=torch.float32)
            if pending_weights is None:
                pending_weights = []

        labels = batch.get("label")
        if labels is not None:
            labels = labels.detach().cpu()
            if pending_labels is None:
                pending_labels = []

        paths = batch["image_path"]
        if isinstance(paths, str):
            paths = [paths]

        for sample_index in range(batch_size):
            pending_features.append(features[sample_index])
            if pending_masks is not None and masks is not None:
                pending_masks.append(masks[sample_index])
            if pending_weights is not None and weights is not None:
                pending_weights.append(weights[sample_index])
            if pending_labels is not None and labels is not None:
                pending_labels.append(labels[sample_index])
            pending_paths.append(paths[sample_index])
            total += 1

            if len(pending_features) >= shard_size:
                shard_path = flush_shard(
                    out_dir,
                    shard_index,
                    pending_features,
                    pending_masks,
                    pending_weights,
                    pending_labels,
                    pending_paths,
                    dtype,
                )
                shard_names.append(shard_path.name)
                shard_sizes.append(len(pending_features))
                shard_index += 1
                pending_features = []
                pending_masks = [] if pending_masks is not None else None
                pending_weights = [] if pending_weights is not None else None
                pending_labels = [] if pending_labels is not None else None
                pending_paths = []

        if (batch_index + 1) % 20 == 0:
            print(f"  cached {total} samples ({batch_index + 1} batches)")

    if pending_features:
        shard_path = flush_shard(
            out_dir,
            shard_index,
            pending_features,
            pending_masks,
            pending_weights,
            pending_labels,
            pending_paths,
            dtype,
        )
        shard_names.append(shard_path.name)
        shard_sizes.append(len(pending_features))

    manifest = {
        "count": total,
        "shards": shard_names,
        "shard_sizes": shard_sizes,
        "feature_shape": feature_shape,
        "mask_shape": mask_shape,
        "dtype": "fp16" if dtype == torch.float16 else "fp32",
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def tasks_for_args(args: argparse.Namespace) -> list[str]:
    if args.task == "both":
        return ["foot", "ulcer"]
    if args.task == "all":
        return ["foot", "ulcer", "dfu"]
    return [args.task]


def splits_for_args(args: argparse.Namespace) -> list[str]:
    if args.split == "both":
        return ["train", "val"]
    return [args.split]


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    dtype = storage_dtype(args.dtype)

    backbone = load_backbone(args, device)
    top_manifest: dict[str, Any] = {
        "image_size": args.image_size,
        "feature_dim": backbone.feature_dim,
        "backbone_checkpoint": str(args.dinov3_checkpoint),
        "trained_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "tasks": {},
    }

    for task in tasks_for_args(args):
        top_manifest["tasks"][task] = {}
        for split in splits_for_args(args):
            print(f"Caching {task}/{split} -> {output_dir / task / split}")
            if task == "dfu":
                dataset = ClassificationImageDataset(
                    root=args.dfu_root,
                    csv_path=args.dfu_csv,
                    split=split,
                    image_size=args.image_size,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                    classes=args.dfu_class,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
                    drop_last=False,
                )
                top_manifest["tasks"][task]["classes"] = list(dataset.classes)
                top_manifest["tasks"][task]["id2label"] = dataset.id2label
            else:
                loader = make_loader(task, split, args, shuffle=False)
            split_manifest = cache_split(
                backbone=backbone,
                loader=loader,
                out_dir=output_dir / task / split,
                dtype=dtype,
                shard_size=max(1, args.shard_size),
                device=device,
                use_amp=use_amp,
                limit_batches=args.limit_batches,
            )
            if task == "dfu":
                split_manifest["classes"] = list(dataset.classes)
                split_manifest["id2label"] = dataset.id2label
                with (output_dir / task / split / "manifest.json").open("w", encoding="utf-8") as handle:
                    json.dump(split_manifest, handle, indent=2, ensure_ascii=False)
            top_manifest["tasks"][task][split] = split_manifest
            print(f"  done: {split_manifest['count']} samples, {len(split_manifest['shards'])} shards")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(top_manifest, handle, indent=2, ensure_ascii=False)
    print(f"Saved feature cache to {output_dir}")


if __name__ == "__main__":
    main()
