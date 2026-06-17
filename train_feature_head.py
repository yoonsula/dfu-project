from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import CachedFeatureDataset
from losses import binary_segmentation_metrics, segmentation_loss
from models import DFUFeatureClassifierHead, FastInstFootHead, FastInstUlcerHead
from paths import FEATURE_CACHE_DIR as DEFAULT_FEATURE_CACHE_DIR
from paths import TRAIN_OUTPUT_DIR as DEFAULT_TRAIN_OUTPUT_DIR
from utils.runtime import resolve_device
from utils.runtime import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one head from frozen DINOv3 feature cache.")
    parser.add_argument("--task", choices=("foot", "ulcer", "dfu"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--foot-num-queries", type=int, default=8)
    parser.add_argument("--ulcer-num-queries", type=int, default=16)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"{args.task}_head_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return Path(args.output_dir) / name


def cache_split_dir(args: argparse.Namespace, split: str) -> Path:
    root = Path(args.cache_dir)
    nested = root / args.task / split
    return nested if nested.exists() else root / split


def make_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    dataset = CachedFeatureDataset(cache_split_dir(args, split))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )


def make_head(args: argparse.Namespace, train_manifest: dict[str, Any]) -> torch.nn.Module:
    if args.task == "foot":
        return FastInstFootHead(
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_queries=args.foot_num_queries,
        )
    if args.task == "ulcer":
        return FastInstUlcerHead(
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_queries=args.ulcer_num_queries,
        )

    classes = train_manifest.get("classes")
    num_classes = len(classes) if isinstance(classes, list) and classes else 3
    return DFUFeatureClassifierHead(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
    )


def _limited(loader: DataLoader, limit: int | None):
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def train_epoch(
    head: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    task: str,
    device: torch.device,
    limit_batches: int | None,
) -> dict[str, float]:
    head.train()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    metric_batches: list[dict[str, float]] = []
    steps = 0

    for batch in _limited(loader, limit_batches):
        features = batch["features"].to(device)
        optimizer.zero_grad(set_to_none=True)
        if task == "dfu":
            labels = batch["label"].to(device)
            logits = head(features)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            total_correct += float((preds == labels).sum().item())
            total_count += int(labels.numel())
        else:
            masks = batch["mask"].to(device)
            loss_weight = batch.get("loss_weight")
            if loss_weight is not None:
                loss_weight = loss_weight.to(device)
            logits = head(features)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits, masks, sample_weights=loss_weight)
            metric_batches.append(binary_segmentation_metrics(logits, masks))

        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())
        steps += 1

    metrics = {"loss": total_loss / max(steps, 1)}
    if task == "dfu":
        metrics["accuracy"] = total_correct / max(total_count, 1)
    elif metric_batches:
        metrics["dice"] = sum(item["dice"] for item in metric_batches) / len(metric_batches)
        metrics["iou"] = sum(item["iou"] for item in metric_batches) / len(metric_batches)
    return metrics


@torch.no_grad()
def validate(
    head: torch.nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
    limit_batches: int | None,
) -> dict[str, float]:
    head.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    metric_batches: list[dict[str, float]] = []
    steps = 0

    for batch in _limited(loader, limit_batches):
        features = batch["features"].to(device)
        if task == "dfu":
            labels = batch["label"].to(device)
            logits = head(features)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            total_correct += float((preds == labels).sum().item())
            total_count += int(labels.numel())
        else:
            masks = batch["mask"].to(device)
            loss_weight = batch.get("loss_weight")
            if loss_weight is not None:
                loss_weight = loss_weight.to(device)
            logits = head(features)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits, masks, sample_weights=loss_weight)
            metric_batches.append(binary_segmentation_metrics(logits, masks))

        total_loss += float(loss.item())
        steps += 1

    metrics = {"val_loss": total_loss / max(steps, 1)}
    if task == "dfu":
        metrics["val_accuracy"] = total_correct / max(total_count, 1)
    elif metric_batches:
        metrics["val_dice"] = sum(item["dice"] for item in metric_batches) / len(metric_batches)
        metrics["val_iou"] = sum(item["iou"] for item in metric_batches) / len(metric_batches)
    return metrics


def score_for_task(metrics: dict[str, float], task: str) -> float:
    if task == "dfu":
        return metrics.get("val_accuracy", 0.0)
    return metrics.get("val_dice", 0.0)


def save_head_checkpoint(
    path: Path,
    head: torch.nn.Module,
    args: argparse.Namespace,
    train_manifest: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task": args.task,
        "head_state_dict": head.state_dict(),
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
    }
    if args.task == "dfu":
        classes = train_manifest.get("classes", ["TS6_normal skin", "diabetic ulcer", "other_injury"])
        payload["classes"] = classes
        payload["id2label"] = {index: label for index, label in enumerate(classes)}
        payload["dropout"] = args.dropout
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args, "train", shuffle=True)
    val_loader = make_loader(args, "val", shuffle=False)
    head = make_head(args, train_loader.dataset.manifest).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False, default=str)

    best_score = -1.0
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            head,
            train_loader,
            optimizer,
            args.task,
            device,
            args.limit_train_batches,
        )
        val_metrics = validate(head, val_loader, args.task, device, args.limit_val_batches)
        metrics = {**train_metrics, **val_metrics}
        score = score_for_task(metrics, args.task)
        is_best = score > best_score
        if is_best:
            best_score = score
            save_head_checkpoint(
                output_dir / "best.pt",
                head,
                args,
                train_loader.dataset.manifest,
                epoch,
                metrics,
            )
        save_head_checkpoint(
            output_dir / "last.pt",
            head,
            args,
            train_loader.dataset.manifest,
            epoch,
            metrics,
        )
        metric_text = " | ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        print(f"epoch={epoch:03d} | best={best_score:.4f} | {metric_text}")

    elapsed = perf_counter() - started
    print(f"Training complete in {elapsed:.1f}s. Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
