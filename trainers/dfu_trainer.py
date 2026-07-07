from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from datasets import ClassificationImageDataset
from models import DINOv3Backbone, DFUFeatureClassifierHead
from inference.checkpoints import DEFAULT_IMAGE_SIZE
from paths import DEFAULT_DFU_CLASSIFICATION_DATA_ROOT
from trainers.training_log import (
    TrainingLogger,
    collect_environment_info,
    count_model_parameters,
)
from trainers.common import add_common_args
from trainers.common import format_metrics
from trainers.common import limited_batches
from trainers.common import prepare_run
from utils.runtime import autocast_context
from utils.runtime import make_grad_scaler

DFU_CLASSES = ("dfu", "other")
TASK = "dfu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the DFU classification head on shared frozen DINOv3 features.",
    )
    add_common_args(parser, default_batch_size=32, default_lr=5.0e-4)
    parser.add_argument(
        "--dfu-root",
        type=Path,
        default=DEFAULT_DFU_CLASSIFICATION_DATA_ROOT,
        help="Binary ImageFolder root for DFU classification.",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        choices=("linear", "mlp"),
        default="linear",
        help="Classification head architecture. 'linear' matches the notebook-style frozen backbone classifier.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size for --head-type mlp.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for --head-type mlp.")
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="Optionally use inverse-frequency class weights.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for the per-step cosine scheduler.",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        choices=("accuracy", "f1"),
        default="f1",
        help="Checkpoint selection metric.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def move_classification_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return images, labels


def predict_dfu_logits(
    backbone: DINOv3Backbone,
    head: DFUFeatureClassifierHead,
    images: torch.Tensor,
) -> torch.Tensor:
    features = backbone(images)
    return head(features)


def classification_counts(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    predictions = logits.argmax(dim=1)
    dfu_label = 0
    return {
        "correct": int((predictions == labels).sum().item()),
        "total": int(labels.numel()),
        "tp": int(((predictions == dfu_label) & (labels == dfu_label)).sum().item()),
        "fp": int(((predictions == dfu_label) & (labels != dfu_label)).sum().item()),
        "fn": int(((predictions != dfu_label) & (labels == dfu_label)).sum().item()),
    }


def finalize_classification_metrics(counts: dict[str, int]) -> dict[str, float]:
    true_positive = counts["tp"]
    false_positive = counts["fp"]
    false_negative = counts["fn"]
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1.0e-8)
    return {
        "accuracy": counts["correct"] / max(counts["total"], 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def accumulate_classification_counts(
    total: dict[str, int],
    batch_counts: dict[str, int],
) -> None:
    for key in total:
        total[key] += batch_counts[key]


def _average_losses_and_global_classification_metrics(
    losses: list[float],
    count_batches: list[dict[str, int]],
) -> dict[str, float]:
    if not count_batches:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    totals = {"correct": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0}
    for batch_counts in count_batches:
        accumulate_classification_counts(totals, batch_counts)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        **finalize_classification_metrics(totals),
    }


def train_dfu_epoch(
    backbone: DINOv3Backbone,
    head: DFUFeatureClassifierHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    class_weights: torch.Tensor | None,
    limit_batches: int | None,
    scheduler: LRScheduler | None = None,
) -> dict[str, float]:
    backbone.eval()
    head.train()
    losses: list[float] = []
    count_batches: list[dict[str, int]] = []

    for batch in limited_batches(loader, limit_batches):
        images, labels = move_classification_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            logits = predict_dfu_logits(backbone, head, images)
            loss = F.cross_entropy(logits, labels, weight=class_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.detach().item()))
        count_batches.append(classification_counts(logits.detach(), labels))

    averaged = _average_losses_and_global_classification_metrics(losses, count_batches)
    return {
        "train_loss": averaged["loss"],
        "dfu_train_accuracy": averaged["accuracy"],
        "dfu_train_precision": averaged["precision"],
        "dfu_train_recall": averaged["recall"],
        "dfu_train_f1": averaged["f1"],
        "train_accuracy": averaged["accuracy"],
        "train_f1": averaged["f1"],
    }


@torch.no_grad()
def validate_dfu(
    backbone: DINOv3Backbone,
    head: DFUFeatureClassifierHead,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    class_weights: torch.Tensor | None,
    limit_batches: int | None,
) -> dict[str, float]:
    backbone.eval()
    head.eval()
    losses: list[float] = []
    count_batches: list[dict[str, int]] = []

    for batch in limited_batches(loader, limit_batches):
        images, labels = move_classification_batch(batch, device)
        with autocast_context(device, use_amp):
            logits = predict_dfu_logits(backbone, head, images)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        losses.append(float(loss.item()))
        count_batches.append(classification_counts(logits, labels))

    averaged = _average_losses_and_global_classification_metrics(losses, count_batches)
    return {
        "dfu_val_loss": averaged["loss"],
        "dfu_val_accuracy": averaged["accuracy"],
        "dfu_val_precision": averaged["precision"],
        "dfu_val_recall": averaged["recall"],
        "dfu_val_f1": averaged["f1"],
    }


def save_dfu_checkpoint(
    path: Path,
    head: DFUFeatureClassifierHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    scheduler: LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task": TASK,
        "epoch": epoch,
        "head_state_dict": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "classes": DFU_CLASSES,
        "id2label": {index: label for index, label in enumerate(DFU_CLASSES)},
        "feature_dim": 384,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "head_type": args.head_type,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def make_classification_loader(
    split: str,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = ClassificationImageDataset(
        root=args.dfu_root,
        split=split,
        image_size=args.image_size,
        classes=DFU_CLASSES,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )


def make_dfu_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    args: argparse.Namespace,
) -> LRScheduler | None:
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler != "cosine":
        raise ValueError("DFU training currently supports --lr-scheduler cosine or none.")

    steps_per_epoch = len(train_loader)
    if args.limit_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.limit_train_batches)
    total_training_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = int(args.warmup_ratio * total_training_steps)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )


def classification_class_counts(dataset: ClassificationImageDataset) -> dict[str, int]:
    counts = Counter(int(sample.label) for sample in dataset.samples)
    return {
        dataset.id2label[index]: counts.get(index, 0)
        for index in range(len(dataset.classes))
    }


def make_class_weights(dataset: ClassificationImageDataset, device: torch.device) -> torch.Tensor | None:
    counts = Counter(int(sample.label) for sample in dataset.samples)
    total = sum(counts.values())
    weights = [
        total / max(len(dataset.classes) * counts.get(index, 0), 1)
        for index in range(len(dataset.classes))
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def collect_dfu_dataset_info(
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict[str, Any]:
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    return {
        "classes": DFU_CLASSES,
        "splits": {
            "train": {
                "count": len(train_dataset),
                "class_counts": classification_class_counts(train_dataset),
                "batch_size": train_loader.batch_size,
                "num_batches": len(train_loader),
            },
            "val": {
                "count": len(val_dataset),
                "class_counts": classification_class_counts(val_dataset),
                "batch_size": val_loader.batch_size,
                "num_batches": len(val_loader),
            },
        },
        "total_train_samples": len(train_dataset),
        "total_val_samples": len(val_dataset),
    }


def train(args: argparse.Namespace) -> None:
    args.task = TASK
    device, use_amp = prepare_run(args)

    if args.image_size != DEFAULT_IMAGE_SIZE:
        print(
            f"Warning: --image-size {args.image_size} differs from the shared foot/wound pipeline default ({DEFAULT_IMAGE_SIZE}). "
            "Use the same image size across foot, wound, and dfu heads at inference."
        )

    train_loader = make_classification_loader("train", args, shuffle=True)
    val_loader = make_classification_loader("val", args, shuffle=False)

    backbone = DINOv3Backbone(
        model_path=args.dinov3_model,
        freeze=True,
    ).to(device)
    head = DFUFeatureClassifierHead(
        feature_dim=384,
        hidden_dim=args.hidden_dim,
        num_classes=len(DFU_CLASSES),
        dropout=args.dropout,
        head_type=args.head_type,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_dfu_lr_scheduler(optimizer, train_loader, args)
    scaler = make_grad_scaler(use_amp)
    class_weights = (
        make_class_weights(train_loader.dataset, device)
        if args.class_weight == "balanced"
        else None
    )
    best_metric_name = "dfu_val_accuracy" if args.best_metric == "accuracy" else "dfu_val_f1"

    logger = TrainingLogger(args.output_dir)
    logger.write_initial_artifacts(
        args=args,
        dataset_info=collect_dfu_dataset_info(train_loader, val_loader),
        environment=collect_environment_info(device),
        model_info={
            "backbone": backbone.__class__.__name__,
            "head": head.__class__.__name__,
            "head_type": args.head_type,
            "task": TASK,
            "classes": DFU_CLASSES,
            "backbone_frozen": True,
            **count_model_parameters(head),
        },
    )
    print(f"Training logs will be saved to: {args.output_dir}")
    print(
        f"dfu profile: head={args.head_type} lr={args.lr} "
        f"batch={args.batch_size} best_metric={args.best_metric}"
    )
    if class_weights is not None:
        print(f"class_weights={class_weights.detach().cpu().tolist()}")

    best_score = -1.0
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None
    training_started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = perf_counter()
        train_metrics = train_dfu_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            class_weights,
            args.limit_train_batches,
            scheduler,
        )
        task_metrics = validate_dfu(
            backbone,
            head,
            val_loader,
            device,
            use_amp,
            class_weights,
            args.limit_val_batches,
        )
        metrics = {**train_metrics, **task_metrics}
        metrics["val_accuracy"] = metrics["dfu_val_accuracy"]
        metrics["val_f1"] = metrics["dfu_val_f1"]
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        score = metrics[best_metric_name]
        epoch_seconds = perf_counter() - epoch_started
        is_best = score > best_score

        display_metrics = {key: value for key, value in metrics.items() if key != "learning_rate"}
        print(
            f"epoch={epoch:03d} | lr={metrics['learning_rate']:.2e} | {format_metrics(display_metrics)}"
        )
        save_dfu_checkpoint(
            args.output_dir / "last.pt", head, optimizer, epoch, metrics, args, scheduler
        )
        if is_best:
            best_score = score
            epochs_without_improvement = 0
            save_dfu_checkpoint(
                args.output_dir / "best.pt", head, optimizer, epoch, metrics, args, scheduler
            )
        else:
            epochs_without_improvement += 1

        logger.log_epoch(
            epoch=epoch,
            metrics=metrics,
            score=score,
            epoch_seconds=epoch_seconds,
            is_best=is_best,
        )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            stop_epoch = epoch
            print(
                f"Early stopping at epoch {epoch}: "
                f"no {best_metric_name} improvement for {args.early_stopping_patience} epochs "
                f"(best epoch {logger.best_epoch}, score {logger.best_score:.4f})"
            )
            break

    logger.finalize(
        total_seconds=perf_counter() - training_started,
        early_stopping={
            "enabled": args.early_stopping_patience > 0,
            "patience": args.early_stopping_patience,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        },
    )
    if stopped_early:
        print(
            f"Training stopped early at epoch {stop_epoch}. "
            f"Best checkpoint: epoch {logger.best_epoch} ({best_metric_name}={logger.best_score:.4f})"
        )
    print(f"Training complete. Logs and checkpoints saved to: {args.output_dir}")


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
