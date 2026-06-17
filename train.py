from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from cli.dataset_args import add_dataset_args
from data.loaders import make_loader as make_segmentation_loader
from datasets import ClassificationImageDataset
from losses import binary_segmentation_metrics, segmentation_loss
from models import DINOv3Backbone, DFUFeatureClassifierHead, SingleTaskSegModel
from paths import DEFAULT_DFU_CLASSIFICATION_DATA_ROOT
from paths import DINOV3_CHECKPOINT as DEFAULT_DINOV3_CHECKPOINT
from paths import DINOV3_REPO as DEFAULT_DINOV3_REPO
from paths import TRAIN_OUTPUT_DIR as DEFAULT_TRAIN_OUTPUT_DIR
from training_log import (
    TrainingLogger,
    collect_dataset_stats,
    collect_environment_info,
    count_model_parameters,
)
from utils.runtime import autocast_context
from utils.runtime import make_grad_scaler
from utils.runtime import resolve_device
from utils.runtime import seed_everything

DFU_CLASSES = ("dfu", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one DFU task head with a frozen DINOv3 backbone.")
    add_dataset_args(parser)
    parser.add_argument("--task", type=str, choices=("foot", "ulcer", "dfu"), required=True)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subdirectory name under --output-dir. Defaults to a timestamp when --output-dir is the default.",
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--dfu-root",
        type=Path,
        default=DEFAULT_DFU_CLASSIFICATION_DATA_ROOT,
        help="Binary ImageFolder root for --task dfu.",
    )
    parser.add_argument(
        "--dfu-head-type",
        type=str,
        choices=("linear", "mlp"),
        default="linear",
        help="DFU head architecture. 'linear' matches the notebook-style frozen backbone classifier.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size for --dfu-head-type mlp.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for --dfu-head-type mlp.")
    parser.add_argument(
        "--dfu-lr",
        type=float,
        default=5.0e-3,
        help="Learning rate for --task dfu. Notebook default was 5e-3.",
    )
    parser.add_argument(
        "--dfu-batch-size",
        type=int,
        default=32,
        help="Batch size for --task dfu. Notebook default was 32.",
    )
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="For --task dfu, optionally use inverse-frequency class weights.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for --task dfu cosine scheduler.",
    )
    parser.add_argument(
        "--dfu-best-metric",
        type=str,
        choices=("accuracy", "f1"),
        default="f1",
        help="Checkpoint selection metric for --task dfu.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=("none", "cosine", "step"),
        default="cosine",
        help="Learning rate scheduler. 'cosine' decays to --min-lr over --epochs.",
    )
    parser.add_argument("--min-lr", type=float, default=1.0e-6, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--lr-step-size", type=int, default=10, help="StepLR: decay every N epochs.")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="StepLR: multiply LR by this factor.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=7,
        help="Stop if the task validation score does not improve for this many epochs. 0 disables early stopping.",
    )
    return parser.parse_args()


def move_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    loss_weight = batch.get("loss_weight")
    if loss_weight is None:
        return images, masks, None
    return images, masks, loss_weight.to(device, non_blocking=True)


def move_classification_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return images, labels


def predict_task_logits(
    model: SingleTaskSegModel,
    images: torch.Tensor,
    task: str,
) -> torch.Tensor:
    features = model.encode(images)
    output_size = tuple(int(value) for value in images.shape[-2:])
    if task == "foot":
        return model.predict_foot_logits(features, output_size)
    if task == "ulcer":
        return model.predict_ulcer_logits(features, output_size)
    raise ValueError(f"Unsupported task: {task}")


def predict_dfu_logits(
    backbone: DINOv3Backbone,
    head: DFUFeatureClassifierHead,
    images: torch.Tensor,
) -> torch.Tensor:
    features = backbone(images)
    return head(features)


def configure_task_training(model: SingleTaskSegModel, task: str) -> None:
    for parameter in model.foot_head.parameters():
        parameter.requires_grad = task == "foot"
    for parameter in model.ulcer_head.parameters():
        parameter.requires_grad = task == "ulcer"


def train_one_task_batch(
    model: SingleTaskSegModel,
    batch: dict,
    task: str,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, float]]:
    images, masks, loss_weight = move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)

    with autocast_context(device, use_amp):
        logits = predict_task_logits(model, images, task)
        loss = segmentation_loss(logits, masks, sample_weights=loss_weight)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        metrics = binary_segmentation_metrics(logits, masks)
    return float(loss.detach().item()), metrics


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


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    return finalize_classification_metrics(classification_counts(logits, labels))


def accumulate_classification_counts(
    total: dict[str, int],
    batch_counts: dict[str, int],
) -> None:
    for key in total:
        total[key] += batch_counts[key]


def _average_metric_batches(metric_batches: list[dict[str, float]]) -> dict[str, float]:
    if not metric_batches:
        return {"dice": 0.0, "iou": 0.0, "accuracy": 0.0}
    keys = metric_batches[0].keys()
    return {key: sum(batch[key] for batch in metric_batches) / len(metric_batches) for key in keys}


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


def train_epoch(
    model: SingleTaskSegModel,
    loader: DataLoader,
    task: str,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    limit_batches: int | None,
) -> dict[str, float]:
    model.train()
    total = 0.0
    steps = 0
    metric_batches: list[dict[str, float]] = []

    for batch in _limited(loader, limit_batches):
        loss, metrics = train_one_task_batch(
            model, batch, task, optimizer, scaler, device, use_amp
        )
        total += loss
        steps += 1
        metric_batches.append(metrics)

    loss = total / max(steps, 1)
    task_metrics = _average_metric_batches(metric_batches)
    return {
        "train_loss": loss,
        f"{task}_loss": loss,
        "train_dice": task_metrics["dice"],
        "train_iou": task_metrics["iou"],
        "train_accuracy": task_metrics["accuracy"],
        f"{task}_train_dice": task_metrics["dice"],
        f"{task}_train_iou": task_metrics["iou"],
        f"{task}_train_accuracy": task_metrics["accuracy"],
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

    for batch in _limited(loader, limit_batches):
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
def validate_task(
    model: SingleTaskSegModel,
    loader: DataLoader,
    task: str,
    device: torch.device,
    limit_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_accuracy = 0.0
    steps = 0

    for batch in _limited(loader, limit_batches):
        images, masks, loss_weight = move_batch(batch, device)
        logits = predict_task_logits(model, images, task)
        metrics = binary_segmentation_metrics(logits, masks)
        total_loss += float(segmentation_loss(logits, masks, sample_weights=loss_weight).item())
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_accuracy += metrics["accuracy"]
        steps += 1

    denom = max(steps, 1)
    return {
        f"{task}_val_loss": total_loss / denom,
        f"{task}_val_dice": total_dice / denom,
        f"{task}_val_iou": total_iou / denom,
        f"{task}_val_accuracy": total_accuracy / denom,
        f"{task}_dice": total_dice / denom,
        f"{task}_iou": total_iou / denom,
        f"{task}_accuracy": total_accuracy / denom,
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

    for batch in _limited(loader, limit_batches):
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
        "dfu_accuracy": averaged["accuracy"],
        "dfu_f1": averaged["f1"],
    }


def _limited(loader: Iterable, limit: int | None) -> Iterable:
    if limit is None:
        yield from loader
        return
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        yield batch


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )
    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def save_checkpoint(
    path: Path,
    model: SingleTaskSegModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    task_head = model.foot_head if args.task == "foot" else model.ulcer_head
    payload: dict[str, Any] = {
        "task": args.task,
        "epoch": epoch,
        "model": model.state_dict(),
        "head_state_dict": task_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def save_dfu_checkpoint(
    path: Path,
    head: DFUFeatureClassifierHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task": "dfu",
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
        "head_type": args.dfu_head_type,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if args.run_name:
        return output_dir / args.run_name
    if output_dir.resolve() == DEFAULT_TRAIN_OUTPUT_DIR.resolve():
        return output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir


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
        batch_size=args.dfu_batch_size,
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
        raise ValueError("--task dfu currently supports --lr-scheduler cosine or none.")

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


def collect_task_dataset_info(
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict[str, Any]:
    return {
        "splits": {
            "train": collect_dataset_stats(train_loader.dataset, train_loader),
            "val": collect_dataset_stats(val_loader.dataset, val_loader),
        },
        "total_train_samples": len(train_loader.dataset),
        "total_val_samples": len(val_loader.dataset),
    }


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


def train_dfu_head(args: argparse.Namespace, device: torch.device, use_amp: bool) -> None:
    if args.image_size != 384:
        print(
            f"Warning: --image-size {args.image_size} differs from the shared foot/ulcer pipeline default (384). "
            "Use the same image size across foot, ulcer, and dfu heads at inference."
        )

    train_loader = make_classification_loader("train", args, shuffle=True)
    val_loader = make_classification_loader("val", args, shuffle=False)

    backbone = DINOv3Backbone(
        repo_dir=args.dinov3_repo,
        checkpoint_path=args.dinov3_checkpoint,
        freeze=True,
    ).to(device)
    head = DFUFeatureClassifierHead(
        feature_dim=384,
        hidden_dim=args.hidden_dim,
        num_classes=len(DFU_CLASSES),
        dropout=args.dropout,
        head_type=args.dfu_head_type,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.dfu_lr, weight_decay=args.weight_decay)
    scheduler = make_dfu_lr_scheduler(optimizer, train_loader, args)
    scaler = make_grad_scaler(use_amp)
    class_weights = (
        make_class_weights(train_loader.dataset, device)
        if args.class_weight == "balanced"
        else None
    )
    best_metric_name = "dfu_val_accuracy" if args.dfu_best_metric == "accuracy" else "dfu_val_f1"

    logger = TrainingLogger(args.output_dir)
    logger.write_initial_artifacts(
        args=args,
        dataset_info=collect_dfu_dataset_info(train_loader, val_loader),
        environment=collect_environment_info(device),
        model_info={
            "backbone": backbone.__class__.__name__,
            "head": head.__class__.__name__,
            "head_type": args.dfu_head_type,
            "task": args.task,
            "classes": DFU_CLASSES,
            "backbone_frozen": True,
            **count_model_parameters(head),
        },
    )
    print(f"Training logs will be saved to: {args.output_dir}")
    print(
        f"dfu profile: head={args.dfu_head_type} lr={args.dfu_lr} "
        f"batch={args.dfu_batch_size} best_metric={args.dfu_best_metric}"
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


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_output_dir(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    if args.task == "dfu":
        train_dfu_head(args, device, use_amp)
        return

    train_loader = make_segmentation_loader(args.task, "train", args, shuffle=True)
    val_loader = make_segmentation_loader(args.task, "val", args, shuffle=False)

    backbone = DINOv3Backbone(
        repo_dir=args.dinov3_repo,
        checkpoint_path=args.dinov3_checkpoint,
        freeze=not args.unfreeze_backbone,
    )
    model = SingleTaskSegModel(backbone=backbone).to(device)
    configure_task_training(model, args.task)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_lr_scheduler(optimizer, args)
    scaler = make_grad_scaler(use_amp)

    logger = TrainingLogger(args.output_dir)
    logger.write_initial_artifacts(
        args=args,
        dataset_info=collect_task_dataset_info(train_loader, val_loader),
        environment=collect_environment_info(device),
        model_info={
            "model": model.__class__.__name__,
            "task": args.task,
            "backbone_frozen": not args.unfreeze_backbone,
            **count_model_parameters(model),
        },
    )
    print(f"Training logs will be saved to: {args.output_dir}")

    best_score = -1.0
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None
    training_started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = perf_counter()
        train_metrics = train_epoch(
            model,
            train_loader,
            args.task,
            optimizer,
            scaler,
            device,
            use_amp,
            args.limit_train_batches,
        )
        task_metrics = validate_task(model, val_loader, args.task, device, args.limit_val_batches)
        metrics = {**train_metrics, **task_metrics}
        metrics["val_dice"] = metrics[f"{args.task}_val_dice"]
        metrics["val_iou"] = metrics[f"{args.task}_val_iou"]
        metrics["val_accuracy"] = metrics[f"{args.task}_val_accuracy"]
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        score = metrics["val_dice"]
        epoch_seconds = perf_counter() - epoch_started
        is_best = score > best_score

        display_metrics = {key: value for key, value in metrics.items() if key != "learning_rate"}
        print(
            f"epoch={epoch:03d} | lr={metrics['learning_rate']:.2e} | {format_metrics(display_metrics)}"
        )
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, epoch, metrics, args, scheduler
        )
        if is_best:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, epoch, metrics, args, scheduler
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

        if scheduler is not None:
            scheduler.step()

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            stop_epoch = epoch
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val_dice improvement for {args.early_stopping_patience} epochs "
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
            f"Best checkpoint: epoch {logger.best_epoch} (val_dice={logger.best_score:.4f})"
        )
    print(f"Training complete. Logs and checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
