from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from cli.dataset_args import add_dataset_args
from data.loaders import make_loader
from losses import binary_segmentation_metrics, segmentation_loss
from models import DINOv3Backbone, MultiTaskSegModel
from paths import DINOV3_CHECKPOINT as DEFAULT_DINOV3_CHECKPOINT
from paths import DINOV3_REPO as DEFAULT_DINOV3_REPO
from paths import TRAIN_OUTPUT_DIR as DEFAULT_TRAIN_OUTPUT_DIR
from training_log import (
    TrainingLogger,
    collect_dataset_info,
    collect_environment_info,
    count_model_parameters,
)
from utils.runtime import autocast_context
from utils.runtime import make_grad_scaler
from utils.runtime import resolve_device
from utils.runtime import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 multi-task DFU segmentation.")
    add_dataset_args(parser)
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
        "--lr-scheduler",
        type=str,
        choices=("none", "cosine", "step"),
        default="cosine",
        help="Learning rate scheduler. 'cosine' decays to --min-lr over --epochs.",
    )
    parser.add_argument("--min-lr", type=float, default=1.0e-6, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--lr-step-size", type=int, default=10, help="StepLR: decay every N epochs.")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="StepLR: multiply LR by this factor.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
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
    parser.add_argument(
        "--ulcer-epoch-multiplier",
        type=int,
        default=2,
        help="Repeat ulcer training batches this many times per epoch.",
    )
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=7,
        help="Stop if val_dice does not improve for this many epochs. 0 disables early stopping.",
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


def train_one_task_batch(
    model: MultiTaskSegModel,
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
        outputs = model(images)
        logits = outputs[task]
        loss = segmentation_loss(logits, masks, sample_weights=loss_weight)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        metrics = binary_segmentation_metrics(logits, masks)
    return float(loss.detach().item()), metrics


def _average_metric_batches(metric_batches: list[dict[str, float]]) -> dict[str, float]:
    if not metric_batches:
        return {"dice": 0.0, "iou": 0.0, "accuracy": 0.0}
    keys = metric_batches[0].keys()
    return {key: sum(batch[key] for batch in metric_batches) / len(metric_batches) for key in keys}


def train_epoch(
    model: MultiTaskSegModel,
    foot_loader: DataLoader,
    ulcer_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    limit_batches: int | None,
    ulcer_epoch_multiplier: int = 1,
) -> dict[str, float]:
    model.train()
    foot_total = 0.0
    ulcer_total = 0.0
    foot_steps = 0
    ulcer_steps = 0
    foot_metric_batches: list[dict[str, float]] = []
    ulcer_metric_batches: list[dict[str, float]] = []

    for batch in _limited(foot_loader, limit_batches):
        loss, metrics = train_one_task_batch(
            model, batch, "foot", optimizer, scaler, device, use_amp
        )
        foot_total += loss
        foot_steps += 1
        foot_metric_batches.append(metrics)

    ulcer_passes = max(1, ulcer_epoch_multiplier)
    for _ in range(ulcer_passes):
        for batch in _limited(ulcer_loader, limit_batches):
            loss, metrics = train_one_task_batch(
                model, batch, "ulcer", optimizer, scaler, device, use_amp
            )
            ulcer_total += loss
            ulcer_steps += 1
            ulcer_metric_batches.append(metrics)

    foot_loss = foot_total / max(foot_steps, 1)
    ulcer_loss = ulcer_total / max(ulcer_steps, 1)
    foot_metrics = _average_metric_batches(foot_metric_batches)
    ulcer_metrics = _average_metric_batches(ulcer_metric_batches)
    return {
        "foot_loss": foot_loss,
        "ulcer_loss": ulcer_loss,
        "train_loss": 0.5 * (foot_loss + ulcer_loss),
        "foot_train_dice": foot_metrics["dice"],
        "foot_train_iou": foot_metrics["iou"],
        "foot_train_accuracy": foot_metrics["accuracy"],
        "ulcer_train_dice": ulcer_metrics["dice"],
        "ulcer_train_iou": ulcer_metrics["iou"],
        "ulcer_train_accuracy": ulcer_metrics["accuracy"],
        "train_dice": 0.5 * (foot_metrics["dice"] + ulcer_metrics["dice"]),
        "train_iou": 0.5 * (foot_metrics["iou"] + ulcer_metrics["iou"]),
        "train_accuracy": 0.5 * (foot_metrics["accuracy"] + ulcer_metrics["accuracy"]),
    }


@torch.no_grad()
def validate_task(
    model: MultiTaskSegModel,
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
        outputs = model(images)
        logits = outputs[task]
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
    model: MultiTaskSegModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
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


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_output_dir(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    foot_train = make_loader("foot", "train", args, shuffle=True)
    ulcer_train = make_loader("ulcer", "train", args, shuffle=True)
    foot_val = make_loader("foot", "val", args, shuffle=False)
    ulcer_val = make_loader("ulcer", "val", args, shuffle=False)

    backbone = DINOv3Backbone(
        repo_dir=args.dinov3_repo,
        checkpoint_path=args.dinov3_checkpoint,
        freeze=not args.unfreeze_backbone,
    )
    model = MultiTaskSegModel(backbone=backbone).to(device)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_lr_scheduler(optimizer, args)
    scaler = make_grad_scaler(use_amp)

    logger = TrainingLogger(args.output_dir)
    logger.write_initial_artifacts(
        args=args,
        dataset_info=collect_dataset_info(args, foot_train, ulcer_train, foot_val, ulcer_val),
        environment=collect_environment_info(device),
        model_info={
            "model": model.__class__.__name__,
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
            foot_train,
            ulcer_train,
            optimizer,
            scaler,
            device,
            use_amp,
            args.limit_train_batches,
            ulcer_epoch_multiplier=args.ulcer_epoch_multiplier,
        )
        foot_metrics = validate_task(model, foot_val, "foot", device, args.limit_val_batches)
        ulcer_metrics = validate_task(model, ulcer_val, "ulcer", device, args.limit_val_batches)
        metrics = {**train_metrics, **foot_metrics, **ulcer_metrics}
        metrics["val_dice"] = 0.5 * (metrics["foot_val_dice"] + metrics["ulcer_val_dice"])
        metrics["val_iou"] = 0.5 * (metrics["foot_val_iou"] + metrics["ulcer_val_iou"])
        metrics["val_accuracy"] = 0.5 * (metrics["foot_val_accuracy"] + metrics["ulcer_val_accuracy"])
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
