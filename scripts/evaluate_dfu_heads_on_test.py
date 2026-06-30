#!/usr/bin/env python3
"""Evaluate all DFU head checkpoints on the cropped DFU test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import ClassificationImageDataset
from eval.runners import _load_dfu_stack, save_eval_report
from paths import DINOV3_MODEL_PATH
from trainers.dfu_trainer import DFU_CLASSES, validate_dfu
from utils.runtime import resolve_device

HEAD_COLORS = {"linear": "#3b82f6", "mlp": "#22c55e"}
TOP_COLOR = "#f59e0b"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Directory containing dfu_head_* checkpoint folders.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT.parents[1] / "03_데이터" / "dfu_classification_cropped" / "test",
        help="Path to cropped DFU test folder, or its parent containing test/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "head_checkpoint_comparison" / "test_eval",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    return parser


def resolve_test_root(data_root: Path) -> tuple[Path, Path]:
    """Return (dataset_root, displayed_test_root) for ClassificationImageDataset split='test'."""
    data_root = data_root.resolve()
    if data_root.name == "test":
        return data_root.parent, data_root
    return data_root, data_root / "test"


def find_checkpoints(checkpoints_root: Path) -> list[Path]:
    checkpoint_paths: list[Path] = []
    for checkpoint_dir in sorted(checkpoints_root.iterdir()):
        if not checkpoint_dir.is_dir():
            continue
        if not (checkpoint_dir.name.startswith("dfu_head_linear_") or checkpoint_dir.name.startswith("dfu_head_mlp_")):
            continue
        checkpoint = checkpoint_dir / "best.pt"
        if checkpoint.exists():
            checkpoint_paths.append(checkpoint)
        else:
            print(f"[skip] no best.pt: {checkpoint_dir.name}")
    return checkpoint_paths


def load_train_config(checkpoint_dir: Path) -> dict[str, Any]:
    log_path = checkpoint_dir / "train_log.json"
    if not log_path.exists():
        matches = sorted(checkpoint_dir.glob("*train_log.json"))
        log_path = matches[0] if matches else log_path
    if not log_path.exists():
        return {}
    with log_path.open(encoding="utf-8") as handle:
        return (json.load(handle).get("config") or {})


def infer_checkpoint_meta(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = payload.get("args") or {}
    return {
        "image_size": int(args.get("image_size", payload.get("image_size", 512))),
        "head": str(payload.get("head_type", args.get("head_type", "-"))),
        "hidden_dim": payload.get("hidden_dim", args.get("hidden_dim")),
        "dropout": payload.get("dropout", args.get("dropout")),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    meta = infer_checkpoint_meta(checkpoint)
    dataset = ClassificationImageDataset(
        root=dataset_root,
        split="test",
        image_size=meta["image_size"],
        classes=DFU_CLASSES,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    backbone, head = _load_dfu_stack(
        checkpoint,
        dinov3_model=DINOV3_MODEL_PATH,
        device=device,
    )
    metrics = validate_dfu(backbone, head, loader, device, use_amp, None, None)
    class_counts = {
        dataset.id2label[index]: sum(1 for sample in dataset.samples if sample.label == index)
        for index in range(len(dataset.classes))
    }
    return {
        "task": "dfu",
        "checkpoint": str(checkpoint),
        "dataset_root_used": str(dataset_root),
        "split": "test",
        "image_size": meta["image_size"],
        "num_samples": len(dataset),
        "class_counts": class_counts,
        "metrics": metrics,
        "amp": use_amp,
    }


def fmt(value: Any, kind: str = "default") -> str:
    if value is None:
        return "-"
    if kind == "score":
        return f"{value:.4f}"
    if kind == "lr":
        return f"{value:.0e}"
    return str(value)


def write_notion_table(runs: list[dict[str, Any]], output_path: Path, test_root: Path) -> None:
    sample_count = runs[0]["num_samples"] if runs else 0
    dfu_count = runs[0]["dfu_count"] if runs else 0
    other_count = runs[0]["other_count"] if runs else 0
    headers = [
        "Run",
        "Head",
        "LR",
        "Aug",
        "Hidden dim",
        "Dropout",
        "Weight decay",
        "Samples",
        "dfu",
        "other",
        "Test F1",
        "Accuracy",
        "Precision",
        "Recall",
        "Test loss",
    ]
    lines = [
        "# DFU Head Test Evaluation Summary",
        "",
        f"- Test data: `{test_root}`",
        f"- Samples: {sample_count} total, dfu {dfu_count}, other {other_count}",
        "- 정렬 기준: Test F1 내림차순",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for run in runs:
        row = [
            run["run"],
            run["head"],
            fmt(run["lr"], "lr"),
            run["aug"],
            fmt(run["hidden_dim"]),
            fmt(run["dropout"]),
            fmt(run["weight_decay"], "lr"),
            fmt(run["num_samples"]),
            fmt(run["dfu_count"]),
            fmt(run["other_count"]),
            fmt(run["test_f1"], "score"),
            fmt(run["test_accuracy"], "score"),
            fmt(run["test_precision"], "score"),
            fmt(run["test_recall"], "score"),
            fmt(run["test_loss"], "score"),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Compact View", ""])
    compact_headers = ["Run", "Head", "LR", "Aug", "Test F1", "Accuracy", "Precision", "Recall", "Test loss"]
    lines.extend([
        "| " + " | ".join(compact_headers) + " |",
        "| " + " | ".join(["---"] * len(compact_headers)) + " |",
    ])
    for run in runs:
        row = [
            run["run"],
            run["head"],
            fmt(run["lr"], "lr"),
            run["aug"],
            fmt(run["test_f1"], "score"),
            fmt(run["test_accuracy"], "score"),
            fmt(run["test_precision"], "score"),
            fmt(run["test_recall"], "score"),
            fmt(run["test_loss"], "score"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_f1_ranking(runs: list[dict[str, Any]], output_path: Path) -> None:
    ranked = sorted(runs, key=lambda row: row["test_f1"] or 0)
    labels = [row["run"].replace("dfu_head_", "") for row in ranked]
    values = [row["test_f1"] for row in ranked]
    colors = [
        TOP_COLOR if index == len(ranked) - 1 else HEAD_COLORS.get(row["head"], "#64748b")
        for index, row in enumerate(ranked)
    ]

    fig, ax = plt.subplots(figsize=(11.5, max(5.8, len(labels) * 0.56)))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    bars = ax.barh(labels, values, color=colors, edgecolor="#1f2937", linewidth=0.8, zorder=3)
    for bar, row in zip(bars, ranked):
        if row["aug"] == "Y":
            bar.set_hatch("///")
        if row is ranked[-1]:
            bar.set_edgecolor("#78350f")
            bar.set_linewidth(2.2)
    for bar, value in zip(bars, values):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", ha="left", va="center", fontsize=9)

    ax.set_title(f"Test F1 by Checkpoint\n{runs[0]['num_samples']} images, sorted by test F1", loc="left", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Test F1", fontsize=11, fontweight="bold")
    ax.set_xlim(min(values) - 0.035, max(values) + 0.055)
    ax.grid(axis="x", linestyle="--", alpha=0.28, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=TOP_COLOR, edgecolor="#78350f", label="top score"),
            mpatches.Patch(facecolor=HEAD_COLORS["linear"], label="linear"),
            mpatches.Patch(facecolor=HEAD_COLORS["mlp"], label="mlp"),
            mpatches.Patch(facecolor="white", hatch="///", edgecolor="#1f2937", label="augmented"),
        ],
        loc="lower right",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#e5e7eb",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_grouped_metrics(runs: list[dict[str, Any]], output_path: Path) -> None:
    ranked = sorted(runs, key=lambda row: row["test_f1"] or 0, reverse=True)
    metric_keys = [
        ("test_f1", "F1"),
        ("test_accuracy", "Accuracy"),
        ("test_precision", "Precision"),
        ("test_recall", "Recall"),
    ]
    labels = [row["run"].replace("dfu_head_", "").replace("_", "\n", 1) for row in ranked]
    x = list(range(len(labels)))
    width = 0.18
    offsets = [(-1.5 + index) * width for index in range(len(metric_keys))]
    colors = ["#2563eb", "#059669", "#ea580c", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(max(13, len(labels) * 1.45), 7.2))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    ax.axvspan(-0.5, 0.5, color="#fef3c7", alpha=0.55, zorder=0)

    for index, (key, label) in enumerate(metric_keys):
        values = [row[key] for row in ranked]
        best_value = max(value for value in values if value is not None)
        bars = ax.bar(
            [position + offsets[index] for position in x],
            values,
            width=width,
            label=label,
            color=colors[index],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            is_best = abs(value - best_value) < 1e-12
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if is_best else "normal",
                color="#111827" if is_best else "#6b7280",
                zorder=4,
            )

    ax.set_title(
        f"DFU Head Checkpoints - Test Metrics\n{runs[0]['num_samples']} images, sorted by test F1",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    ax.set_ylabel("Test score", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.72, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.28, zorder=0)
    ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    for index, row in enumerate(ranked):
        ax.text(
            index,
            0.727,
            row["head"].upper(),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=HEAD_COLORS.get(row["head"], "#64748b"),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    dataset_root, test_root = resolve_test_root(args.data_root)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for checkpoint in find_checkpoints(args.checkpoints_root):
        run_name = checkpoint.parent.name
        print(f"[eval] {run_name}", flush=True)
        report = evaluate_checkpoint(
            checkpoint,
            dataset_root=dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            use_amp=use_amp,
        )
        report["data_root"] = str(test_root)
        save_eval_report(report, args.output_dir / f"{run_name}_test_report.json")

        config = load_train_config(checkpoint.parent)
        checkpoint_meta = infer_checkpoint_meta(checkpoint)
        metrics = report["metrics"]
        rows.append(
            {
                "run": run_name,
                "head": config.get("head_type", checkpoint_meta["head"]),
                "lr": config.get("lr"),
                "aug": "Y" if run_name.endswith("_aug") else "N",
                "hidden_dim": config.get("hidden_dim", checkpoint_meta["hidden_dim"]),
                "dropout": config.get("dropout", checkpoint_meta["dropout"]),
                "weight_decay": config.get("weight_decay"),
                "num_samples": report["num_samples"],
                "dfu_count": report["class_counts"].get("dfu"),
                "other_count": report["class_counts"].get("other"),
                "test_loss": metrics.get("dfu_val_loss"),
                "test_accuracy": metrics.get("dfu_val_accuracy"),
                "test_precision": metrics.get("dfu_val_precision"),
                "test_recall": metrics.get("dfu_val_recall"),
                "test_f1": metrics.get("dfu_val_f1"),
            }
        )

    rows.sort(key=lambda row: row["test_f1"] or 0, reverse=True)
    (args.output_dir / "test_eval_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notion_table(rows, args.output_dir / "test_eval_notion.md", test_root)
    plot_f1_ranking(rows, args.output_dir / "test_f1_ranking.png")
    plot_grouped_metrics(rows, args.output_dir / "test_metrics_grouped.png")

    print(f"[done] saved: {args.output_dir}")
    for row in rows:
        print(
            f"{row['run']}: f1={fmt(row['test_f1'], 'score')} "
            f"acc={fmt(row['test_accuracy'], 'score')} "
            f"prec={fmt(row['test_precision'], 'score')} "
            f"rec={fmt(row['test_recall'], 'score')}"
        )


if __name__ == "__main__":
    main()
