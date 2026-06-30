#!/usr/bin/env python3
"""Visualize best validation metrics from dfu_head_linear_* / dfu_head_mlp_* checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

METRIC_KEYS = [
    ("dfu_val_f1", "F1"),
    ("dfu_val_accuracy", "Accuracy"),
    ("dfu_val_precision", "Precision"),
    ("dfu_val_recall", "Recall"),
]

RUN_PATTERN = re.compile(r"^dfu_head_(linear|mlp)_(.+)$")
HEAD_COLORS = {"linear": "#3b82f6", "mlp": "#22c55e"}
BEST_COLOR = "#f59e0b"
TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#6b7280"


def find_train_log(checkpoint_dir: Path) -> Path | None:
    direct = checkpoint_dir / "train_log.json"
    if direct.is_file():
        return direct
    matches = sorted(checkpoint_dir.glob("*train_log.json"))
    return matches[0] if matches else None


def parse_run_name(run_name: str) -> dict:
    match = RUN_PATTERN.match(run_name)
    if not match:
        return {"head": "?", "lr_tag": run_name, "aug": False, "label": run_name}

    head, suffix = match.group(1), match.group(2)
    aug = suffix.endswith("_aug")
    lr_tag = suffix.removesuffix("_aug") if aug else suffix
    aug_label = " +aug" if aug else ""
    return {
        "head": head,
        "lr_tag": lr_tag,
        "aug": aug,
        "label": f"{head}\n{lr_tag}{aug_label}",
        "short": f"{head}_{lr_tag}{'_aug' if aug else ''}",
    }


def load_runs(checkpoints_root: Path) -> list[dict]:
    runs: list[dict] = []
    if not checkpoints_root.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_root}")

    for checkpoint_dir in sorted(checkpoints_root.iterdir()):
        if not checkpoint_dir.is_dir():
            continue
        name = checkpoint_dir.name
        if not (name.startswith("dfu_head_linear_") or name.startswith("dfu_head_mlp_")):
            continue

        log_path = find_train_log(checkpoint_dir)
        if log_path is None:
            print(f"[skip] no train_log.json: {checkpoint_dir}")
            continue

        with log_path.open(encoding="utf-8") as f:
            data = json.load(f)

        best = data.get("best_metrics") or {}
        meta = parse_run_name(name)
        config = data.get("config") or {}

        runs.append(
            {
                "run": name,
                **meta,
                "lr": config.get("lr"),
                "best_epoch": data.get("best_epoch"),
                "best_score": data.get("best_score"),
                "val_loss": best.get("dfu_val_loss"),
                **{key: best.get(key) for key, _ in METRIC_KEYS},
            }
        )

    # linear first, then mlp; within each group sort by lr then aug
    head_order = {"linear": 0, "mlp": 1}
    runs.sort(key=lambda r: (head_order.get(r["head"], 9), r["lr_tag"], r["aug"]))
    return runs


def plot_metrics(runs: list[dict], output_path: Path) -> None:
    if not runs:
        raise ValueError("No checkpoint runs to plot.")

    ranked_runs = sorted(runs, key=lambda r: r["dfu_val_f1"] or 0, reverse=True)
    labels = [r["label"] for r in ranked_runs]
    x = np.arange(len(labels))
    n_metrics = len(METRIC_KEYS)
    width = 0.18
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * width

    colors = ["#2563eb", "#059669", "#ea580c", "#7c3aed"]
    fig, ax = plt.subplots(figsize=(max(13, len(labels) * 1.45), 7.2))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    best_f1_idx = 0
    ax.axvspan(best_f1_idx - 0.5, best_f1_idx + 0.5, color="#fef3c7", alpha=0.55, zorder=0)

    for i, (key, title) in enumerate(METRIC_KEYS):
        values = [r[key] for r in ranked_runs]
        best_value = max(v for v in values if v is not None)
        bars = ax.bar(
            x + offsets[i],
            values,
            width=width,
            label=title,
            color=colors[i],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            if value is None:
                continue
            is_metric_best = abs(value - best_value) < 1e-12
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if is_metric_best else "normal",
                color=TEXT_COLOR if is_metric_best else MUTED_TEXT_COLOR,
                rotation=0,
                zorder=4,
            )

    ax.set_ylabel("Validation score", fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.set_title(
        "DFU Head Checkpoints - Validation Metrics\n"
        "Sorted by validation F1",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=TEXT_COLOR)
    ax.set_ylim(0.72, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.28, zorder=0)
    ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")

    for i, r in enumerate(ranked_runs):
        head_color = HEAD_COLORS.get(r["head"], "#64748b")
        ax.text(
            i,
            0.727,
            r["head"].upper(),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=head_color,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_f1_comparison(runs: list[dict], output_path: Path) -> None:
    ranked_runs = sorted(runs, key=lambda r: r["dfu_val_f1"] or 0)
    labels = [r["short"] for r in ranked_runs]
    f1_values = [r["dfu_val_f1"] for r in ranked_runs]
    best_run = ranked_runs[-1]

    colors = []
    for r in ranked_runs:
        if r is best_run:
            colors.append(BEST_COLOR)
        else:
            colors.append(HEAD_COLORS.get(r["head"], "#64748b"))

    fig, ax = plt.subplots(figsize=(11.5, max(5.8, len(labels) * 0.56)))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    bars = ax.barh(labels, f1_values, color=colors, edgecolor="#1f2937", linewidth=0.8, zorder=3)
    for bar, run in zip(bars, ranked_runs):
        if run["aug"]:
            bar.set_hatch("///")
        if run is best_run:
            bar.set_edgecolor("#78350f")
            bar.set_linewidth(2.2)

    for bar, value, run in zip(bars, f1_values, ranked_runs):
        is_best = run is best_run
        ax.text(
            value + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}  ep.{run['best_epoch']}",
            ha="left",
            va="center",
            fontsize=9.5 if is_best else 8.5,
            fontweight="bold" if is_best else "normal",
            color="#92400e" if is_best else TEXT_COLOR,
        )

    ax.set_xlabel("Best validation F1", fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.set_title(
        "Validation F1 by Checkpoint\nSorted by validation F1",
        loc="left",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=16,
    )
    ax.set_xlim(min(f1_values) - 0.035, max(f1_values) + 0.055)
    ax.grid(axis="x", linestyle="--", alpha=0.28, zorder=0)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT_COLOR)
    ax.tick_params(axis="x", colors=MUTED_TEXT_COLOR)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")

    legend_items = [
        mpatches.Patch(facecolor=BEST_COLOR, edgecolor="#78350f", label="top score"),
        mpatches.Patch(facecolor=HEAD_COLORS["linear"], label="linear"),
        mpatches.Patch(facecolor=HEAD_COLORS["mlp"], label="mlp"),
        mpatches.Patch(facecolor="white", hatch="///", edgecolor="#1f2937", label="augmented"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary_table(runs: list[dict], output_path: Path) -> None:
    headers = ["run", "head", "lr", "aug", "best_epoch", "val_f1", "val_acc", "val_prec", "val_rec", "val_loss"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in runs:
        row = [
            r["run"],
            r["head"],
            f"{r['lr']:.0e}" if r.get("lr") is not None else "-",
            "Y" if r["aug"] else "N",
            str(r.get("best_epoch", "-")),
            f"{r['dfu_val_f1']:.4f}" if r.get("dfu_val_f1") is not None else "-",
            f"{r['dfu_val_accuracy']:.4f}" if r.get("dfu_val_accuracy") is not None else "-",
            f"{r['dfu_val_precision']:.4f}" if r.get("dfu_val_precision") is not None else "-",
            f"{r['dfu_val_recall']:.4f}" if r.get("dfu_val_recall") is not None else "-",
            f"{r['val_loss']:.4f}" if r.get("val_loss") is not None else "-",
        ]
        lines.append("| " + " | ".join(row) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "analysis" / "head_checkpoint_comparison",
    )
    args = parser.parse_args()

    runs = load_runs(args.checkpoints_root)
    if not runs:
        raise SystemExit(f"No matching runs under {args.checkpoints_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics(runs, args.output_dir / "best_val_metrics_grouped.png")
    plot_f1_comparison(runs, args.output_dir / "best_val_f1.png")
    write_summary_table(runs, args.output_dir / "summary.md")

    best = max(runs, key=lambda r: r["dfu_val_f1"] or 0)
    print(f"Loaded {len(runs)} runs from {args.checkpoints_root}")
    print(f"Best F1: {best['run']} = {best['dfu_val_f1']:.4f} (epoch {best['best_epoch']})")
    print(f"Saved charts to {args.output_dir}")


if __name__ == "__main__":
    main()
