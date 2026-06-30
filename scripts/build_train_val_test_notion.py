#!/usr/bin/env python3
"""Build train/val/test comparison table for Notion from train_log + test_eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_head_checkpoints import find_train_log, parse_run_name


def fmt(value, kind: str = "default") -> str:
    if value is None:
        return "-"
    if kind == "score":
        return f"{value:.4f}"
    if kind == "lr":
        return f"{value:.0e}"
    if kind == "gap":
        return f"{value:+.4f}"
    return str(value)


def load_train_val_runs(checkpoints_root: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for checkpoint_dir in sorted(checkpoints_root.iterdir()):
        if not checkpoint_dir.is_dir():
            continue
        name = checkpoint_dir.name
        if not (name.startswith("dfu_head_linear_") or name.startswith("dfu_head_mlp_")):
            continue
        log_path = find_train_log(checkpoint_dir)
        if log_path is None:
            print(f"[skip] no train_log.json: {name}")
            continue

        data = json.loads(log_path.read_text(encoding="utf-8"))
        best = data.get("best_metrics") or {}
        config = data.get("config") or {}
        meta = parse_run_name(name)

        runs[name] = {
            "run": name,
            "head": config.get("head_type", meta["head"]),
            "lr": config.get("lr"),
            "aug": "Y" if meta["aug"] else "N",
            "best_epoch": data.get("best_epoch"),
            "train_f1": best.get("dfu_train_f1"),
            "train_acc": best.get("dfu_train_accuracy"),
            "train_precision": best.get("dfu_train_precision"),
            "train_recall": best.get("dfu_train_recall"),
            "train_loss": best.get("train_loss"),
            "val_f1": best.get("dfu_val_f1"),
            "val_acc": best.get("dfu_val_accuracy"),
            "val_precision": best.get("dfu_val_precision"),
            "val_recall": best.get("dfu_val_recall"),
            "val_loss": best.get("dfu_val_loss"),
        }
    return runs


def load_test_runs(test_summary_path: Path) -> dict[str, dict]:
    if not test_summary_path.exists():
        raise FileNotFoundError(
            f"Test summary not found: {test_summary_path}\n"
            "Run scripts/evaluate_dfu_heads_on_test.py first."
        )
    rows = json.loads(test_summary_path.read_text(encoding="utf-8"))
    return {row["run"]: row for row in rows}


def merge_runs(train_val: dict[str, dict], test: dict[str, dict]) -> list[dict]:
    merged: list[dict] = []
    for run_name in sorted(set(train_val) & set(test)):
        row = {**train_val[run_name], **test[run_name]}
        train_f1 = row.get("train_f1")
        val_f1 = row.get("val_f1")
        test_f1 = row.get("test_f1")
        row["train_test_f1_gap"] = (train_f1 - test_f1) if train_f1 is not None and test_f1 is not None else None
        row["val_test_f1_gap"] = (val_f1 - test_f1) if val_f1 is not None and test_f1 is not None else None
        merged.append(row)

    missing_test = sorted(set(train_val) - set(test))
    missing_train = sorted(set(test) - set(train_val))
    for name in missing_test:
        print(f"[warn] no test result: {name}")
    for name in missing_train:
        print(f"[warn] no train_log: {name}")

    merged.sort(key=lambda row: row.get("test_f1") or 0, reverse=True)
    return merged


def write_notion(rows: list[dict], output_path: Path, test_root: Path) -> None:
    sample_count = rows[0].get("num_samples") if rows else 0
    dfu_count = rows[0].get("dfu_count") if rows else 0
    other_count = rows[0].get("other_count") if rows else 0

    lines = [
        "# DFU Head Train / Validation / Test Summary",
        "",
        "- 정렬 기준: Test F1 내림차순",
        f"- Test data: `{test_root}`",
        f"- Test samples: {sample_count} total, dfu {dfu_count}, other {other_count}",
        "",
        "## F1 중심 요약",
        "",
        "| Run | Head | LR | Aug | Best epoch | Train F1 | Val F1 | Test F1 | Train-Test F1 gap | Val-Test F1 gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"],
                    row["head"],
                    fmt(row["lr"], "lr"),
                    row["aug"],
                    fmt(row["best_epoch"]),
                    fmt(row["train_f1"], "score"),
                    fmt(row["val_f1"], "score"),
                    fmt(row["test_f1"], "score"),
                    fmt(row["train_test_f1_gap"], "gap"),
                    fmt(row["val_test_f1_gap"], "gap"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 전체 지표",
            "",
            "| Run | Head | LR | Aug | Best epoch | Train F1 | Val F1 | Test F1 | Train Acc | Val Acc | Test Acc | Train Prec. | Val Prec. | Test Prec. | Train Recall | Val Recall | Test Recall | Train loss | Val loss | Test loss |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"],
                    row["head"],
                    fmt(row["lr"], "lr"),
                    row["aug"],
                    fmt(row["best_epoch"]),
                    fmt(row["train_f1"], "score"),
                    fmt(row["val_f1"], "score"),
                    fmt(row["test_f1"], "score"),
                    fmt(row["train_acc"], "score"),
                    fmt(row["val_acc"], "score"),
                    fmt(row["test_accuracy"], "score"),
                    fmt(row["train_precision"], "score"),
                    fmt(row["val_precision"], "score"),
                    fmt(row["test_precision"], "score"),
                    fmt(row["train_recall"], "score"),
                    fmt(row["val_recall"], "score"),
                    fmt(row["test_recall"], "score"),
                    fmt(row["train_loss"], "score"),
                    fmt(row["val_loss"], "score"),
                    fmt(row["test_loss"], "score"),
                ]
            )
            + " |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--test-summary",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "head_checkpoint_comparison" / "test_eval" / "test_eval_summary.json",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=PROJECT_ROOT.parents[1] / "03_데이터" / "dfu_classification_cropped" / "test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "head_checkpoint_comparison" / "train_val_test_notion.md",
    )
    args = parser.parse_args()

    train_val = load_train_val_runs(args.checkpoints_root)
    test = load_test_runs(args.test_summary)
    rows = merge_runs(train_val, test)
    if not rows:
        raise SystemExit("No runs with both train_log.json and test evaluation results.")

    write_notion(rows, args.output, args.test_root.resolve())
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
