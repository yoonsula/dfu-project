#!/usr/bin/env python3
"""Build a PDF (or optional HTML) report of DFU test misclassifications."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import ClassificationImageDataset
from eval.runners import _load_dfu_stack
from paths import DINOV3_MODEL_PATH
from trainers.dfu_trainer import DFU_CLASSES, predict_dfu_logits
from utils.runtime import resolve_device

ERROR_LABELS = {
    "fp": "False Positive (other → dfu)",
    "fn": "False Negative (dfu → other)",
}


@dataclass
class Misclassification:
    image_path: str
    image_name: str
    true_label: str
    predicted_label: str
    error_type: str
    prob_dfu: float
    prob_other: float
    confidence: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "dfu_head_mlp_1e3_aug" / "best.pt",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT.parents[1] / "03_데이터" / "dfu_classification_cropped",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "head_checkpoint_comparison" / "test_eval" / "error_report",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument(
        "--format",
        choices=("pdf", "html", "both"),
        default="pdf",
        help="Output format. PDF uses the same styled HTML rendered via headless Chrome.",
    )
    return parser


def collect_misclassifications(
    checkpoint: Path,
    dataset_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[Misclassification], dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    image_size = int(payload.get("args", {}).get("image_size", payload.get("image_size", 512)))
    id2label = {int(k): str(v) for k, v in payload.get("id2label", {i: c for i, c in enumerate(DFU_CLASSES)}).items()}

    dataset = ClassificationImageDataset(
        root=dataset_root,
        split="test",
        image_size=image_size,
        classes=DFU_CLASSES,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    backbone, head = _load_dfu_stack(checkpoint, dinov3_model=DINOV3_MODEL_PATH, device=device)

    misclassified: list[Misclassification] = []
    total = 0
    correct = 0

    backbone.eval()
    head.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = predict_dfu_logits(backbone, head, images)
            probs = F.softmax(logits, dim=1)

            predictions = logits.argmax(dim=1)
            total += labels.numel()
            correct += int((predictions == labels).sum().item())

            for index in range(labels.size(0)):
                true_idx = int(labels[index].item())
                pred_idx = int(predictions[index].item())
                if true_idx == pred_idx:
                    continue

                true_label = id2label[true_idx]
                pred_label = id2label[pred_idx]
                prob_dfu = float(probs[index, 0].item())
                prob_other = float(probs[index, 1].item())
                image_path = batch["image_path"][index]
                error_type = "fn" if true_label == "dfu" else "fp"

                misclassified.append(
                    Misclassification(
                        image_path=image_path,
                        image_name=Path(image_path).name,
                        true_label=true_label,
                        predicted_label=pred_label,
                        error_type=error_type,
                        prob_dfu=prob_dfu,
                        prob_other=prob_other,
                        confidence=max(prob_dfu, prob_other),
                    )
                )

    summary = {
        "checkpoint": str(checkpoint),
        "run_name": checkpoint.parent.name,
        "image_size": image_size,
        "total_samples": total,
        "correct": correct,
        "incorrect": len(misclassified),
        "accuracy": correct / max(total, 1),
        "false_positive": sum(1 for row in misclassified if row.error_type == "fp"),
        "false_negative": sum(1 for row in misclassified if row.error_type == "fn"),
        "test_root": str(dataset_root / "test"),
    }
    return misclassified, summary


def make_thumbnail_data_uri(image_path: Path, thumb_size: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_report_html(
    rows: list[Misclassification],
    summary: dict,
    *,
    thumb_size: int,
    for_pdf: bool = False,
) -> str:
    cards: list[dict] = []
    for row in rows:
        image_path = Path(row.image_path)
        cards.append(
            {
                **asdict(row),
                "image_src": make_thumbnail_data_uri(image_path, thumb_size),
            }
        )
    cards.sort(key=lambda item: (-item["confidence"], item["error_type"], item["image_name"]))

    def card_html(card: dict) -> str:
        error_label = ERROR_LABELS.get(card["error_type"], card["error_type"])
        return f"""
        <article class="card" data-type="{card['error_type']}">
          <img src="{card['image_src']}" alt="{card['image_name']}">
          <div class="meta">
            <div class="filename">{card['image_name']}</div>
            <div class="labels">
              <span class="tag true">정답: {card['true_label']}</span>
              <span class="tag pred">예측: {card['predicted_label']}</span>
              <span class="tag err">{error_label}</span>
            </div>
            <div class="probs">
              <div class="prob-row">
                <span>dfu</span>
                <div class="bar"><div class="fill dfu" style="width:{card['prob_dfu'] * 100:.1f}%"></div></div>
                <span>{card['prob_dfu']:.4f}</span>
              </div>
              <div class="prob-row">
                <span>other</span>
                <div class="bar"><div class="fill other" style="width:{card['prob_other'] * 100:.1f}%"></div></div>
                <span>{card['prob_other']:.4f}</span>
              </div>
            </div>
          </div>
        </article>
        """

    body_class = "export-pdf" if for_pdf else ""
    toolbar_html = "" if for_pdf else f"""
  <div class="toolbar">
    <button class="filter active" data-filter="all">전체 ({len(cards)})</button>
    <button class="filter" data-filter="fp">FP ({summary['false_positive']})</button>
    <button class="filter" data-filter="fn">FN ({summary['false_negative']})</button>
  </div>"""
    script_html = "" if for_pdf else """
  <script>
    const buttons = document.querySelectorAll('.filter');
    const cards = document.querySelectorAll('.card');
    buttons.forEach((button) => {
      button.addEventListener('click', () => {
        buttons.forEach((b) => b.classList.remove('active'));
        button.classList.add('active');
        const filter = button.dataset.filter;
        cards.forEach((card) => {
          const show = filter === 'all' || card.dataset.type === filter;
          card.classList.toggle('hidden', !show);
        });
      });
    });
  </script>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DFU Test Misclassification Report - {summary['run_name']}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: #e5e7eb;
      --dfu: #2563eb;
      --other: #059669;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans KR", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      padding: 24px 28px;
      background: var(--card);
      border-bottom: 1px solid var(--border);
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .stat {{
      background: #f3f4f6;
      border-radius: 10px;
      padding: 12px 14px;
    }}
    .stat .label {{ color: var(--muted); font-size: 13px; }}
    .stat .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      padding: 16px 28px;
      position: sticky;
      top: 0;
      background: rgba(248, 250, 252, 0.95);
      backdrop-filter: blur(6px);
      border-bottom: 1px solid var(--border);
      z-index: 10;
    }}
    button {{
      border: 1px solid var(--border);
      background: var(--card);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 999px;
      cursor: pointer;
      font-size: 14px;
    }}
    button.active {{
      background: #111827;
      color: white;
      border-color: #111827;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
      padding: 20px 28px 40px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      background: #e5e7eb;
      display: block;
    }}
    .meta {{ padding: 14px; }}
    .filename {{ font-weight: 600; font-size: 14px; word-break: break-all; }}
    .labels {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }}
    .tag {{
      font-size: 12px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #f3f4f6;
    }}
    .tag.true {{ background: #eff6ff; color: #1d4ed8; }}
    .tag.pred {{ background: #ecfdf5; color: #047857; }}
    .tag.err {{ background: #fef2f2; color: #b91c1c; }}
    .probs {{ margin-top: 8px; }}
    .prob-row {{
      display: grid;
      grid-template-columns: 42px 1fr 56px;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .bar {{
      height: 8px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
    }}
    .fill {{ height: 100%; }}
    .fill.dfu {{ background: var(--dfu); }}
    .fill.other {{ background: var(--other); }}
    .hidden {{ display: none; }}

    body.export-pdf .grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      padding: 16px 20px 28px;
    }}
    body.export-pdf .card {{
      break-inside: avoid;
      page-break-inside: avoid;
    }}
    body.export-pdf header {{
      padding: 20px 24px;
    }}

    @media print {{
      body {{ background: white; }}
      .toolbar {{ display: none !important; }}
      .grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }}
      .card {{
        break-inside: avoid;
        page-break-inside: avoid;
        box-shadow: none;
      }}
    }}
  </style>
</head>
<body class="{body_class}">
  <header>
    <h1>DFU Test Misclassification Report</h1>
    <div>Checkpoint: <code>{summary['run_name']}</code></div>
    <div>Test samples: {summary['total_samples']} (dfu/other classification)</div>
    <div class="summary">
      <div class="stat"><div class="label">Total</div><div class="value">{summary['total_samples']}</div></div>
      <div class="stat"><div class="label">Correct</div><div class="value">{summary['correct']}</div></div>
      <div class="stat"><div class="label">Incorrect</div><div class="value">{summary['incorrect']}</div></div>
      <div class="stat"><div class="label">Accuracy</div><div class="value">{summary['accuracy']:.4f}</div></div>
      <div class="stat"><div class="label">False Positive</div><div class="value">{summary['false_positive']}</div></div>
      <div class="stat"><div class="label">False Negative</div><div class="value">{summary['false_negative']}</div></div>
    </div>
  </header>
{toolbar_html}
  <section class="grid" id="grid">
    {''.join(card_html(card) for card in cards)}
  </section>
{script_html}
</body>
</html>
"""


def render_pdf_from_html(html: str, output_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.pdf(
            path=str(output_path),
            format="A4",
            print_background=True,
            margin={"top": "12mm", "right": "10mm", "bottom": "12mm", "left": "10mm"},
        )
        browser.close()


def save_sidecar_files(rows: list[Misclassification], summary: dict, output_dir: Path, run_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{run_name}_misclassified.json").write_text(
        json.dumps({"summary": summary, "items": [asdict(row) for row in rows]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{run_name}_misclassified.csv").write_text(
        "image_name,true_label,predicted_label,error_type,prob_dfu,prob_other,confidence,image_path\n"
        + "\n".join(
            f"{row.image_name},{row.true_label},{row.predicted_label},{row.error_type},"
            f"{row.prob_dfu:.6f},{row.prob_other:.6f},{row.confidence:.6f},{row.image_path}"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    rows, summary = collect_misclassifications(
        args.checkpoint.resolve(),
        args.data_root.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    run_name = summary["run_name"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_sidecar_files(rows, summary, args.output_dir, run_name)

    html_interactive = build_report_html(rows, summary, thumb_size=args.thumb_size, for_pdf=False)
    html_for_pdf = build_report_html(rows, summary, thumb_size=args.thumb_size, for_pdf=True)

    if args.format in {"html", "both"}:
        html_path = args.output_dir / f"{run_name}_test_errors.html"
        html_path.write_text(html_interactive, encoding="utf-8")
        print(f"HTML: {html_path}")

    if args.format in {"pdf", "both"}:
        pdf_path = args.output_dir / f"{run_name}_test_errors.pdf"
        render_pdf_from_html(html_for_pdf, pdf_path)
        print(f"PDF:  {pdf_path}")

    print(f"[done] {summary['incorrect']} misclassified / {summary['total_samples']} total")
    print(f"JSON: {args.output_dir / f'{run_name}_misclassified.json'}")
    print(f"CSV:  {args.output_dir / f'{run_name}_misclassified.csv'}")


if __name__ == "__main__":
    main()
