from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from models.dfu_classifier import DinoV3LinearClassifier
from paths import DEFAULT_CLASSIFICATION_CHECKPOINT
from paths import DINOV3_HF_MODEL_DIR as DEFAULT_DINOV3_HF_MODEL_DIR
from utils.image_io import iter_images
from utils.runtime import resolve_device
from utils.runtime import synchronize_if_needed


@dataclass(frozen=True)
class ClassScore:
    class_name: str
    probability: float


@dataclass(frozen=True)
class ClassificationResult:
    enabled: bool
    predicted_class: str | None
    confidence: float | None
    top_k: tuple[ClassScore, ...]
    classification_ms: float
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class ClassificationBundle:
    model: DinoV3LinearClassifier
    image_processor: AutoImageProcessor
    id2label: dict[int, str]
    classes: tuple[str, ...]
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DFU image classification inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CLASSIFICATION_CHECKPOINT,
        help="Classification checkpoint (.pt).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_DINOV3_HF_MODEL_DIR,
        help="Local Hugging Face DINOv3 model directory.",
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image file or directory.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def resolve_model_dir(model_dir: str | Path, fallback: Path) -> Path:
    path = Path(model_dir)
    if path.is_absolute() and path.exists():
        return path.resolve()

    candidates = [
        fallback,
        DEFAULT_DINOV3_HF_MODEL_DIR.parent / path.name,
        DEFAULT_DINOV3_HF_MODEL_DIR,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"DINOv3 Hugging Face model directory not found: {model_dir}\n"
        f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )


def load_classification_bundle(
    checkpoint_path: Path,
    device: torch.device,
    model_dir: Path | None = None,
) -> ClassificationBundle:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classification checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint missing config dict: {checkpoint_path}")

    classes = tuple(config["classes"])
    id2label = {int(key): value for key, value in config["id2label"].items()}
    num_classes = int(config["num_classes"])
    freeze_backbone = bool(config.get("freeze_backbone", True))

    resolved_model_dir = resolve_model_dir(
        config.get("model_dir", DEFAULT_DINOV3_HF_MODEL_DIR.name),
        fallback=model_dir or DEFAULT_DINOV3_HF_MODEL_DIR,
    )
    backbone = AutoModel.from_pretrained(resolved_model_dir, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(resolved_model_dir, local_files_only=True)

    model = DinoV3LinearClassifier(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return ClassificationBundle(
        model=model,
        image_processor=image_processor,
        id2label=id2label,
        classes=classes,
        checkpoint_path=checkpoint_path,
    )


@torch.inference_mode()
def classify_image(
    image: Image.Image,
    bundle: ClassificationBundle | None,
    device: torch.device,
    *,
    enabled: bool = True,
    top_k: int = 3,
) -> ClassificationResult:
    checkpoint_path = str(bundle.checkpoint_path) if bundle is not None else None
    if not enabled or bundle is None:
        return ClassificationResult(
            enabled=False,
            predicted_class=None,
            confidence=None,
            top_k=(),
            classification_ms=0.0,
            checkpoint_path=checkpoint_path,
        )

    start = perf_counter()
    inputs = bundle.image_processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    logits = bundle.model(pixel_values)
    probabilities = F.softmax(logits, dim=-1)[0]

    top_k = min(top_k, len(bundle.classes))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)
    scores = tuple(
        ClassScore(
            class_name=bundle.id2label[int(index)],
            probability=float(prob),
        )
        for prob, index in zip(top_probs.cpu(), top_indices.cpu())
    )
    synchronize_if_needed(device)
    classification_ms = (perf_counter() - start) * 1000.0

    return ClassificationResult(
        enabled=True,
        predicted_class=scores[0].class_name if scores else None,
        confidence=scores[0].probability if scores else None,
        top_k=scores,
        classification_ms=round(classification_ms, 2),
        checkpoint_path=str(bundle.checkpoint_path),
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    bundle = load_classification_bundle(args.checkpoint, device, model_dir=args.model_dir)
    image_paths = list(iter_images(args.image))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.image}")

    for image_path in image_paths:
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        result = classify_image(image, bundle, device, enabled=True, top_k=args.top_k)
        print(
            f"{image_path}: class={result.predicted_class} "
            f"confidence={result.confidence:.4f} "
            f"classification_ms={result.classification_ms:.2f}"
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
