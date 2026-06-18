from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from inference.checkpoints import checkpoint_state_dict, strip_first_matching_prefix
from models import DFUFeatureClassifierHead

DFU_CLASSES = ("dfu", "other")


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
class DFUHeadBundle:
    head: DFUFeatureClassifierHead
    id2label: dict[int, str]
    classes: tuple[str, ...]
    checkpoint_path: Path


def load_dfu_head_bundle(
    checkpoint_path: Path,
    device: torch.device,
) -> DFUHeadBundle:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DFU head checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"DFU head checkpoint must be a dict: {checkpoint_path}")

    classes = tuple(checkpoint.get("classes", DFU_CLASSES))
    id2label_raw = checkpoint.get("id2label", {index: label for index, label in enumerate(classes)})
    id2label = {int(key): str(value) for key, value in id2label_raw.items()}
    feature_dim = int(checkpoint.get("feature_dim", 384))
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    dropout = float(checkpoint.get("dropout", 0.2))
    head_type = str(checkpoint.get("head_type", "mlp"))
    head = DFUFeatureClassifierHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=len(classes),
        dropout=dropout,
        head_type=head_type,
    ).to(device)
    state_dict = strip_first_matching_prefix(
        checkpoint_state_dict(checkpoint),
        ("classification_head.", "dfu_head.", "head."),
    )
    head.load_state_dict(state_dict, strict=True)
    head.eval()
    return DFUHeadBundle(
        head=head,
        id2label=id2label,
        classes=classes,
        checkpoint_path=checkpoint_path,
    )


@torch.inference_mode()
def classify_shared_features(
    features: torch.Tensor,
    bundle: DFUHeadBundle | None,
    *,
    enabled: bool,
    top_k: int,
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
    logits = bundle.head(features)
    probabilities = F.softmax(logits, dim=-1)[0]
    k = min(top_k, len(bundle.classes))
    top_probs, top_indices = torch.topk(probabilities, k=k)
    scores = tuple(
        ClassScore(
            class_name=bundle.id2label[int(index)],
            probability=float(prob),
        )
        for prob, index in zip(top_probs.cpu(), top_indices.cpu())
    )
    classification_ms = (perf_counter() - start) * 1000.0
    return ClassificationResult(
        enabled=True,
        predicted_class=scores[0].class_name if scores else None,
        confidence=scores[0].probability if scores else None,
        top_k=scores,
        classification_ms=round(classification_ms, 2),
        checkpoint_path=checkpoint_path,
    )
