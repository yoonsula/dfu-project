"""Central path configuration. Paths default to project-relative locations (see .env.example)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

# Allow `python ../dfu-project/infer.py` or running from any cwd.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_path(env_key: str, default: Path) -> Path:
    """Resolve env path; relative values are interpreted from PROJECT_ROOT."""
    raw = os.environ.get(env_key)
    if raw is None:
        return default.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


# --- Inference assets (bundled under assets/ + checkpoints/) ---

DINOV3_REPO = _resolve_path("DINOV3_REPO", ASSETS_DIR / "dinov3")

_DINOV3_CHECKPOINT_NAME = os.environ.get(
    "DINOV3_CHECKPOINT_NAME",
    "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
)
DINOV3_CHECKPOINT = _resolve_path(
    "DINOV3_CHECKPOINT",
    DINOV3_REPO / "checkpoint" / _DINOV3_CHECKPOINT_NAME,
)

DINOV3_HF_MODEL_DIR = _resolve_path(
    "DINOV3_HF_MODEL_DIR",
    ASSETS_DIR / "dinov3-hf" / "dinov3-vits16-pretrain-lvd1689m",
)

CHECKPOINT_DIR = _resolve_path("DFU_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")
DEFAULT_CHECKPOINT = _resolve_path("DFU_CHECKPOINT", CHECKPOINT_DIR / "best.pt")
DEFAULT_CLASSIFICATION_CHECKPOINT = _resolve_path(
    "DFU_CLASSIFICATION_CHECKPOINT",
    CHECKPOINT_DIR / "dinov3_linear_best_0.001.pt",
)

INFERENCE_OUTPUT_DIR = _resolve_path(
    "DFU_INFERENCE_OUTPUT_DIR",
    PROJECT_ROOT / "output" / "inference",
)

TRAIN_OUTPUT_DIR = _resolve_path(
    "DFU_TRAIN_OUTPUT_DIR",
    PROJECT_ROOT / "output" / "train",
)

# --- Training data (not bundled; override via .env when retraining) ---

DATA_ROOT = _resolve_path("DFU_DATA_ROOT", PROJECT_ROOT / ".." / ".." / "03_데이터")
DEFAULT_FOOT_ROOT = _resolve_path("DFU_FOOT_ROOT", DATA_ROOT / "roboflow-foot")
DEFAULT_BODY_ROOT = _resolve_path("DFU_BODY_ROOT", DATA_ROOT / "roboflow-body")
DEFAULT_HUMANBODY_ROOT = _resolve_path("DFU_HUMANBODY_ROOT", DATA_ROOT / "roboflow-humanbody")
DEFAULT_CLOSEUP_NEGATIVE_ROOT = _resolve_path(
    "DFU_CLOSEUP_NEGATIVE_ROOT",
    DATA_ROOT / "closeup-negative",
)
DEFAULT_ULCER_ROOT = _resolve_path(
    "DFU_ULCER_ROOT",
    DATA_ROOT / "wound-segmentation" / "data" / "Foot Ulcer Segmentation Challenge",
)
DEFAULT_WOUND_IMAGE_ROOT = _resolve_path(
    "DFU_WOUND_IMAGE_ROOT",
    DATA_ROOT / "Wound Image Dataset",
)
