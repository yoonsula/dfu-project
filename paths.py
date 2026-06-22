"""Path configuration for the AI Hub DFU classification trainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_path(env_key: str, default: Path) -> Path:
    raw = os.environ.get(env_key)
    if raw is None:
        return default.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DINOV3_MODEL_PATH = _resolve_path(
    "DINOV3_MODEL_PATH",
    ASSETS_DIR / "dinov3-hf",
)

CHECKPOINT_DIR = _resolve_path("DFU_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")
TRAIN_OUTPUT_DIR = _resolve_path("DFU_TRAIN_OUTPUT_DIR", CHECKPOINT_DIR)

DATA_ROOT = _resolve_path("DFU_DATA_ROOT", PROJECT_ROOT / ".." / ".." / "03_데이터")
DEFAULT_DFU_CLASSIFICATION_DATA_ROOT = _resolve_path(
    "DFU_CLASSIFICATION_DATA_ROOT",
    DATA_ROOT / "dfu_classification_data",
)
