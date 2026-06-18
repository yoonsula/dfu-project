"""Check that bundled models and assets are present before inference."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import DINOV3_CHECKPOINT, DINOV3_REPO


def main() -> int:
    required = [
        ("DINOv3 repo (segmentation)", DINOV3_REPO),
        ("DINOv3 backbone weights", DINOV3_CHECKPOINT),
    ]
    missing: list[tuple[str, Path]] = []
    for label, path in required:
        if not path.exists():
            missing.append((label, path))

    if missing:
        print("Missing required assets:")
        for label, path in missing:
            print(f"  - {label}: {path}")
        return 1

    print("All required assets are present.")
    for label, path in required:
        print(f"  OK  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
