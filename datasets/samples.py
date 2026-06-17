from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path | None = None
    image_id: int | None = None
    annotations: tuple[dict[str, Any], ...] = ()
    augment_profile: str = "none"

    @property
    def is_negative(self) -> bool:
        if self.mask_path is not None:
            return False
        return len(self.annotations) == 0
