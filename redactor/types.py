"""Shared prompts and results used by selection, tracking, and evaluation."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass
class TargetPrompt:
    instance_id: str
    frame_index: int
    mask: np.ndarray | None = None
    box: tuple[float, float, float, float] | None = None
    label: str = "object"
    source: str = "manual"
    action: Literal["select", "end"] = "select"

    def selection(self, shape: tuple[int, int]) -> np.ndarray:
        if not self.instance_id or self.frame_index < 0:
            raise ValueError("Prompts need an instance id and a nonnegative frame index")
        if self.action not in ("select", "end"):
            raise ValueError("Prompt action must be select or end")
        if self.action == "end":
            if self.mask is not None or self.box is not None:
                raise ValueError("An end prompt must not include selection geometry")
            return np.zeros(shape, np.uint8)
        if (self.mask is None) == (self.box is None):
            raise ValueError("Provide exactly one of mask or box")
        height, width = shape
        if self.mask is not None:
            if self.mask.shape != shape or not np.any(self.mask):
                raise ValueError("Prompt mask must match the frame and contain foreground")
            return np.where(self.mask > 0, 255, 0).astype(np.uint8)
        if not np.all(np.isfinite(self.box)):
            raise ValueError("Box coordinates must be finite")
        x1, y1, x2, y2 = self.box
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Box must have positive area inside the frame")
        selection = np.zeros(shape, np.uint8)
        selection[y1:y2, x1:x2] = 255
        return selection


@dataclass
class TrackResult:
    instance_id: str
    mask: np.ndarray
    confidence: float = 1.0
    state: Literal["tracking", "recovered", "uncertain", "lost", "absent"] = "tracking"
    reason: str = ""


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return an exclusive xyxy box; an empty mask has box (0, 0, 0, 0)."""
    x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
    return x, y, x + width, y + height
