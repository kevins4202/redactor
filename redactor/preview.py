"""Segment one prompted frame without propagating or exporting a video."""

from dataclasses import replace
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .baseline import segment


def preview_mask(frame, prompts, backend="classical"):
    if not prompts:
        raise ValueError("No targets on this frame; draw a selection or try another description")
    if any(prompt.action != "select" for prompt in prompts):
        raise ValueError("An end marker has no segmentation to preview")
    if backend == "sam2":
        from .sam_backend import Sam2Session
        # The video backend receives the same quality-100 JPEG preparation.
        with tempfile.TemporaryDirectory(prefix="redactor-preview-") as directory:
            if not cv2.imwrite(str(Path(directory) / "000000.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise OSError("Cannot prepare the preview frame")
            session = Sam2Session(directory)
            try:
                results = session.step(frame, [replace(prompt, frame_index=0) for prompt in prompts])
                masks = [result.mask for result in results]
            finally:
                session.close()
    elif backend in ("classical", "baseline", "learned"):
        # Learned recovery changes re-detection only, not initial segmentation.
        masks = [segment(frame, prompt.selection(frame.shape[:2])) for prompt in prompts]
    else:
        raise ValueError(f"Unknown preview backend: {backend}")
    union = np.zeros(frame.shape[:2], np.uint8)
    for mask in masks:
        union |= mask
    if not np.any(union):
        raise ValueError("Segmentation found no foreground; redraw the selection")
    return union
