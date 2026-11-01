"""Preview a selection with the same model and input preparation as processing."""

from dataclasses import replace
from pathlib import Path
import tempfile

import cv2

from .sam_backend import VideoSegmenter


def preview_mask(frame, prompts):
    if len(prompts) != 1 or prompts[0].action != 'select':
        raise ValueError("Draw one selection on this frame")
    with tempfile.TemporaryDirectory(prefix='redactor-preview-') as directory:
        if not cv2.imwrite(str(Path(directory) / '000000.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
            raise OSError('Could not prepare the preview')
        model = VideoSegmenter()
        try:
            _, mask, _ = next(model.propagate(directory, [replace(prompts[0], frame_index=0)], frame.shape[:2]))
        finally:
            model.close()
    if not mask.any():
        raise ValueError('No foreground found; adjust the box and try again')
    return mask
