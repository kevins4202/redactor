"""Apply a solid mask with a small source-resolution coverage margin."""

import cv2
import numpy as np


def coverage_margin(shape):
    return max(4, round(min(shape) * 0.01))


def expanded_mask(mask):
    radius = coverage_margin(mask.shape)
    return cv2.dilate(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))


def redact_frame(frame, mask):
    if mask.shape != frame.shape[:2]:
        raise ValueError("Mask must match the frame")
    result = frame.copy()
    result[expanded_mask(mask) > 0] = 0
    return result
