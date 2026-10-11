"""Redaction and explicit handling of uncertain tracking."""

import cv2
import numpy as np


class ReviewRequired(RuntimeError):
    pass


def coverage_mask(results, shape, on_failure="cover"):
    if on_failure not in ("cover", "stop", "predict"):
        raise ValueError("Failure policy must be cover, stop, or predict")
    union = np.zeros(shape, np.uint8)
    for result in results:
        if result.mask.shape != shape:
            raise ValueError("Tracking mask has the wrong frame dimensions")
        if result.state in ("uncertain", "lost"):
            if on_failure == "stop":
                raise ReviewRequired(f"Target {result.instance_id} needs a new prompt: {result.reason}")
            if on_failure == "cover":
                return np.full(shape, 255, np.uint8), True
        union = cv2.bitwise_or(union, result.mask)
    return union, False


def redact_frame(frame, mask, style="blur", dilation=9, blur_kernel=41):
    if mask.shape != frame.shape[:2]:
        raise ValueError("Mask must match the frame")
    if dilation < 1 or dilation % 2 == 0 or blur_kernel < 3 or blur_kernel % 2 == 0:
        raise ValueError("Dilation and blur kernels must be positive odd numbers (blur >= 3)")
    expanded = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8)) > 0
    if style == "blur":
        replacement = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
    elif style == "solid":
        replacement = np.zeros_like(frame)
    elif style == "pixelate":
        height, width = frame.shape[:2]
        replacement = cv2.resize(frame, (max(1, width // 24), max(1, height // 24)), interpolation=cv2.INTER_AREA)
        replacement = cv2.resize(replacement, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Unknown redaction style: {style}")
    return np.where(expanded[:, :, None], replacement, frame)
