"""Metrics distinguish localization, pixel exposure, and excess redaction."""

import cv2
import numpy as np


def frame_metrics(prediction, truth, dilation=9):
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("Prediction and ground truth must be matching 2D masks")
    if dilation < 1 or dilation % 2 == 0:
        raise ValueError("Dilation must be a positive odd integer")
    predicted, visible = prediction > 0, truth > 0
    covered = cv2.dilate(predicted.astype(np.uint8), np.ones((dilation, dilation), np.uint8)) > 0
    intersection = np.count_nonzero(predicted & visible)
    union = np.count_nonzero(predicted | visible)
    visible_pixels = int(np.count_nonzero(visible))
    leaked_pixels = int(np.count_nonzero(visible & ~covered))
    excess_pixels = int(np.count_nonzero(covered & ~visible))
    return {
        "iou": float(intersection / union) if union else 1.0,
        "visible_pixels": visible_pixels,
        "leaked_pixels": leaked_pixels,
        "leak_rate": leaked_pixels / visible_pixels if visible_pixels else None,
        "excess_pixels": excess_pixels,
        "excess_frame_fraction": excess_pixels / visible.size,
    }


def summarize(rows, fps=30.0):
    if not rows:
        raise ValueError("Cannot summarize an empty sequence")
    visible = [row for row in rows if row["visible_pixels"] > 0]
    streak = longest = 0
    for row in rows:
        streak = streak + 1 if row["leaked_pixels"] > 0 else 0
        longest = max(longest, streak)
    pixels = sum(row["visible_pixels"] for row in rows)
    leaks = sum(row["leaked_pixels"] for row in rows)
    return {
        "frames": len(rows),
        "visible_frames": len(visible),
        "mean_iou": float(np.mean([row["iou"] for row in visible])) if visible else None,
        "mean_leak_rate": float(np.mean([row["leak_rate"] for row in visible])) if visible else None,
        "pixel_leak_rate": leaks / pixels if pixels else None,
        "worst_leak_rate": max((row["leak_rate"] for row in visible), default=None),
        "leaking_frame_fraction": sum(row["leaked_pixels"] > 0 for row in visible) / len(visible) if visible else None,
        "longest_leak_frames": longest,
        "longest_leak_seconds": longest / fps,
        "mean_excess_frame_fraction": float(np.mean([row["excess_frame_fraction"] for row in rows])),
    }
