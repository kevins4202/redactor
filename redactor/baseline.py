"""The proposal's classical baselines, retained for controlled comparisons."""

import cv2
import numpy as np

from .types import TrackResult, mask_box


def segment(frame, selection, mode="lasso"):
    selection = np.where(selection > 0, 255, 0).astype(np.uint8)
    if selection.shape != frame.shape[:2] or not np.any(selection) or np.all(selection):
        raise ValueError("Selection must match the frame and include foreground and background")
    mask = np.full(frame.shape[:2], cv2.GC_BGD, np.uint8)
    background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    if mode == "box":
        x1, y1, x2, y2 = mask_box(selection)
        cv2.grabCut(frame, mask, (x1, y1, x2 - x1, y2 - y1), background, foreground, 2, cv2.GC_INIT_WITH_RECT)
    else:
        expanded = cv2.dilate(selection, np.ones((9, 9), np.uint8))
        distance = cv2.distanceTransform(selection, cv2.DIST_L2, 5)
        mask[expanded > 0] = cv2.GC_PR_BGD
        mask[selection > 0] = cv2.GC_PR_FGD
        mask[distance > distance.max() * 0.35] = cv2.GC_FGD
        cv2.grabCut(frame, mask, None, background, foreground, 2, cv2.GC_INIT_WITH_MASK)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def translate(previous_frame, frame, mask, gate=False):
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(previous_gray, 100, 0.01, 5, mask=mask)
    if points is None:
        return mask, False
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
    if moved is None or status is None:
        return mask, False
    good = status.reshape(-1) == 1
    if gate:
        returned, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, moved, None)
        if returned is None or back_status is None:
            return mask, False
        error = np.linalg.norm(points - returned, axis=2).reshape(-1)
        good &= (back_status.reshape(-1) == 1) & (error < 1.5)
    if np.count_nonzero(good) < (3 if gate else 1):
        return mask, False
    dx, dy = np.median(moved[good] - points[good], axis=0).reshape(2)
    if gate and np.hypot(dx, dy) > 20:
        return mask, False
    matrix = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    height, width = mask.shape
    return cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST), True


class BaselineTracker:
    def __init__(self, mode="lasso", gate=False):
        self.mode = mode
        self.gate = gate

    def initialize(self, frame, prompt):
        self.instance_id = prompt.instance_id
        self.mask = segment(frame, prompt.selection(frame.shape[:2]), self.mode)
        self.previous = frame.copy()
        return TrackResult(self.instance_id, self.mask.copy())

    def update(self, frame):
        mask, valid = translate(self.previous, frame, self.mask, self.gate)
        self.mask = mask
        if valid or not self.gate:
            self.previous = frame.copy()
        return TrackResult(self.instance_id, mask.copy(), float(valid), "tracking" if valid else "uncertain")
