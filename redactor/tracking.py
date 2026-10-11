"""Scale-aware classical tracking with appearance checks and re-detection.

Motion is estimated from forward/backward-checked LK correspondences using a
RANSAC similarity transform. Uncertain motion is never silently called a success.
A frozen first-frame appearance template provides a recovery anchor; it is not
updated from uncertain observations and therefore cannot learn an occluder.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .baseline import segment
from .types import TrackResult, mask_box


@dataclass
class TrackerConfig:
    refine_every: int = 5
    min_points: int = 4
    fb_error: float = 1.5
    min_inlier_ratio: float = 0.55
    min_appearance: float = 0.60
    recovery_threshold: float = 0.90
    cut_threshold: float = 0.60
    search_every: int = 1
    search_size: int = 480


def scene_difference(previous, frame):
    """Histogram distance plus spatial change avoids declaring a pan a cut."""
    small_previous = cv2.resize(previous, (64, 48))
    small_frame = cv2.resize(frame, (64, 48))
    histograms = []
    for image in (small_previous, small_frame):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        histograms.append(cv2.normalize(hist, None, norm_type=cv2.NORM_L1))
    distance = cv2.compareHist(*histograms, cv2.HISTCMP_BHATTACHARYYA)
    difference = float(np.mean(np.abs(small_previous.astype(float) - small_frame))) / 255
    return float(distance) if difference > 0.12 else 0.0


def similarity_motion(previous, frame, mask, config):
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8))
    points = cv2.goodFeaturesToTrack(previous_gray, 180, 0.01, 3, mask=interior)
    if points is None or len(points) < config.min_points:
        return None, 0.0, "too few features"
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
    if moved is None:
        return None, 0.0, "optical flow failed"
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, moved, None)
    if returned is None:
        return None, 0.0, "backward flow failed"
    error = np.linalg.norm(points - returned, axis=2).ravel()
    good = (status.ravel() > 0) & (backward.ravel() > 0) & (error < config.fb_error)
    if np.count_nonzero(good) < config.min_points:
        return None, 0.0, "inconsistent optical flow"
    matrix, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC,
                                                ransacReprojThreshold=2.5, maxIters=1000)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None, 0.0, "motion fit failed"
    ratio = float(inliers.mean())
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if ratio < config.min_inlier_ratio or not 0.80 < scale < 1.25:
        return None, ratio, "implausible motion"
    # Support falls as an object disappears, even when the surviving points fit.
    confidence = ratio * min(1.0, np.count_nonzero(good) / max(8, len(points) * 0.6))
    return matrix.astype(np.float32), confidence, ""


class ClassicalTracker:
    def __init__(self, config=None, recover=True):
        self.config = config or TrackerConfig()
        self.recover = recover

    def initialize(self, frame, prompt):
        self.instance_id = prompt.instance_id
        self.mask = segment(frame, prompt.selection(frame.shape[:2]))
        if not np.any(self.mask):
            raise ValueError("Segmentation produced an empty target; redraw the selection")
        self.previous = frame.copy()
        self.index = 0
        self.missing = 0
        self.velocity = np.zeros(2, np.float32)
        self.anchor_mask = self.mask.copy()
        x1, y1, x2, y2 = mask_box(self.mask)
        self.template = frame[y1:y2, x1:x2].copy()
        self.template_mask = self.mask[y1:y2, x1:x2].copy()
        self.initial_area = np.count_nonzero(self.mask)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], self.mask, [30, 32], [0, 180, 0, 256])
        histogram = cv2.GaussianBlur(histogram, (3, 3), 0)
        self.histogram = cv2.normalize(histogram, None, 0, 255, cv2.NORM_MINMAX)
        return TrackResult(self.instance_id, self.mask.copy())

    def appearance(self, frame, mask):
        if not np.any(mask):
            return 0.0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        projection = cv2.calcBackProject([hsv], [0, 1], self.histogram, [0, 180, 0, 256], 1)
        # Fraction of predicted pixels compatible with the initial appearance.
        return float(np.mean(projection[mask > 0] > 12))

    def refine(self, frame, propagated):
        if np.count_nonzero(propagated) < 12:
            return propagated
        # Restrict GrabCut to a padded crop; full-HD global fitting is wasteful.
        x1, y1, x2, y2 = mask_box(propagated)
        padding = max(8, int(max(x2 - x1, y2 - y1) * 0.15))
        height, width = propagated.shape
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
        region = propagated[y1:y2, x1:x2]
        core = cv2.erode(region, np.ones((3, 3), np.uint8))
        if not np.any(core):
            return propagated
        labels = np.full(region.shape, cv2.GC_BGD, np.uint8)
        labels[cv2.dilate(region, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_PR_BGD
        labels[region > 0] = cv2.GC_PR_FGD
        labels[core > 0] = cv2.GC_FGD
        if not np.any(labels == cv2.GC_BGD):
            return propagated
        cv2.grabCut(frame[y1:y2, x1:x2], labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 1, cv2.GC_INIT_WITH_MASK)
        refined = np.zeros_like(propagated)
        refined[y1:y2, x1:x2] = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0)
        ratio = np.count_nonzero(refined) / np.count_nonzero(propagated)
        return refined if 0.65 < ratio < 1.5 else propagated

    def redetect(self, frame):
        """Search independently of the old location, including after a cut.

        Masked color-template error is checked at multiple scales. Ambiguous
        matches are rejected. This is an explicit classical baseline, not a
        learned identity recognizer; large pose/viewpoint changes can defeat it.
        """
        height, width = frame.shape[:2]
        search_scale = min(1.0, self.config.search_size / max(height, width))
        search_frame = cv2.resize(frame, None, fx=search_scale, fy=search_scale) if search_scale < 1 else frame
        search_height, search_width = search_frame.shape[:2]
        template_height, template_width = self.template.shape[:2]
        best = None
        candidates = []
        current_scale = np.sqrt(max(1, np.count_nonzero(self.mask)) / self.initial_area)
        scales = sorted(set([*np.linspace(0.40, 1.45, 15), float(current_scale), 1.0]))
        for scale in scales:
            size = (max(5, round(template_width * scale * search_scale)), max(5, round(template_height * scale * search_scale)))
            if size[0] > search_width or size[1] > search_height:
                continue
            template = cv2.resize(self.template, size)
            mask = cv2.resize(self.template_mask, size, interpolation=cv2.INTER_NEAREST)
            response = cv2.matchTemplate(search_frame, template, cv2.TM_SQDIFF_NORMED, mask=mask)
            response = np.nan_to_num(response, nan=2.0, posinf=2.0, neginf=2.0)
            minimum, _, location, _ = cv2.minMaxLoc(response)
            score = max(0.0, 1 - float(minimum))
            x, y = location
            candidates.append((score, x, y, size, mask))
            if best is None or score > best[0]:
                best = candidates[-1]
            # The second match at the SAME scale matters too (e.g. two people
            # wearing the same outfit). Suppress nearby overlapping placements.
            response[max(0, y - size[1] // 2):y + size[1] // 2 + 1,
                     max(0, x - size[0] // 2):x + size[0] // 2 + 1] = 2
            second, _, other_location, _ = cv2.minMaxLoc(response)
            candidates.append((max(0.0, 1 - float(second)), *other_location, size, mask))
        if best is None or best[0] < self.config.recovery_threshold:
            return None, 0.0
        score, x, y, size, mask = best
        # Reject competing locations with almost identical evidence.
        for other, ox, oy, osize, _ in candidates:
            distance = np.hypot(ox + osize[0] / 2 - x - size[0] / 2,
                                oy + osize[1] / 2 - y - size[1] / 2)
            if distance > max(size) and other > score - 0.015:
                return None, score
        recovered = np.zeros(search_frame.shape[:2], np.uint8)
        recovered[y:y + size[1], x:x + size[0]] = mask
        if search_scale < 1:
            recovered = cv2.resize(recovered, (width, height), interpolation=cv2.INTER_NEAREST)
        if self.appearance(frame, recovered) < self.config.min_appearance:
            return None, score
        return self.refine(frame, recovered), score

    def update(self, frame):
        self.index += 1
        cut = scene_difference(self.previous, frame) > self.config.cut_threshold
        candidate = None
        confidence = 0.0
        reason = "scene cut" if cut else "target not located"
        if not cut and self.missing == 0:
            matrix, confidence, reason = similarity_motion(self.previous, frame, self.mask, self.config)
            if matrix is not None:
                height, width = self.mask.shape
                propagated = cv2.warpAffine(self.mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
                appearance = self.appearance(frame, propagated)
                confidence = min(confidence, appearance)
                if confidence >= self.config.min_appearance:
                    candidate = self.refine(frame, propagated) if self.index % self.config.refine_every == 0 else propagated
                    old_box, new_box = mask_box(self.mask), mask_box(candidate)
                    self.velocity = (np.array(new_box[:2]) + new_box[2:] - np.array(old_box[:2]) - old_box[2:]) / 2
                else:
                    reason = "appearance or feature support dropped"
        recovered = False
        if candidate is None and self.recover and self.index % self.config.search_every == 0:
            candidate, score = self.redetect(frame)
            if candidate is not None:
                recovered, confidence = True, score
                self.velocity[:] = 0
        self.previous = frame.copy()
        if candidate is not None and np.any(candidate):
            self.mask = candidate
            self.missing = 0
            return TrackResult(self.instance_id, candidate.copy(), float(confidence), "recovered" if recovered else "tracking")
        self.missing += 1
        # Keep a predicted location for review, but label it uncertain. The export
        # policy decides whether to cover the frame or require a corrective prompt.
        if not cut and self.missing <= 10:
            matrix = np.array([[1, 0, self.velocity[0]], [0, 1, self.velocity[1]]], np.float32)
            self.mask = cv2.warpAffine(self.mask, matrix, (self.mask.shape[1], self.mask.shape[0]), flags=cv2.INTER_NEAREST)
        return TrackResult(self.instance_id, self.mask.copy(), 0.0,
                           "uncertain" if self.missing <= 10 else "lost", reason)
