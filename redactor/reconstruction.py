"""Recover masked pixels from observed backgrounds in other video frames."""

from dataclasses import dataclass
import warnings

import cv2
import numpy as np


@dataclass
class Reconstruction:
    image: np.ndarray
    observed: np.ndarray
    source_count: np.ndarray
    accepted_sources: int


class BackgroundReconstructor:
    """Cache background features, then reconstruct each frame independently."""

    def __init__(self, frames, masks):
        if len(frames) != len(masks) or not frames:
            raise ValueError("Supply one mask per frame")
        self.shape = frames[0].shape[:2]
        if any(frame.shape != (*self.shape, 3) or frame.dtype != np.uint8 for frame in frames):
            raise ValueError("Frames must be equally sized uint8 BGR images")
        if any(mask.shape != self.shape for mask in masks):
            raise ValueError("Mask dimensions must match frames")
        self.frames = frames
        self.masks = [np.where(mask > 0, 255, 0).astype(np.uint8) for mask in masks]
        orb = cv2.ORB_create(nfeatures=1800, fastThreshold=10)
        self.features = []
        for frame, mask in zip(frames, self.masks):
            safe = cv2.dilate(mask, np.ones((9, 9), np.uint8))
            self.features.append(orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 255-safe))

    def alignment(self, source, target):
        a, da = self.features[source]
        b, db = self.features[target]
        if da is None or db is None or min(len(da), len(db)) < 12:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        matches = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < .75*n.distance]
        if len(matches) < 12:
            return None
        src = np.float32([a[m.queryIdx].pt for m in matches])
        dst = np.float32([b[m.trainIdx].pt for m in matches])
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        if matrix is None or not np.isfinite(matrix).all() or int(inliers.sum()) < 10 or inliers.mean() < .45:
            return None
        height, width = self.shape
        corners = np.float32([[[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]])
        warped = cv2.perspectiveTransform(corners, matrix)[0]
        ratio = abs(cv2.contourArea(warped)) / (width*height)
        if not np.isfinite(warped).all() or not cv2.isContourConvex(warped) or not .5 < ratio < 2:
            return None
        return matrix

    def reconstruct(self, target, method="aligned", candidates=16):
        if method not in ("spatial", "unaligned", "aligned"):
            raise ValueError("Unknown reconstruction method")
        frame, mask = self.frames[target], self.masks[target]
        missing = mask > 0
        count = np.zeros(self.shape, np.uint16)
        if not missing.any():
            return Reconstruction(frame.copy(), missing, count, 0)
        filled = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        if method == "spatial":
            return Reconstruction(filled, np.zeros(self.shape, bool), count, 0)
        # Spread candidates across the whole shot so a slowly moving target can
        # reveal background outside a short local window. No target frame copy.
        indices = np.unique(np.linspace(0, len(self.frames)-1, min(candidates+1, len(self.frames))).round().astype(int))
        indices = [int(i) for i in indices if i != target]
        height, width = self.shape
        ys, xs = np.nonzero(missing)
        observations, accepted = [], 0
        for source in indices:
            matrix = np.eye(3) if method == "unaligned" else self.alignment(source, target)
            if matrix is None:
                continue
            image = cv2.warpPerspective(self.frames[source], matrix, (width, height), flags=cv2.INTER_LINEAR)
            safe = cv2.erode(255-self.masks[source], np.ones((3, 3), np.uint8))
            valid = cv2.warpPerspective(safe, matrix, (width, height), flags=cv2.INTER_LINEAR) == 255
            observation = image[ys, xs].astype(np.float32)
            observation[~valid[ys, xs]] = np.nan
            observations.append(observation)
            accepted += 1
        if observations:
            stack = np.stack(observations)
            support = np.isfinite(stack[:, :, 0]).sum(axis=0)
            # Require independent agreement opportunities rather than trusting
            # one warped observation. Unsupported pixels retain spatial filling.
            usable = support >= 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(stack, axis=0)
            filled[ys[usable], xs[usable]] = np.clip(median[usable], 0, 255).astype(np.uint8)
            count[ys, xs] = support
        observed = missing & (count >= 2)
        filled[~missing] = frame[~missing]
        return Reconstruction(filled, observed, count, accepted)
