import cv2
import numpy as np


class SegmentTracker:
    def __init__(self, first_frame, initial_box, grabcut_iterations=2):
        self.grabcut_iterations = grabcut_iterations
        self.previous_frame = first_frame.copy()
        self.mask = self._initialize_mask(first_frame, initial_box)

    def _initialize_mask(self, frame, box):
        height, width = frame.shape[:2]
        x, y, box_width, box_height = box
        x = max(0, min(x, width - 2))
        y = max(0, min(y, height - 2))
        box_width = max(1, min(box_width, width - x - 1))
        box_height = max(1, min(box_height, height - y - 1))
        grabcut_mask = np.zeros((height, width), dtype=np.uint8)
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            frame,
            grabcut_mask,
            (x, y, box_width, box_height),
            background_model,
            foreground_model,
            self.grabcut_iterations,
            cv2.GC_INIT_WITH_RECT,
        )
        return np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

    def _estimate_motion(self, previous_gray, current_gray):
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            mask=self.mask,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7,
        )
        if points is not None and len(points) >= 3:
            moved_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )
            valid = status.reshape(-1) == 1
            if np.count_nonzero(valid) >= 3:
                matrix, _ = cv2.estimateAffinePartial2D(
                    points[valid],
                    moved_points[valid],
                    method=cv2.RANSAC,
                )
                if matrix is not None:
                    return matrix

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        vectors = flow[self.mask > 0]
        if len(vectors) == 0:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        dx, dy = np.median(vectors, axis=0)
        return np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

    def _refine_mask(self, frame, propagated):
        kernel = np.ones((7, 7), dtype=np.uint8)
        expanded = cv2.dilate(propagated, kernel, iterations=2)
        core = cv2.erode(propagated, kernel, iterations=1)
        grabcut_mask = np.full(propagated.shape, cv2.GC_BGD, dtype=np.uint8)
        grabcut_mask[expanded > 0] = cv2.GC_PR_BGD
        grabcut_mask[propagated > 0] = cv2.GC_PR_FGD
        grabcut_mask[core > 0] = cv2.GC_FGD
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            frame,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            1,
            cv2.GC_INIT_WITH_MASK,
        )
        refined = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        if cv2.countNonZero(refined) < 20:
            return propagated
        return refined

    def update(self, frame):
        previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matrix = self._estimate_motion(previous_gray, current_gray)
        height, width = frame.shape[:2]
        propagated = cv2.warpAffine(
            self.mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        self.mask = self._refine_mask(frame, propagated)
        self.previous_frame = frame.copy()
        return self.mask


def redact_frame(frame, mask, blur_kernel=41, dilation=9):
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    expanded = cv2.dilate(mask, np.ones((dilation, dilation), dtype=np.uint8))
    blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
    return np.where(expanded[:, :, None] > 0, blurred, frame)

