from pathlib import Path

import cv2
import numpy as np


INPUT = "data/demo.mp4"
OUTPUT = "output/demo-redacted.mp4"
INITIAL_BOX = (28, 68, 94, 94)


def segment(frame, box):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(frame, mask, box, background, foreground, 2, cv2.GC_INIT_WITH_RECT)
    return np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)


def track(previous_frame, frame, mask):
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(previous_gray, 100, 0.01, 5, mask=mask)
    if points is None:
        return mask

    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
    good = status.reshape(-1) == 1
    if not np.any(good):
        return mask

    dx, dy = np.median(moved[good] - points[good], axis=0).reshape(2)
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    height, width = mask.shape
    return cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)


def blur(frame, mask):
    mask = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8))
    blurred = cv2.GaussianBlur(frame, (41, 41), 0)
    return np.where(mask[:, :, None] > 0, blurred, frame)


def run():
    capture = cv2.VideoCapture(INPUT)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not read {INPUT}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    height, width = frame.shape[:2]
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    mask = segment(frame, INITIAL_BOX)
    writer.write(blur(frame, mask))
    previous_frame = frame
    frame_count = 1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        mask = track(previous_frame, frame, mask)
        writer.write(blur(frame, mask))
        previous_frame = frame
        frame_count += 1

    capture.release()
    writer.release()
    print(f"wrote {frame_count} frames to {OUTPUT}")


if __name__ == "__main__":
    run()

