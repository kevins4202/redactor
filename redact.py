import argparse
from pathlib import Path

import cv2
import numpy as np


def segment(frame, box):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(frame, mask, box, background, foreground, 2, cv2.GC_INIT_WITH_RECT)
    return np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)


def choose_mask(frame):
    while True:
        box = cv2.selectROI(
            "Draw a box around the object, then press Enter",
            frame,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyAllWindows()
        if box[2] == 0 or box[3] == 0:
            raise RuntimeError("no region selected")

        mask = segment(frame, box)
        preview = frame.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Segmentation: Enter to accept, R to redraw", preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key in (10, 13):
            return mask
        if key != ord("r"):
            raise RuntimeError("selection cancelled")


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


def run(input_path, output_path, box=None):
    capture = cv2.VideoCapture(input_path)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not read {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    height, width = frame.shape[:2]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    mask = segment(frame, box) if box else choose_mask(frame)
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
    print(f"wrote {frame_count} frames to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--box", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, tuple(args.box) if args.box else None)
