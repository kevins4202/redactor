import argparse
from pathlib import Path

import cv2
import numpy as np


def segment(frame, selection):
    kernel = np.ones((9, 9), dtype=np.uint8)
    expanded = cv2.dilate(selection, kernel)
    distance = cv2.distanceTransform(selection, cv2.DIST_L2, 5)
    mask = np.full(frame.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    mask[expanded > 0] = cv2.GC_PR_BGD
    mask[selection > 0] = cv2.GC_PR_FGD
    mask[distance > distance.max() * 0.35] = cv2.GC_FGD
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(frame, mask, None, background, foreground, 2, cv2.GC_INIT_WITH_MASK)
    return np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)


def draw_shape(frame):
    title = "Hold left mouse to draw, Enter to finish, R to reset"
    state = {"drawing": False, "points": [], "image": frame.copy()}

    def mouse(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["points"] = [(x, y)]
            state["image"] = frame.copy()
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            previous = state["points"][-1]
            state["points"].append((x, y))
            cv2.line(state["image"], previous, (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["points"].append((x, y))
            if len(state["points"]) > 2:
                cv2.line(state["image"], state["points"][-1], state["points"][0], (0, 255, 0), 2)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse)
    while True:
        cv2.imshow(title, state["image"])
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and len(state["points"]) > 2:
            selection = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(selection, [np.array(state["points"], dtype=np.int32)], 255)
            cv2.destroyAllWindows()
            return selection
        if key == ord("r"):
            state["points"] = []
            state["image"] = frame.copy()
        if key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("selection cancelled")


def choose_mask(frame):
    height, width = frame.shape[:2]
    scale = min(1.0, 1200 / width, 800 / height)
    display = cv2.resize(frame, None, fx=scale, fy=scale)

    while True:
        selection = draw_shape(display)
        mask = segment(display, selection)

        preview = display.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Segmentation: Enter to accept, R to redraw", preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key in (10, 13):
            return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
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


def run(input_path):
    output_path = str(Path("output") / Path(input_path).name)
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

    mask = choose_mask(frame)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input)
