"""Lightweight OpenCV selection UI with explicit cancellation and previews."""

import cv2
import numpy as np

from .baseline import segment


def draw_shape(frame):
    title = "Select target | Drag to outline | Enter: preview | R: reset | Esc: cancel"
    state = {"drawing": False, "points": [], "image": frame.copy()}

    def mouse(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(drawing=True, points=[(x, y)], image=frame.copy())
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            cv2.line(state["image"], state["points"][-1], (x, y), (65, 220, 140), 2)
            state["points"].append((x, y))
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            state["points"].append((x, y))
            if len(state["points"]) > 2:
                cv2.line(state["image"], state["points"][-1], state["points"][0], (65, 220, 140), 2)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            cv2.imshow(title, state["image"])
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Selection cancelled")
            if key == ord("r"):
                state.update(drawing=False, points=[], image=frame.copy())
            if key in (10, 13) and len(state["points"]) > 2:
                selection = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillPoly(selection, [np.array(state["points"], np.int32)], 255)
                if 0 < np.count_nonzero(selection) < selection.size:
                    return selection
    finally:
        cv2.destroyAllWindows()


def choose_selection(frame):
    height, width = frame.shape[:2]
    scale = min(1.0, 1200 / width, 800 / height)
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    while True:
        selection = draw_shape(display)
        try:
            mask = segment(display, selection)
        except (ValueError, cv2.error):
            print("Could not segment that outline. Draw slightly outside the object.")
            continue
        preview = display.copy()
        tint = np.full_like(preview, (65, 220, 140))
        mixed = cv2.addWeighted(preview, 0.65, tint, 0.35, 0)
        preview[mask > 0] = mixed[mask > 0]
        title = "Preview | Enter: accept | R: redraw | Esc: cancel"
        cv2.imshow(title, preview)
        try:
            while True:
                key = cv2.waitKey(20) & 0xFF
                if key in (10, 13):
                    return cv2.resize(selection, (width, height), interpolation=cv2.INTER_NEAREST)
                if key == ord("r"):
                    break
                if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    raise ValueError("Selection cancelled")
        finally:
            cv2.destroyAllWindows()


def choose_mask(frame):
    return segment(frame, choose_selection(frame))
