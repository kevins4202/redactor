import cv2
import numpy as np

from tracker import SegmentTracker, redact_frame


def make_frame(center_x, center_y, width=240, height=180):
    rng = np.random.default_rng(4)
    frame = rng.integers(10, 55, (height, width, 3), dtype=np.uint8)
    cv2.circle(frame, (center_x, center_y), 28, (20, 80, 240), -1)
    cv2.circle(frame, (center_x - 9, center_y - 6), 4, (250, 250, 250), -1)
    cv2.circle(frame, (center_x + 9, center_y - 6), 4, (250, 250, 250), -1)
    cv2.line(frame, (center_x - 12, center_y + 10), (center_x + 12, center_y + 10), (10, 10, 10), 3)
    return frame


def mask_center(mask):
    moments = cv2.moments(mask)
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def test_tracks_segmented_object():
    first = make_frame(60, 90)
    tracker = SegmentTracker(first, (27, 57, 66, 66))
    for step in range(1, 9):
        expected = (60 + 5 * step, 90 - 2 * step)
        mask = tracker.update(make_frame(*expected))
    center = mask_center(mask)
    assert abs(center[0] - expected[0]) < 8
    assert abs(center[1] - expected[1]) < 8
    assert cv2.countNonZero(mask) > 1500


def test_redaction_only_changes_masked_region():
    frame = make_frame(60, 90)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (60, 90), 28, 255, -1)
    redacted = redact_frame(frame, mask, dilation=1)
    assert np.array_equal(redacted[0, 0], frame[0, 0])
    assert not np.array_equal(redacted[90, 60], frame[90, 60])

