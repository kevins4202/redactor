"""Check the encoded demo against every source frame and its static typography."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
capture = cv2.VideoCapture(str(ROOT / "redactor-demo.mp4"))
if not capture.isOpened():
    raise RuntimeError("Render redactor-demo.mp4 first")
report = {"frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
          "fps": capture.get(cv2.CAP_PROP_FPS),
          "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
          "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), "checks": []}
assert (report["frames"], report["fps"], report["width"], report["height"]) == (360, 30, 1920, 1080)
shots = []
for name in ("original", "redacted"):
    source = cv2.VideoCapture(str(ROOT / "assets" / f"{name}.mp4"))
    errors, title_errors = [], []
    reference_title = None
    for index in range(150):
        ok, image = capture.read()
        source_ok, expected = source.read()
        assert ok and source_ok, (name, index)
        observed = image[240:996, 120:1800]
        error = float(np.mean(cv2.absdiff(expected, observed)))
        errors.append(error)
        assert error < 8, (name, index, error)
        title = image[40:220, 100:1820]
        if reference_title is None:
            reference_title = title.copy()
        title_error = float(np.mean(cv2.absdiff(reference_title, title)))
        title_errors.append(title_error)
        assert title_error < 1.5, ("unstable title", name, index, title_error)
        if index == 75:
            shots.append(cv2.resize(image, (640, 360)))
    source.release()
    report["checks"].append({"panel": name, "frames_checked": len(errors),
        "mean_absolute_pixel_error": float(np.mean(errors)),
        "max_frame_pixel_error": max(errors), "max_title_pixel_change": max(title_errors)})
closing = None
for index in range(60):
    ok, frame = capture.read()
    assert ok
    if closing is None:
        closing = frame.copy()
    assert float(np.mean(cv2.absdiff(closing, frame))) < 1.5, ("unstable closing", index)
    if index == 30:
        shots.append(cv2.resize(frame, (640, 360)))
capture.release()
cv2.imwrite(str(ROOT.parents[1] / "docs/images/demo-encoded.jpg"), np.concatenate(shots, axis=0))
report["sha256"] = hashlib.sha256((ROOT / "redactor-demo.mp4").read_bytes()).hexdigest()
(ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
