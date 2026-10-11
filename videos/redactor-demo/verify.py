"""Verify the encoded demo and compare its video panels with source frames."""

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
assert (report["frames"], report["fps"], report["width"], report["height"]) == (900, 30, 1920, 1080)
for time in (8.2, 10.5, 13, 17.5):
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(time * 30))
    ok, image = capture.read()
    assert ok
    for name, left in (("original", 135), ("redacted", 1035)):
        source = cv2.VideoCapture(str(ROOT / "assets" / f"{name}.mp4"))
        source.set(cv2.CAP_PROP_POS_FRAMES, int((time - 8) * .5 * 30 + 1e-6))
        ok, frame = source.read()
        source.release()
        assert ok
        expected = cv2.resize(frame, (750, 675))[15:-15, 15:-15]
        observed = image[284:959, left:left + 750][15:-15, 15:-15]
        error = float(np.mean(np.abs(expected.astype(float) - observed)))
        report["checks"].append({"time": time, "panel": name, "mean_absolute_pixel_error": error})
        assert error < 12, (time, name, error)
shots = []
for time in (4, 13, 24):
    capture.set(cv2.CAP_PROP_POS_FRAMES, time * 30)
    ok, frame = capture.read()
    assert ok
    shots.append(cv2.resize(frame, (640, 360)))
capture.release()
cv2.imwrite(str(ROOT.parents[1] / "docs/images/demo-encoded.jpg"), np.concatenate(shots, axis=0))
report["sha256"] = hashlib.sha256((ROOT / "redactor-demo.mp4").read_bytes()).hexdigest()
(ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
