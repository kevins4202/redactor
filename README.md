# Video Redactor Baseline

This repository contains a minimal video-redaction prototype. A hardcoded rectangle identifies the target in the first frame. GrabCut converts that rectangle into a pixel mask, Lucas-Kanade optical flow moves the mask between frames, and Gaussian blur conceals the masked region.

## Run the included demo

```bash
/opt/anaconda3/bin/python scripts/generate_demo.py
/opt/anaconda3/bin/python redact.py
```

The result is written to `output/demo-redacted.mp4`.

## Run on another video

Place the video in `data/`, then edit `INPUT`, `OUTPUT`, and `INITIAL_BOX` at the top of `redact.py`. The box uses `(x, y, width, height)` and should contain the complete target with a little surrounding background.

The current baseline tracks translation only. It does not handle large size changes, rotation, complete occlusion, or scene cuts.

## Test

```bash
/opt/anaconda3/bin/python -m pytest -q
```
