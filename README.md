# Video Redactor Baseline

This repository contains a minimal video-redaction prototype. A hardcoded rectangle identifies the target in the first frame. GrabCut converts that rectangle into a pixel mask, Lucas-Kanade optical flow estimates the target motion between frames, and GrabCut refines the propagated mask before Gaussian blur is applied.

## Run the included demo

```bash
/opt/anaconda3/bin/python scripts/generate_demo.py
/opt/anaconda3/bin/python main.py
```

The result is written to `output/demo-redacted.mp4`.

## Run on another video

Place the video in `data/`, inspect its first frame, and set `INITIAL_BOX` in `config.py` to `(x, y, width, height)` around the target. You can also override it without changing the file:

```bash
/opt/anaconda3/bin/python main.py --input data/input.mp4 --output output/redacted.mp4 --box 100,80,160,220
```

The target should be fully contained in the box with some visible background around it. The current baseline handles one continuously visible object and does not recover after a complete occlusion or a scene cut.

## Test

```bash
/opt/anaconda3/bin/python -m pytest -q
```
