# Video Redactor Baseline

This repository contains a minimal video-redaction prototype. You draw a rectangle around an object in the first frame, GrabCut snaps it to the object's boundary, Lucas-Kanade optical flow moves the resulting mask between frames, and Gaussian blur conceals the masked region.

## Run

```bash
/opt/anaconda3/bin/python redact.py data/input.mp4
```

Draw a rectangle around the object and press Enter. The green outline shows the refined GrabCut segmentation. Press Enter to accept it or `R` to draw the rectangle again. The result keeps the input filename and is written to `output/input.mp4`.

For a noninteractive run, provide the rectangle as `x y width height`:

```bash
/opt/anaconda3/bin/python redact.py data/demo.mp4 --box 28 68 94 94
```

The tracker rejects inconsistent optical flow and keeps the mask at its last reliable position during a short occlusion. It still tracks translation only and does not handle large size changes, rotation, long occlusions with target movement, or scene cuts.

## Optional folders

`scripts/` only generates the synthetic demonstration video. `tests/` only checks the segmentation and tracking functions. Neither folder is needed to run `redact.py`, so both can be deleted. Keeping `tests/` is useful while modifying the project.

## Test

```bash
/opt/anaconda3/bin/python -m pytest -q
```
