# Video Redactor Baseline

This repository contains a minimal video-redaction prototype. You draw a freehand shape around an object in the first frame, GrabCut snaps it to the object's boundary, Lucas-Kanade optical flow moves the resulting mask between frames, and Gaussian blur conceals the masked region.

## Run

```bash
/opt/anaconda3/bin/python redact.py data/input.mp4
```

Hold the left mouse button and draw a circle or irregular closed shape slightly outside the object. Press Enter when the outline is complete. The green preview shows the refined GrabCut segmentation. Press Enter to accept it or `R` to draw again. The result keeps the input filename and is written to `output/input.mp4`.

The current baseline tracks translation only. It does not handle large size changes, rotation, occlusion, or scene cuts.

## Optional folders

`scripts/` only generates the synthetic demonstration video. `tests/` only checks the segmentation and tracking functions. Neither folder is needed to run `redact.py`, so both can be deleted. Keeping `tests/` is useful while modifying the project.

## Test

```bash
/opt/anaconda3/bin/python -m pytest -q
```
