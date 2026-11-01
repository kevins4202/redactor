# Redactor

Draw a box around one object, follow it with **SAM 2.1 Base+**, review the result,
and export a video with the visible target covered by a solid mask.

There is one tracking model. No optical flow, GrabCut, detector presets, backend
menu, or experimental appearance recovery runs in this version. A box on another
frame corrects the same target. All corrections are supplied before processing;
a selection on a later frame can also propagate backward through that shot.

## Install and run

Use Python 3.11, 3.12, or 3.13 and install [FFmpeg](https://ffmpeg.org/download.html).
CUDA and Apple Silicon GPUs are selected automatically; otherwise processing
runs on CPU. PyTorch is pinned to the version verified with this model. Processing
can take substantially longer than the clip's playback duration.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/setup_models.py
python app.py
```

On Windows, activate with `.venv\Scripts\activate`. Install a device-appropriate
PyTorch build if the default wheel does not support your CUDA environment.
Setup downloads the official 309 MiB checkpoint, checks its SHA-256, and installs
a pinned revision of the official SAM 2 source. No footage is uploaded.

Open **http://127.0.0.1:8765**:

1. Open a short clip or use the crosswalk sample.
2. Draw a box around one target on a clear frame. **Check selection** previews
   its initial segmentation; redraw the box if it includes the wrong object.
3. **Process clip**, then watch the result.
4. If needed, jump to a flagged frame, draw a corrective box, and process again.
5. Download the result. If frames were flagged, correct them or explicitly mark
   the preview reviewed before the final download becomes available.

The preview keeps the scene visible during tracking uncertainty. It never blacks
out a frame as a fallback. An empty predicted mask is flagged for review; the app
cannot automatically tell full occlusion from a missed target. Shots without a
selection are flagged and are not tracked using memory from another shot.

The model mask is expanded by 1% of the smaller source dimension, with a minimum
of four pixels on each side. This covers small boundary errors but cannot repair
an incorrect identity. Review checks flag empty, unstable, oversized, or abruptly
changing masks. They are heuristics and cannot detect every identity switch.

## Command line

```sh
python redact.py data/crosswalk-short.mp4 --box 1386 730 1660 1080 -o output/example.mp4
python redact.py clip.mp4 --prompts selections.json -o output/corrected.mp4
```

Selections use exclusive `[x1, y1, x2, y2]` boxes in original source pixels and
zero-based frame indices. Later entries correct the same target:

```json
{"version": 1, "prompts": [
  {"id": "target", "frame": 0, "box": [100, 80, 220, 300]},
  {"id": "target", "frame": 60, "box": [160, 90, 260, 290]}
]}
```

Existing polygon/mask selections are read as bounding boxes for replay. Only one target
is supported. Source audio is preserved. Output paths cannot overwrite the source;
existing results require `--overwrite`.

Processing saves a review JSON, replay selections, raw masks in `NAME.masks/`,
and `NAME-preview.mp4`. If no review flags are raised, `NAME.mp4` is also
published. Otherwise the CLI exits with status 2 and leaves the preview for
inspection, without publishing the final video. The browser provides the review
and approval workflow. Sessions are held in memory, so keep the app running
until processing and review are finished.

## Validation

```sh
python -m pytest -q
REDACTOR_MODEL_TESTS=1 python -m pytest -q -m model
python scripts/evaluate.py --output results/rewrite-synthetic.json
```

The first command checks application behavior without loading model weights.
The second runs the actual model on translation, backward propagation, scale,
and corrective prompts. Evaluation scores raw localization, visible-pixel leakage,
and excess coverage separately; a blackout is never credited as better tracking.

The old reports under `docs/` and existing `results/` files describe the previous
research prototype. They are historical evidence, not measurements of this rewrite.
New validation is recorded in [rewrite results](docs/rewrite-results.md).
