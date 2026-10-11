# Redactor

Select an object once, propagate its mask through a video, and export a redacted
copy. The CPU backend uses GrabCut, RANSAC similarity transforms, periodic
segmentation, appearance monitoring, and multiscale re-detection. The original
translation-only tracker remains available for comparison.

## Run

```bash
python -m pip install -r requirements.txt
python redact.py data/crosswalk-short.mp4
```

Draw a freehand outline slightly outside the object. Press Enter to preview,
Enter to accept, R to redraw, or Escape to cancel. The default output is
`output/crosswalk-short-redacted.mp4`. Each export also saves its prompts and a
frame-by-frame review log. FFmpeg preserves input audio and produces H.264 MP4;
use `--no-audio` to permit a video-only export without FFmpeg.

```sh
python redact.py clip.mp4 --box 100 80 220 300 --style solid
python redact.py clip.mp4 --prompts selections.json --mask-dir output/masks
python redact.py clip.mp4 --backend baseline --on-failure predict
python redact.py data/crosswalk-short.mp4 --backend sam2 --prompts examples/crosswalk.json
python redact.py clip.mp4 --backend sam2 --text "the woman in a white skirt"
python redact.py clip.mp4 --detect faces plates
python redact.py clip.mp4 --detect faces --detector haar
```

Boxes are exclusive `[x1, y1, x2, y2]` coordinates in the original video. Repeat
`--box` for multiple targets. A JSON prompt file can add targets or correct an
existing target at any frame:

```json
{"version": 1, "prompts": [
  {"id": "person", "frame": 0, "box": [100, 80, 220, 300]},
  {"id": "person", "frame": 60, "box": [160, 90, 260, 290]},
  {"id": "second", "frame": 40, "polygon": [[10, 20], [80, 20], [80, 150], [10, 150]]}
]}
```

Masks can also be supplied as a `"mask": "relative/path.png"` entry. Existing
outputs are protected unless `--overwrite` is supplied. Input and output must
always be different files.

SAM 2 and Grounding DINO require the optional [model setup](docs/models.md).
Grounding DINO accepts descriptions or the face/license-plate presets; the Haar
face option works with the base dependencies. Detection runs on every frame by
default, assigns separate instance IDs, and picks up later arrivals. Increase
`--detect-every` for speed at the cost of potentially missing new targets between
detection frames. Detector failures and identity ambiguity still need review.

## Review and limitations

Uncertain or lost tracking produces a solid covered frame by default, recorded
in the review log. Use `--on-failure stop` to abort and add a corrective prompt.
`--on-failure predict` exports uncertain predicted locations for research and
can expose the target. Covered frames are a request for review, not evidence
that tracking worked. Undetected errors remain possible with all policies.

Classical appearance recovery is limited by pose changes, similar-looking
objects, and partial occlusion. A Gaussian blur is not a guarantee of anonymity;
use `--style solid` when identifying detail must be completely covered, and
review the entire output before sharing.

## Evaluation

See [evaluation notes](docs/evaluation.md) for metrics, reconstructed synthetic
fixtures, and the DAVIS reader. The original proposal is [581proposal.pdf](581proposal.pdf).

```sh
python scripts/evaluate.py --output results/comparison.json
```

## Test

```bash
python -m pytest -q
```
