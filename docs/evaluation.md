# Evaluation

Run the CPU benchmark from the repository root:

```sh
python -m pip install -r requirements.txt
python scripts/evaluate.py --output results/baseline.json
python -m pytest -q
```

The `person-v1` synthetic fixture reconstructs the proposal's four experimental
conditions: translation, shrinking to 45% scale, complete occlusion behind a
pillar, and a scene cut halfway through a 90-frame clip. It is **not** the missing
original fixture, and its numbers must not be presented as reproductions of
Table 1 in the proposal. Texture moves with the subject and the renderer supplies
exact visible-object masks. Initial prompts use 5, 9, and 13 pixels of slack.
Only the initial prompt uses ground truth; propagation never sees later labels.

The retained methods are box-initialized GrabCut, lasso-seeded GrabCut with median
translation, and the historical forward/backward flow gate. `affine` adds
scale-aware motion and refinement; `classical` adds appearance recovery.
`--methods sam2` evaluates the optional SAM 2.1 backend with the same coarse
prompt, using its own initial segmentation. Each run resets
OpenCV's random seed and uses one OpenCV thread. JSON records per-frame masks'
scores and summary statistics, parameters, source digest, commit, dependency
versions, and timing. A sibling CSV contains one summary row per run. Timing
includes segmentation and propagation but excludes fixture construction and
video encoding; it is not end-to-end export speed. Model loading and SAM 2 frame
preparation are reported separately as `setup_seconds`. Raw predicted masks are
scored even on uncertain frames; the full-frame export guard is not counted as
successful tracking.

## Metrics

- **IoU:** intersection over union before safety dilation. Mean IoU is calculated
  over frames where the object is visible.
- **Leak rate:** fraction of visible object pixels outside the predicted mask
  after 9×9 dilation. Both a mean over visible frames and a pixel-weighted rate
  are reported. Fully occluded frames have no leak denominator, rather than being
  counted as successful visible frames.
- **Temporal exposure:** fraction of visible frames with any uncovered pixels,
  worst-frame leak, and the longest consecutive exposure in frames and seconds.
- **Excess redaction:** fraction of the entire frame covered outside the visible
  ground-truth object. This penalizes masking the whole image to get zero leakage.

Coverage scores do not establish that a Gaussian blur removes identifying detail.
Aggregate per-clip results as well as per-frame scores; an average can hide a
short period of complete exposure.

## DAVIS 2017

Use the official [train/validation images and annotations](https://davischallenge.org/davis2017/code.html).
Pass the extracted `DAVIS` directory:

```sh
python scripts/evaluate.py --davis data/DAVIS --sequences blackswan bmx-trees --output results/davis.json
```

The reader uses `JPEGImages/480p`, indexed `Annotations/480p`, and, when sequence
names are omitted, `ImageSets/2017/val.txt`. Palette indices are preserved as
instance IDs. Every frame must have an annotation. Each object is scored
separately and receives its first prompt on its first visible frame. Missing or
malformed annotations fail explicitly. The large dataset is not committed.
