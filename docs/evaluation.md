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

Final reports also include `guarded_output` scores and the fraction of processed
frames flagged for review. Those scores account for full-frame covering and its
large excess-redaction cost. JSON is checkpointed after every run; `complete`
must be true before summarizing a report.

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

## Crosswalk footage

`data/crosswalk-annotations.json` contains six manually traced polygons (frames
0, 30, 60, 90, 120, 149) for the woman in the white skirt. These are approximate
development annotations, not expert-reviewed dense ground truth. They include
visible clothing and the shoulder bag, and exclude the carried pink box where
separable. No model-generated mask is used as a ground-truth annotation.

```sh
python scripts/evaluate.py --annotations data/crosswalk-annotations.json --methods lasso classical sam2 --slack 5 --output results/crosswalk.json
```

Tracking still runs through every source frame; only annotated frames receive a
score. Continuous exposure duration is deliberately unavailable for sparse
labels. `examples/crosswalk.json` is a separate, replayable initial box for the
demo; evaluation initializes from the annotated silhouette plus the chosen slack.

## Reproduce the final comparison

```sh
python scripts/download_davis.py --help
python scripts/evaluate.py --methods box lasso gate affine classical sam2 --output results/final-synthetic.json
python scripts/evaluate.py --davis data/DAVIS --sequences blackswan bmx-trees breakdance car-roundabout drift-chicane libby horsejump-high soapbox --methods lasso classical sam2 --slack 5 --output results/final-davis.json
python scripts/evaluate.py --annotations data/crosswalk-annotations.json --methods lasso classical sam2 --slack 5 --output results/final-crosswalk.json
python -m pip install "matplotlib>=3.8"
python scripts/summarize_results.py
```

See [the final report](final-report.md) for subset selection, aggregation,
development-set limitations, runtime interpretation, and failure cases. The
synthetic and real-video runs have different source digests because detector
provenance was added between them; the measured tracking algorithms did not
change. Dirty flags can reflect untracked result and documentation artifacts;
the recorded source digest identifies the exact Python files used.
