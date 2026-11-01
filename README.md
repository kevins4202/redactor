# Text-guided video object removal

Experimental branch: `codex/temporal-object-removal`.

Describe an object, obtain its box with **Grounding DINO**, track its masks with
**SAM 2.1 Base+**, and reconstruct the background using our OpenCV pipeline.
The reconstruction uses ORB background matches, RANSAC homographies, masked
pixel medians across frames, and Telea inpainting where observations are absent.
There is no generative video model and no model training.

## Run

Use the existing Python 3.11-3.13 environment, FFmpeg, and SAM setup:

```sh
python -m pip install -r requirements-lock.txt
python -m pip install -e ".[text]"
python scripts/setup_models.py
python scripts/remove_object.py clip.mp4 --text "red car" -o output/removed.mp4
```

The text detector downloads its pinned pretrained checkpoint on first use.
Footage is processed locally on the machine running the script. Check the saved
`removed-assets/selection.jpg`: text selection uses the highest-scoring first-frame
candidate and can select the wrong instance. This prototype does not pause for
interactive confirmation. Targets must be visible in the first frame.

For a manual selection, use `--box x1 y1 x2 y2` in original image coordinates.
For reconstruction-only comparisons, use `--masks path/to/original-resolution-masks`
instead. These three input methods are mutually exclusive. The output resolution
is capped at 640 pixels wide; clips must be a single shot with at most 300 frames.
Existing output artifacts require `--overwrite`.

Outputs include the removed video with source audio, a four-panel comparison,
selection/mask artifacts, and a JSON report. Raw masks are expanded by 3 working
pixels; all three comparison methods receive exactly those masks. Reconstruction
uses up to 17 distributed source frames, excludes the frame being reconstructed,
and requires at least two unmasked observations before copying a temporal median.
Unrecoverable pixels use single-frame Telea filling. Pixels outside the expanded
mask are unchanged before lossy video encoding. No separate seam blending is
implemented yet.

## Results and reproducibility

[Measured comparisons and meeting notes](docs/removal-results.md)

```sh
python -m pytest -p no:capture -q -m "not model"
python scripts/benchmark_removal.py
```

The benchmark compares single-frame inpainting, unaligned temporal medians,
and aligned temporal medians with identical masks and source access. Clean
reference pixels are used only for scoring. This is a new reconstruction
benchmark, not a continuation of the proposal's tracking IoU table.

## Scope

This is a first geometry-based reconstruction experiment, not a new foundation
model or a claim of research novelty. Strong parallax, moving non-target objects,
shadows, inaccurate masks, and never-observed background remain difficult.
Observation support is not a calibrated confidence or proof of reconstruction
correctness. The text selector has only been smoke-tested on a small number of
prompts; broad referring-expression accuracy has not been established.

The earlier redaction application remains available through `python app.py`.
Its [workflow](docs/redaction-workflow.md) and historical reports are retained
for comparison; they do not describe the removal CLI. New removal code is limited
to `grounding.py`, `reconstruction.py`, and the two scripts above. We have not
reintroduced old tracker, recovery, or configuration menus.
