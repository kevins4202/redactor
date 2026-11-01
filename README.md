# Text-guided video object removal

Describe an object. Grounding DINO selects it, SAM 2 tracks its mask, and our
OpenCV pipeline reconstructs the background from other frames. Unrecoverable
pixels use spatial inpainting. No generative video model is used.

## Run

Run commands from this project folder. Python 3.11-3.13 and FFmpeg are required.

```sh
source .venv/bin/activate
python -m pip install -e ".[test]"
python scripts/remove_object.py examples/crosswalk.mp4 \
  --text "woman wearing a white skirt" -o output/example.mp4
```

For a fresh machine, first create the environment and install the tracking model:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/setup_models.py
```

The text detector downloads its pinned weights on first use. CUDA or the Mac GPU
is selected automatically for SAM; local processing can be slow. Footage stays on
the machine running the program.

Inspect `output/example-assets/selection.jpg` to check that the correct object was
selected. That folder also contains a four-panel comparison video and masks.
`output/example.mp4` is the final reconstruction; `output/example.json` records
settings and observation support. Add `--overwrite` to replace existing outputs.

For an easier example:

```sh
python scripts/remove_object.py examples/dog.mp4 --text dog -o output/dog.mp4
```

Instead of text, use `--box x1 y1 x2 y2` in original source coordinates. `--masks`
reuses original-resolution masks for controlled reconstruction comparisons.

## Results and tests

[Open the completed demos](results/demo.html) · [Read the measured results](results/README.md)

```sh
python scripts/benchmark_removal.py
python -m pytest -p no:capture -q
REDACTOR_MODEL_TESTS=1 python -m pytest -p no:capture -q
```

The first test command skips the three optional actual-model tests. The benchmark
compares spatial filling, unaligned temporal reconstruction, and aligned temporal
reconstruction with identical masks. It evaluates background reconstruction,
not the proposal's tracking IoU/leakage tests.

## Files

```text
redactor/   selection, tracking, reconstruction, video I/O
scripts/    run removal, run benchmark, set up model
tests/     focused reconstruction and model tests
examples/   crosswalk and dog clips, existing annotations
results/    current measurements, report, demo page
reference/  proposal and supplied past-project examples
output/     generated videos/masks; old artifacts in legacy/ (ignored)
models/     downloaded model source and weights (ignored)
```

The old browser app, classical trackers, duplicate reports, and promotional-video
project have been removed from this branch. Their tracked versions remain in Git
history. Existing completed demos are preserved under `output/removal/`.

This is a short-clip prototype: one target visible in the first frame, one shot,
at most 300 frames, and output capped at 640 pixels wide. Text uses the highest
scoring first-frame candidate and can be wrong. Reconstruction uses background
alignment and a masked temporal median; moving bystanders, parallax, seams,
shadows, and never-visible background remain limitations. Temporal source
optimization and separate seam blending are not implemented.
