# Optional models

The CPU classical backend needs only `requirements.txt`. SAM 2 and text detection
are optional local models. They do not upload footage. Model downloads require
internet access; inference runs locally after weights have been cached.

Use Python 3.11 or 3.12 for the widest compatibility. Install a matching PyTorch
and TorchVision build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).
For this project's tested NVIDIA environment:

```sh
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-ml.txt
git clone https://github.com/facebookresearch/sam2.git models/sam2
git -C models/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
```

On Windows PowerShell, disable SAM 2's optional CUDA extension before installing:

```powershell
$env:SAM2_BUILD_CUDA = '0'
python -m pip install --no-build-isolation -e models/sam2
python scripts/setup_models.py
```

On macOS/Linux, use `SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e models/sam2`.
This backend disables the optional hole-filling postprocessor, so its custom CUDA
extension is not required. CPU inference is available but slower. The default
checkpoint is SAM 2.1 Hiera Tiny; `--checkpoint`, `--model-config`, and `--device`
allow another compatible checkpoint or CPU execution.

SAM 2 uses its image prompting and video propagation APIs. Frames and memory are
offloaded to CPU to fit smaller GPUs. Source frames are extracted as quality-100
JPEGs for the official frame-directory API. This uses temporary disk space and
CPU RAM proportional to clip length. Later prompts are inserted only when their
frame is reached; future corrections do not leak into earlier results.

An empty mask is marked uncertain. Scene cuts remain flagged until a new prompt
for that instance confirms the target in the new shot. As with the classical
backend, default export covers uncertain frames and the review log explains why.

The default Grounding DINO model is `IDEA-Research/grounding-dino-tiny`, pinned
to revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. This project tested
Transformers 5.19.0. [Environment metadata](../results/environment.json) records
all major tested packages. The SAM 2 checkpoint SHA-256 is
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.

After setup, use `python scripts/check_models.py` for real text, face, and plate
smoke checks. The plate example requires DAVIS `car-roundabout` locally. Model
inference remains local, but these detectors can miss small/profile faces,
unusual plates, and ambiguous descriptions. The smoke checks are not a recall
benchmark. Automatic corrections can be reviewed in the browser by switching
back to **Draw a selection** after the first export.

References: [SAM 2 source and checkpoint license](https://github.com/facebookresearch/sam2),
[Grounding DINO integration](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino).

## Learned appearance recovery

`--backend learned` uses [TorchVision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
with the explicitly selected `ResNet18_Weights.IMAGENET1K_V1` weights. TorchVision
downloads the 44.7 MB checkpoint to its normal torch cache on first use; subsequent
inference is local. Install the existing optional torch/torchvision dependencies.
To cache the model before opening the studio:

```sh
python -c "from redactor.appearance import AppearanceEncoder; model = AppearanceEncoder(); print(model.metadata); model.close()"
```

This experiment uses frozen layer3 features at stride 16. Input is converted to
RGB, resized with preserved approximate aspect ratio (multiples of 32, longest
side at most 640), scaled to [0,1], and normalized with ImageNet channel means
and standard deviations. It deliberately omits the classification center crop
so objects can be searched across the frame. A soft foreground-weighted feature
template is frozen on the first prompt, then matched by spatial cosine
correlation over scales 0.4–1.5 plus the current scale. Competing non-overlapping
matches within 0.04 score are rejected; accepted matches require score ≥0.80.
These are fixed experimental thresholds, not calibrated probabilities.

LK motion, histogram monitoring, and GrabCut refinement remain identical to
`classical`; only the recovery search changes. A learned match transfers the
initial mask shape before refinement, which still limits large pose changes.
The features are not a trained person re-identification system. The encoder is
shared across targets in an export, has no online weight updates, and never
uses later benchmark labels. Model URL, checkpoint SHA-256, dependency versions,
preprocessing, and thresholds are saved with each evaluation/export.

The paired experiment did not improve average DAVIS tracking. The option stays
explicitly experimental rather than replacing either default backend.
