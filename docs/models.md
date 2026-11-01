# Tracking model

The runtime uses one checkpoint: **SAM 2.1 Hiera Base+**.

- Source: https://github.com/facebookresearch/sam2
- Pinned revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Configuration: `configs/sam2.1/sam2.1_hiera_b+.yaml`
- Checkpoint: `models/sam2.1_hiera_base_plus.pt`
- SHA-256: `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`

Install the dependencies in the root README, then run
`python scripts/setup_models.py`. The setup script checks the downloaded weights
and installs the pinned source without its optional CUDA extension. Inference
uses CUDA or Apple Silicon GPU acceleration when available, otherwise CPU.
The application pins PyTorch 2.7.1 and TorchVision 0.22.1. A newer PyTorch 2.15.0
build returned empty masks on the local Mac GPU after initialization; the pinned
build matched CPU foreground-pixel counts on a three-frame SAM 2.1 Large diagnostic. Model provenance is included in each
processing report. No Grounding DINO or separate appearance model is installed.

Masks come from temporal video segmentation, not from warping an initial mask.
The source frames are supplied as quality-100 JPEGs. All user selections and
corrections are added before propagation. Each shot has an independent inference
state. A first selection on a later frame propagates forward and backward within
its shot; no model result is reused across a detected cut.

Mask stability compares positive regions at two logit thresholds. It is a
boundary heuristic, not an identity probability or a guarantee of full coverage.
Empty masks remain empty in the preview and trigger review rather than blackout.
