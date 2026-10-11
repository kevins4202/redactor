"""Optional frozen ImageNet features for classical recovery experiments.

The encoder is not trained on this project's clips. Dense cosine matching keeps
the initial target's spatial layout; it is not a person re-identification model.
"""

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .tracking import ClassicalTracker
from .types import mask_box


class AppearanceEncoder:
    def __init__(self, device="auto", size=640):
        try:
            import torch
            import torch.nn.functional as functional
            import torchvision
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise RuntimeError("Learned recovery requires torch and torchvision; see docs/models.md") from error
        self.torch, self.functional = torch, functional
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights, progress=False).eval().to(self.device)
        self.model = torch.nn.Sequential(*list(model.children())[:7])  # layer3, stride 16
        self.model.requires_grad_(False)
        self.size = size
        self.mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        with checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        self.metadata = {"architecture": "resnet18.layer3", "weights": str(weights), "url": weights.url,
                         "checkpoint_sha256": checksum, "torch": torch.__version__, "torchvision": torchvision.__version__,
                         "device": self.device, "max_side": size, "preprocessing": "RGB, ImageNet normalization, aspect-preserving resize to multiples of 32; no center crop",
                         "recovery_threshold": .80, "ambiguity_margin": .04, "training": "frozen ImageNet weights; no project fine-tuning"}
        self.last_frame, self.last_features = None, None

    def features(self, frame):
        # Multi-instance processing can reuse the SAME immutable source array.
        if frame is self.last_frame:
            return self.last_features
        height, width = frame.shape[:2]
        scale = min(1., self.size / max(height, width))
        size = (max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32))
        rgb = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            tensor = self.torch.from_numpy(rgb).to(self.device, dtype=self.torch.float32).permute(2, 0, 1)[None] / 255
            features = self.functional.normalize(self.model((tensor - self.mean) / self.std), dim=1)
        self.last_frame, self.last_features = frame, features
        return features

    def close(self):
        self.last_frame = self.last_features = None
        del self.model
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


class LearnedRecoveryTracker(ClassicalTracker):
    """Identical motion/segmentation to classical; replace only recovery evidence."""

    def __init__(self, encoder, config=None):
        super().__init__(config)
        self.encoder = encoder
        self.metadata = encoder.metadata

    def initialize(self, frame, prompt):
        result = super().initialize(frame, prompt)
        features = self.encoder.features(frame)
        height, width = features.shape[-2:]
        weights = cv2.resize(self.mask.astype(np.float32) / 255, (width, height), interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = mask_box((weights > .1).astype(np.uint8))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Target is too small for learned recovery; use SAM 2 or classical tracking")
        self.feature_template = features[:, :, y1:y2, x1:x2].clone()
        self.feature_weights = self.encoder.torch.as_tensor(weights[y1:y2, x1:x2], device=self.encoder.device)[None, None]
        # Preserve exact image-to-feature placement, including empty border cells.
        self.feature_mask = self.mask[round(y1 * frame.shape[0] / height):round(y2 * frame.shape[0] / height),
                                      round(x1 * frame.shape[1] / width):round(x2 * frame.shape[1] / width)].copy()
        return result

    def redetect(self, frame):
        torch, functional = self.encoder.torch, self.encoder.functional
        features = self.encoder.features(frame)
        height, width = features.shape[-2:]
        candidates = []
        with torch.inference_mode():
            current_scale = np.sqrt(max(1, np.count_nonzero(self.mask)) / self.initial_area)
            sizes = {(max(2, round(self.feature_template.shape[-2] * scale)),
                      max(2, round(self.feature_template.shape[-1] * scale)))
                     for scale in [*np.linspace(.4, 1.5, 12), current_scale, 1.]}
            for h, w in sorted(sizes):
                if h > height or w > width:
                    continue
                template = functional.normalize(functional.interpolate(self.feature_template, size=(h, w), mode="bilinear", align_corners=False), dim=1)
                weights = functional.interpolate(self.feature_weights, size=(h, w), mode="bilinear", align_corners=False)
                response = functional.conv2d(features, template * weights / weights.sum().clamp_min(1e-6))[0, 0]
                for _ in range(2):
                    index = int(response.argmax().item())
                    y, x = divmod(index, response.shape[1])
                    score = float(response[y, x].item())
                    candidates.append((score, x, y, w, h))
                    response[max(0, y-h//2):y+h//2+1, max(0, x-w//2):x+w//2+1] = -1
        if not candidates:
            return None, 0.
        candidates.sort(reverse=True)
        score, x, y, w, h = candidates[0]
        if score < self.metadata["recovery_threshold"]:
            return None, score
        for other, ox, oy, ow, oh in candidates[1:]:
            # Treat overlapping scales of the same location as one hypothesis.
            overlap = max(0, min(x+w, ox+ow)-max(x, ox)) * max(0, min(y+h, oy+oh)-max(y, oy))
            if overlap / max(1, min(w*h, ow*oh)) < .25 and other > score - self.metadata["ambiguity_margin"]:
                return None, score
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = round(x*fw/width), round(y*fh/height), round((x+w)*fw/width), round((y+h)*fh/height)
        recovered = np.zeros((fh, fw), np.uint8)
        recovered[y1:y2, x1:x2] = cv2.resize(self.feature_mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        return self.refine(frame, recovered), score
