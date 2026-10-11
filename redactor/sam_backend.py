"""Optional SAM 2.1 segmentation and video propagation through its public API."""

from contextlib import nullcontext
import hashlib
from pathlib import Path

import cv2
import numpy as np

from .tracking import scene_difference
from .types import TrackResult, mask_box


DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_CHECKPOINT = "models/sam2.1_hiera_tiny.pt"


def prompt_geometry(prompt, shape):
    selection = prompt.selection(shape)
    box = np.asarray(prompt.box if prompt.box is not None else mask_box(selection), np.float32)
    if prompt.mask is None:
        return box, None, None
    distance = cv2.distanceTransform(selection, cv2.DIST_L2, 5)
    _, maximum, _, location = cv2.minMaxLoc(distance)
    if maximum <= 0:
        raise ValueError("Prompt mask is empty")
    points, labels = [location], [1]
    x1, y1, x2, y2 = box.astype(int)
    # Negative points in the coarse mask's holes help separate nearby people.
    for y in np.linspace(y1 + 1, y2 - 2, 5).astype(int):
        for x in np.linspace(x1 + 1, x2 - 2, 5).astype(int):
            if 0 <= y < shape[0] and 0 <= x < shape[1] and selection[y, x] == 0:
                points.append((x, y))
                labels.append(0)
    return box, np.asarray(points, np.float32), np.asarray(labels, np.int32)


class Sam2Session:
    def __init__(self, frame_dir, checkpoint=DEFAULT_CHECKPOINT, config=DEFAULT_CONFIG, device="auto"):
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as error:
            raise RuntimeError("SAM 2 is not installed. Follow docs/models.md to install the optional backend.") from error
        if not Path(checkpoint).is_file():
            raise ValueError(f"SAM 2 checkpoint not found: {checkpoint}. Run python scripts/setup_models.py.")
        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.predictor = build_sam2_video_predictor(config, str(checkpoint), device=self.device,
                                                   apply_postprocessing=False)
        with Path(checkpoint).open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        self.metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": checksum,
                         "config": config, "device": self.device, "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name() if self.device == "cuda" else None,
                         "postprocessing": False, "frame_input": "quality-100 JPEG"}
        self.context = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        with torch.inference_mode(), self.context():
            self.state = self.predictor.init_state(str(frame_dir), offload_video_to_cpu=True,
                                                   offload_state_to_cpu=True)
        self.iterator = None
        self.last_index = -1
        self.previous = None
        self.review_ids = set()
        self.trackers = {}  # Last results, also used by detector association.

    def step(self, frame, prompts=()):
        index = self.last_index + 1
        if self.previous is not None and scene_difference(self.previous, frame) > 0.60:
            self.review_ids.update(self.trackers)
        self.previous = frame.copy()
        with self.torch.inference_mode(), self.context():
            if prompts:
                if self.iterator is not None:
                    self.iterator.close()
                    self.iterator = None
                for prompt in prompts:
                    if prompt.frame_index != index:
                        raise ValueError("SAM 2 prompts must be applied on their specified frame")
                    box, points, labels = prompt_geometry(prompt, frame.shape[:2])
                    self.predictor.add_new_points_or_box(self.state, index, prompt.instance_id,
                                                         box=box, points=points, labels=labels)
                    self.review_ids.discard(prompt.instance_id)
            self.last_index = index
            if not self.state["obj_ids"]:
                return []
            if self.iterator is None:
                self.iterator = self.predictor.propagate_in_video(self.state, start_frame_idx=index)
            out_index, ids, logits = next(self.iterator)
            if out_index != index:
                raise RuntimeError(f"SAM 2 frame mismatch: expected {index}, received {out_index}")
            masks = (logits > 0).to("cpu").numpy()[:, 0]
            results = []
            for offset, instance_id in enumerate(ids):
                mask = masks[offset]
                if np.any(mask):
                    positive = logits[offset, 0][logits[offset, 0] > 0]
                    confidence = float(positive.sigmoid().mean().item())
                else:
                    confidence = 0.0
                uncertain = not np.any(mask) or confidence < 0.75 or instance_id in self.review_ids
                reason = "scene cut requires a new prompt" if instance_id in self.review_ids else "no confident visible mask" if uncertain else ""
                result = TrackResult(instance_id, mask.astype(np.uint8) * 255, confidence,
                                     "uncertain" if uncertain else "tracking", reason)
                results.append(result)
                self.trackers[instance_id] = result
            return results

    def close(self):
        if self.iterator is not None:
            self.iterator.close()
        self.predictor.reset_state(self.state)
        self.state.clear()
        self.trackers.clear()


def extract_frames(input_path, destination):
    from .video import VideoReader
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with VideoReader(input_path) as reader:
        for index, frame in reader:
            if not cv2.imwrite(str(destination / f"{index:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise OSError(f"Could not prepare SAM 2 frame {index}")
