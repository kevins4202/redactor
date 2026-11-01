"""Optional SAM 2.1 segmentation and video propagation through its public API."""

from contextlib import nullcontext
import hashlib
from pathlib import Path

import cv2
import numpy as np

from .tracking import ClassicalTracker, scene_difference
from .types import TrackResult, mask_box


DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_CHECKPOINT = "models/sam2.1_hiera_tiny.pt"
CONFIDENT = 0.75  # Mean foreground probability needed to call a mask visible.
RECOVER_AFTER = 2  # Frames without a confident mask before re-detection runs.


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
    def __init__(self, frame_dir, checkpoint=DEFAULT_CHECKPOINT, config=DEFAULT_CONFIG, device="auto", recover=True):
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
        # Recovery prompts a frame SAM 2 already tracked. Without this, that
        # correction is stored as ordinary memory and overwritten on re-propagation.
        self.predictor.add_all_frames_to_correct_as_cond = True
        with Path(checkpoint).open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        self.metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": checksum,
                         "config": config, "device": self.device, "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name() if self.device == "cuda" else None,
                         "postprocessing": False, "frame_input": "quality-100 JPEG",
                         "recovery": f"classical template re-detection after {RECOVER_AFTER} missing frames, "
                                     "re-prompted as a mask" if recover else None}
        self.context = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        with torch.inference_mode(), self.context():
            self.state = self.predictor.init_state(str(frame_dir), offload_video_to_cpu=True,
                                                   offload_state_to_cpu=True)
        self.iterator = None
        self.last_index = -1
        self.previous = None
        self.review_ids = set()
        self.trackers = {}  # Last results, also used by detector association.
        self.recover = recover
        self.references = {}  # Frozen appearance templates for re-detection.
        self.reference_pending = set()
        self.missing = {}

    def step(self, frame, prompts=()):
        index = self.last_index + 1
        if self.previous is not None and scene_difference(self.previous, frame) > 0.60:
            self.review_ids.update(self.trackers)
        self.previous = frame.copy()
        with self.torch.inference_mode(), self.context():
            if prompts:
                self.restart_propagation()
                for prompt in prompts:
                    if prompt.frame_index != index:
                        raise ValueError("SAM 2 prompts must be applied on their specified frame")
                    if prompt.action == "end":
                        self.predictor.remove_object(self.state, prompt.instance_id, strict=True, need_output=False)
                        self.trackers.pop(prompt.instance_id, None)
                        self.review_ids.discard(prompt.instance_id)
                        self.references.pop(prompt.instance_id, None)
                        self.missing.pop(prompt.instance_id, None)
                        self.reference_pending.discard(prompt.instance_id)
                        continue
                    box, points, labels = prompt_geometry(prompt, frame.shape[:2])
                    self.predictor.add_new_points_or_box(self.state, index, prompt.instance_id,
                                                         box=box, points=points, labels=labels)
                    self.review_ids.discard(prompt.instance_id)
                    # A user prompt replaces the re-detection template once SAM 2 is confident.
                    self.reference_pending.add(prompt.instance_id)
                    self.missing[prompt.instance_id] = 0
            self.last_index = index
            if not self.state["obj_ids"]:
                return []
            outputs = self.propagate(index)
            recovered = self.recover_missing(frame, index, outputs) if self.recover else set()
            if recovered:
                self.restart_propagation()
                outputs = self.propagate(index)
            results = []
            for instance_id, (mask, confidence) in outputs.items():
                visible = confidence >= CONFIDENT
                if visible:
                    self.missing[instance_id] = 0
                    self.remember(instance_id, frame, mask)
                else:
                    self.missing[instance_id] = self.missing.get(instance_id, 0) + 1
                if instance_id in self.review_ids:
                    state, reason = "uncertain", "scene cut requires a new prompt"
                elif not visible:
                    state, reason = "uncertain", "no confident visible mask"
                elif instance_id in recovered:
                    state, reason = "recovered", "re-detected by appearance template after loss"
                else:
                    state, reason = "tracking", ""
                result = TrackResult(instance_id, mask.astype(np.uint8) * 255, confidence, state, reason)
                results.append(result)
                self.trackers[instance_id] = result
            return results

    def restart_propagation(self):
        if self.iterator is not None:
            self.iterator.close()
            self.iterator = None

    def propagate(self, index):
        if self.iterator is None:
            self.iterator = self.predictor.propagate_in_video(self.state, start_frame_idx=index)
        out_index, ids, logits = next(self.iterator)
        if out_index != index:
            raise RuntimeError(f"SAM 2 frame mismatch: expected {index}, received {out_index}")
        masks = (logits > 0).to("cpu").numpy()[:, 0]
        outputs = {}
        for offset, instance_id in enumerate(ids):
            mask = masks[offset]
            confidence = 0.0
            if np.any(mask):
                positive = logits[offset, 0][logits[offset, 0] > 0]
                confidence = float(positive.sigmoid().mean().item())
            outputs[instance_id] = (mask, confidence)
        return outputs

    def remember(self, instance_id, frame, mask):
        mask = mask.astype(np.uint8) * 255
        if instance_id in self.reference_pending:
            reference = self.references.get(instance_id) or ClassicalTracker()
            reference.set_reference(frame, mask)
            self.references[instance_id] = reference
            self.reference_pending.discard(instance_id)
        elif instance_id in self.references:
            # Only the size estimate follows the target; the template stays frozen.
            self.references[instance_id].mask = mask

    def recover_missing(self, frame, index, outputs):
        """Re-find targets SAM 2 has dropped and give them back as mask prompts.

        During a long occlusion SAM 2's recent memory fills with empty frames,
        so it often never re-acquires the target. A classical template search
        supplies a new conditioning mask, which SAM 2 keeps attending to.
        """
        recovered = set()
        for instance_id, (_, confidence) in outputs.items():
            reference = self.references.get(instance_id)
            if reference is None or confidence >= CONFIDENT:
                continue
            if self.missing.get(instance_id, 0) + 1 < RECOVER_AFTER:
                continue
            candidate, _ = reference.redetect(frame)
            if candidate is None or not np.any(candidate):
                continue
            self.predictor.add_new_mask(self.state, index, instance_id, candidate > 0)
            recovered.add(instance_id)
        return recovered

    def close(self):
        if self.iterator is not None:
            self.iterator.close()
        self.predictor.reset_state(self.state)
        self.state.clear()
        self.trackers.clear()
        self.references.clear()
        del self.predictor
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def extract_frames(input_path, destination):
    from .video import VideoReader
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with VideoReader(input_path) as reader:
        for index, frame in reader:
            if not cv2.imwrite(str(destination / f"{index:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise OSError(f"Could not prepare SAM 2 frame {index}")
