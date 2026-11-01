"""One SAM 2.1 Base+ model for selection and offline video propagation."""

from contextlib import nullcontext
import hashlib
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .types import mask_box


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_CHECKPOINT = ROOT / "models/sam2.1_hiera_base_plus.pt"
SAM_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


def prompt_box(prompt, shape):
    selection = prompt.selection(shape)
    return np.asarray(prompt.box if prompt.box is not None else mask_box(selection), np.float32)


def check_cancelled(cancelled):
    if cancelled and cancelled():
        raise RuntimeError("Processing cancelled")


class FrameStore:
    """Keep at most two normalized frames resident instead of the entire clip."""

    def __init__(self, directory, image_size):
        import torch
        self.paths = sorted(Path(directory).glob('*.jpg'))
        self.image_size = image_size
        self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([.229, .224, .225], dtype=torch.float32)[:, None, None]
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from sam2.utils.misc import _load_img_as_tensor
        if index not in self.cache:
            image, _, _ = _load_img_as_tensor(str(self.paths[index]), self.image_size)
            self.cache[index] = (image.float() - self.mean) / self.std
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


class VideoSegmenter:
    def __init__(self):
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as error:
            raise RuntimeError("Install the tracking model with python scripts/setup_models.py first.") from error
        if not DEFAULT_CHECKPOINT.is_file():
            raise RuntimeError("Model weights are missing. Run python scripts/setup_models.py first.")
        if torch.__version__.split("+")[0] != "2.7.1":
            raise RuntimeError("Use the tested model runtime: python -m pip install -r requirements-lock.txt")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.predictor = build_sam2_video_predictor(
            DEFAULT_CONFIG, str(DEFAULT_CHECKPOINT), device=self.device,
            apply_postprocessing=False,
            hydra_overrides_extra=[
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            ],
        )
        with DEFAULT_CHECKPOINT.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        self.metadata = {"name": "SAM 2.1 Base+", "config": DEFAULT_CONFIG,
                         "checkpoint_sha256": checksum, "source_revision": SAM_REVISION,
                         "device": self.device, "torch": torch.__version__}

    def context(self):
        return self.torch.autocast("cuda", dtype=self.torch.bfloat16) if self.device == "cuda" else nullcontext()

    def propagate(self, frame_dir, prompts, shape, cancelled=None):
        """Yield each frame once, with all corrections available before propagation."""
        state = None
        with self.torch.inference_mode(), self.context():
            try:
                check_cancelled(cancelled)
                images = FrameStore(frame_dir, self.predictor.image_size)
                with tempfile.TemporaryDirectory(prefix='redactor-initialize-') as directory:
                    shutil.copyfile(images.paths[0], Path(directory) / '000000.jpg')
                    state = self.predictor.init_state(directory, offload_video_to_cpu=True,
                                                       offload_state_to_cpu=True)
                state['images'] = images
                state['num_frames'] = len(images)
                for prompt in prompts:
                    check_cancelled(cancelled)
                    box = prompt_box(prompt, shape)
                    self.predictor.add_new_points_or_box(state, prompt.frame_index, 1,
                                                         box=box)
                start = min(prompt.frame_index for prompt in prompts)
                for reverse in ([False, True] if start > 0 else [False]):
                    iterator = self.predictor.propagate_in_video(state, start_frame_idx=start, reverse=reverse)
                    try:
                        for index, ids, logits in iterator:
                            check_cancelled(cancelled)
                            if reverse and index == start:
                                continue
                            scores = logits[ids.index(1), 0].float().cpu().numpy()
                            if not np.isfinite(scores).all():
                                raise RuntimeError("The tracking model returned invalid values; no final video was published")
                            mask = (scores > 0).astype(np.uint8) * 255
                            outer = np.count_nonzero(scores > -1)
                            stability = np.count_nonzero(scores > 1) / max(1, outer)
                            yield index, mask, float(stability)
                    finally:
                        iterator.close()
            finally:
                if state is not None:
                    self.predictor.reset_state(state)
                    state.clear()

    def close(self):
        del self.predictor
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        elif self.device == "mps":
            self.torch.mps.empty_cache()
