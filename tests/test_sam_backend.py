import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from redactor.sam_backend import FrameStore, VideoSegmenter, prompt_box, propagate_shots
from redactor.types import TargetPrompt, mask_box
from redactor.datasets import synthetic_sequence
from redactor.metrics import frame_metrics


def test_legacy_outline_uses_the_same_box_prompt_as_the_new_ui():
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)
    box = prompt_box(TargetPrompt('one', 0, mask=mask), mask.shape)
    assert tuple(box) == (30, 30, 71, 71)


@pytest.fixture(scope='module')
def model():
    if os.environ.get('REDACTOR_MODEL_TESTS') != '1':
        pytest.skip('Set REDACTOR_MODEL_TESTS=1 to run the real model')
    segmenter = VideoSegmenter()
    yield segmenter
    segmenter.close()


def frames_to_disk(frames, directory):
    for index, frame in enumerate(frames):
        assert cv2.imwrite(str(directory / f'{index:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])


def test_shots_never_share_tracking_memory(tmp_path):
    frames_to_disk([np.full((20, 30, 3), i * 20, np.uint8) for i in range(6)], tmp_path)
    class RecordingModel:
        def __init__(self):
            self.calls = []
        def propagate(self, directory, prompts, shape, cancelled=None):
            paths = sorted(Path(directory).glob('*.jpg'))
            self.calls.append(([p.frame_index for p in prompts], len(paths)))
            for index in range(len(paths)):
                yield index, np.full(shape, 255, np.uint8), .95
    model = RecordingModel()
    prompt = TargetPrompt('one', 4, box=(1, 1, 5, 5))
    results = list(propagate_shots(model, tmp_path, [prompt], (20, 30), 6, [3]))
    assert model.calls == [([1], 3)]
    assert [i for i, _, _ in results] == list(range(6))
    assert all(not mask.any() for _, mask, _ in results[:3])
    assert all(mask.any() for _, mask, _ in results[3:])


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("REDACTOR_MODEL_TESTS") != "1", reason="Requires the installed model source")
def test_lazy_frames_match_official_preprocessing(tmp_path):
    import torch
    from sam2.utils.misc import load_video_frames
    frames = synthetic_sequence('translation', count=4).frames
    frames_to_disk(frames, tmp_path)
    official, _, _ = load_video_frames(str(tmp_path), image_size=64, offload_video_to_cpu=True)
    lazy = FrameStore(tmp_path, 64)
    for index in [0, 1, 3, 0, 2]:
        assert torch.equal(lazy[index], official[index])
    assert len(lazy.cache) == 2


@pytest.mark.model
def test_actual_model_tracks_translation_in_both_directions(model, tmp_path):
    sequence = synthetic_sequence('translation', count=30)
    frames = sequence.frames[:8]
    frames_to_disk(frames, tmp_path)
    prompt = TargetPrompt('one', 4, box=mask_box(sequence.masks[4]))
    results = list(model.propagate(tmp_path, [prompt], frames[0].shape[:2]))
    assert sorted(index for index, _, _ in results) == list(range(8))
    assert len(results) == 8
    for index, mask, _ in results:
        assert frame_metrics(mask, sequence.masks[index])['iou'] > 0.70


@pytest.mark.model
def test_actual_model_handles_scale_with_later_correction(model, tmp_path):
    sequence = synthetic_sequence('scale', count=24)
    frames_to_disk(sequence.frames, tmp_path)
    prompts = [TargetPrompt('one', index, box=mask_box(sequence.masks[index])) for index in [0, 16]]
    results = list(model.propagate(tmp_path, prompts, sequence.frames[0].shape[:2]))
    assert len(results) == len(sequence.frames)
    for index, mask, _ in results:
        assert frame_metrics(mask, sequence.masks[index])['iou'] > 0.65
