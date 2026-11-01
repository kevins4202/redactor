import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from redactor.sam_backend import FrameStore, VideoSegmenter, prompt_box
from redactor.types import TargetPrompt, mask_box
from tracking_fixtures import synthetic_sequence
def frame_metrics(prediction, truth):
    predicted, actual = prediction > 0, truth > 0
    return {'iou': np.count_nonzero(predicted & actual) / max(1, np.count_nonzero(predicted | actual))}


def test_mask_selection_converts_to_a_tight_box():
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
