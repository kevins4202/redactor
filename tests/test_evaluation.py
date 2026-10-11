import cv2
import numpy as np
import pytest

from redactor.datasets import synthetic_sequence
from redactor.metrics import frame_metrics, summarize
from redactor.types import TargetPrompt


def test_metrics_distinguish_exposure_from_excess_blur():
    truth = np.zeros((20, 20), np.uint8)
    truth[5:15, 5:15] = 255
    empty = np.zeros_like(truth)
    exposed = frame_metrics(empty, truth, dilation=1)
    covered = frame_metrics(np.full_like(truth, 255), truth, dilation=1)
    assert exposed["leak_rate"] == 1
    assert covered["leak_rate"] == 0
    assert covered["iou"] == 0.25
    assert covered["excess_pixels"] == 300
    hidden = frame_metrics(empty, empty, dilation=1)
    summary = summarize([exposed, exposed, hidden, exposed], fps=2)
    assert summary["longest_leak_seconds"] == 1
    assert summary["visible_frames"] == 3
    assert summary["leaking_frame_fraction"] == 1


def test_synthetic_occlusion_has_true_disappearance_and_reappearance():
    sequence = synthetic_sequence("occlusion")
    areas = [cv2.countNonZero(mask) for mask in sequence.masks]
    assert areas[0] > 1000 and areas[-1] > 1000
    assert 0 in areas
    again = synthetic_sequence("occlusion")
    assert np.array_equal(sequence.frames[50], again.frames[50])


def test_prompt_clips_boxes_and_rejects_empty_selection():
    prompt = TargetPrompt("person", 0, box=(-4, -3, 10, 8))
    assert np.count_nonzero(prompt.selection((20, 20))) == 80
    with pytest.raises(ValueError):
        TargetPrompt("person", 0, box=(50, 50, 60, 60)).selection((20, 20))
    with pytest.raises(ValueError):
        TargetPrompt("person", 0, mask=np.zeros((20, 20))).selection((20, 20))
