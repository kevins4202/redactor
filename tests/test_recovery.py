import cv2
import numpy as np

from redactor.datasets import synthetic_sequence
from redactor.metrics import frame_metrics
from redactor.tracking import ClassicalTracker
from redactor.types import TargetPrompt


def run_sequence(name):
    cv2.setRNGSeed(581)
    sequence = synthetic_sequence(name)
    tracker = ClassicalTracker()
    selection = cv2.dilate(sequence.masks[0], np.ones((11, 11), np.uint8))
    first = tracker.initialize(sequence.frames[0], TargetPrompt("person", 0, mask=selection))
    results = [first] + [tracker.update(frame) for frame in sequence.frames[1:]]
    return sequence, results


def test_scale_adaptation_preserves_coverage_without_original_sized_mask():
    sequence, results = run_sequence("scale")
    final = frame_metrics(results[-1].mask, sequence.masks[-1])
    assert final["iou"] > 0.65
    assert final["leak_rate"] < 0.03
    assert np.count_nonzero(results[-1].mask) < np.count_nonzero(results[0].mask) * 0.4


def test_recovers_after_full_occlusion():
    sequence, results = run_sequence("occlusion")
    assert any(result.state in ("uncertain", "lost") for result in results)
    assert any(result.state == "recovered" for result in results[45:])
    assert frame_metrics(results[-1].mask, sequence.masks[-1])["iou"] > 0.65


def test_recovers_after_scene_cut():
    sequence, results = run_sequence("cut")
    assert results[45].state == "recovered"
    assert frame_metrics(results[45].mask, sequence.masks[45])["leak_rate"] < 0.03
    assert frame_metrics(results[-1].mask, sequence.masks[-1])["iou"] > 0.65
