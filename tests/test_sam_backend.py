import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from redactor.sam_backend import Sam2Session, prompt_geometry
from redactor.types import TargetPrompt, mask_box
from redactor.datasets import synthetic_sequence
from redactor.metrics import frame_metrics


def test_lasso_prompt_supplies_foreground_point_inside_outline():
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)
    box, points, labels = prompt_geometry(TargetPrompt("one", 0, mask=mask), mask.shape)
    assert tuple(box) == (30, 30, 71, 71)
    assert mask[round(points[0, 1]), round(points[0, 0])] == 255
    assert labels[0] == 1
    assert 0 in labels


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("REDACTOR_MODEL_TESTS") != "1", reason="Set REDACTOR_MODEL_TESTS=1 to test installed SAM 2 weights")
def test_sam2_tracks_and_adds_midclip_instance(tmp_path):
    sequence = synthetic_sequence("translation", count=12)
    for index, frame in enumerate(sequence.frames):
        assert cv2.imwrite(str(tmp_path / f"{index:06d}.jpg"), frame)
    session = Sam2Session(tmp_path)
    try:
        for index, frame in enumerate(sequence.frames):
            prompts = []
            if index in (0, 6):
                prompts = [TargetPrompt("one" if index == 0 else "two", index, box=mask_box(sequence.masks[index]))]
            if index == 10:
                prompts = [TargetPrompt("two", index, action="end")]
            results = session.step(frame, prompts)
            assert len(results) == (2 if 6 <= index < 10 else 1)
            assert frame_metrics(results[0].mask, sequence.masks[index])["iou"] > 0.60
    finally:
        session.close()
