import os

import cv2
import numpy as np
import pytest

from redactor.datasets import synthetic_sequence
from redactor.metrics import frame_metrics
from redactor.types import TargetPrompt


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("REDACTOR_MODEL_TESTS") != "1", reason="Requires optional pretrained ResNet weights")
def test_learned_recovery_relocates_target_after_cut_and_keeps_frozen_anchor():
    from redactor.appearance import AppearanceEncoder, LearnedRecoveryTracker
    sequence = synthetic_sequence("cut")
    encoder = AppearanceEncoder()
    try:
        tracker = LearnedRecoveryTracker(encoder)
        selection = cv2.dilate(sequence.masks[0], np.ones((11, 11), np.uint8))
        tracker.initialize(sequence.frames[0], TargetPrompt("person", 0, mask=selection))
        anchor = tracker.feature_template.clone()
        result = tracker.update(sequence.frames[45])
        assert result.state == "recovered"
        assert frame_metrics(result.mask, sequence.masks[45])["iou"] > .65
        assert encoder.torch.equal(anchor, tracker.feature_template)
        assert len(encoder.metadata["checkpoint_sha256"]) == 64
        assert not any(parameter.requires_grad for parameter in encoder.model.parameters())
    finally:
        encoder.close()


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("REDACTOR_MODEL_TESTS") != "1", reason="Requires optional pretrained ResNet weights")
def test_learned_recovery_rejects_two_identical_candidates():
    from redactor.appearance import AppearanceEncoder, LearnedRecoveryTracker
    sequence = synthetic_sequence("translation")
    frame, mask = sequence.frames[0], sequence.masks[0]
    # Duplicate a full source frame: two equally plausible copies of the target.
    duplicated = np.concatenate([frame, frame], axis=1)
    initial = np.concatenate([frame, np.zeros_like(frame)], axis=1)
    selection = np.concatenate([cv2.dilate(mask, np.ones((11, 11), np.uint8)), np.zeros_like(mask)], axis=1)
    encoder = AppearanceEncoder()
    try:
        tracker = LearnedRecoveryTracker(encoder)
        tracker.initialize(initial, TargetPrompt("person", 0, mask=selection))
        recovered, _ = tracker.redetect(duplicated)
        assert recovered is None
    finally:
        encoder.close()
