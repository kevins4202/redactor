import cv2
import numpy as np
import pytest

from redactor.reconstruction import BackgroundReconstructor


def test_hidden_pixels_never_used_and_outside_unchanged():
    rng = np.random.default_rng(10)
    background = rng.integers(0, 256, (96, 144, 3), dtype=np.uint8)
    frames, masks = [], []
    for x in (10, 50, 90, 110):
        mask = np.zeros(background.shape[:2], np.uint8)
        mask[30:60, x:x+15] = 255
        frame = background.copy()
        frame[mask > 0] = (0, 0, 255)
        frames.append(frame)
        masks.append(mask)
    model = BackgroundReconstructor(frames, masks)
    result = model.reconstruct(1, "unaligned")
    assert np.array_equal(result.image, background)
    assert result.observed[masks[1] > 0].all()
    assert np.array_equal(result.image[masks[1] == 0], frames[1][masks[1] == 0])


def test_permanent_hole_is_not_claimed_observed():
    frame = np.full((64, 96, 3), 70, np.uint8)
    mask = np.zeros((64, 96), np.uint8)
    mask[20:40, 30:50] = 255
    frame[mask > 0] = 200
    result = BackgroundReconstructor([frame]*4, [mask]*4).reconstruct(0, "unaligned")
    assert not result.observed.any()
    assert not result.source_count.any()
    assert np.array_equal(result.image[mask == 0], frame[mask == 0])


def test_alignment_recovers_known_camera_translation():
    rng = np.random.default_rng(11)
    background = rng.integers(0, 256, (160, 240, 3), dtype=np.uint8)
    matrix = np.float32([[1, 0, 8], [0, 1, 4]])
    moved = cv2.warpAffine(background, matrix, (240, 160))
    masks = [np.zeros((160, 240), np.uint8)]*2
    model = BackgroundReconstructor([background, moved], masks)
    estimate = model.alignment(0, 1)
    assert estimate is not None
    points = np.float32([[[50, 50], [100, 100], [180, 70]]])
    predicted = cv2.perspectiveTransform(points, estimate)
    assert np.max(np.abs(predicted-(points+[8, 4]))) < 1.5


def test_validation():
    with pytest.raises(ValueError):
        BackgroundReconstructor([], [])


def test_removal_cli_reuses_masks_without_loading_models(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    import sys
    from redactor.video import VideoReader, VideoWriter

    source = tmp_path/'source.mp4'
    masks = tmp_path/'masks'
    masks.mkdir()
    with VideoWriter(source, tmp_path/'unused.mp4', 3, (96, 64), audio=False) as writer:
        for t in range(3):
            frame = np.full((64, 96, 3), 80, np.uint8)
            mask = np.zeros((64, 96), np.uint8)
            mask[20:40, 10+t*25:25+t*25] = 255
            frame[mask > 0] = (0, 0, 200)
            writer.write(frame)
            cv2.imwrite(str(masks/f'{t:06d}.png'), mask)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path/'removed.mp4'
    script = Path(__file__).resolve().parents[1]/'scripts/remove_object.py'
    subprocess.run([sys.executable, str(script), str(source), '--masks', str(masks), '-o', str(output)], check=True, capture_output=True)
    report = json.loads(output.with_suffix('.json').read_text())
    assert report['complete'] and report['frames'] == 3
    assert report['tracking_model'] is None
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with VideoReader(output) as reader:
        assert len(list(reader)) == 3
    collision = subprocess.run([sys.executable, str(script), str(source), '--masks', str(masks), '-o', str(source), '--overwrite'], capture_output=True)
    assert collision.returncode != 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
