import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from redactor.pipeline import MultiTracker, process_video
from redactor.prompts import load_prompts, save_prompts
from redactor.render import ReviewRequired, coverage_mask
from redactor.types import TargetPrompt, TrackResult
from redactor.video import VideoWriter


def test_uncertain_targets_require_cover_or_explicit_policy():
    mask = np.zeros((24, 32), np.uint8)
    result = TrackResult("person", mask, 0, "lost", "scene cut")
    cover, guarded = coverage_mask([result], mask.shape)
    assert guarded and np.all(cover == 255)
    with pytest.raises(ReviewRequired):
        coverage_mask([result], mask.shape, "stop")
    predicted, guarded = coverage_mask([result], mask.shape, "predict")
    assert not guarded and not np.any(predicted)


def test_prompt_roundtrip_and_midclip_correction(tmp_path):
    frame = np.full((60, 80, 3), 40, np.uint8)
    cv2.rectangle(frame, (10, 10), (30, 40), (30, 80, 220), -1)
    first = TargetPrompt("one", 0, box=(8, 8, 33, 43))
    second = TargetPrompt("two", 5, box=(8, 8, 33, 43))
    path = tmp_path / "prompts.json"
    save_prompts(path, [first, second])
    prompts = load_prompts(path, frame.shape[:2])
    pipeline = MultiTracker()
    assert len(pipeline.step(frame, [prompts[0]])) == 1
    assert len(pipeline.step(frame, [prompts[1]])) == 2
    correction = TargetPrompt("one", 6, box=(8, 8, 33, 43))
    assert len(pipeline.step(frame, [correction])) == 2
    end = TargetPrompt("one", 7, action="end")
    assert [result.instance_id for result in pipeline.step(frame, [end])] == ["two"]
    save_prompts(path, [first, second, end])
    assert load_prompts(path, frame.shape[:2])[-1].action == "end"


def test_video_export_preserves_frame_count_and_writes_replay_metadata(tmp_path):
    source = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (80, 60))
    assert writer.isOpened()
    for index in range(12):
        frame = np.full((60, 80, 3), 30, np.uint8)
        cv2.rectangle(frame, (10 + index, 10), (30 + index, 40), (40, 80, 230), -1)
        cv2.circle(frame, (20 + index, 18), 3, (240, 240, 240), -1)
        writer.write(frame)
    writer.release()
    output = tmp_path / "result.mp4"
    report = process_video(source, output, [TargetPrompt("one", 0, box=(8, 8, 33, 43))], style="solid", audio=False, margin=8)
    assert report["frames"] == 12
    assert report["margin_pixels"] == 8
    capture = cv2.VideoCapture(str(output))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 12
    assert capture.get(cv2.CAP_PROP_FPS) == 12
    ok, frame = capture.read()
    capture.release()
    assert ok and int(frame[25, 20].max()) < 10
    assert int(frame[25, 5].max()) < 10  # Within the configured eight-pixel margin.
    assert int(frame[25, 65].min()) > 15  # Distant background remains visible.
    assert len(load_prompts(output.with_suffix(".prompts.json"), (60, 80))) == 1
    assert len(json.loads(output.with_suffix(".json").read_text())["per_frame"]) == 12
    with pytest.raises(FileExistsError):
        process_video(source, output, [], audio=False)


def test_failed_export_does_not_replace_existing_output(tmp_path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep this")
    with pytest.raises(RuntimeError, match="failure"):
        with VideoWriter(output, tmp_path / "input.mp4", 30, (80, 60), overwrite=True, audio=False) as writer:
            writer.write(np.zeros((60, 80, 3), np.uint8))
            raise RuntimeError("failure")
    assert output.read_bytes() == b"keep this"
    assert not list(tmp_path.glob(".redactor-*"))


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is not installed")
def test_audio_is_preserved_and_input_cannot_be_overwritten(tmp_path):
    source = tmp_path / "with-audio.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=80x60:r=10:d=0.4",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(source)], check=True, capture_output=True)
    output = tmp_path / "redacted.mp4"
    with VideoWriter(output, source, 10, (80, 60), audio=True) as writer:
        for _ in range(4):
            writer.write(np.zeros((60, 80, 3), np.uint8))
    info = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(output)],
                          check=True, capture_output=True, text=True)
    types = {stream["codec_type"] for stream in json.loads(info.stdout)["streams"]}
    assert types == {"audio", "video"}
    with pytest.raises(ValueError, match="must differ"):
        VideoWriter(source, source, 10, (80, 60), overwrite=True)
