import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from redactor.pipeline import process_video, publish_preview, review_records, validate_prompts
from redactor.render import redact_frame
from redactor.types import TargetPrompt
from redactor.video import VideoWriter


@pytest.fixture
def clip(tmp_path):
    source = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 12, (80, 60))
    for _ in range(4):
        writer.write(np.full((60, 80, 3), 100, np.uint8))
    writer.release()
    return source


def install_model(monkeypatch, masks):
    class Model:
        metadata = {'name': 'test double, not accuracy evidence'}
        def propagate(self, frame_dir, prompts, shape, cancelled=None):
            for index, mask in enumerate(masks):
                yield index, mask, 0.95 if mask.any() else 0.0
        def close(self):
            pass
    monkeypatch.setattr('redactor.pipeline.VideoSegmenter', Model)


def test_empty_prediction_never_blacks_out_frame_or_publishes_final(clip, tmp_path, monkeypatch):
    mask = np.zeros((60, 80), np.uint8)
    mask[10:40, 20:35] = 255
    install_model(monkeypatch, [mask, mask, np.zeros_like(mask), mask])
    output = tmp_path / 'result.mp4'
    report = process_video(clip, output, [TargetPrompt('one', 0, box=(18, 8, 37, 42))], audio=False)
    assert report['review_frames'] == [2]
    assert not output.exists()
    preview = tmp_path / 'result-preview.mp4'
    capture = cv2.VideoCapture(str(preview))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 4
    capture.set(cv2.CAP_PROP_POS_FRAMES, 2)
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.mean() > 80
    assert (tmp_path / 'result.masks/000002.png').is_file()
    publish_preview(output)
    assert output.read_bytes() == preview.read_bytes()


def test_complete_analysis_exports_video_and_replay(clip, tmp_path, monkeypatch):
    mask = np.zeros((60, 80), np.uint8)
    mask[10:40, 20:35] = 255
    install_model(monkeypatch, [mask] * 4)
    output = tmp_path / 'result.mp4'
    phases = []
    report = process_video(clip, output, [TargetPrompt('one', 2, box=(18, 8, 37, 42))],
                           audio=False, progress=lambda stage, done, total: phases.append(stage))
    assert report['frames'] == 4 and report['ready_for_export']
    assert set(phases) == {'prepare', 'track', 'render'}
    assert output.exists() and output.with_suffix('.prompts.json').exists()
    assert json.loads(output.with_suffix('.json').read_text())['review_frames'] == []
    capture = cv2.VideoCapture(str(output))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame[20, 25].max() < 10 and frame[20, 70].min() > 80
    with pytest.raises(FileExistsError):
        process_video(clip, output, [], audio=False)


def test_incomplete_analysis_preserves_existing_output(clip, tmp_path, monkeypatch):
    output = tmp_path / 'result.mp4'
    output.write_bytes(b'original')
    install_model(monkeypatch, [np.zeros((60, 80), np.uint8)])
    with pytest.raises(RuntimeError, match='every frame'):
        process_video(clip, output, [TargetPrompt('one', 0, box=(10, 10, 30, 40))], overwrite=True, audio=False)
    assert output.read_bytes() == b'original'
    assert not list(tmp_path.glob('.redactor-*'))


def test_cancel_before_model_load_preserves_existing_output(clip, tmp_path):
    output = tmp_path / 'result.mp4'
    output.write_bytes(b'original')
    with pytest.raises(RuntimeError, match='cancelled'):
        process_video(clip, output, [TargetPrompt('one', 0, box=(10, 10, 30, 40))],
                      overwrite=True, audio=False, cancelled=lambda: True)
    assert output.read_bytes() == b'original'


def test_one_target_and_valid_corrections_are_required():
    with pytest.raises(ValueError, match='one target'):
        validate_prompts([TargetPrompt('a', 0, box=(1, 1, 5, 5)), TargetPrompt('b', 2, box=(1, 1, 5, 5))], (20, 20), 4)
    with pytest.raises(ValueError, match='outside'):
        validate_prompts([TargetPrompt('a', 4, box=(1, 1, 5, 5))], (20, 20), 4)
    with pytest.raises(ValueError, match='one selection per frame'):
        validate_prompts([TargetPrompt('a', 0, box=(1, 1, 5, 5))] * 2, (20, 20), 4)


def test_preview_and_mask_artifacts_cannot_overwrite_source(clip, tmp_path):
    prompt = TargetPrompt('one', 0, box=(10, 10, 30, 40))
    preview_source = tmp_path / 'result-preview.mp4'
    shutil.copyfile(clip, preview_source)
    original = preview_source.read_bytes()
    with pytest.raises(ValueError, match='overwrite the input'):
        process_video(preview_source, tmp_path / 'result.mp4', [prompt], overwrite=True, audio=False)
    assert preview_source.read_bytes() == original
    inside_masks = tmp_path / 'result.masks/input.mp4'
    inside_masks.parent.mkdir()
    shutil.copyfile(clip, inside_masks)
    with pytest.raises(ValueError, match='overwrite the input'):
        process_video(inside_masks, tmp_path / 'result.mp4', [prompt], overwrite=True, audio=False)
    assert inside_masks.read_bytes() == original


def test_new_shot_requires_an_anchor_but_does_not_flag_anchored_shots():
    rows = [{'frame': i, 'area': 50, 'frame_area': 400, 'stability': .95} for i in range(6)]
    prompts = [TargetPrompt('a', 0, box=(1, 1, 5, 5))]
    assert [r['frame'] for r in review_records(rows, prompts, cuts=[3]) if r['review_reasons']] == [3, 4, 5]
    prompts.append(TargetPrompt('a', 4, box=(1, 1, 5, 5)))
    assert not any(r['review_reasons'] for r in review_records(rows, prompts, cuts=[3]))


def test_redaction_changes_only_expanded_mask_region():
    frame = np.full((100, 100, 3), 120, np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[20:40, 20:40] = 255
    result = redact_frame(frame, mask)
    assert result[25, 25].max() == 0
    assert np.array_equal(result[80, 80], frame[80, 80])
    assert np.array_equal(redact_frame(frame, np.zeros_like(mask)), frame)


def test_failed_encoding_does_not_replace_existing_output(tmp_path):
    output = tmp_path / 'existing.mp4'
    output.write_bytes(b'keep this')
    with pytest.raises(RuntimeError, match='failure'):
        with VideoWriter(output, tmp_path / 'input.mp4', 30, (80, 60), overwrite=True, audio=False) as writer:
            writer.write(np.zeros((60, 80, 3), np.uint8))
            raise RuntimeError('failure')
    assert output.read_bytes() == b'keep this'
    assert not list(tmp_path.glob('.redactor-*'))


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg is not installed')
def test_audio_preservation_and_source_protection(tmp_path):
    source = tmp_path / 'with-audio.mp4'
    subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=80x60:r=10:d=0.4',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.4', '-c:v', 'libx264', '-c:a', 'aac',
                    '-shortest', str(source)], check=True, capture_output=True)
    output = tmp_path / 'redacted.mp4'
    with VideoWriter(output, source, 10, (80, 60), audio=True) as writer:
        for _ in range(4):
            writer.write(np.zeros((60, 80, 3), np.uint8))
    info = subprocess.run([shutil.which('ffprobe'), '-v', 'error', '-show_streams', '-of', 'json', str(output)],
                          check=True, capture_output=True, text=True)
    assert {stream['codec_type'] for stream in json.loads(info.stdout)['streams']} == {'audio', 'video'}
    with pytest.raises(ValueError, match='must differ'):
        VideoWriter(source, source, 10, (80, 60), overwrite=True)
