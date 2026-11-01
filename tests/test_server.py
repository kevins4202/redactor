import base64
import json
import time

import cv2
import numpy as np
import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient
from redactor.server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    source = tmp_path / 'demo.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (80, 60))
    for _ in range(4):
        writer.write(np.full((60, 80, 3), 100, np.uint8))
    writer.release()
    class Model:
        metadata = {'name': 'test double'}
        def propagate(self, frame_dir, prompts, shape, cancelled=None):
            for index in range(4):
                mask = np.zeros(shape, np.uint8)
                mask[10:40, 12:32] = 255
                yield index, mask, .95
        def close(self):
            pass
    monkeypatch.setattr('redactor.pipeline.VideoSegmenter', Model)
    with TestClient(create_app(tmp_path / 'studio', source)) as client:
        yield client


def await_job(client, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.get(f'/api/exports/{job_id}').json()
        if status['status'] not in ('queued', 'running'):
            return status
        time.sleep(.05)
    pytest.fail('Processing did not finish')


def test_origins_frame_bounds_and_prompt_file_access(client):
    assert client.get('/').status_code == 200
    assert client.get('/api/videos').json()[0]['id'] == 'demo'
    assert client.get('/api/videos/demo/frames/0').headers['content-type'] == 'image/jpeg'
    assert client.get('/api/videos/demo/frames/99').status_code == 400
    assert client.post('/api/exports', headers={'Origin': 'http://unrelated.example'}, json={}).status_code == 403
    malicious = {'video_id': 'demo', 'prompts': [{'id': 'one', 'frame': 0, 'mask': '../../private.png'}]}
    assert client.post('/api/exports', json=malicious).status_code == 400


def test_old_options_are_rejected_instead_of_silently_ignored(client):
    body = {'video_id': 'demo', 'backend': 'classical', 'prompts': [{'id': 'one', 'frame': 0, 'box': [10, 8, 35, 48]}]}
    assert client.post('/api/exports', json=body).status_code == 422


def test_upload_validates_media(client, tmp_path):
    assert client.post('/api/videos', files={'file': ('bad.mp4', b'not a video', 'video/mp4')}).status_code == 400
    response = client.post('/api/videos', files={'file': ('clip.mp4', (tmp_path / 'demo.mp4').read_bytes(), 'video/mp4')})
    assert response.status_code == 200
    entry = response.json()
    assert entry['frames'] == 4 and 'path' not in entry


def test_processing_downloads_and_selection_replay(client):
    request = {'video_id': 'demo', 'prompts': [{'id': 'one', 'frame': 0, 'box': [10, 8, 35, 48]}]}
    response = client.post('/api/exports', json=request)
    assert response.status_code == 202
    job_id = response.json()['id']
    status = await_job(client, job_id)
    assert status['status'] == 'complete', status
    assert client.get(f'/api/exports/{job_id}/file/video').status_code == 200
    assert client.get(f'/api/exports/{job_id}/file/report').json()['frames'] == 4
    assert client.get(f'/api/exports/{job_id}/file/prompts').json()['prompts'] == request['prompts']


def test_uncertain_output_requires_review_before_final_download(client, monkeypatch):
    def analyze(source, output, prompts, **kwargs):
        output.with_name(output.stem + '-preview.mp4').write_bytes(b'preview')
        report = {'review_frames': [1], 'ready_for_export': False}
        output.with_suffix('.json').write_text(json.dumps(report))
        return report
    monkeypatch.setattr('redactor.server.process_video', analyze)
    body = {'video_id': 'demo', 'prompts': [{'id': 'one', 'frame': 0, 'box': [10, 8, 35, 48]}]}
    job_id = client.post('/api/exports', json=body).json()['id']
    assert await_job(client, job_id)['status'] == 'review'
    assert client.get(f'/api/exports/{job_id}/file/video').status_code == 409
    assert client.get(f'/api/exports/{job_id}/file/preview').content == b'preview'
    assert client.post(f'/api/exports/{job_id}/approve').status_code == 200
    assert client.get(f'/api/exports/{job_id}/file/video').content == b'preview'
    report = client.get(f'/api/exports/{job_id}/file/report').json()
    assert report['user_reviewed_frames'] == [1] and report['ready_for_export']


def test_selection_preview_and_lock_release_on_failure(client, monkeypatch):
    mask = np.zeros((60, 80), np.uint8)
    mask[10:40, 12:32] = 255
    monkeypatch.setattr('redactor.preview.preview_mask', lambda *args: mask)
    body = {'video_id': 'demo', 'frame': 2, 'prompts': [{'id': 'one', 'frame': 2, 'box': [10, 8, 35, 48]}]}
    response = client.post('/api/previews', json=body)
    assert response.status_code == 200
    overlay = cv2.imdecode(np.frombuffer(base64.b64decode(response.json()['mask'].split(',')[1]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(overlay[:, :, 3] == 220, mask > 0)
    body['frame'] = 1
    assert client.post('/api/previews', json=body).status_code == 400
    def fail(*args):
        raise RuntimeError('test failure')
    monkeypatch.setattr('redactor.preview.preview_mask', fail)
    body['frame'] = 2
    assert client.post('/api/previews', json=body).status_code == 400
    assert client.post('/api/exports', json={'video_id': 'demo', 'prompts': body['prompts']}).status_code == 202
