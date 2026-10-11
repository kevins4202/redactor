import importlib.util
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from redactor.server import create_app


@pytest.fixture
def client(tmp_path):
    source = tmp_path / "demo.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60))
    for _ in range(4):
        frame = np.full((60, 80, 3), 35, np.uint8)
        cv2.rectangle(frame, (12, 10), (32, 45), (40, 90, 230), -1)
        cv2.circle(frame, (20, 20), 4, (230, 230, 230), -1)
        writer.write(frame)
    writer.release()
    with TestClient(create_app(tmp_path / "studio", source)) as client:
        yield client


def test_server_validates_origins_and_prompt_file_access(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/videos").json()[0]["id"] == "demo"
    assert client.get("/api/videos/demo/frames/0").headers["content-type"] == "image/jpeg"
    assert client.get("/api/videos/demo/frames/99").status_code == 400
    assert client.post("/api/exports", headers={"Origin": "http://unrelated.example"}, json={}).status_code == 403
    malicious = {"video_id": "demo", "prompts": [{"id": "one", "frame": 0, "mask": "../../private.png"}]}
    assert client.post("/api/exports", json=malicious).status_code == 400


def test_upload_rejects_invalid_media_and_registers_valid_video(client, tmp_path):
    assert client.post("/api/videos", files={"file": ("bad.mp4", b"not a video", "video/mp4")}).status_code == 400
    source = tmp_path / "demo.mp4"
    response = client.post("/api/videos", files={"file": ("clip.mp4", source.read_bytes(), "video/mp4")})
    assert response.status_code == 200
    entry = response.json()
    assert entry["frames"] == 4 and "path" not in entry
    assert client.get(f"/api/videos/{entry['id']}/frames/3").status_code == 200


def test_browser_export_produces_download_and_review_log(client):
    request = {"video_id": "demo", "backend": "classical", "style": "solid", "audio": False,
               "prompts": [{"id": "one", "frame": 0, "polygon": [[10, 8], [35, 8], [35, 48], [10, 48]]}]}
    response = client.post("/api/exports", json=request)
    assert response.status_code == 202
    job_id = response.json()["id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.get(f"/api/exports/{job_id}").json()
        if status["status"] in ("complete", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "complete", status
    assert client.get(f"/api/exports/{job_id}/file/video").status_code == 200
    assert client.get(f"/api/exports/{job_id}/file/report").json()["frames"] == 4
    saved = client.get(f"/api/exports/{job_id}/file/prompts").json()["prompts"][0]
    assert saved == request["prompts"][0]  # no inaccessible PNG sidecar
