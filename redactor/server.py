"""A loopback-only review application. Videos stay on the local machine."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import threading
import tempfile
from urllib.parse import urlparse
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from typing import Literal

from .pipeline import process_video
from .prompts import load_prompts
from .video import VideoReader


class ExportRequest(BaseModel):
    video_id: str
    prompts: list[dict] = Field(default_factory=list)
    backend: Literal["classical", "sam2", "baseline", "learned"] = "classical"
    style: Literal["blur", "solid", "pixelate"] = "blur"
    mode: Literal["manual", "text", "faces", "plates"] = "manual"
    text: str = ""
    audio: bool = True
    margin: int = Field(default=4, ge=0, le=128, strict=True)


class PreviewRequest(ExportRequest):
    frame: int = Field(default=0, ge=0)


def make_detector(options):
    if options.mode == "manual":
        return None
    from .detection import DetectorSource, FaceDetector, GroundingDetector
    if options.mode == "faces" and importlib.util.find_spec("transformers") is None:
        return DetectorSource(FaceDetector())
    description = options.text if options.mode == "text" else "a human face" if options.mode == "faces" else "a vehicle license plate"
    if not description.strip():
        raise ValueError("Describe what you want to hide")
    required = ["face" if options.mode == "faces" else "plate"] if options.mode in ("faces", "plates") else None
    return DetectorSource(GroundingDetector(description, required_labels=required))


def create_app(storage=None, demo=None):
    storage = Path(storage or "output/studio").resolve()
    storage.mkdir(parents=True, exist_ok=True)
    videos, jobs = {}, {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="redactor-export")
    lock = threading.Lock()
    compute = threading.Lock()  # Preview and export must not load models concurrently.

    @asynccontextmanager
    async def lifespan(app):
        yield
        for job in jobs.values():
            job["cancel"].set()
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Redactor", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_requests(request, call_next):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin requests are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def register(path, name, video_id=None):
        with VideoReader(path) as reader:
            if reader.count <= 0:
                raise ValueError("The review interface requires a video with a known frame count")
            entry = {"id": video_id or uuid4().hex, "name": name, "path": str(Path(path).resolve()),
                     "width": reader.width, "height": reader.height, "frames": reader.count, "fps": reader.fps}
        videos[entry["id"]] = entry
        return {key: value for key, value in entry.items() if key != "path"}

    demo = Path(demo or Path(__file__).resolve().parents[1] / "data/crosswalk-short.mp4")
    if demo.is_file():
        register(demo, "Crosswalk · sample clip", "demo")

    def video(video_id):
        if video_id not in videos:
            raise HTTPException(404, "Video not found; upload it again if the app was restarted")
        return videos[video_id]

    def job_entry(job_id):
        if job_id not in jobs:
            raise HTTPException(404, "Export not found")
        return jobs[job_id]

    @app.get("/api/videos")
    def list_videos():
        return [{key: value for key, value in entry.items() if key != "path"} for entry in videos.values()]

    @app.get("/api/capabilities")
    def capabilities():
        return {"sam2": importlib.util.find_spec("sam2") is not None and Path("models/sam2.1_hiera_tiny.pt").is_file(),
                "learned": importlib.util.find_spec("torchvision") is not None,
                "grounding": importlib.util.find_spec("transformers") is not None}

    @app.post("/api/videos")
    async def upload(file: UploadFile):
        video_id = uuid4().hex
        directory = storage / video_id
        directory.mkdir()
        path = directory / "input.mp4"
        size = 0
        try:
            with path.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024**3:
                        raise ValueError("Please choose a video smaller than 2 GB")
                    stream.write(chunk)
            return register(path, Path(file.filename or "Video").name, video_id)
        except (ValueError, OSError) as error:
            path.unlink(missing_ok=True)
            raise HTTPException(400, str(error)) from error
        finally:
            await file.close()

    @app.get("/api/videos/{video_id}/frames/{index}")
    def frame(video_id: str, index: int):
        entry = video(video_id)
        if not 0 <= index < entry["frames"]:
            raise HTTPException(400, "Frame is outside the video")
        capture = cv2.VideoCapture(entry["path"])
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise HTTPException(400, "Could not decode this frame")
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise HTTPException(500, "Could not prepare the preview")
            return Response(encoded.tobytes(), media_type="image/jpeg")
        finally:
            capture.release()

    @app.post("/api/previews")
    def preview(options: PreviewRequest):
        from .preview import preview_mask
        entry = video(options.video_id)
        if options.frame >= entry["frames"]:
            raise HTTPException(400, "Frame is outside the video")
        if not compute.acquire(blocking=False):
            raise HTTPException(409, "An export or preview is running; wait for it to finish")
        detector = None
        try:
            with tempfile.TemporaryDirectory(prefix="redactor-selection-") as directory:
                if any("mask" in prompt for prompt in options.prompts):
                    raise ValueError("The browser accepts boxes and outlines only")
                if options.mode == "manual":
                    if any(prompt.get("frame", 0) != options.frame or prompt.get("action") == "end" for prompt in options.prompts):
                        raise ValueError("Preview only selections drawn on the displayed frame")
                    path = Path(directory) / "prompts.json"
                    path.write_text(json.dumps({"version": 1, "prompts": options.prompts}), encoding="utf-8")
                    prompts = load_prompts(path, (entry["height"], entry["width"]))
                capture = cv2.VideoCapture(entry["path"])
                try:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, options.frame)
                    ok, image = capture.read()
                finally:
                    capture.release()
                if not ok:
                    raise ValueError("Could not decode this frame")
                detector = make_detector(options)
                if detector is not None:
                    prompts = detector.prompts(image, options.frame, {})
                mask = preview_mask(image, prompts, options.backend)
                overlay = np.zeros((*mask.shape, 4), np.uint8)
                overlay[:, :, :3] = (102, 205, 60)  # BGRA: opaque mask, faint coverage margin.
                expanded = cv2.dilate(mask, np.ones((2 * options.margin + 1, 2 * options.margin + 1), np.uint8))
                overlay[:, :, 3] = np.where(mask > 0, 255, np.where(expanded > 0, 80, 0)).astype(np.uint8)
                ok, png = cv2.imencode(".png", overlay)
                if not ok:
                    raise OSError("Could not encode the preview")
                return {"mask": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                        "targets": len(prompts), "frame": options.frame, "backend": options.backend, "margin": options.margin}
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
            raise HTTPException(400, str(error)) from error
        finally:
            try:
                if detector is not None:
                    detector.close()
            finally:
                compute.release()

    def run_job(job_id, options, prompts):
        job = jobs[job_id]
        job["status"] = "running"
        detector = None
        try:
            detector = make_detector(options)
            def progress(done, total):
                job.update(done=done, total=total)
            report = process_video(video(options.video_id)["path"], job["output"], prompts,
                                   backend=options.backend, style=options.style, audio=options.audio,
                                   detector=detector, progress=progress, cancelled=job["cancel"].is_set, margin=options.margin)
            job.update(status="complete", report=report, review_frames=[row["frame"] for row in report["per_frame"]
                                                                       if row["covered_for_review"]],
                       review_targets={str(row["frame"]): [target["id"] for target in row["targets"]
                                                            if target["state"] in ("uncertain", "lost")]
                                       for row in report["per_frame"] if row["covered_for_review"]})
        except Exception as error:
            job.update(status="cancelled" if job["cancel"].is_set() else "failed", error=str(error))
        finally:
            try:
                if detector is not None:
                    detector.close()
            finally:
                compute.release()

    @app.post("/api/exports", status_code=202)
    def start_export(options: ExportRequest):
        entry = video(options.video_id)
        if options.mode == "text" and not options.text.strip():
            raise HTTPException(400, "Describe what you want to hide")
        with lock:
            if any(job["status"] in ("queued", "running") for job in jobs.values()):
                raise HTTPException(409, "An export is already running; finish or cancel it first")
            job_id = uuid4().hex
            directory = storage / job_id
            directory.mkdir()
            prompt_path = directory / "selection.json"
            # UI accepts geometry only; do not let HTTP callers read arbitrary mask files.
            for prompt in options.prompts:
                if "mask" in prompt:
                    raise HTTPException(400, "The browser accepts boxes and outlines only")
            prompt_path.write_text(json.dumps({"version": 1, "prompts": options.prompts}))
            try:
                prompts = load_prompts(prompt_path, (entry["height"], entry["width"])) if options.mode == "manual" else []
                if any(prompt.frame_index >= entry["frames"] for prompt in prompts):
                    raise ValueError("Selection is outside the video")
            except (ValueError, TypeError, KeyError) as error:
                raise HTTPException(400, str(error)) from error
            job = {"id": job_id, "status": "queued", "done": 0, "total": entry["frames"],
                   "cancel": threading.Event(), "output": directory / "redacted.mp4", "review_frames": [],
                   "mode": options.mode}
            if not compute.acquire(blocking=False):
                raise HTTPException(409, "A preview is running; wait for it to finish")
            jobs[job_id] = job
            try:
                executor.submit(run_job, job_id, options, prompts)
            except Exception:
                jobs.pop(job_id, None)
                compute.release()
                raise
        return {"id": job_id}

    @app.get("/api/exports/{job_id}")
    def export_status(job_id):
        job = job_entry(job_id)
        return {key: value for key, value in job.items() if key not in ("cancel", "output", "report")}

    @app.post("/api/exports/{job_id}/cancel")
    def cancel(job_id):
        job_entry(job_id)["cancel"].set()
        return {"ok": True}

    @app.get("/api/exports/{job_id}/file/{kind}")
    def download(job_id, kind: Literal["video", "report", "prompts"]):
        job = job_entry(job_id)
        if job["status"] != "complete":
            raise HTTPException(409, "Export is not ready")
        output = job["output"]
        path = output if kind == "video" else output.with_suffix(".json" if kind == "report" else ".prompts.json")
        if kind == "prompts" and job["mode"] == "manual":
            # Keep original polygon geometry so this download is self-contained.
            # The pipeline's local replay also supports PNG sidecars.
            path = output.parent / "selection.json"
        return FileResponse(path, media_type="video/mp4" if kind == "video" else "application/json")

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="Open the local Redactor review interface")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port} in your browser. Videos stay on this computer.")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
