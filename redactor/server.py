"""Local single-target selection, processing, and review."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import threading
import tempfile
from typing import Literal
from urllib.parse import urlparse
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .pipeline import process_video, publish_preview, validate_prompts
from .prompts import load_prompts
from .render import expanded_mask
from .sam_backend import DEFAULT_CHECKPOINT
from .video import VideoReader


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    video_id: str
    prompts: list[dict] = Field(min_length=1)


class PreviewRequest(ExportRequest):
    frame: int = Field(default=0, ge=0)


def create_app(storage=None, demo=None):
    storage = Path(storage or 'output/studio').resolve()
    storage.mkdir(parents=True, exist_ok=True)
    videos, jobs = {}, {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='redactor')
    compute = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        for job in jobs.values():
            job['cancel'].set()
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title='Redactor', lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

    @app.middleware('http')
    async def local_requests(request, call_next):
        origin = request.headers.get('origin')
        if origin and urlparse(origin).netloc != request.headers.get('host'):
            return JSONResponse({'detail': 'Cross-origin requests are not allowed'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    def register(path, name, video_id=None):
        with VideoReader(path) as reader:
            if reader.count <= 0:
                raise ValueError('This video has no known frame count')
            entry = {'id': video_id or uuid4().hex, 'name': name, 'path': str(Path(path).resolve()),
                     'width': reader.width, 'height': reader.height, 'frames': reader.count, 'fps': reader.fps}
        videos[entry['id']] = entry
        return {key: value for key, value in entry.items() if key != 'path'}

    demo = Path(demo or Path(__file__).resolve().parents[1] / 'data/crosswalk-short.mp4')
    if demo.is_file():
        register(demo, 'Crosswalk sample', 'demo')

    def video(video_id):
        if video_id not in videos:
            raise HTTPException(404, 'Video not found; open it again after restarting the app')
        return videos[video_id]

    def job_entry(job_id):
        if job_id not in jobs:
            raise HTTPException(404, 'Processing session not found')
        return jobs[job_id]

    def selections(options, entry):
        if any('mask' in prompt for prompt in options.prompts):
            raise ValueError('Browser selections must use boxes or polygons')
        with tempfile.TemporaryDirectory(prefix='redactor-selection-') as directory:
            path = Path(directory) / 'prompts.json'
            path.write_text(json.dumps({'version': 1, 'prompts': options.prompts}), encoding='utf-8')
            prompts = load_prompts(path, (entry['height'], entry['width']))
        return validate_prompts(prompts, (entry['height'], entry['width']), entry['frames'])

    @app.get('/api/videos')
    def list_videos():
        return [{key: value for key, value in entry.items() if key != 'path'} for entry in videos.values()]

    @app.get('/api/capabilities')
    def capabilities():
        ready = importlib.util.find_spec('sam2') is not None and DEFAULT_CHECKPOINT.is_file()
        return {'ready': ready, 'model': 'SAM 2.1 Base+'}

    @app.post('/api/videos')
    async def upload(file: UploadFile):
        video_id = uuid4().hex
        directory = storage / video_id
        directory.mkdir()
        path = directory / 'input.mp4'
        size = 0
        try:
            with path.open('wb') as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024**3:
                        raise ValueError('Choose a video smaller than 2 GB')
                    stream.write(chunk)
            return register(path, Path(file.filename or 'Video').name, video_id)
        except (ValueError, OSError) as error:
            path.unlink(missing_ok=True)
            raise HTTPException(400, str(error)) from error
        finally:
            await file.close()

    @app.get('/api/videos/{video_id}/frames/{index}')
    def frame(video_id: str, index: int):
        entry = video(video_id)
        if not 0 <= index < entry['frames']:
            raise HTTPException(400, 'Frame is outside the video')
        capture = cv2.VideoCapture(entry['path'])
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise HTTPException(400, 'Could not decode this frame')
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise HTTPException(500, 'Could not prepare the frame')
            return Response(encoded.tobytes(), media_type='image/jpeg')
        finally:
            capture.release()

    @app.post('/api/previews')
    def preview(options: PreviewRequest):
        from .preview import preview_mask
        entry = video(options.video_id)
        try:
            prompts = selections(options, entry)
            if any(prompt.frame_index != options.frame for prompt in prompts):
                raise ValueError('Preview a selection drawn on the displayed frame')
            if not 0 <= options.frame < entry['frames']:
                raise ValueError('Frame is outside the video')
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(400, str(error)) from error
        if not compute.acquire(blocking=False):
            raise HTTPException(409, 'Processing is running; wait or cancel it first')
        capture = cv2.VideoCapture(entry['path'])
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, options.frame)
            ok, image = capture.read()
            if not ok:
                raise ValueError('Could not decode this frame')
            mask = preview_mask(image, prompts)
            overlay = np.zeros((*mask.shape, 4), np.uint8)
            overlay[:, :, :3] = (102, 205, 60)
            overlay[:, :, 3] = np.where(mask > 0, 220, np.where(expanded_mask(mask) > 0, 80, 0))
            ok, png = cv2.imencode('.png', overlay)
            if not ok:
                raise OSError('Could not encode the mask')
            return {'mask': 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')}
        except (ValueError, RuntimeError, OSError) as error:
            raise HTTPException(400, str(error)) from error
        finally:
            capture.release()
            compute.release()

    def run_job(job_id, options, prompts):
        job = jobs[job_id]
        job['status'] = 'running'
        try:
            def progress(stage, done, total):
                job.update(stage=stage, done=done, total=total)
            report = process_video(video(options.video_id)['path'], job['output'], prompts,
                                   progress=progress, cancelled=job['cancel'].is_set)
            job.update(status='review' if report['review_frames'] else 'complete',
                       review_frames=report['review_frames'], report=report,
                       review_reasons={str(row['frame']): row['review_reasons']
                                       for row in report.get('per_frame', []) if row['review_reasons']})
        except Exception as error:
            job.update(status='cancelled' if job['cancel'].is_set() else 'failed', error=str(error))
        finally:
            compute.release()

    @app.post('/api/exports', status_code=202)
    def start_export(options: ExportRequest):
        entry = video(options.video_id)
        try:
            prompts = selections(options, entry)
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(400, str(error)) from error
        if not compute.acquire(blocking=False):
            raise HTTPException(409, 'Processing is running; wait or cancel it first')
        try:
            job_id = uuid4().hex
            directory = storage / job_id
            directory.mkdir()
            (directory / 'selection.json').write_text(json.dumps({'version': 1, 'prompts': options.prompts}), encoding='utf-8')
            jobs[job_id] = {'id': job_id, 'status': 'queued', 'stage': 'prepare', 'done': 0,
                            'total': entry['frames'], 'cancel': threading.Event(),
                            'output': directory / 'redacted.mp4', 'review_frames': []}
            executor.submit(run_job, job_id, options, prompts)
        except Exception:
            compute.release()
            raise
        return {'id': job_id}

    @app.get('/api/exports/{job_id}')
    def export_status(job_id):
        return {key: value for key, value in job_entry(job_id).items() if key not in ('cancel', 'output', 'report')}

    @app.post('/api/exports/{job_id}/cancel')
    def cancel(job_id):
        job_entry(job_id)['cancel'].set()
        return {'ok': True}

    @app.post('/api/exports/{job_id}/approve')
    def approve(job_id):
        job = job_entry(job_id)
        if job['status'] != 'review':
            raise HTTPException(409, 'This preview is not awaiting review')
        publish_preview(job['output'])
        report = job['report']
        report.update(ready_for_export=True, user_reviewed_frames=job['review_frames'])
        job['output'].with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        job['status'] = 'complete'
        return {'ok': True}

    @app.get('/api/exports/{job_id}/file/{kind}')
    def download(job_id, kind: Literal['video', 'preview', 'report', 'prompts']):
        job = job_entry(job_id)
        if job['status'] not in ('complete', 'review') or (kind == 'video' and job['status'] != 'complete'):
            raise HTTPException(409, 'Review or correct the preview before downloading the final video')
        output = job['output']
        paths = {'video': output, 'preview': output.with_name(output.stem + '-preview.mp4'),
                 'report': output.with_suffix('.json'), 'prompts': output.parent / 'selection.json'}
        return FileResponse(paths[kind], media_type='video/mp4' if kind in ('video', 'preview') else 'application/json')

    app.mount('/', StaticFiles(directory=Path(__file__).parent / 'web', html=True), name='web')
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description='Open the local redaction app')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(f'Open http://127.0.0.1:{args.port}')
    uvicorn.run(create_app(), host='127.0.0.1', port=args.port)
