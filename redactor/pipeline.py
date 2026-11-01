"""Analyze first, review uncertain intervals, then publish the redacted clip."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

from .prompts import save_prompts
from .render import coverage_margin, redact_frame
from .sam_backend import VideoSegmenter, check_cancelled, extract_frames, propagate_shots
from .types import mask_box
from .video import VideoReader, VideoWriter


def validate_prompts(prompts, shape, count):
    if not prompts:
        raise ValueError("Select the object before processing")
    if len({prompt.instance_id for prompt in prompts}) != 1:
        raise ValueError("Select one target; later selections correct that same target")
    frames = set()
    for prompt in prompts:
        prompt.selection(shape)
        if prompt.action != "select":
            raise ValueError("Only target selections and corrections are supported")
        if prompt.frame_index >= count:
            raise ValueError(f"Selection frame {prompt.frame_index} is outside the video")
        if prompt.frame_index in frames:
            raise ValueError("Use one selection per frame")
        frames.add(prompt.frame_index)
    return sorted(prompts, key=lambda prompt: prompt.frame_index)


def review_records(records, prompts, cuts=()):
    anchors = {prompt.frame_index for prompt in prompts}
    previous = None
    for row in records:
        reasons = []
        area = row["area"]
        if not area:
            reasons.append("No visible mask: check occlusion, departure, or tracking failure")
        elif area > row["frame_area"] * 0.80:
            reasons.append("Mask covers most of the frame")
        elif row["stability"] < 0.80:
            reasons.append("Unstable mask boundary")
        if previous and area and previous["area"]:
            ratio = area / previous["area"]
            if ratio < 0.30 or ratio > 3.0:
                reasons.append("Abrupt change in mask size")
        shot_start = max((cut for cut in cuts if cut <= row["frame"]), default=0)
        shot_end = min((cut for cut in cuts if cut > row["frame"]), default=len(records))
        if not any(shot_start <= anchor < shot_end for anchor in anchors):
            reasons.append("This shot needs a target selection")
        row["review_reasons"] = reasons
        row["state"] = "review" if reasons else "tracking"
        previous = row
    return records


def process_video(input_path, output_path, prompts, overwrite=False, audio=True,
                  progress=None, cancelled=None):
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if source == output:
        raise ValueError("Input and output paths must differ")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must use the .mp4 extension")
    artifacts = [output, output.with_name(output.stem + '-preview.mp4'), output.with_suffix('.json'),
                 output.with_suffix('.prompts.json'), output.with_suffix('.masks')]
    if any(path.resolve() == source for path in artifacts) or source.is_relative_to(artifacts[-1].resolve()):
        raise ValueError("Output artifacts must not overwrite the input video")
    if not overwrite and any(path.exists() for path in artifacts):
        raise FileExistsError("Output already exists; choose another name or use --overwrite")
    if audio and not shutil.which('ffmpeg'):
        raise RuntimeError("Install FFmpeg before processing so source audio can be preserved")
    with VideoReader(source) as reader:
        shape, count, fps = (reader.height, reader.width), reader.count, reader.fps
    if count <= 0:
        raise ValueError("The video must have a known frame count")
    if any(value % 2 for value in shape):
        raise ValueError("MP4 export requires even frame dimensions")
    prompts = validate_prompts(list(prompts), shape, count)
    for index, prompt in enumerate(prompts):
        sidecar = output.with_name(f'{output.stem}.prompts-{index:03d}.png')
        if prompt.mask is not None and sidecar.resolve() == source:
            raise ValueError("Selection artifacts must not overwrite the input video")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".redactor-analysis-", dir=output.parent) as directory:
        work = Path(directory)
        frames, masks = work / 'frames', work / 'masks'
        masks.mkdir()
        decoded, cuts = extract_frames(source, frames, cancelled, progress)
        if decoded != count:
            raise ValueError("Decoded frame count does not match the video")
        if progress:
            progress("track", 0, count)
        model = VideoSegmenter()
        records = {}
        try:
            for index, mask, stability in propagate_shots(model, frames, prompts, shape, count, cuts, cancelled):
                if index in records or not 0 <= index < count or mask.shape != shape:
                    raise RuntimeError("The model returned invalid frame data")
                if not cv2.imwrite(str(masks / f"{index:06d}.png"), mask):
                    raise OSError(f"Could not save mask {index}")
                records[index] = {"frame": index, "area": int(np.count_nonzero(mask)),
                                  "frame_area": int(mask.size), "box": list(mask_box(mask)),
                                  "stability": stability}
                if progress:
                    progress("track", len(records), count)
            metadata = model.metadata
        finally:
            model.close()
        if len(records) != count:
            raise RuntimeError("The model did not return a mask for every frame")
        rows = [records[index] for index in range(count)]
        with VideoReader(source) as reader, VideoWriter(work / 'preview.mp4', source, fps,
                                                       (shape[1], shape[0]), audio=audio) as writer:
            for index, frame in reader:
                check_cancelled(cancelled)
                mask = cv2.imread(str(masks / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
                writer.write(redact_frame(frame, mask))
                if progress:
                    progress("render", index + 1, count)
        review_records(rows, prompts, cuts)
        review_frames = [row['frame'] for row in rows if row['review_reasons']]
        with source.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        report = {"version": 2, "input": str(source), "input_sha256": checksum,
                  "output": str(output), "model": metadata, "frames": count, "fps": fps,
                  "seconds": time.perf_counter() - started, "margin_pixels": coverage_margin(shape),
                  "review_frames": review_frames, "ready_for_export": not review_frames,
                  "scene_cuts": cuts, "per_frame": rows,
                  "warnings": ["Review the whole clip. These checks cannot detect every identity switch or missed boundary."]}
        (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        save_prompts(work / output.with_suffix('.prompts.json').name, prompts)
        check_cancelled(cancelled)
        destination = output.with_suffix('.masks')
        if destination.exists():
            shutil.rmtree(destination)
        masks.replace(destination)
        for path in work.glob('*.png'):
            path.replace(output.parent / path.name)
        (work / output.with_suffix('.prompts.json').name).replace(output.with_suffix('.prompts.json'))
        (work / 'report.json').replace(output.with_suffix('.json'))
        preview = output.with_name(output.stem + '-preview.mp4')
        (work / 'preview.mp4').replace(preview)
        if not review_frames:
            publish_preview(output)
        elif output.exists():
            output.unlink()
    return report


def publish_preview(output_path):
    output = Path(output_path)
    preview = output.with_name(output.stem + '-preview.mp4')
    with tempfile.TemporaryDirectory(prefix='.redactor-publish-', dir=output.parent) as directory:
        staged = Path(directory) / output.name
        shutil.copyfile(preview, staged)
        staged.replace(output)
