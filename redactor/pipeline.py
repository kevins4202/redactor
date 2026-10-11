"""Multi-instance processing with corrections, saved prompts, and review logs."""

import json
import hashlib
from contextlib import ExitStack
from pathlib import Path
import tempfile
import time

import cv2

from .baseline import BaselineTracker
from .prompts import save_prompts
from .render import coverage_mask, redact_frame
from .tracking import ClassicalTracker
from .types import mask_box
from .video import VideoReader, VideoWriter


class MultiTracker:
    def __init__(self, factory=ClassicalTracker):
        self.factory = factory
        self.trackers = {}

    def step(self, frame, prompts=()):
        results = {}
        for prompt in prompts:
            if prompt.action == "end":
                if prompt.instance_id not in self.trackers:
                    raise ValueError(f"Cannot end inactive target {prompt.instance_id}")
                del self.trackers[prompt.instance_id]
                continue
            tracker = self.factory()
            results[prompt.instance_id] = tracker.initialize(frame, prompt)
            self.trackers[prompt.instance_id] = tracker
        for instance_id, tracker in self.trackers.items():
            if instance_id not in results:
                results[instance_id] = tracker.update(frame)
        return list(results.values())


def process_video(input_path, output_path, prompts, backend="classical", style="blur",
                  on_failure="cover", overwrite=False, audio=True, mask_dir=None,
                  progress=None, detector=None, detect_every=1, checkpoint=None,
                  model_config=None, device="auto", cancelled=None):
    if backend not in ("classical", "baseline", "sam2"):
        raise ValueError(f"Unsupported streaming backend: {backend}")
    if detect_every < 1:
        raise ValueError("Detection interval must be positive")
    prompt_list = list(prompts)
    prompt_schedule = {}
    for prompt in prompt_list:
        prompt_schedule.setdefault(prompt.frame_index, []).append(prompt)
    pipeline = MultiTracker(ClassicalTracker if backend == "classical" else BaselineTracker)
    records = []
    had_targets = False
    started = time.perf_counter()
    report_path = Path(output_path).with_suffix(".json")
    prompt_path = Path(output_path).with_suffix(".prompts.json")
    for path in (report_path, prompt_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output metadata exists: {path}; use --overwrite")
    if mask_dir:
        mask_dir = Path(mask_dir)
        if mask_dir.exists() and any(mask_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Mask directory is not empty: {mask_dir}")
        mask_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources, VideoReader(input_path) as reader:
        for prompt in prompt_list:
            prompt.selection((reader.height, reader.width))
            if reader.count and prompt.frame_index >= reader.count:
                raise ValueError(f"Prompt frame {prompt.frame_index} is outside the video")
        if backend == "sam2":
            from .sam_backend import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, Sam2Session, extract_frames
            directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="redactor-sam2-"))
            extract_frames(input_path, directory)
            pipeline = Sam2Session(directory, checkpoint or DEFAULT_CHECKPOINT, model_config or DEFAULT_CONFIG, device)
            resources.callback(pipeline.close)
        with VideoWriter(output_path, input_path, reader.fps, (reader.width, reader.height), overwrite, audio) as writer:
            for index, frame in reader:
                if cancelled and cancelled():
                    raise RuntimeError("Export cancelled")
                current_prompts = list(prompt_schedule.get(index, []))
                if detector is not None and index % detect_every == 0:
                    detected = detector.prompts(frame, index, pipeline.trackers)
                    current_prompts.extend(detected)
                    prompt_list.extend(detected)
                results = pipeline.step(frame, current_prompts)
                had_targets = had_targets or bool(results)
                union, covered = coverage_mask(results, frame.shape[:2], on_failure)
                writer.write(redact_frame(frame, union, "solid" if covered else style))
                if mask_dir and not cv2.imwrite(str(mask_dir / f"{index:06d}.png"), union):
                    raise OSError(f"Cannot save mask at frame {index}")
                records.append({"frame": index, "covered_for_review": covered, "targets": [
                    {"id": result.instance_id, "state": result.state, "confidence": result.confidence,
                     "reason": result.reason, "box": list(mask_box(result.mask))} for result in results]})
                if progress:
                    progress(index + 1, reader.count)
            if not had_targets:
                raise ValueError("No targets were selected or detected; no redacted video was exported")
        seconds = time.perf_counter() - started
        with Path(input_path).open("rb") as stream:
            input_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {"version": 1, "input": str(Path(input_path).resolve()), "input_sha256": input_hash,
                  "output": str(Path(output_path).resolve()), "model": getattr(pipeline, "metadata", None),
                  "backend": backend, "style": style, "failure_policy": on_failure, "fps": reader.fps,
                  "frames": len(records), "seconds": seconds, "processing_fps": len(records) / seconds,
                  "review_frames": sum(row["covered_for_review"] for row in records),
                  "warnings": ["Unflagged tracking errors remain possible. Review the entire output before sharing."],
                  "per_frame": records}
    save_prompts(prompt_path, prompt_list)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
