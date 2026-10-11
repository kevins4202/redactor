"""Run: python scripts/evaluate.py --output results/synthetic.json"""

import argparse
import csv
from contextlib import ExitStack
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from redactor.baseline import BaselineTracker
from redactor.datasets import SCENARIOS, SYNTHETIC_VERSION, annotated_video, davis_sequences, synthetic_sequence
from redactor.metrics import frame_metrics, summarize
from redactor.types import TargetPrompt


def make_tracker(method):
    if method in ("box", "lasso", "gate"):
        return BaselineTracker("box" if method == "box" else "lasso", gate=method == "gate")
    if method in ("affine", "classical"):
        from redactor.tracking import ClassicalTracker
        return ClassicalTracker(recover=method == "classical")
    raise ValueError(f"Unknown method: {method}")


def evaluate(sequence, method, slack=9, checkpoint=None, model_config=None, device="auto"):
    # A late appearing object is first prompted when it is actually visible.
    start = next(index for index, mask in enumerate(sequence.masks) if mask is not None and np.any(mask))
    selection = cv2.dilate(sequence.masks[start], np.ones((2 * slack + 1, 2 * slack + 1), np.uint8))
    prompt = TargetPrompt(str(sequence.object_id), start, mask=selection)
    rows = []
    guarded_rows = []
    review_frames = 0
    setup_started = time.perf_counter()
    model_metadata = None
    with ExitStack() as resources:
        if method == "sam2":
            from redactor.sam_backend import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, Sam2Session
            directory = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix="redactor-eval-")))
            for index, frame in enumerate(sequence.frames):
                if not cv2.imwrite(str(directory / f"{index:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                    raise OSError("Could not prepare SAM 2 evaluation input")
            tracker = Sam2Session(directory, checkpoint or DEFAULT_CHECKPOINT, model_config or DEFAULT_CONFIG, device)
            resources.callback(tracker.close)
            model_metadata = tracker.metadata
        else:
            tracker = make_tracker(method)
        setup_seconds = time.perf_counter() - setup_started
        started = time.perf_counter()
        for index in range(0 if method == "sam2" else start, len(sequence.frames)):
            frame = sequence.frames[index]
            if method == "sam2":
                results = tracker.step(frame, [prompt] if index == start else [])
                if index < start:
                    continue
                result = results[0]
            else:
                result = tracker.initialize(frame, prompt) if index == start else tracker.update(frame)
            guarded = result.state in ("uncertain", "lost")
            review_frames += int(guarded)
            if sequence.masks[index] is not None:
                rows.append({"frame": index, "state": result.state, "confidence": result.confidence,
                             **frame_metrics(result.mask, sequence.masks[index])})
                guarded_rows.append({"frame": index, **frame_metrics(
                    np.full_like(result.mask, 255) if guarded else result.mask, sequence.masks[index])})
        elapsed = time.perf_counter() - started
    return {"sequence": sequence.name, "method": method, "slack": slack,
            "model": model_metadata,
            "prompt_frame": start, "setup_seconds": setup_seconds, "seconds": elapsed,
            "processed_frames": len(sequence.frames) - start, "fps": (len(sequence.frames) - start) / elapsed,
            "review_frames": review_frames, "review_frame_fraction": review_frames / (len(sequence.frames) - start),
            "guarded_output": summarize(guarded_rows, sequence.fps),
            **summarize(rows, sequence.fps), "per_frame": rows}


def provenance():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    digest = hashlib.sha256()
    for path in sorted([*root.glob("redactor/*.py"), Path(__file__).resolve()]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_sha256": digest.hexdigest(), "python": platform.python_version(),
            "opencv": cv2.__version__, "numpy": np.__version__, "platform": platform.platform()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic.json"))
    parser.add_argument("--methods", nargs="+", choices=["box", "lasso", "gate", "affine", "classical", "sam2"], default=["box", "lasso", "gate", "affine", "classical"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-config")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--slack", type=int, nargs="+", default=[5, 9, 13])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument("--davis", type=Path)
    data_source.add_argument("--annotations", type=Path, help="Sparse annotations of a local video")
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args()
    if any(value < 0 for value in args.slack):
        parser.error("Slack must be nonnegative")
    cv2.setNumThreads(1)
    sequences = davis_sequences(args.davis, args.sequences) if args.davis else [annotated_video(args.annotations)] if args.annotations else (synthetic_sequence(name) for name in args.scenarios)
    runs = []
    report = {"dataset": str(args.davis.resolve()) if args.davis else str(args.annotations) if args.annotations else SYNTHETIC_VERSION,
              "notes": "Top-level metrics score raw tracking masks. guarded_output separately scores full-frame coverage on uncertain/lost states; it can hide most of a frame and is not a tracking accuracy improvement. Metrics measure mask coverage, not anonymity. Synthetic person-v1 reconstructs the experiment but is not the proposal's original fixture; crosswalk labels are approximate sparse polygons.",
              "environment": provenance(), "complete": False, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        summaries = [{key: value for key, value in run.items() if key not in ("per_frame", "model", "guarded_output")} for run in runs]
        if summaries:
            with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
    for sequence in sequences:
        for method in args.methods:
            for slack in args.slack:
                cv2.setRNGSeed(581)
                run = evaluate(sequence, method, slack, args.checkpoint, args.model_config, args.device)
                runs.append(run)
                save()
                print(f"{sequence.name:20s} {method:10s} slack={slack:2d} IoU={run['mean_iou']:.3f} leak={run['mean_leak_rate']:.1%} {run['fps']:.1f} fps", flush=True)
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
