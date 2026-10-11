"""Run: python scripts/evaluate.py --output results/synthetic.json"""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from redactor.baseline import BaselineTracker
from redactor.datasets import SCENARIOS, SYNTHETIC_VERSION, davis_sequences, synthetic_sequence
from redactor.metrics import frame_metrics, summarize
from redactor.types import TargetPrompt


def make_tracker(method):
    if method in ("box", "lasso", "gate"):
        return BaselineTracker("box" if method == "box" else "lasso", gate=method == "gate")
    if method in ("affine", "classical"):
        from redactor.tracking import ClassicalTracker
        return ClassicalTracker(recover=method == "classical")
    raise ValueError(f"Unknown method: {method}")


def evaluate(sequence, method, slack=9):
    tracker = make_tracker(method)
    # A late appearing object is first prompted when it is actually visible.
    start = next(index for index, mask in enumerate(sequence.masks) if np.any(mask))
    selection = cv2.dilate(sequence.masks[start], np.ones((2 * slack + 1, 2 * slack + 1), np.uint8))
    prompt = TargetPrompt(str(sequence.object_id), start, mask=selection)
    rows = []
    started = time.perf_counter()
    for index in range(start, len(sequence.frames)):
        frame = sequence.frames[index]
        result = tracker.initialize(frame, prompt) if index == start else tracker.update(frame)
        rows.append({"frame": index, "state": result.state, "confidence": result.confidence,
                     **frame_metrics(result.mask, sequence.masks[index])})
    elapsed = time.perf_counter() - started
    return {"sequence": sequence.name, "method": method, "slack": slack,
            "prompt_frame": start, "seconds": elapsed, "fps": len(rows) / elapsed,
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
    parser.add_argument("--methods", nargs="+", choices=["box", "lasso", "gate", "affine", "classical"], default=["box", "lasso", "gate", "affine", "classical"])
    parser.add_argument("--slack", type=int, nargs="+", default=[5, 9, 13])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--davis", type=Path)
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args()
    if any(value < 0 for value in args.slack):
        parser.error("Slack must be nonnegative")
    cv2.setNumThreads(1)
    sequences = davis_sequences(args.davis, args.sequences) if args.davis else (synthetic_sequence(name) for name in args.scenarios)
    runs = []
    for sequence in sequences:
        for method in args.methods:
            for slack in args.slack:
                cv2.setRNGSeed(581)
                run = evaluate(sequence, method, slack)
                runs.append(run)
                print(f"{sequence.name:20s} {method:10s} slack={slack:2d} IoU={run['mean_iou']:.3f} leak={run['mean_leak_rate']:.1%} {run['fps']:.1f} fps", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"dataset": str(args.davis.resolve()) if args.davis else SYNTHETIC_VERSION,
              "notes": "Reconstructed benchmark; not the proposal's original fixtures. Metrics measure mask coverage, not anonymity.",
              "environment": provenance(), "runs": runs}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summaries = [{key: value for key, value in run.items() if key != "per_frame"} for run in runs]
    if summaries:
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
