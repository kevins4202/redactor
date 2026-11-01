"""Evaluate the single model on deterministic scenes or independently annotated footage."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from redactor.datasets import annotated_video, synthetic_sequence
from redactor.metrics import frame_metrics, summarize
from redactor.render import coverage_margin
from redactor.sam_backend import VideoSegmenter
from redactor.types import TargetPrompt, mask_box


def provenance():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.glob('redactor/*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    return {'commit': commit, 'source_sha256': digest.hexdigest(), 'python': platform.python_version(),
            'opencv': cv2.__version__, 'numpy': np.__version__, 'platform': platform.platform()}


def evaluate(model, sequence):
    start = next(i for i, mask in enumerate(sequence.masks) if mask is not None and mask.any())
    prompt = TargetPrompt('target', start, box=mask_box(sequence.masks[start]))
    rows = []
    began = time.perf_counter()
    shape = sequence.frames[0].shape[:2]
    dilation = 2 * coverage_margin(shape) + 1
    with tempfile.TemporaryDirectory(prefix='redactor-evaluate-') as directory:
        for index, frame in enumerate(sequence.frames):
            if not cv2.imwrite(str(Path(directory) / f'{index:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise OSError('Could not prepare evaluation frame')
        for index, mask, stability in model.propagate(directory, [prompt], shape):
            truth = sequence.masks[index]
            if truth is not None:
                rows.append({'frame': index, 'stability': stability,
                             **frame_metrics(mask, truth, dilation=dilation)})
    rows.sort(key=lambda row: row['frame'])
    return {'sequence': sequence.name, 'seconds': time.perf_counter() - began,
            'prompt_frame': start, 'margin_pixels': (dilation - 1) // 2,
            **summarize(rows, sequence.fps), 'per_frame': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/rewrite-synthetic.json'))
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--scenarios', nargs='+', choices=['translation', 'scale', 'occlusion'],
                        default=['translation', 'scale', 'occlusion'])
    parser.add_argument('--frames', type=int, default=30)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sequences = [annotated_video(args.annotations)] if args.annotations else [
        synthetic_sequence(name, count=args.frames) for name in args.scenarios]
    model = VideoSegmenter()
    report = {'environment': provenance(), 'model': model.metadata, 'runs': [],
              'notes': 'One initial box; no later labels supplied to propagation. Raw IoU and expanded-mask leakage. No full-frame blackout guard.'}
    try:
        for sequence in sequences:
            result = evaluate(model, sequence)
            report['runs'].append(result)
            print(sequence.name, result['mean_iou'], result['mean_leak_rate'], flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    finally:
        model.close()


if __name__ == '__main__':
    main()
