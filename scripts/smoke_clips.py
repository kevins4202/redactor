"""Run two additional real-video smoke tests from the pinned SAM 2 gallery."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from redactor.pipeline import process_video
from redactor.types import TargetPrompt


ROOT = Path(__file__).resolve().parents[1]
CASES = [('01_dog', (430, 365, 720, 565)), ('04_coffee', (380, 290, 840, 650))]


def main():
    work = ROOT / 'output/rewrite-smoke'
    work.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, box in CASES:
        original = ROOT / f'models/sam2/demo/data/gallery/{name}.mp4'
        source = work / f'{name}-source.mp4'
        capture = cv2.VideoCapture(str(original))
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not capture.isOpened():
            raise ValueError('Run model setup to fetch the official example clips first')
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), fps / 8, size)
        if not writer.isOpened():
            raise RuntimeError('Could not prepare the smoke clip')
        count = 0
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % 8 == 0:
                    writer.write(frame)
                    count += 1
                index += 1
        finally:
            capture.release()
            writer.release()
        def progress(stage, done, total):
            if done == 1 or done % 10 == 0 or done == total:
                print(name, stage, done, total, flush=True)
        report = process_video(source, work / f'{name}.mp4', [TargetPrompt('target', 0, box=box)],
                               overwrite=True, progress=progress)
        with original.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        summaries.append({'name': name, 'original_sha256': checksum, 'source': str(original.relative_to(ROOT)),
                          'source_stride': 8, 'source_fps': fps, 'frames': count, 'initial_box': list(box),
                          'model': report['model'], 'review_frames': report['review_frames'],
                          'seconds': report['seconds'], 'output': report['output']})
        destination = ROOT / 'results/rewrite-real-smoke.json'
        destination.write_text(json.dumps({'runs': summaries,
                              'limits': 'Subsampled official example videos, one initial box each. Visual smoke checks, not dense annotated benchmarks.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
