"""Check the failing clip's saved masks against independent source-image points."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from redactor.render import expanded_mask


def video_checks(path):
    capture = cv2.VideoCapture(str(path))
    black = 0
    frames = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            black += int(np.mean(np.max(frame, axis=2) < 10) > .99)
            frames += 1
    finally:
        capture.release()
    return {'frames': frames, 'fps': fps, 'black_frames': black}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--checkpoints', type=Path, default=Path('data/light-jacket-checkpoints.json'))
    parser.add_argument('--output', type=Path, default=Path('results/rewrite-light-jacket.json'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    labels = json.loads(args.checkpoints.read_text())
    if report['input_sha256'] != labels['input_sha256']:
        raise ValueError('These checkpoints describe a different source video')
    rows = []
    for row in labels['checkpoints']:
        mask = cv2.imread(str(args.report.with_suffix('.masks') / f'{row["frame"]:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Missing mask for frame {row["frame"]}')
        if mask.shape != (labels['height'], labels['width']):
            raise ValueError('Checkpoint dimensions do not match the masks')
        covered = expanded_mask(mask) > 0
        hits = [bool(covered[y, x]) for x, y in row['target_points']]
        other_hits = [bool(covered[y, x]) for x, y in row['other_points']]
        rows.append({'frame': row['frame'], 'target_points': len(hits), 'covered_target_points': sum(hits),
                     'uncovered_points': [point for point, hit in zip(row['target_points'], hits) if not hit],
                     'other_points': len(other_hits), 'covered_other_points': sum(other_hits)})
    video = args.report.with_name(args.report.stem + '-preview.mp4')
    result = {'model': report['model'], 'frames': report['frames'], 'review_frames': report['review_frames'],
              'video': video_checks(video), 'seconds': report['seconds'],
              'source_points': str(args.checkpoints), 'annotation_method': labels['method'],
              'target_points': sum(r['target_points'] for r in rows),
              'covered_target_points': sum(r['covered_target_points'] for r in rows),
              'other_points': sum(r['other_points'] for r in rows),
              'covered_other_points': sum(r['covered_other_points'] for r in rows),
              'per_checkpoint': rows,
              'limits': 'Interior point checks and visual review are not dense boundary ground truth. No inference about unannotated clips.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if result['video']['black_frames'] or result['video']['frames'] != report['frames'] or result['covered_target_points'] != result['target_points'] or result['covered_other_points']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
