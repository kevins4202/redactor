"""Matched background-reconstruction comparisons with withheld clean ground truth."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from redactor.reconstruction import BackgroundReconstructor
from redactor.video import VideoWriter

METHODS = ("spatial", "unaligned", "aligned")


def fixture(seed, moving, permanent=False, count=36):
    rng = np.random.default_rng(seed)
    height, width = 192, 320
    world = rng.integers(35, 220, (240, 420, 3), dtype=np.uint8)
    world = cv2.GaussianBlur(world, (3, 3), .6)
    for _ in range(65):
        x, y = rng.integers(0, 400), rng.integers(0, 220)
        color = tuple(int(c) for c in rng.integers(15, 240, 3))
        cv2.rectangle(world, (x, y), (x+int(rng.integers(5, 40)), y+int(rng.integers(5, 35))), color, -1)
    truth, frames, masks = [], [], []
    for t in range(count):
        progress = t/(count-1)
        x0 = round(55*progress) if moving else 0
        clean = world[15:15+height, x0:x0+width].copy()
        mask = np.zeros((height, width), np.uint8)
        x = 160 if permanent else round(35+250*progress)
        cv2.ellipse(mask, (x, 115), (22, 49), 0, 0, 360, 255, -1)
        frame = clean.copy()
        frame[mask > 0] = (30, 45, 230)
        truth.append(clean)
        frames.append(frame)
        masks.append(mask)
    return truth, frames, masks


def panel(images, labels):
    tiles = []
    for image, label in zip(images, labels):
        tile = cv2.copyMakeBorder(image, 26, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def run_case(name, truth, frames, masks, output, fps=18):
    began = time.perf_counter()
    cv2.setRNGSeed(581)
    model = BackgroundReconstructor(frames, masks)
    setup_seconds = time.perf_counter()-began
    results, predictions = {}, {}
    for method in METHODS:
        sums = dict(absolute=0., squared=0., pixels=0, observed=0, missing=0, outside_max=0, temporal_absolute=0., temporal_values=0)
        rows, outputs, previous = [], [], None
        began = time.perf_counter()
        for t, (clean, mask) in enumerate(zip(truth, masks)):
            result = model.reconstruct(t, method, candidates=17)
            missing = mask > 0
            error = result.image.astype(np.float32)-clean.astype(np.float32)
            values = error[missing]
            sums['absolute'] += float(np.abs(values).sum())
            sums['squared'] += float((values*values).sum())
            sums['pixels'] += values.size
            sums['missing'] += int(missing.sum())
            sums['observed'] += int(result.observed.sum())
            sums['outside_max'] = max(sums['outside_max'], int(np.abs(result.image.astype(np.int16)-frames[t].astype(np.int16))[~missing].max(initial=0)))
            if previous is not None:
                shared = missing & (masks[t-1] > 0)
                delta_error = error-previous
                sums['temporal_absolute'] += float(np.abs(delta_error[shared]).sum())
                sums['temporal_values'] += int(shared.sum())*3
            previous = error
            outputs.append(result.image)
            rows.append({'frame': t, 'mae': float(np.abs(values).mean()), 'observed_fraction': float(result.observed[missing].mean()), 'accepted_sources': result.accepted_sources})
        mse = sums['squared']/sums['pixels']
        results[method] = {'masked_mae': sums['absolute']/sums['pixels'], 'masked_psnr_db': float(10*np.log10(255**2/max(mse, 1e-12))),
                           'observed_fraction': sums['observed']/sums['missing'], 'outside_max_change': sums['outside_max'],
                           'temporal_residual_mae': sums['temporal_absolute']/max(1, sums['temporal_values']),
                           'seconds': time.perf_counter()-began, 'per_frame': rows}
        predictions[method] = outputs
        print(name, method, {k: round(v, 3) if isinstance(v, float) else v for k, v in results[method].items() if k != 'per_frame'}, flush=True)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        first = panel([truth[0], frames[0], *[predictions[m][0] for m in METHODS]], ['Withheld ground truth', 'Corrupted input', 'Single-frame inpainting', 'Unaligned temporal median', 'Aligned temporal median'])
        video = output/f'{name}.mp4'
        with VideoWriter(video, video.with_suffix('.source.mp4'), fps, (first.shape[1], first.shape[0]), overwrite=True, audio=False) as writer:
            for t in range(len(frames)):
                image = panel([truth[t], frames[t], *[predictions[m][t] for m in METHODS]], ['Withheld ground truth', 'Corrupted input', 'Single-frame inpainting', 'Unaligned temporal median', 'Aligned temporal median'])
                writer.write(image)
                if t in (0, len(frames)//2, len(frames)-1):
                    cv2.imwrite(str(output/f'{name}-{t:03d}.jpg'), image)
    return {'name': name, 'frames': len(frames), 'shape': list(frames[0].shape), 'feature_setup_seconds': setup_seconds, 'methods': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/removal-benchmark.json'))
    parser.add_argument('--artifacts', type=Path, default=Path('output/benchmarks'))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    report = {'protocol': 'reconstruction-v1', 'notes': '36 frames; exact shared masks; full-shot offline access; at most 17 evenly distributed candidate indices excluding target; support >=2; no clean pixels inside holes available to algorithm; Telea fallback; no learned inpainting. Test seeds fixed before execution. Not the proposal tracking benchmark.',
              'environment': {'python': platform.python_version(), 'opencv': cv2.__version__, 'platform': platform.platform()},
              'source_sha256': hashlib.sha256(Path('redactor/reconstruction.py').read_bytes()).hexdigest(), 'runs': []}
    for seed in (581, 582, 583):
        for name, moving, permanent in [('stationary', False, False), ('camera-pan', True, False), ('never-visible', False, True)]:
            truth, frames, masks = fixture(seed, moving, permanent)
            report['runs'].append(run_case(f'{name}-{seed}', truth, frames, masks, args.artifacts if seed == 581 else None))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
    # Natural moving footage with artificial corruption has observable reference
    # pixels too. Its underlying moving people remain in the reference video.
    cap = cv2.VideoCapture('examples/crosswalk.mp4')
    truth, frames, masks = [], [], []
    for t in range(36):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t*4)
        ok, image = cap.read()
        if not ok:
            raise RuntimeError('Missing natural-video benchmark frame')
        clean = cv2.resize(image, (480, 224))
        mask = np.zeros(clean.shape[:2], np.uint8)
        x = round(40+390*t/35)
        cv2.ellipse(mask, (x, 105), (18, 32), 0, 0, 360, 255, -1)
        frame = clean.copy()
        frame[mask > 0] = (30, 45, 230)
        truth.append(clean)
        frames.append(frame)
        masks.append(mask)
    cap.release()
    report['runs'].append(run_case('natural-crosswalk-artificial-hole', truth, frames, masks, args.artifacts, fps=7.5))
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')

if __name__ == '__main__':
    main()
