"""Measure reconstruction optimizations and smaller-model quality on shared inputs."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from redactor.reconstruction import BackgroundReconstructor
from redactor.sam_backend import VideoSegmenter
from redactor.types import TargetPrompt
from redactor.video import VideoReader

ROOT = Path(__file__).resolve().parents[1]


def reconstruction_comparison():
    capture = cv2.VideoCapture(str(ROOT/'examples/crosswalk.mp4'))
    frames, masks, truth = [], [], []
    for t in range(36):
        capture.set(cv2.CAP_PROP_POS_FRAMES, t*4)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError('Cannot decode reconstruction fixture')
        clean = cv2.resize(frame, (480, 224))
        mask = np.zeros(clean.shape[:2], np.uint8)
        cv2.ellipse(mask, (round(40+390*t/35), 105), (18, 32), 0, 0, 360, 255, -1)
        corrupted = clean.copy()
        corrupted[mask > 0] = (30, 45, 230)
        truth.append(clean)
        frames.append(corrupted)
        masks.append(mask)
    capture.release()
    settings = [('previous', 1, 17, True), ('final_only', 1, 17, False),
                ('eight_sources', 1, 8, False), ('four_threads', 4, 8, False)]
    runs = []
    for name, threads, sources, comparisons in settings:
        cv2.setNumThreads(threads)
        cv2.setRNGSeed(581)
        start = time.perf_counter()
        model = BackgroundReconstructor(frames, masks)
        setup = time.perf_counter()-start
        absolute, pixels, observed, total = 0., 0, 0, 0
        for t, (clean, mask) in enumerate(zip(truth, masks)):
            if comparisons:
                model.reconstruct(t, 'spatial', candidates=sources)
                model.reconstruct(t, 'unaligned', candidates=sources)
            result = model.reconstruct(t, candidates=sources)
            missing = mask > 0
            absolute += float(np.abs(result.image.astype(np.float32)-clean)[missing].sum())
            pixels += int(missing.sum())*3
            observed += int(result.observed.sum())
            total += int(missing.sum())
        run = dict(name=name, threads_requested=threads, threads_reported=cv2.getNumThreads(), maximum_sources=sources, comparison_methods=comparisons,
                   seconds=time.perf_counter()-start, feature_setup_seconds=setup,
                   masked_mae=absolute/pixels, observed_fraction=observed/total)
        print('reconstruction', json.dumps(run), flush=True)
        runs.append(run)
    return runs


def model_comparison(variants=("tiny", "base_plus"), mps_half=False):
    cv2.setNumThreads(4)
    annotations = json.loads((ROOT/'examples/crosswalk-annotations.json').read_text())
    labeled = {row['frame']:row for row in annotations['annotations']}
    indices = sorted(set(range(0, 150, 5)) | set(labeled))
    with VideoReader(ROOT/'examples/crosswalk.mp4') as reader:
        frames = [cv2.resize(frame, (640, 298)) for i, frame in reader if i in indices]
    # Freeze the first M2 DINO selection so every model receives the same box.
    box = (55.95099639892578, 203.77769470214844, 95.1392822265625, 295.8820495605469)
    runs = []
    with tempfile.TemporaryDirectory(prefix='runtime-models-') as directory:
        for i, frame in enumerate(frames):
            cv2.imwrite(str(Path(directory)/f'{i:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        for variant in variants:
            start = time.perf_counter()
            model = VideoSegmenter(variant=variant, image_size=512, mps_half=mps_half)
            load = time.perf_counter()-start
            print('model', model.metadata, flush=True)
            rows = []
            start = time.perf_counter()
            try:
                for i, mask, stability in model.propagate(directory, [TargetPrompt('target', 0, box=box)], frames[0].shape[:2]):
                    source_index = indices[i]
                    if source_index not in labeled:
                        continue
                    truth = np.zeros((annotations['height'], annotations['width']), np.uint8)
                    cv2.fillPoly(truth, [np.asarray(labeled[source_index]['polygon'], np.int32)], 255)
                    truth = cv2.resize(truth, (640, 298), interpolation=cv2.INTER_NEAREST) > 0
                    predicted = mask > 0
                    cover = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
                    rows.append({'frame': source_index, 'iou': float(np.count_nonzero(predicted & truth)/max(1,np.count_nonzero(predicted | truth))),
                                 'leak_rate': float(np.count_nonzero(truth & ~cover)/max(1,np.count_nonzero(truth))), 'stability': stability})
                tracking = time.perf_counter()-start
                run = dict(variant=variant, model=model.metadata, sampled_frames=len(frames), indices=indices, prompt_box=list(box),
                           load_seconds=load, tracking_seconds=tracking,
                           mean_sparse_iou=float(np.mean([r['iou'] for r in rows])),
                           mean_sparse_leak=float(np.mean([r['leak_rate'] for r in rows])), per_frame=rows)
                print('model_result', json.dumps(run), flush=True)
                runs.append(run)
            finally:
                model.close()
    return runs


def main():
    output = ROOT/'results/runtime-benchmark.json'
    report = {'notes':'Reconstruction: identical 36-frame natural fixture, before encoding, one pass per setting; no model inference in those timings. Model: same 31 sampled crosswalk frames, same initial box, both at 512 internal resolution; six approximate polygon labels scored. Sampling changes temporal difficulty versus full video. No download time included. Single-run timings, not statistical confidence intervals.',
              'environment':{'platform':platform.platform(),'opencv':cv2.__version__},
              'reconstruction_source_sha256':hashlib.sha256((ROOT/'redactor/reconstruction.py').read_bytes()).hexdigest()}
    report['reconstruction'] = reconstruction_comparison()
    output.write_text(json.dumps(report,indent=2)+'\n')
    report['models'] = model_comparison()
    report['models'] += model_comparison(variants=('base_plus',), mps_half=True)
    report['complete'] = True
    output.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
