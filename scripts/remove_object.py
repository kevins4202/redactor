"""Offline text/box selection, SAM masks, and observed-background reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from redactor.reconstruction import BackgroundReconstructor, configure_threads
from redactor.video import VideoReader, VideoWriter, scene_cut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--text')
    selection.add_argument('--box', type=float, nargs=4)
    selection.add_argument('--masks', type=Path, help='Reuse one original- or working-resolution PNG per frame')
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Input and output must differ')
    if args.output.suffix.lower() != '.mp4':
        raise ValueError('Output must be MP4')
    assets = args.output.with_suffix('')
    assets = assets.parent/(assets.name+'-assets')
    protected = [args.output, args.output.with_suffix('.json'), assets]
    if not args.overwrite and any(path.exists() for path in protected):
        raise FileExistsError('Output or companion artifacts already exist; use --overwrite')
    if args.input.resolve() in [path.resolve() for path in protected]:
        raise ValueError('Output artifacts collide with the source')
    began = time.perf_counter()
    configure_threads()
    timings = {}
    stage = time.perf_counter()
    cv2.setRNGSeed(581)
    print('decoding', args.input, flush=True)
    with VideoReader(args.input) as reader:
        fps = reader.fps
        original_shape = (reader.height, reader.width)
        # Bounded working resolution for the CPU reconstruction prototype.
        scale = min(1., 640/reader.width)
        width = max(2, round(reader.width*scale/2)*2)
        height = max(2, round(reader.height*scale/2)*2)
        frames = []
        for index, frame in reader:
            if index >= 300:
                raise ValueError('This in-memory prototype supports short clips up to 300 frames')
            frames.append(cv2.resize(frame, (width, height)))
    if not frames:
        raise ValueError('Video has no frames')
    cuts = [i for i in range(1, len(frames)) if scene_cut(frames[i-1], frames[i])]
    if cuts:
        raise ValueError(f'Use a single shot for this prototype; scene cuts found at {cuts}')
    timings['decode_seconds'] = time.perf_counter()-stage
    assets.mkdir(parents=True, exist_ok=True)
    selection_info = {'source': 'cached masks', 'path': str(args.masks)}
    model_info = None
    stage = time.perf_counter()
    if args.masks:
        paths = sorted(args.masks.glob('*.png'))
        if len(paths) != len(frames):
            raise ValueError('Cached mask count must equal original video frame count')
        masks = []
        for path in paths:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape not in (original_shape, (height, width)):
                raise ValueError(f'Invalid original- or working-resolution mask: {path}')
            masks.append(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST))
    else:
        from redactor.sam_backend import VideoSegmenter
        from redactor.types import TargetPrompt
        if args.text:
            print('selecting target from text', flush=True)
            from redactor.grounding import ground_target
            selection_info = ground_target(frames[0], args.text)
            box = selection_info['selected']['box']
        else:
            box = [args.box[0]*width/original_shape[1], args.box[1]*height/original_shape[0],
                   args.box[2]*width/original_shape[1], args.box[3]*height/original_shape[0]]
            selection_info = {'source': 'manual box', 'selected': {'box': box}}
        preview = frames[0].copy()
        x1, y1, x2, y2 = [round(value) for value in box]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(assets/'selection.jpg'), preview)
        (assets/'selection.json').write_text(json.dumps(selection_info, indent=2)+'\n')
        print('selection', json.dumps(selection_info), flush=True)
        timings['selection_seconds'] = time.perf_counter()-stage
        masks = [None]*len(frames)
        stage = time.perf_counter()
        model = VideoSegmenter()
        timings['model_load_seconds'] = time.perf_counter()-stage
        print('tracking', model.metadata['name'], 'on', model.device, 'at', model.metadata['image_size'], 'pixels', flush=True)
        stage = time.perf_counter()
        model_info = model.metadata
        try:
            with tempfile.TemporaryDirectory(prefix='removal-frames-') as directory:
                for i, frame in enumerate(frames):
                    if not cv2.imwrite(str(Path(directory)/f'{i:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                        raise OSError('Failed to prepare inference frame')
                for i, mask, stability in model.propagate(directory, [TargetPrompt('target', 0, box=tuple(box))], (height, width)):
                    masks[i] = mask
        finally:
            model.close()
        timings['tracking_and_frame_preparation_seconds'] = time.perf_counter()-stage
        if any(mask is None for mask in masks) or not masks[0].any():
            raise RuntimeError('Incomplete or empty initial segmentation')
    stage = time.perf_counter()
    np.savez_compressed(assets/'masks.npz', packed=np.packbits(np.stack(masks) > 0, axis=-1), width=width)
    expanded = [cv2.dilate(mask, np.ones((7, 7), np.uint8)) for mask in masks]
    timings['mask_preparation_seconds'] = time.perf_counter()-stage
    stage = time.perf_counter()
    configure_threads()
    model = BackgroundReconstructor(frames, expanded)
    timings['background_features_seconds'] = time.perf_counter()-stage
    rows = []
    reconstruction_seconds = 0.
    stage = time.perf_counter()
    with VideoWriter(args.output, args.input, fps, (width, height), overwrite=args.overwrite) as writer:
        parallelism = configure_threads()
        opencv_threads = parallelism['reported']
        print('reconstructing with at most 8 sources;', parallelism, flush=True)
        for i, frame in enumerate(frames):
            tick = time.perf_counter()
            result = model.reconstruct(i)
            reconstruction_seconds += time.perf_counter()-tick
            writer.write(result.image)
            missing = int((expanded[i] > 0).sum())
            rows.append({'frame': i, 'mask_pixels': missing, 'observed_pixels': int(result.observed.sum()),
                         'accepted_sources': result.accepted_sources,
                         'fallback_pixels': missing-int(result.observed.sum())})
            if i % 20 == 0:
                print('reconstructed', i, '/', len(frames), flush=True)
    timings['reconstruction_seconds'] = reconstruction_seconds
    timings['render_and_encode_seconds'] = time.perf_counter()-stage-reconstruction_seconds
    report = {'input': str(args.input), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'selection': selection_info, 'tracking_model': model_info, 'frames': len(frames), 'fps': fps,
              'working_resolution': [width, height], 'source_resolution': original_shape[::-1],
              'mask_expansion_pixels': 3, 'maximum_source_candidates': 8, 'opencv_threads': opencv_threads, 'opencv_parallelism': parallelism, 'timings': timings, 'minimum_source_support': 2,
              'method': 'ORB + RANSAC homography; masked temporal median; Telea fallback; hard composite inside expanded mask',
              'notes': 'Prototype. First-frame highest-score text selection can be wrong. Non-target moving objects are not masked. Homographies assume approximately planar background/pure rotation. Observation support does not imply correctness. No claim of hidden-background accuracy on real removal.',
              'seconds': time.perf_counter()-began, 'per_frame': rows, 'complete': True}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print('saved', args.output, 'seconds', round(report['seconds'], 2), flush=True)
    print('timings', json.dumps(timings), flush=True)

if __name__ == '__main__':
    main()
