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
from redactor.reconstruction import BackgroundReconstructor
from redactor.video import VideoReader, VideoWriter, scene_cut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--text')
    selection.add_argument('--box', type=float, nargs=4)
    selection.add_argument('--masks', type=Path, help='Reuse one PNG per original frame, for matched reconstruction comparisons')
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
    cv2.setNumThreads(1)
    cv2.setRNGSeed(581)
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
    assets.mkdir(parents=True, exist_ok=True)
    selection_info = {'source': 'cached masks', 'path': str(args.masks)}
    model_info = None
    if args.masks:
        paths = sorted(args.masks.glob('*.png'))
        if len(paths) != len(frames):
            raise ValueError('Cached mask count must equal original video frame count')
        masks = []
        for path in paths:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != original_shape:
                raise ValueError(f'Invalid original-resolution mask: {path}')
            masks.append(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST))
    else:
        from redactor.sam_backend import VideoSegmenter
        from redactor.types import TargetPrompt
        if args.text:
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
        masks = [None]*len(frames)
        model = VideoSegmenter()
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
        if any(mask is None for mask in masks) or not masks[0].any():
            raise RuntimeError('Incomplete or empty initial segmentation')
    mask_dir = assets/'masks'
    mask_dir.mkdir(exist_ok=True)
    for i, mask in enumerate(masks):
        cv2.imwrite(str(mask_dir/f'{i:06d}.png'), mask)
    expanded = [cv2.dilate(mask, np.ones((7, 7), np.uint8)) for mask in masks]
    model = BackgroundReconstructor(frames, expanded)
    rows = []
    comparison = assets/'comparison.mp4'
    labels = ('Input', 'Single-frame fill', 'Unaligned reconstruction', 'Aligned reconstruction')
    with VideoWriter(args.output, args.input, fps, (width, height), overwrite=args.overwrite) as writer, VideoWriter(comparison, args.input, fps, (width*2, (height+26)*2), overwrite=args.overwrite) as demo:
        for i, frame in enumerate(frames):
            outputs = [model.reconstruct(i, method) for method in ('spatial', 'unaligned', 'aligned')]
            writer.write(outputs[-1].image)
            tiles = []
            for image, label in zip([frame, *[r.image for r in outputs]], labels):
                tile = cv2.copyMakeBorder(image, 26, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(tile, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            montage = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
            demo.write(montage)
            if i in (0, len(frames)//2, len(frames)-1):
                cv2.imwrite(str(assets/f'comparison-{i:03d}.jpg'), montage)
            missing = int((expanded[i] > 0).sum())
            rows.append({'frame': i, 'mask_pixels': missing, 'observed_pixels': int(outputs[-1].observed.sum()),
                         'accepted_sources': outputs[-1].accepted_sources,
                         'fallback_pixels': missing-int(outputs[-1].observed.sum())})
            if i % 20 == 0:
                print('reconstructed', i, '/', len(frames), flush=True)
    report = {'input': str(args.input), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'selection': selection_info, 'tracking_model': model_info, 'frames': len(frames), 'fps': fps,
              'working_resolution': [width, height], 'source_resolution': original_shape[::-1],
              'mask_expansion_pixels': 3, 'maximum_source_candidates': 17, 'minimum_source_support': 2,
              'method': 'ORB + RANSAC homography; masked temporal median; Telea fallback; hard composite inside expanded mask',
              'notes': 'Prototype. First-frame highest-score text selection can be wrong. Non-target moving objects are not masked. Homographies assume approximately planar background/pure rotation. Observation support does not imply correctness. No claim of hidden-background accuracy on real removal.',
              'seconds': time.perf_counter()-began, 'per_frame': rows, 'complete': True}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print('saved', args.output, 'seconds', report['seconds'], flush=True)

if __name__ == '__main__':
    main()
