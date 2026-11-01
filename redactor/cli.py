"""Select one target, process the clip, and inspect any review intervals."""

import argparse
from pathlib import Path
import sys

from .pipeline import process_video
from .prompts import load_prompts
from .types import TargetPrompt
from .video import VideoReader


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Track and redact one selected object with SAM 2.1 Base+.')
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--box', nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'))
    selection.add_argument('--prompts', type=Path, help='Saved selection and later corrections for the same target')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or Path('output') / f'{args.input.stem}-redacted.mp4'
    try:
        with VideoReader(args.input) as reader:
            prompts = load_prompts(args.prompts, (reader.height, reader.width)) if args.prompts else [
                TargetPrompt('target', 0, box=tuple(args.box))]
        def progress(stage, done, total):
            if done == 1 or done % 30 == 0 or done == total:
                print(f'{stage}: {done}/{total}', flush=True)
        report = process_video(args.input, output, prompts, overwrite=args.overwrite, progress=progress)
        if report['review_frames']:
            preview = output.with_name(output.stem + '-preview.mp4')
            print(f'Preview: {preview}\n{len(report["review_frames"])} frames need review. Add corrections and rerun, or review and approve in the browser.')
            return 2
        print(f'Saved {output} ({report["frames"]} frames). Review the clip before sharing.')
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f'Redactor: {error}', file=sys.stderr)
        return 1
