import argparse
from pathlib import Path
import sys

from .pipeline import process_video
from .prompts import load_prompts
from .types import TargetPrompt
from .ui import choose_selection
from .video import VideoReader


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Select an object once and redact it throughout a video.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--backend", choices=["classical", "baseline"], default="classical")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--prompts", type=Path, help="JSON prompts, including corrections and later arrivals")
    selection.add_argument("--box", nargs=4, type=int, action="append", metavar=("X1", "Y1", "X2", "Y2"), help="Exclusive xyxy box in source pixels; repeat for multiple objects")
    parser.add_argument("--style", choices=["blur", "solid", "pixelate"], default="blur")
    parser.add_argument("--on-failure", choices=["cover", "stop", "predict"], default="cover",
                        help="Cover uncertain frames (default), stop for a correction, or export predicted masks for research")
    parser.add_argument("--mask-dir", type=Path, help="Save undilated union masks for inspection")
    parser.add_argument("--no-audio", action="store_true", help="Export without audio; permits running without FFmpeg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or Path("output") / f"{args.input.stem}-redacted.mp4"
    try:
        with VideoReader(args.input) as reader:
            if args.prompts:
                prompts = load_prompts(args.prompts, (reader.height, reader.width))
            elif args.box:
                prompts = [TargetPrompt(str(index + 1), 0, box=tuple(box)) for index, box in enumerate(args.box)]
            else:
                _, frame = next(iter(reader))
                prompts = [TargetPrompt("1", 0, mask=choose_selection(frame))]
        def progress(done, total):
            if done == 1 or done % 30 == 0 or done == total:
                print(f"Processed {done}/{total or '?'} frames", flush=True)
        report = process_video(args.input, output, prompts, backend=args.backend, style=args.style,
                               on_failure=args.on_failure, overwrite=args.overwrite, audio=not args.no_audio,
                               mask_dir=args.mask_dir, progress=progress)
        print(f"Saved {output} ({report['frames']} frames; {report['review_frames']} covered frames need review)")
        return 0
    except (ValueError, RuntimeError, OSError, StopIteration) as error:
        print(f"Redactor: {error}", file=sys.stderr)
        return 1
