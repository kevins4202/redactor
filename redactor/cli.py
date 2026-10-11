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
    parser.add_argument("--backend", choices=["classical", "baseline", "sam2"], default="classical")
    parser.add_argument("--checkpoint", type=Path, help="SAM 2 checkpoint (default: models/sam2.1_hiera_tiny.pt)")
    parser.add_argument("--model-config", help="SAM 2 Hydra config path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--prompts", type=Path, help="JSON prompts, including corrections and later arrivals")
    selection.add_argument("--box", nargs=4, type=int, action="append", metavar=("X1", "Y1", "X2", "Y2"), help="Exclusive xyxy box in source pixels; repeat for multiple objects")
    selection.add_argument("--text", help="Detect a described object with Grounding DINO, e.g. 'the woman in a white skirt'")
    selection.add_argument("--detect", choices=["faces", "plates"], nargs="+", help="Find faces and/or license plates throughout the video")
    parser.add_argument("--detector", choices=["grounding", "haar"], default="grounding", help="Haar is a lightweight option for faces only")
    parser.add_argument("--detect-every", type=int, default=1, help="Run detection every N frames; larger intervals can miss new arrivals")
    parser.add_argument("--threshold", type=float, default=0.30, help="Grounding DINO detection threshold")
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
        detector = None
        if args.detect or args.text:
            from .detection import DetectorSource, FaceDetector, GroundingDetector
            if args.detector == "haar":
                if args.text or args.detect != ["faces"]:
                    raise ValueError("The Haar detector supports --detect faces only")
                detector = DetectorSource(FaceDetector())
            else:
                description = args.text or ". ".join("a human face" if label == "faces" else "a vehicle license plate" for label in args.detect)
                required = ["face" if name == "faces" else "plate" for name in args.detect] if args.detect else None
                detector = DetectorSource(GroundingDetector(description, args.device, args.threshold, required_labels=required))
        with VideoReader(args.input) as reader:
            if args.prompts:
                prompts = load_prompts(args.prompts, (reader.height, reader.width))
            elif args.box:
                prompts = [TargetPrompt(str(index + 1), 0, box=tuple(box)) for index, box in enumerate(args.box)]
            elif detector is not None:
                prompts = []
            else:
                _, frame = next(iter(reader))
                prompts = [TargetPrompt("1", 0, mask=choose_selection(frame))]
        def progress(done, total):
            if done == 1 or done % 30 == 0 or done == total:
                print(f"Processed {done}/{total or '?'} frames", flush=True)
        report = process_video(args.input, output, prompts, backend=args.backend, style=args.style,
                               on_failure=args.on_failure, overwrite=args.overwrite, audio=not args.no_audio,
                               mask_dir=args.mask_dir, progress=progress, checkpoint=args.checkpoint,
                               model_config=args.model_config, device=args.device, detector=detector,
                               detect_every=args.detect_every)
        print(f"Saved {output} ({report['frames']} frames; {report['review_frames']} covered frames need review)")
        return 0
    except (ValueError, RuntimeError, OSError, StopIteration) as error:
        print(f"Redactor: {error}", file=sys.stderr)
        return 1
