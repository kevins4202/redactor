import argparse
from pathlib import Path

import cv2

from config import BLUR_KERNEL, GRABCUT_ITERATIONS, INITIAL_BOX, MASK_DILATION
from tracker import SegmentTracker, redact_frame


def parse_box(value):
    values = tuple(int(part) for part in value.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("box must be x,y,width,height")
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/demo.mp4")
    parser.add_argument("--output", default="output/demo-redacted.mp4")
    parser.add_argument("--box", type=parse_box, default=INITIAL_BOX)
    return parser.parse_args()


def process_video(input_path, output_path, initial_box):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {input_path}")

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("video has no readable frames")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    height, width = first_frame.shape[:2]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create {output_path}")

    tracker = SegmentTracker(first_frame, initial_box, GRABCUT_ITERATIONS)
    writer.write(redact_frame(first_frame, tracker.mask, BLUR_KERNEL, MASK_DILATION))
    frame_count = 1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        mask = tracker.update(frame)
        writer.write(redact_frame(frame, mask, BLUR_KERNEL, MASK_DILATION))
        frame_count += 1

    capture.release()
    writer.release()
    return frame_count, width, height, fps


def main():
    args = parse_args()
    result = process_video(args.input, args.output, args.box)
    frame_count, width, height, fps = result
    print(f"wrote {frame_count} frames at {width}x{height} and {fps:.2f} fps to {args.output}")


if __name__ == "__main__":
    main()

