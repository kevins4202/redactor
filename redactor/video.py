"""Video I/O with checked writers, audio preservation, and atomic publication."""

import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2


class VideoReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Cannot open video: {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.capture.release()
            raise ValueError("Video has no valid frame rate")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.capture.release()

    def __iter__(self):
        while True:
            success, frame = self.capture.read()
            if not success:
                # OpenCV reports decode errors and EOF identically. A known
                # advertised frame count lets us reject a truncated export.
                if self.count > 0 and self.index < self.count:
                    raise ValueError(f"Video stopped decoding at frame {self.index} of {self.count}")
                break
            if frame.shape[:2] != (self.height, self.width):
                raise ValueError("Video dimensions changed during decoding")
            yield self.index, frame
            self.index += 1


class VideoWriter:
    def __init__(self, output, source, fps, size, overwrite=False, audio=True):
        self.output, self.source = Path(output).resolve(), Path(source).resolve()
        if self.output == self.source:
            raise ValueError("Input and output paths must differ")
        if self.output.suffix.lower() != ".mp4":
            raise ValueError("Output must use the .mp4 extension")
        if self.output.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {self.output}; use --overwrite to replace it")
        if min(size) <= 0 or any(value % 2 for value in size):
            raise ValueError("MP4 export requires positive, even frame dimensions")
        self.ffmpeg = shutil.which("ffmpeg")
        self.audio = audio
        if audio and not self.ffmpeg:
            raise RuntimeError("FFmpeg is required to preserve audio. Install FFmpeg or use --no-audio.")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix=".redactor-", dir=self.output.parent)
        self.temporary = Path(self.temp.name) / "frames.mp4"
        self.writer = cv2.VideoWriter(str(self.temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            self.temp.cleanup()
            raise RuntimeError("Could not initialize the MP4 encoder")
        self.count = 0

    def __enter__(self):
        return self

    def write(self, frame):
        self.writer.write(frame)
        self.count += 1

    def __exit__(self, error_type, error, traceback):
        self.writer.release()
        try:
            if error_type is not None:
                return False
            if not self.count or not self.temporary.exists() or self.temporary.stat().st_size == 0:
                raise RuntimeError("Encoder produced no video")
            final = self.temporary
            if self.ffmpeg:
                final = Path(self.temp.name) / "final.mp4"
                command = [self.ffmpeg, "-v", "error", "-y", "-i", str(self.temporary)]
                if self.audio:
                    command += ["-i", str(self.source), "-map", "0:v:0", "-map", "1:a?"]
                command += ["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"]
                command += ["-c:a", "aac"] if self.audio else ["-an"]
                command += ["-movflags", "+faststart", str(final)]
                process = subprocess.run(command, capture_output=True, text=True)
                if process.returncode:
                    raise RuntimeError(f"FFmpeg export failed: {process.stderr.strip()}")
            # Never expose the partial, potentially unredacted output on failure.
            final.replace(self.output)
        finally:
            self.temp.cleanup()
