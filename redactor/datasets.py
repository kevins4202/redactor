"""Deterministic synthetic scenes and a strict DAVIS 2017 reader."""

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


SCENARIOS = ("translation", "scale", "occlusion", "cut")
SYNTHETIC_VERSION = "person-v1"


@dataclass
class Sequence:
    name: str
    frames: list[np.ndarray]
    masks: list[np.ndarray | None]
    fps: float = 30.0
    object_id: int = 1


def synthetic_sequence(scenario, count=90, seed=581):
    """Render an instance-textured pedestrian with exact visible masks.

    The texture moves with the object, not with the image coordinates. Ground
    truth is used only to create the initial prompt and to score predictions.
    This is a new, versioned fixture, not the proposal's missing original script.
    """
    if scenario not in SCENARIOS or count < 2:
        raise ValueError("Choose a supported scenario and at least two frames")
    rng = np.random.default_rng(seed)
    height, width = 240, 360
    background = rng.integers(25, 85, (height, width, 3), np.uint8)
    for _ in range(24):
        x, y = rng.integers(0, width), rng.integers(0, height)
        color = tuple(int(c) for c in rng.integers(30, 170, 3))
        cv2.rectangle(background, (x, y), (x + 18, y + 25), color, -1)
    cut_background = rng.integers(100, 155, (height, width, 3), np.uint8)
    silhouette = np.zeros((112, 72), np.uint8)
    cv2.circle(silhouette, (36, 15), 13, 255, -1)
    polygon = np.array([(23, 30), (49, 30), (65, 67), (56, 72), (49, 55),
                        (47, 108), (35, 108), (34, 76), (28, 108), (15, 108),
                        (23, 55), (14, 74), (5, 68)], np.int32)
    cv2.fillPoly(silhouette, [polygon], 255)
    texture = np.full((112, 72, 3), (45, 90, 220), np.int16)
    texture += rng.integers(-30, 31, texture.shape, dtype=np.int16)
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    cv2.circle(texture, (36, 15), 13, (115, 170, 210), -1)
    cv2.circle(texture, (32, 13), 2, (20, 20, 20), -1)
    cv2.circle(texture, (41, 13), 2, (20, 20, 20), -1)
    cv2.line(texture, (25, 40), (45, 70), (240, 240, 240), 4)
    frames, masks = [], []
    for index in range(count):
        progress = index / (count - 1)
        scale = 1 - 0.55 * progress if scenario == "scale" else 1.0
        center_x = 65 + 230 * progress
        center_y = 130 + 7 * np.sin(progress * np.pi * 2)
        after_cut = scenario == "cut" and index >= count // 2
        if after_cut:
            center_x, center_y = 285 - 90 * (progress - 0.5), 100
        matrix = np.array([[scale, 0, center_x - 36 * scale],
                           [0, scale, center_y - 56 * scale]], np.float32)
        mask = cv2.warpAffine(silhouette, matrix, (width, height), flags=cv2.INTER_NEAREST)
        subject = cv2.warpAffine(texture, matrix, (width, height))
        frame = (cut_background if after_cut else background).copy()
        frame[mask > 0] = subject[mask > 0]
        if scenario == "occlusion":
            cv2.rectangle(frame, (139, 35), (220, 220), (100, 110, 105), -1)
            mask[35:221, 139:221] = 0
        frames.append(frame)
        masks.append(mask)
    return Sequence(scenario, frames, masks)


def davis_sequences(root, names=None, split="val", resolution="480p"):
    """Yield one sequence per annotated instance, preserving indexed PNG ids."""
    root = Path(root)
    if names is None:
        names = (root / "ImageSets" / "2017" / f"{split}.txt").read_text().split()
    for name in names:
        image_dir = root / "JPEGImages" / resolution / name
        annotation_dir = root / "Annotations" / resolution / name
        paths = sorted(image_dir.glob("*.jpg"))
        if not paths:
            raise ValueError(f"No frames found for DAVIS sequence {name}: {image_dir}")
        frames, labels = [], []
        for path in paths:
            annotation = annotation_dir / f"{path.stem}.png"
            if not annotation.is_file():
                raise ValueError(f"Missing per-frame annotation: {annotation}")
            frame = cv2.imread(str(path))
            with Image.open(annotation) as image:
                label = np.array(image)
            if frame is None or label.shape != frame.shape[:2]:
                raise ValueError(f"Invalid frame or indexed mask: {path}")
            frames.append(frame)
            labels.append(label)
        ids = sorted(set(int(value) for label in labels for value in np.unique(label)) - {0, 255})
        for object_id in ids:
            masks = [np.where(label == object_id, 255, 0).astype(np.uint8) for label in labels]
            yield Sequence(f"{name}/{object_id}", frames, masks, object_id=object_id)


def annotated_video(path):
    """Load sparse source-frame annotations without inventing in-between labels."""
    from .video import VideoReader
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != 1:
        raise ValueError("Unsupported annotation format")
    with VideoReader(path.parent / document["video"]) as reader:
        frames = [frame for _, frame in reader]
        shape = (reader.height, reader.width)
        if shape != (document["height"], document["width"]):
            raise ValueError("Annotation dimensions do not match the source video")
        masks = [None] * len(frames)
        for annotation in document["annotations"]:
            index = annotation["frame"]
            if index < 0 or index >= len(frames) or masks[index] is not None:
                raise ValueError("Invalid or duplicate annotated frame")
            mask = np.zeros(shape, np.uint8)
            points = np.asarray(annotation["polygon"], np.int32)
            cv2.fillPoly(mask, [points], 255)
            masks[index] = mask
        return Sequence(path.stem, frames, masks, fps=reader.fps)
