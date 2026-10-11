"""Portable JSON prompt files, including corrections and late-arriving targets."""

import json
from pathlib import Path

import cv2
import numpy as np

from .types import TargetPrompt


def load_prompts(path, shape):
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("Prompt file must be an object with version: 1")
    prompts = []
    seen = set()
    for entry in document.get("prompts", []):
        fields = [key for key in ("box", "polygon", "mask") if key in entry]
        if len(fields) != 1:
            raise ValueError("Each prompt needs exactly one box, polygon, or mask")
        mask = None
        if "mask" in entry:
            mask = cv2.imread(str(path.parent / entry["mask"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot read prompt mask: {entry['mask']}")
        elif "polygon" in entry:
            points = np.asarray(entry["polygon"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.all(np.isfinite(points)):
                raise ValueError("Polygon must contain at least three finite x,y pairs")
            mask = np.zeros(shape, np.uint8)
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
        frame_index = entry.get("frame", 0)
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise ValueError("Prompt frame must be an integer")
        instance_id = entry.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("Prompt id must be a nonempty string")
        prompt = TargetPrompt(instance_id, frame_index, mask=mask,
                              box=tuple(entry["box"]) if "box" in entry else None,
                              label=entry.get("label", "object"), source=entry.get("source", "manual"))
        prompt.selection(shape)
        key = (prompt.instance_id, prompt.frame_index)
        if key in seen:
            raise ValueError(f"Duplicate prompt for {key}")
        seen.add(key)
        prompts.append(prompt)
    if not prompts:
        raise ValueError("Prompt file contains no targets")
    return sorted(prompts, key=lambda prompt: prompt.frame_index)


def save_prompts(path, prompts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, prompt in enumerate(prompts):
        entry = {"id": prompt.instance_id, "frame": prompt.frame_index, "label": prompt.label, "source": prompt.source}
        if prompt.mask is not None:
            mask_path = path.with_name(f"{path.stem}-{index:03d}.png")
            if not cv2.imwrite(str(mask_path), prompt.mask):
                raise OSError(f"Cannot write {mask_path}")
            entry["mask"] = mask_path.name
        else:
            entry["box"] = list(prompt.box)
        entries.append(entry)
    path.write_text(json.dumps({"version": 1, "prompts": entries}, indent=2) + "\n", encoding="utf-8")
