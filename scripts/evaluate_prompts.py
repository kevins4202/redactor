"""Paired prompt-geometry ablation on every annotated crowded crosswalk frame."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from redactor.baseline import segment
from redactor.datasets import annotated_video
from redactor.metrics import frame_metrics
from redactor.types import mask_box
from scripts.evaluate import provenance


def compare_frame(frame, truth, slack):
    outline = cv2.dilate(truth, np.ones((2 * slack + 1, 2 * slack + 1), np.uint8))
    rectangle = np.zeros_like(outline)
    x1, y1, x2, y2 = mask_box(outline)
    rectangle[y1:y2, x1:x2] = 255
    rows = []
    masks = {}
    for name, selection, mode in (("box", rectangle, "box"),
                                   ("box_seeded", rectangle, "lasso"),
                                   ("lasso", outline, "lasso")):
        cv2.setRNGSeed(581)
        started = time.perf_counter()
        mask = segment(frame, selection, mode)
        rows.append({"method": name, "slack": slack, "seconds": time.perf_counter() - started,
                     **frame_metrics(mask, truth)})
        masks[name] = mask
    return rows, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/crosswalk-annotations.json"))
    parser.add_argument("--output", type=Path, default=Path("results/prompt-ablation.json"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sequence = annotated_video(args.annotations)
    rows = []
    panels = []
    for index, (frame, truth) in enumerate(zip(sequence.frames, sequence.masks)):
        if truth is None or not np.any(truth):
            continue
        for slack in (5, 9, 13):
            scores, masks = compare_frame(frame, truth, slack)
            rows.extend({"frame": index, **row} for row in scores)
            if slack == 9:
                x1, y1, x2, y2 = mask_box(truth)
                x1, y1 = max(0, x1 - 65), max(0, y1 - 35)
                x2, y2 = min(frame.shape[1], x2 + 65), min(frame.shape[0], y2 + 35)
                tiles = []
                for name, mask in [("truth", truth), *masks.items()]:
                    overlay = frame.copy()
                    overlay[mask > 0] = (overlay[mask > 0] * .45 + np.array([70, 190, 90]) * .55).astype(np.uint8)
                    crop = overlay[y1:y2, x1:x2]
                    scale = min(240 / crop.shape[1], 320 / crop.shape[0])
                    crop = cv2.resize(crop, None, fx=scale, fy=scale)
                    tile = np.full((356, 260, 3), 245, np.uint8)
                    tile[32:32+crop.shape[0], 10:10+crop.shape[1]] = crop
                    cv2.putText(tile, f"{index}: {name}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (25, 35, 25), 1, cv2.LINE_AA)
                    tiles.append(tile)
                panels.append(np.concatenate(tiles, axis=1))
        print(f"Scored frame {index}", flush=True)
    summary = {}
    for name in ("box", "box_seeded", "lasso"):
        selected = [row for row in rows if row["method"] == name]
        summary[name] = {"samples": len(selected), **{key: float(np.mean([row[key] for row in selected]))
                         for key in ("iou", "leak_rate", "excess_frame_fraction")}}
    report = {"complete": True, "environment": provenance([Path(__file__).resolve()]), "annotations": str(args.annotations),
              "annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
              "protocol": "Independent segmentation per annotated frame, paired slack 5/9/13. box uses rectangle-initialized GrabCut; box_seeded and lasso use identical seeded GrabCut. Prompts are generated from approximate truth plus dilation, not freehand user input. No propagation. Six frames of one clip are correlated; no significance or annotation-effort claim.",
              "summary": summary, "per_frame": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if panels:
        image_path = Path("docs/images/prompt-ablation.jpg")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), np.concatenate(panels, axis=0)):
            raise OSError(f"Cannot save {image_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
