"""Optional real Grounding DINO smoke checks; not a detector recall benchmark."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from redactor.detection import GroundingDetector, box_iou


def main():
    cases = [
        ("description", "data/crosswalk-short.mp4", "the woman in a white skirt", None, [210, 740, 343, 1078]),
        ("faces", "data/crosswalk-short.mp4", "a human face", ["face"], [238, 759, 274, 802]),
        ("plates", "data/DAVIS/JPEGImages/480p/car-roundabout/00000.jpg", "a vehicle license plate", ["plate"], [668, 288, 690, 310]),
    ]
    records = []
    for name, source, text, labels, expected in cases:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {source}; see docs/models.md and docs/evaluation.md")
        if path.suffix == ".mp4":
            capture = cv2.VideoCapture(str(path))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise ValueError(f"Cannot decode {source}")
        else:
            frame = cv2.imread(str(path))
        detector = GroundingDetector(text, required_labels=labels)
        try:
            detections = detector.detect(frame)
            overlap = max((box_iou(item.box, expected) for item in detections), default=0)
            record = {"case": name, "source": source, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "model": detector.metadata, "expected_region": expected, "best_overlap": overlap,
                      "passed": overlap > .25, "detections": [asdict(item) for item in detections]}
            records.append(record)
            print(f"{name}: {len(detections)} detections, known-region overlap {overlap:.3f}", flush=True)
        finally:
            detector.close()
    output = Path("results/detector-smoke.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"notes": "Three known examples, approximate expected regions; this demonstrates execution only and does not measure dataset-level precision or recall.", "checks": records}, indent=2) + "\n")
    if not all(record["passed"] for record in records):
        raise RuntimeError("A known-example detector smoke check failed; inspect results/detector-smoke.json")


if __name__ == "__main__":
    main()
