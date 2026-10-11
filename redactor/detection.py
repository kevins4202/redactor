"""Manual-compatible target prompts from local face and text detectors."""

from dataclasses import dataclass

import cv2
import numpy as np

from .types import TargetPrompt, mask_box


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    score: float
    label: str


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area - intersection) if area > intersection else 0.0


def suppress_duplicates(detections, threshold=0.55):
    kept = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if not any(box_iou(detection.box, other.box) > threshold for other in kept):
            kept.append(detection)
    return kept


def contains_box(outer, inner):
    intersection = max(0, min(outer[2], inner[2]) - max(outer[0], inner[0])) * max(0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return bool(area and intersection / area > 0.85)


class FaceDetector:
    """A lightweight frontal-face option; profiles and small faces can be missed."""
    def __init__(self):
        self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.model.empty():
            raise RuntimeError("OpenCV face cascade is unavailable")

    def detect(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        boxes = self.model.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(20, 20))
        return [Detection((float(x), float(y), float(x + width), float(y + height)), 1.0, "face")
                for x, y, width, height in boxes]


class GroundingDetector:
    def __init__(self, text, device="auto", threshold=0.30,
                 model_id="IDEA-Research/grounding-dino-tiny", required_labels=None,
                 revision="a2bb814dd30d776dcf7e30523b00659f4f141c71"):
        if not text.strip() or not 0 < threshold < 1:
            raise ValueError("Detection needs a description and a threshold between 0 and 1")
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Text detection requires requirements-ml.txt; see docs/models.md") from error
        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.text = text.strip().rstrip(".") + "."
        self.threshold = threshold
        self.model_id = model_id
        self.required_labels = required_labels
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=revision).to(self.device).eval()
        self.metadata = {"model_id": model_id, "revision": revision, "text": self.text,
                         "threshold": threshold, "device": self.device, "required_labels": required_labels}

    def detect(self, frame):
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=self.text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.threshold, text_threshold=0.25,
            target_sizes=[frame.shape[:2]])[0]
        labels = result.get("text_labels", result.get("labels"))
        detections = [Detection(tuple(float(value) for value in box.tolist()), float(score.item()), str(label))
                      for box, score, label in zip(result["boxes"], result["scores"], labels)]
        if self.required_labels:
            detections = [item for item in detections if any(word in item.label.lower() for word in self.required_labels)]
        if "." not in self.text.rstrip("."):
            # A single referring expression can produce both the person and a
            # nested clothing box. Keep the enclosing higher-confidence instance.
            detections = [item for item in detections if not any(
                other is not item and other.score > item.score and (contains_box(other.box, item.box) or contains_box(item.box, other.box))
                for other in detections)]
            for item in detections:
                item.label = self.text.rstrip(".")
        return suppress_duplicates(detections)

    def close(self):
        del self.model
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


class DetectorSource:
    """Associate repeated detections; emit corrections and newly entering targets.

    IDs use spatial overlap, not biometric identity. An ambiguous or unmatched
    old track is retained for review rather than silently declaring it absent.
    """
    def __init__(self, detector):
        self.detector = detector
        self.labels = {}
        self.next_id = 1

    def prompts(self, frame, frame_index, trackers):
        detections = suppress_duplicates(self.detector.detect(frame))
        available = {key: mask_box(value.mask) for key, value in trackers.items() if key in self.labels}
        prompts = []
        for detection in detections:
            candidates = [(box_iou(detection.box, box), key) for key, box in available.items()
                          if self.labels[key] == detection.label]
            overlap, instance_id = max(candidates, default=(0, ""))
            if overlap < 0.25:
                while f"detected-{self.next_id}" in trackers:
                    self.next_id += 1
                instance_id = f"detected-{self.next_id}"
                self.next_id += 1
            else:
                del available[instance_id]
            self.labels[instance_id] = detection.label
            prompt = TargetPrompt(instance_id, frame_index, box=detection.box,
                                  label=detection.label, source="detector")
            prompt.selection(frame.shape[:2])
            prompts.append(prompt)
        return prompts

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()
