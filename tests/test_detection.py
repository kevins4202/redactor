from types import SimpleNamespace

import numpy as np

from redactor.detection import Detection, DetectorSource, suppress_duplicates


def test_repeated_detections_keep_ids_and_new_arrivals_get_new_ids():
    class Detector:
        def __init__(self):
            self.boxes = [Detection((10, 10, 30, 40), 0.8, "person")]
        def detect(self, frame):
            return self.boxes
    detector = Detector()
    source = DetectorSource(detector)
    frame = np.zeros((100, 100, 3), np.uint8)
    first = source.prompts(frame, 0, {})[0]
    trackers = {first.instance_id: SimpleNamespace(mask=first.selection(frame.shape[:2]))}
    detector.boxes = [Detection((12, 10, 32, 40), 0.9, "person"), Detection((65, 20, 85, 50), 0.7, "person")]
    later = source.prompts(frame, 5, trackers)
    assert later[0].instance_id == first.instance_id
    assert later[1].instance_id != first.instance_id
    assert later[1].frame_index == 5


def test_overlapping_model_proposals_do_not_create_duplicate_targets():
    detections = [Detection((0, 0, 30, 40), 0.9, "face"), Detection((1, 1, 31, 41), 0.8, "face")]
    assert len(suppress_duplicates(detections)) == 1
