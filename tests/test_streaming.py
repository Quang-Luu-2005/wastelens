"""Geometry, frame validation, identity association and the two-stage contract."""
import base64
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from wastelens.classifier import Prediction
from wastelens.ops import OpsParams
from wastelens.streaming import Detection, ObjectTracker, clip_box, decode_frame, recognize_objects


class StreamingTests(unittest.TestCase):
    def test_clip_and_reject_empty_or_nonfinite_crops(self):
        self.assertEqual(clip_box((-10, -2, 105, 60), 100, 50), (0, 0, 100, 50))
        for box in [(5, 5, 1, 1), (101, 1, 110, 20), (0, 0, float('nan'), 5)]:
            self.assertIsNone(clip_box(box, 100, 50))

    def test_tracker_matches_one_to_one_when_detection_order_changes(self):
        tracker = ObjectTracker()
        first = Detection((0, 0, 30, 30), .9, 'bottle')
        second = Detection((50, 0, 80, 30), .8, 'cup')
        self.assertEqual(tracker.update([first, second]), [1, 2])
        self.assertEqual(tracker.update([second, first]), [2, 1])
        tracker.update([])
        self.assertEqual(tracker.update([first]), [1])
        for _ in range(3):
            tracker.update([])
        self.assertEqual(tracker.update([first]), [3])

    def test_decode_validates_frame_type_size_and_bgr_order(self):
        frame = np.zeros((32, 40, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        _, jpg = cv2.imencode('.jpg', frame)
        decoded = decode_frame({'jpeg': base64.b64encode(jpg).decode()})
        self.assertEqual(decoded.shape, frame.shape)
        self.assertGreater(decoded[0, 0, 2], 250)
        self.assertLess(decoded[0, 0, 0], 5)
        _, png = cv2.imencode('.png', frame)
        for payload in [{'jpeg': 'x' * 2_000_001}, {'jpeg': '%%%invalid'},
                        {'jpeg': base64.b64encode(png).decode()}, {}]:
            with self.assertRaises(ValueError):
                decode_frame(payload)

    def test_classifies_only_detected_crops_and_passes_detector_settings(self):
        frame = np.zeros((80, 100, 3), np.uint8)
        frame[20:40, 30:60, 2] = 255
        detector, classifier = Mock(), Mock()
        detector.detect.return_value = [Detection((30, 20, 60, 40), .9, 'bottle')]
        classifier.classify.return_value = Prediction('plastic', 'plastic', .8, {'plastic': .8})
        objects = recognize_objects(frame, detector, classifier, ObjectTracker(), OpsParams(), .6, 2)
        self.assertEqual(objects[0]['prediction'].label, 'plastic')
        self.assertEqual(objects[0]['box'], [30, 20, 60, 40])
        self.assertEqual(detector.detect.call_args.kwargs, {'threshold': .6, 'max_objects': 2})
        crop = classifier.classify.call_args.args[0]
        self.assertLess(crop.shape[0], frame.shape[0])
        self.assertGreater(crop[:, :, 2].mean(), 200)
        classifier.reset_mock()
        detector.detect.return_value = []
        self.assertEqual(recognize_objects(frame, detector, classifier, ObjectTracker(), OpsParams()), [])
        classifier.classify.assert_not_called()
