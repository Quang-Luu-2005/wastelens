"""Video file mode: frame bookkeeping, EMA smoothing, thresholds, truncation and H.264 output."""

import csv
import io
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from wastelens.classifier import Prediction
from wastelens.ops import OpsParams
from wastelens.profiles import MODEL_PROFILES
from wastelens.streaming import Detection
from wastelens.video import (VideoSettings, fit_size, make_sample_video, objects_csv, probe, process_video,
                             read_frame, records_csv, segment_agreement)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def write_video(path: Path, frames: int, width: int = 100, height: int = 61, fps: int = 10):
    """Synthetic input: a moving bright square on a coloured background."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width // 2 * 2, height // 2 * 2, "yuv420p"
        for index in range(frames):
            image = np.full((stream.height, stream.width, 3), (40, 120, 200), np.uint8)
            image[10:30, 5 + 2 * index:25 + 2 * index] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class ScriptedClassifier:
    """Returns a scripted probability vector per call, mimicking WasteClassifier's interface."""

    def __init__(self, vectors, threshold=0.0):
        self.vectors, self.calls, self.classes, self.threshold = vectors, 0, CLASSES, threshold

    def classify(self, image):
        vector = self.vectors[min(self.calls, len(self.vectors) - 1)]
        self.calls += 1
        index = int(np.argmax(vector))
        return Prediction(label=CLASSES[index], top_class=CLASSES[index], confidence=float(vector[index]),
                          probabilities=dict(zip(CLASSES, map(float, vector))), timings_ms={"total": 1.0})


class FixedDetector:
    """Always finds the same two boxes, most confident first, honouring the detector's arguments."""

    def __init__(self):
        self.calls = []

    def detect(self, frame, threshold=.45, max_objects=3):
        self.calls.append((frame.shape, threshold, max_objects))
        found = [Detection((5, 5, 40, 40), 0.9, "bottle"), Detection((50, 10, 95, 55), 0.6, "cup")]
        return [d for d in found if d.score >= threshold][:max_objects]


def one_hot(index, value=1.0):
    vector = np.full(6, (1 - value) / 5)
    vector[index] = value
    return vector


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "input.mp4"
        write_video(self.source, frames=23)

    def tearDown(self):
        self.directory.cleanup()

    def test_every_frame_is_written_and_every_nth_frame_is_classified(self):
        classifier = ScriptedClassifier([one_hot(0)])
        result = process_video(self.source, self.root / "out.mp4", classifier, OpsParams(crop_keep=0.75),
                               VideoSettings(classify_every=5))
        self.assertEqual((result.frames_read, result.frames_written), (23, 23))
        self.assertEqual([record.frame for record in result.records], [0, 5, 10, 15, 20])
        self.assertFalse(result.truncated)
        self.assertEqual(result.codec, "libx264")
        self.assertTrue(result.browser_playable)
        # The center crop gives an odd size; the encoder needs even dimensions.
        self.assertEqual((result.width % 2, result.height % 2), (0, 0))
        with av.open(str(result.output)) as container:
            stream = container.streams.video[0]
            self.assertEqual(stream.codec_context.name, "h264")
            self.assertEqual(sum(1 for _ in container.decode(stream)), 23)
        self.assertIsNotNone(read_frame(result.output, 12))

    def test_ema_smoothing_delays_a_label_change_and_one_disables_it(self):
        vectors = [one_hot(0), one_hot(1), one_hot(1), one_hot(1)]
        smooth = process_video(self.source, self.root / "smooth.mp4", ScriptedClassifier(vectors),
                               OpsParams(), VideoSettings(classify_every=6, smoothing=0.5))
        self.assertEqual([r.top_class for r in smooth.records], ["cardboard", "glass", "glass", "glass"])
        # After one glass frame the EMA is a 50/50 tie (argmax keeps cardboard); then glass wins.
        self.assertAlmostEqual(smooth.records[1].smoothed_probabilities["glass"], 0.5)
        self.assertEqual([r.smoothed_label for r in smooth.records], ["cardboard", "cardboard", "glass", "glass"])
        raw = process_video(self.source, self.root / "raw.mp4", ScriptedClassifier(vectors),
                            OpsParams(), VideoSettings(classify_every=6, smoothing=1.0))
        self.assertEqual([r.smoothed_label for r in raw.records], ["cardboard", "glass", "glass", "glass"])

    def test_threshold_applies_to_smoothed_confidence(self):
        result = process_video(self.source, self.root / "unsure.mp4",
                               ScriptedClassifier([one_hot(2, 0.5)], threshold=0.6), OpsParams(),
                               VideoSettings(classify_every=10))
        self.assertTrue(all(record.smoothed_label == "uncertain" for record in result.records))
        self.assertEqual(result.label_shares(), {"uncertain": 1.0})

    def test_long_input_is_truncated_at_max_seconds(self):
        result = process_video(self.source, self.root / "short.mp4", ScriptedClassifier([one_hot(3)]),
                               OpsParams(), VideoSettings(classify_every=1, max_seconds=1.0))
        self.assertEqual(result.frames_read, 10)  # 1 s at 10 fps.
        self.assertTrue(result.truncated)
        summary = result.summary()
        self.assertEqual(summary["dominant_label"], "paper")
        self.assertEqual(summary["classified_frames"], 10)

    def test_csv_has_one_row_per_classified_frame_and_all_probabilities(self):
        result = process_video(self.source, self.root / "csv.mp4", ScriptedClassifier([one_hot(4)]),
                               OpsParams(), VideoSettings(classify_every=7))
        rows = list(csv.reader(io.StringIO(records_csv(result, CLASSES))))
        self.assertEqual(len(rows), 1 + len(result.records))
        self.assertEqual(len(rows[0]), 7 + 2 * len(CLASSES))
        self.assertEqual(rows[1][2], "plastic")

    def test_probe_and_size_fitting(self):
        info = probe(self.source)
        self.assertEqual((info["width"], info["height"], info["frames"]), (100, 60, 23))
        self.assertEqual(fit_size(1920, 1080, 960), (960, 540))
        self.assertEqual(fit_size(641, 359, 960), (640, 358))  # Never upscaled, made even.
        with self.assertRaises(ValueError):
            VideoSettings(classify_every=0)

    def test_sample_video_segments_and_agreement(self):
        import cv2
        images = []
        for index, color in enumerate([(0, 0, 255), (0, 255, 0)]):
            path = self.root / f"image{index}.png"
            cv2.imwrite(str(path), np.full((48, 64, 3), color, np.uint8))
            images.append((CLASSES[index], path))
        clip = self.root / "sample.mp4"
        segments = make_sample_video(images, clip, seconds_each=1.0, fps=10)
        self.assertEqual([(s["label"], s["start_s"], s["end_s"]) for s in segments],
                         [("cardboard", 0.0, 1.0), ("glass", 1.0, 2.0)])
        self.assertEqual(probe(clip)["frames"], 20)
        # A classifier that always says cardboard is right on the first segment only.
        result = process_video(clip, self.root / "agree.mp4", ScriptedClassifier([one_hot(0)]), OpsParams(),
                               VideoSettings(classify_every=2, smoothing=1.0))
        self.assertEqual(segment_agreement(result, segments), {"frames": 10, "raw_top1": 0.5, "smoothed": 0.5,
                                                               "smoothed_uncertain": 0.0, "smoothed_wrong": 0.5})
        unsure = process_video(clip, self.root / "unsure2.mp4", ScriptedClassifier([one_hot(0, 0.4)], threshold=0.5),
                               OpsParams(), VideoSettings(classify_every=2, smoothing=1.0))
        agreement = segment_agreement(unsure, segments)
        self.assertEqual((agreement["raw_top1"], agreement["smoothed"], agreement["smoothed_uncertain"]), (0.5, 0.0, 1.0))

    def test_detection_mode_tracks_and_smooths_each_object_separately(self):
        # Calls alternate between the two boxes: the bottle always looks like cardboard, the cup like glass.
        classifier = ScriptedClassifier([one_hot(0), one_hot(1)] * 10)
        detector = FixedDetector()
        result = process_video(self.source, self.root / "objects.mp4", classifier, OpsParams(),
                               VideoSettings(classify_every=5, smoothing=0.5, detect_objects=True), detector=detector)
        self.assertEqual((result.frames_read, result.frames_written, result.analysed_frames), (23, 23, 5))
        self.assertEqual(len(detector.calls), 5)
        self.assertEqual(result.records, [])  # No whole-frame labels in detection mode.
        self.assertEqual([r.frame for r in result.objects], [0, 0, 5, 5, 10, 10, 15, 15, 20, 20])
        by_id = {}
        for record in result.objects:
            by_id.setdefault(record.object_id, set()).add((record.detector_category, record.smoothed_label))
        # IDs survive across frames, and one object's evidence never leaks into the other's EMA.
        self.assertEqual(by_id, {1: {("bottle", "cardboard")}, 2: {("cup", "glass")}})
        self.assertTrue(all(r.smoothed_confidence == 1.0 for r in result.objects))
        summary = result.summary()
        self.assertEqual((summary["mode"], summary["detections"], summary["frames_with_objects"]), ("detection", 10, 5))
        self.assertEqual([(t["object_id"], t["dominant_label"], t["appearances"]) for t in summary["tracks"]],
                         [(1, "cardboard", 5), (2, "glass", 5)])
        rows = list(csv.reader(io.StringIO(objects_csv(result, CLASSES))))
        self.assertEqual(len(rows), 1 + 10)
        self.assertEqual(len(rows[0]), 14 + 2 * len(CLASSES))
        self.assertEqual(rows[1][2:7], ["1", "5", "5", "40", "40"])
        with av.open(str(result.output)) as container:
            self.assertEqual(sum(1 for _ in container.decode(container.streams.video[0])), 23)

    def test_detection_settings_are_passed_on_and_validated(self):
        detector = FixedDetector()
        result = process_video(self.source, self.root / "one.mp4", ScriptedClassifier([one_hot(4)]), OpsParams(),
                               VideoSettings(classify_every=10, detect_objects=True, detector_threshold=0.7,
                                             max_objects=1), detector=detector)
        self.assertEqual({call[1:] for call in detector.calls}, {(0.7, 1)})
        self.assertEqual({r.detector_category for r in result.objects}, {"bottle"})
        with self.assertRaises(ValueError):
            process_video(self.source, self.root / "none.mp4", ScriptedClassifier([one_hot(0)]), OpsParams(),
                          VideoSettings(detect_objects=True))
        with self.assertRaises(ValueError):
            VideoSettings(detector_threshold=1.0)
        with self.assertRaises(ValueError):
            VideoSettings(max_objects=0)

    @unittest.skipUnless(MODEL_PROFILES["improved"].checkpoint.is_file(), "deployed checkpoint not present")
    def test_real_ensemble_produces_normalized_probabilities(self):
        from wastelens.classifier import WasteClassifier
        classifier = WasteClassifier(MODEL_PROFILES["improved"].checkpoint)
        result = process_video(self.source, self.root / "real.mp4", classifier, OpsParams(),
                               VideoSettings(classify_every=5))
        self.assertEqual(len(result.records), 5)
        for record in result.records:
            self.assertAlmostEqual(sum(record.probabilities.values()), 1.0, places=4)
            self.assertAlmostEqual(sum(record.smoothed_probabilities.values()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
