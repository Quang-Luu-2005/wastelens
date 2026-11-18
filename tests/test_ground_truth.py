"""Ground truth is available for dataset samples only, never arbitrary uploads."""

import csv
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from wastelens.classifier import DEFAULT_CHECKPOINT
from wastelens.profiles import MODEL_PROFILES
from wastelens.workbench import assess_prediction

from test_upload_ui import APP, upload


class AssessmentTests(unittest.TestCase):
    def test_accepted_predictions_are_compared_with_ground_truth(self):
        self.assertEqual(assess_prediction("paper", False, "paper"),
                         {"status": "correct", "top1_correct": True, "accepted": True})
        self.assertEqual(assess_prediction("paper", False, "plastic"),
                         {"status": "incorrect", "top1_correct": False, "accepted": True})

    def test_unknown_truth_is_not_treated_as_incorrect(self):
        result = assess_prediction("paper", False, None)
        self.assertEqual(result["status"], "unlabeled")
        self.assertIsNone(result["top1_correct"])

    def test_abstention_is_separate_from_top1_correctness(self):
        for truth, correct in [("paper", True), ("plastic", False)]:
            with self.subTest(truth=truth):
                result = assess_prediction("paper", True, truth)
                self.assertEqual(result["status"], "uncertain")
                self.assertEqual(result["top1_correct"], correct)
                self.assertFalse(result["accepted"])


@unittest.skipUnless(DEFAULT_CHECKPOINT.is_file(), "Install the app checkpoint first")
class GroundTruthUITests(unittest.TestCase):
    def test_random_buttons_preserve_labels_and_do_not_repeat_current_image(self):
        with (APP.parents[1] / "data/splits/trashnet.csv").open(encoding="utf-8", newline="") as file:
            test_labels = {row["path"]: row["label"] for row in csv.DictReader(file) if row["split"] == "test"}
        app = AppTest.from_file(str(APP), default_timeout=60).run()
        for label in dict.fromkeys(test_labels.values()):
            for _ in range(2):
                previous = app.session_state["sample_path"]
                app.button(key=f"sample_{label}").click().run()
                self.assertFalse(app.exception)
                chosen = app.session_state["sample_path"]
                self.assertNotEqual(chosen, previous)
                self.assertEqual(test_labels[chosen], label)
                self.assertEqual(app.session_state["sample_label"], label)
        for _ in range(4):
            previous = app.session_state["sample_path"]
            app.button(key="sample_random").click().run()
            self.assertFalse(app.exception)
            chosen = app.session_state["sample_path"]
            self.assertNotEqual(chosen, previous)
            self.assertEqual(test_labels[chosen], app.session_state["sample_label"])
        self.assertIsNone(app.session_state["result"])

    def test_only_dataset_images_have_ground_truth(self):
        records = []
        original_download = DeltaGenerator.download_button

        def capture(element, label, data, *args, **kwargs):
            if label == "Tải JSON":
                records.append(json.loads(data))
            return original_download(element, label, data, *args, **kwargs)

        def valid(app):
            self.assertFalse(app.exception, [error.message for error in app.exception])
            self.assertFalse(app.error, [error.value for error in app.error])

        with patch.object(DeltaGenerator, "download_button", capture):
            app = AppTest.from_file(str(APP), default_timeout=60).run()
            app.button(key="sample_cardboard").click().run()
            app.button(key="run_inference").click().run()
            valid(app)
            self.assertEqual(records[-1]["ground_truth"]["label"], "cardboard")
            self.assertEqual(records[-1]["ground_truth"]["source"], "dataset")
            self.assertEqual(records[-1]["evaluation"]["top1_correct"],
                             records[-1]["prediction"]["top_class"] == "cardboard")

            with (APP.parents[1] / "data/splits/trashnet.csv").open(encoding="utf-8", newline="") as file:
                test_rows = [row for row in csv.DictReader(file) if row["split"] == "test"]
            self.assertEqual(len(app.selectbox(key="dataset_image").options), len(test_rows))
            selected = test_rows[-1]
            choice = (selected["label"], APP.parents[1] / "data/trashnet" / selected["path"])
            app.selectbox(key="dataset_image").set_value(choice).run()
            app.button(key="use_dataset_image").click().run()
            valid(app)
            self.assertFalse(app.session_state["dataset_browser"])
            self.assertTrue(app.warning)  # The old image's result cannot be reused.
            app.button(key="run_inference").click().run()
            valid(app)
            self.assertEqual(records[-1]["ground_truth"],
                             {"label": selected["label"], "source": "dataset", "path": selected["path"]})

            encoded = io.BytesIO()
            Image.new("RGB", (80, 60), (120, 60, 30)).save(encoded, format="PNG")
            # A filename resembling a dataset label must not manufacture ground truth.
            with patch("streamlit.file_uploader", return_value=upload(encoded.getvalue(), "paper123.png")):
                app.run()
                app.button(key="run_inference").click().run()
                valid(app)
                self.assertEqual(records[-1]["evaluation"]["status"], "unlabeled")
                self.assertIsNone(records[-1]["ground_truth"]["label"])
                self.assertIsNone(records[-1]["ground_truth"]["source"])
                self.assertIsNone(records[-1]["evaluation"]["top1_correct"])
                self.assertFalse(any(box.key == "manual_truth" for box in app.selectbox))
                if all(profile.checkpoint.is_file() for profile in MODEL_PROFILES.values()):
                    app.toggle(key="compare_models").set_value(True).run()
                    valid(app)
                    self.assertEqual(list(app.dataframe[0].value.columns), ["Tên mô hình", "Dự đoán của mô hình"])
                    self.assertTrue(all(item["evaluation"]["top1_correct"] is None
                                        for item in records[-1]["comparisons"].values()))
            app.button(key="sample_cardboard").click().run()
            valid(app)
            self.assertEqual(app.session_state["sample_label"], "cardboard")


if __name__ == "__main__":
    unittest.main()
