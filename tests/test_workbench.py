"""Result validity and explicit inference in the centered workbench."""

import json
import unittest
from unittest.mock import patch

from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from wastelens.ops import OpsParams
from wastelens.profiles import MODEL_PROFILES
from wastelens.workbench import PRESETS, describe_operations, request_signature

from test_upload_ui import APP


class RequestIdentityTests(unittest.TestCase):
    def test_result_is_invalidated_by_input_settings_model_and_checkpoint_changes(self):
        original = request_signature(b"first image", OpsParams(), "improved", 100)
        self.assertEqual(original, request_signature(b"first image", OpsParams(), "improved", 100))
        for arguments in [(b"other image", OpsParams(), "improved", 100),
                          (b"first image", PRESETS["Vừa"], "improved", 100),
                          (b"first image", OpsParams(), "baseline", 100),
                          (b"first image", OpsParams(), "improved", 101)]:
            with self.subTest(arguments=arguments):
                self.assertNotEqual(original, request_signature(*arguments))

    def test_summary_omits_inactive_filters_and_includes_active_geometry(self):
        self.assertEqual(describe_operations(OpsParams(clahe_tiles=12)), "Ảnh gốc · không áp dụng biến đổi")
        summary = describe_operations(OpsParams(gamma=1.15, angle=30, crop_keep=.5))
        self.assertIn("Gamma 1.15", summary)
        self.assertIn("Xoay 30°", summary)
        self.assertIn("Crop 50%", summary)


@unittest.skipUnless(all(profile.checkpoint.is_file() for profile in MODEL_PROFILES.values()),
                     "Install all three app checkpoints first")
class WorkbenchUITests(unittest.TestCase):
    def test_presets_model_changes_comparison_and_custom_apply_require_explicit_inference(self):
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
            valid(app)
            self.assertTrue(app.button(key="run_inference").disabled)
            app.button(key="sample_cardboard").click().run()
            valid(app)
            self.assertFalse(records)
            app.button(key="run_inference").click().run()
            valid(app)
            self.assertEqual(records[-1]["model_details"]["id"], "improved")
            previous = len(records)
            app.button(key="preset_Vừa").click().run()
            valid(app)
            self.assertTrue(app.warning)
            self.assertEqual(len(records), previous)  # No stale result can be exported.
            app.button(key="run_inference").click().run()
            valid(app)
            self.assertEqual(records[-1]["module1"]["gamma"], 1.15)
            app.toggle(key="compare_models").set_value(True).run()
            valid(app)
            self.assertEqual(set(records[-1]["comparisons"]), set(MODEL_PROFILES))
            self.assertEqual(len(app.dataframe[0].value), 3)
            self.assertIn("Đối chiếu", app.dataframe[0].value.columns)
            for comparison in records[-1]["comparisons"].values():
                self.assertEqual(comparison["evaluation"]["top1_correct"], comparison["top_class"] == "cardboard")

            for key in ("baseline", "mobilenet"):
                app.radio(key="model_key").set_value(key).run()
                valid(app)
                self.assertTrue(app.warning)
                app.button(key="run_inference").click().run()
                valid(app)
                self.assertEqual(records[-1]["model_details"]["id"], key)
                self.assertFalse(records[-1]["prediction"]["calibrated"])

            next(button for button in app.button if button.label == "Chỉnh sửa").click().run()
            valid(app)
            app.slider(key="edit_brightness").set_value(-40)
            app.slider(key="edit_angle").set_value(15)
            next(button for button in app.button if button.label == "Áp dụng").click().run()
            valid(app)
            self.assertTrue(app.warning)
            self.assertEqual(app.session_state["processing_level"], "Tùy chỉnh")
            app.button(key="run_inference").click().run()
            valid(app)
            self.assertEqual(records[-1]["module1"]["brightness"], -40)
            self.assertEqual(records[-1]["module1"]["angle"], 15)

            app.button(key="clear_image").click().run()
            valid(app)
            self.assertTrue(app.button(key="run_inference").disabled)
            self.assertIsNone(app.session_state["result"])


if __name__ == "__main__":
    unittest.main()
