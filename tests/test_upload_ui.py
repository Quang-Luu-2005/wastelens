"""Exercise real image decoding and recovery through Streamlit's app runner.

Requires the deployed improved checkpoint (see scripts/prepare_app_models.py).
Only the file-picker input is substituted; the app's decoder and inference run normally.
"""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from wastelens.classifier import DEFAULT_CHECKPOINT

APP = Path(__file__).resolve().parents[1] / "app/app.py"
ERROR = "Không đọc được ảnh. Vui lòng chọn lại tệp JPG hoặc PNG hợp lệ."


def upload(data: bytes, name: str) -> io.BytesIO:
    file = io.BytesIO(data)
    file.name = name
    return file


@unittest.skipUnless(DEFAULT_CHECKPOINT.is_file(), "Install the app checkpoint first")
class UploadUITests(unittest.TestCase):
    def test_invalid_upload_then_valid_upload_recovers_prediction_and_export(self):
        app = AppTest.from_file(str(APP), default_timeout=60)
        with patch("streamlit.file_uploader", return_value=upload(b"not an image", "broken.jpg")):
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual([error.value for error in app.error], [ERROR])
        self.assertFalse(app.metric)

        encoded = io.BytesIO()
        Image.new("RGB", (80, 60), (120, 60, 30)).save(encoded, format="PNG")
        records = []
        original_download = DeltaGenerator.download_button

        def capture_download(element, label, data, *args, **kwargs):
            if label == "Tải kết quả (JSON)":
                records.append(json.loads(data))
            return original_download(element, label, data, *args, **kwargs)

        with patch("streamlit.file_uploader", return_value=upload(encoded.getvalue(), "valid.png")), \
             patch.object(DeltaGenerator, "download_button", capture_download):
            app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.metric), 2)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], "valid.png")
        self.assertEqual(len(records[0]["prediction"]["probabilities"]), 6)
        self.assertAlmostEqual(sum(records[0]["prediction"]["probabilities"].values()), 1, places=6)

    def test_truncated_image_that_opens_but_cannot_decode_shows_friendly_error(self):
        encoded = io.BytesIO()
        Image.new("RGB", (80, 60), (120, 60, 30)).save(encoded, format="BMP")
        truncated = encoded.getvalue()[:64]
        # Ensure this fixture tests deferred pixel decoding rather than Image.open().
        with Image.open(io.BytesIO(truncated)) as image:
            with self.assertRaises(OSError):
                image.load()
        with patch("streamlit.file_uploader", return_value=upload(truncated, "truncated.png")):
            app = AppTest.from_file(str(APP), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual([error.value for error in app.error], [ERROR])


if __name__ == "__main__":
    unittest.main()
