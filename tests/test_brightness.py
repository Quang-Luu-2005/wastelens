"""Regressions for Module 1's signed brightness/contrast transform."""

import unittest

import numpy as np

from wastelens.ops import OpsParams, adjust_brightness_contrast, apply


class BrightnessTests(unittest.TestCase):
    def test_darken_clips_negative_values_instead_of_reflecting_them(self):
        image = np.array([[[0, 20, 50], [100, 150, 255]]], dtype=np.uint8)
        original = image.copy()
        actual = apply(image, OpsParams(brightness=-100))
        np.testing.assert_array_equal(actual, [[[0, 0, 0], [0, 50, 155]]])
        np.testing.assert_array_equal(image, original)
        self.assertEqual(actual.dtype, np.uint8)

    def test_brighten_saturates_without_wrapping(self):
        image = np.array([[[0, 20, 50], [100, 150, 255]]], dtype=np.uint8)
        actual = adjust_brightness_contrast(image, contrast=2, brightness=100)
        np.testing.assert_array_equal(actual, [[[100, 140, 200], [255, 255, 255]]])

    def test_dimming_never_increases_any_pixel(self):
        ramp = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
        for brightness in (-100, -50, -1):
            with self.subTest(brightness=brightness):
                self.assertTrue(np.all(adjust_brightness_contrast(ramp, 1, brightness) <= ramp))


if __name__ == "__main__":
    unittest.main()
