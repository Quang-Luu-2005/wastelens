"""Filter x model experiment: condition definitions, extra operations, Holm and batched inference."""

import unittest

import cv2
import numpy as np
import torch

from wastelens.model import load_classifier, predict_probabilities
from wastelens.profiles import MODEL_PROFILES
from wastelens.robustness import (CORRUPTIONS, RESCUES, build_conditions, color_cast, gray_world, low_contrast,
                                  predict_batch)
from wastelens.stats import holm


def textured(seed=0, shape=(96, 128, 3)):
    return np.random.default_rng(seed).integers(30, 220, shape, dtype=np.uint8)


class ConditionTests(unittest.TestCase):
    def test_condition_set_and_identity(self):
        conditions = build_conditions()
        names = [c.name for c in conditions]
        self.assertEqual(len(names), len(set(names)))
        families = [c.family for c in conditions]
        self.assertEqual(families.count("corruption"), 5 * len(CORRUPTIONS))
        self.assertEqual(families.count("rescue"), sum(len(r) for _, _, r in RESCUES))
        image = textured()
        self.assertTrue(np.array_equal(conditions[0].transform(image, 0), image))
        for condition in conditions:
            out = condition.transform(image, 3)
            self.assertEqual((out.dtype, out.ndim, out.shape[2]), (np.uint8, 3, 3), condition.name)

    def test_severity_increases_damage_and_noise_is_reproducible(self):
        conditions = {c.name: c for c in build_conditions()}
        image = textured()
        errors = [np.abs(conditions[f"gaussian_noise_{s}"].transform(image, 5).astype(int) - image).mean()
                  for s in range(1, 6)]
        self.assertEqual(errors, sorted(errors))
        noisy = conditions["gaussian_noise_3"]
        self.assertTrue(np.array_equal(noisy.transform(image, 5), noisy.transform(image, 5)))
        self.assertFalse(np.array_equal(noisy.transform(image, 5), noisy.transform(image, 6)))

    def test_rescues_bind_their_own_restorer(self):
        conditions = {c.name: c for c in build_conditions()}
        image = textured()
        a = conditions["rescue_salt_pepper_3_median_3"].transform(image, 1)
        b = conditions["rescue_salt_pepper_3_median_5"].transform(image, 1)
        self.assertFalse(np.array_equal(a, b))
        # On a smooth image, median filtering removes most salt-and-pepper outliers.
        smooth = cv2.GaussianBlur(textured(), (0, 0), 6)
        repaired = conditions["rescue_salt_pepper_3_median_3"].transform(smooth, 1)
        damaged = conditions["salt_pepper_3"].transform(smooth, 1)
        self.assertLess(np.abs(repaired.astype(int) - smooth).mean(), np.abs(damaged.astype(int) - smooth).mean())

    def test_extra_operations(self):
        image = textured()
        flat = low_contrast(image, 0.3)
        self.assertLess(flat.std(), image.std())
        self.assertAlmostEqual(flat.mean(), 0.3 * image.mean() + 0.7 * 128, delta=1.0)
        warm = color_cast(image, 0.2)
        self.assertGreater(warm[:, :, 2].mean(), image[:, :, 2].mean())  # Red channel (BGR index 2) rises.
        self.assertLess(warm[:, :, 0].mean(), image[:, :, 0].mean())
        balanced = gray_world(warm).reshape(-1, 3).mean(axis=0)
        self.assertLess(balanced.max() - balanced.min(), 2.0)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(holm([0.5, 0.9]), [1.0, 1.0])


@unittest.skipUnless(all(p.checkpoint.is_file() for p in MODEL_PROFILES.values()), "checkpoints not present")
class BatchInferenceTests(unittest.TestCase):
    def test_batched_probabilities_match_single_image_api(self):
        from PIL import Image
        torch.set_num_threads(2)
        images = [textured(seed) for seed in range(5)]
        for key, profile in MODEL_PROFILES.items():
            model, _ = load_classifier(profile.checkpoint)
            batched = predict_batch(model, images, batch_size=2)
            single = np.stack([predict_probabilities(model, Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))).numpy()
                               for i in images])
            np.testing.assert_allclose(batched, single, atol=1e-5, err_msg=key)


if __name__ == "__main__":
    unittest.main()
