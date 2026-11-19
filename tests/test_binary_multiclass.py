"""Binary vs multi-class: label maps, gradients, softmax/sigmoid equivalence, metrics and a training smoke test."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from wastelens.binary_multiclass import SETUPS, initial_gradient_check, logit_gradient, task_labels, train_fold
from wastelens.stats import agreement_metrics


def features(n=300, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 2772)).astype(np.float32)


class BinaryMulticlassTests(unittest.TestCase):
    def test_label_maps(self):
        six = np.arange(6)
        np.testing.assert_array_equal(task_labels("six", six), six)
        np.testing.assert_array_equal(task_labels("fiber", six), [1, 0, 0, 1, 0, 0])  # cardboard, paper
        np.testing.assert_array_equal(task_labels("trash", six), [0, 0, 0, 0, 0, 1])

    def test_analytic_logit_gradient_matches_autograd_for_every_head(self):
        six = np.random.default_rng(1).integers(0, 6, 300)
        for name, setup in SETUPS.items():
            check = initial_gradient_check(setup, features(), task_labels(setup.task, six))
            self.assertLess(check["analytic_vs_autograd_max_error"], 1e-5, name)

    def test_initial_softmax_loss_and_gradient_are_close_to_uniform_guess_values(self):
        six = np.random.default_rng(2).integers(0, 6, 300)
        check = initial_gradient_check(SETUPS["six_softmax"], features(), six)
        # Uniform softmax over 6 classes: loss ln 6 = 1.792, gradient norm sqrt(5/6) = 0.913.
        self.assertAlmostEqual(check["initial_loss"], np.log(6), delta=0.6)
        self.assertAlmostEqual(check["initial_gradient_norm"], np.sqrt(5 / 6), delta=0.15)

    def test_two_logit_softmax_equals_sigmoid_of_logit_difference(self):
        logits = torch.randn(50, 2)
        np.testing.assert_allclose(logits.softmax(1)[:, 1], torch.sigmoid(logits[:, 1] - logits[:, 0]), atol=1e-6)
        # Its logit gradient is antisymmetric: (p1 - y) on one logit, -(p1 - y) on the other.
        targets = torch.randint(0, 2, (50,))
        gradient = logit_gradient(SETUPS["fiber_softmax2"], logits, targets)
        np.testing.assert_allclose(gradient[:, 0], -gradient[:, 1], atol=1e-6)
        sigmoid = torch.sigmoid(logits[:, 1] - logits[:, 0]) - targets.float()
        np.testing.assert_allclose(gradient[:, 1], sigmoid, atol=1e-6)
        self.assertTrue(torch.allclose(gradient.sum(1), torch.zeros(50), atol=1e-6))

    def test_training_smoke_and_head_sizes(self):
        rng = np.random.default_rng(3)
        six = rng.integers(0, 6, 200)
        x = features(200, 3) + six[:, None].astype(np.float32)  # Learnable signal.
        sizes = {}
        for name in ("six_softmax", "fiber_sigmoid", "fiber_softmax2"):
            setup = SETUPS[name]
            y = task_labels(setup.task, six)
            result = train_fold(setup, x[:150], x[:150], y[:150], x[150:], y[150:], seed=0, epochs=3, record_every=1)
            self.assertEqual(result["probabilities"].shape, (50, 6 if setup.task == "six" else 2))
            np.testing.assert_allclose(result["probabilities"].sum(1), 1, atol=1e-5)
            self.assertEqual(len(result["history"]), 3)
            sizes[name] = result["head_parameters"]
        self.assertEqual(sizes, {"six_softmax": 198, "fiber_sigmoid": 33, "fiber_softmax2": 66})

    def test_agreement_metrics_reward_minority_detection(self):
        labels = np.array([0] * 90 + [1] * 10)
        always_negative = agreement_metrics(labels, np.zeros(100, int), 2)
        # 80 true negatives, 10 false positives, 8 of 10 positives found: 88% accuracy, below 90%.
        detector = agreement_metrics(labels, np.array([0] * 80 + [1] * 10 + [1] * 8 + [0] * 2), 2)
        self.assertGreater(always_negative["accuracy"], detector["accuracy"])
        self.assertGreater(detector["kappa"], always_negative["kappa"])
        self.assertGreater(detector["mcc"], 0.5)


if __name__ == "__main__":
    unittest.main()
