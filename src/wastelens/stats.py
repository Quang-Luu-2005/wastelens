"""Small-sample statistics for comparing classifiers, using only math and NumPy.

Exact binomial tails are summed in log space, so the functions stay accurate for
the few thousand predictions in this project without depending on SciPy.
"""

import math

import numpy as np


def _log_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _log_sum(values: list[float]) -> float:
    peak = max(values)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, math.exp(_log_sum([_log_pmf(i, n, p) for i in range(k + 1)])))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided confidence interval for a binomial proportion k / n."""
    if not 0 <= k <= n or n < 1:
        raise ValueError("Expected 0 <= k <= n and n >= 1")

    def solve(too_small):
        # Bisection on p in [0, 1]; too_small(p) is True below the bound and False above it.
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2
            low, high = (middle, high) if too_small(middle) else (low, middle)
        return (low + high) / 2

    # Lower bound: the p at which observing k or more successes has probability alpha / 2.
    lower = 0.0 if k == 0 else solve(lambda p: 1.0 - binomial_cdf(k - 1, n, p) < alpha / 2)
    # Upper bound: the p at which observing k or fewer successes has probability alpha / 2.
    upper = 1.0 if k == n else solve(lambda p: binomial_cdf(k, n, p) > alpha / 2)
    return lower, upper


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired per-image correctness of two classifiers."""
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b
    # Under the null hypothesis each discordant image favours either model with probability 1/2.
    p_value = 1.0 if discordant == 0 else min(1.0, 2 * binomial_cdf(min(only_a, only_b), discordant, 0.5))
    return {"only_a_correct": only_a, "only_b_correct": only_b, "discordant": discordant, "p_value": p_value}


def confusion(labels: np.ndarray, predicted: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns predictions."""
    return np.bincount(num_classes * labels + predicted, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def accuracy_and_macro_f1(labels: np.ndarray, predicted: np.ndarray, num_classes: int) -> tuple[float, float]:
    matrix = confusion(labels, predicted, num_classes)
    correct = np.diag(matrix)
    # Same definition as train.classification_metrics: F1 = 2TP / (support + predicted).
    f1 = 2 * correct / np.maximum(matrix.sum(axis=1) + matrix.sum(axis=0), 1)
    return correct.sum() / matrix.sum(), f1.mean()


def paired_bootstrap(labels: np.ndarray, predicted_a: np.ndarray, predicted_b: np.ndarray | None,
                     num_classes: int, resamples: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CIs for accuracy and macro-F1, and for B minus A when B is given.

    The same resampled images are used for both models, so the difference CI accounts for pairing.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws = {"accuracy_a": [], "macro_f1_a": [], "accuracy_delta": [], "macro_f1_delta": []}
    for _ in range(resamples):
        index = rng.integers(0, n, n)
        accuracy_a, f1_a = accuracy_and_macro_f1(labels[index], predicted_a[index], num_classes)
        draws["accuracy_a"].append(accuracy_a)
        draws["macro_f1_a"].append(f1_a)
        if predicted_b is not None:
            accuracy_b, f1_b = accuracy_and_macro_f1(labels[index], predicted_b[index], num_classes)
            draws["accuracy_delta"].append(accuracy_b - accuracy_a)
            draws["macro_f1_delta"].append(f1_b - f1_a)
    quantiles = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {name: [float(value) for value in np.percentile(values, quantiles)]
            for name, values in draws.items() if values}
