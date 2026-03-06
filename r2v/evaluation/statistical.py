"""
Statistical testing for rigorous experiment comparison.

Implements:
- Bootstrap confidence intervals (paired bootstrap over tasks)
- McNemar's test for per-task success comparison
- Multiple comparison correction (Holm-Bonferroni)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        statistic: Function to compute on each bootstrap sample
        num_bootstrap: Number of bootstrap resamples
        confidence_level: CI level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    point_estimate = float(statistic(data))

    bootstrap_stats = np.zeros(num_bootstrap)
    for i in range(num_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        bootstrap_stats[i] = statistic(sample)

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


def paired_mcnemar_test(
    successes_a: np.ndarray,
    successes_b: np.ndarray,
) -> tuple[float, float]:
    """McNemar's test for comparing two methods on paired binary outcomes.

    Tests whether the marginal probabilities of success differ between
    the two methods. Uses exact binomial test for small samples.

    Args:
        successes_a: Binary success array for method A
        successes_b: Binary success array for method B

    Returns:
        Tuple of (test statistic, p-value)
    """
    assert len(successes_a) == len(successes_b), "Arrays must be same length"

    a_bool = successes_a.astype(bool)
    b_bool = successes_b.astype(bool)

    # Discordant pairs
    b_not_a = np.sum(~a_bool & b_bool)   # B succeeds, A fails
    a_not_b = np.sum(a_bool & ~b_bool)   # A succeeds, B fails

    n_discordant = b_not_a + a_not_b

    if n_discordant == 0:
        return 0.0, 1.0

    # Use exact binomial test for small samples
    if n_discordant < 25:
        p_value = stats.binom_test(b_not_a, n_discordant, 0.5)
        statistic = float(b_not_a)
    else:
        # Chi-squared approximation (with continuity correction)
        statistic = (abs(b_not_a - a_not_b) - 1) ** 2 / (b_not_a + a_not_b)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return float(statistic), float(p_value)
