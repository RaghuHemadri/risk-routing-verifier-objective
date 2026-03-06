"""
Router calibration metrics: ECE and Brier score.
"""

from __future__ import annotations

import numpy as np


def compute_ece(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.

    Measures how well the router's predicted fallback probabilities
    match the actual fallback necessity.

    ECE = Σ_b (|B_b| / N) |acc(B_b) - conf(B_b)|
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(predicted_probs)

    for i in range(num_bins):
        mask = (predicted_probs >= bin_boundaries[i]) & (
            predicted_probs < bin_boundaries[i + 1]
        )
        if mask.sum() == 0:
            continue

        bin_acc = true_labels[mask].mean()
        bin_conf = predicted_probs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Brier = (1/N) Σ (p_i - y_i)^2
    Lower is better. Decomposes into calibration + refinement.
    """
    return float(np.mean((predicted_probs - true_labels) ** 2))


def compute_calibration_metrics(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    num_bins: int = 15,
) -> dict[str, float]:
    """Compute all calibration metrics at once."""
    return {
        "ece": compute_ece(predicted_probs, true_labels, num_bins),
        "brier": compute_brier(predicted_probs, true_labels),
        "mean_confidence": float(predicted_probs.mean()),
        "mean_accuracy": float(true_labels.mean()),
    }
