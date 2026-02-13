"""
Robustness metrics: CVaR, worst-case, and tail analysis.

Implements the robust success metrics that are central to the paper:
- min_z SR(z): worst-seed success rate
- CVaR_α(1 - S_z): tail average failure rate
- Bottom-k% seed success rate
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from r2v.data.trajectory import Episode


def compute_per_seed_success_rates(episodes: list[Episode]) -> dict[int, float]:
    """Compute success rate grouped by perturbation seed.

    Returns dict mapping seed → success rate.
    """
    seed_results: dict[int, list[float]] = defaultdict(list)

    for ep in episodes:
        seed = ep.perturbation_seed or 0
        seed_results[seed].append(float(ep.success))

    return {
        seed: np.mean(results) for seed, results in seed_results.items()
    }


def compute_worst_seed_sr(episodes: list[Episode]) -> dict[int, float]:
    """Compute per-seed success rates (for worst-seed analysis)."""
    return compute_per_seed_success_rates(episodes)


def compute_cvar_failure(
    episodes: list[Episode],
    alpha: float = 0.2,
) -> float:
    """Compute CVaR_α(1 - S_z): tail average failure rate.

    Groups episodes by perturbation seed, computes per-seed failure rate,
    then returns the average of the worst α-fraction of seeds.

    Args:
        episodes: List of evaluated episodes
        alpha: Tail probability (e.g., 0.2 = worst 20% of seeds)

    Returns:
        CVaR failure rate (lower is better)
    """
    seed_srs = compute_per_seed_success_rates(episodes)
    if not seed_srs:
        return 1.0

    # Convert to failure rates
    seed_failures = {seed: 1.0 - sr for seed, sr in seed_srs.items()}

    # Sort by failure rate (descending = worst first)
    sorted_failures = sorted(seed_failures.values(), reverse=True)

    # Take worst α-fraction
    k = max(1, int(len(sorted_failures) * alpha))
    tail_failures = sorted_failures[:k]

    return float(np.mean(tail_failures))


def compute_bottom_k_sr(
    episodes: list[Episode],
    k_fraction: float = 0.1,
) -> float:
    """Compute average success rate of the worst k% of seeds.

    Args:
        episodes: List of evaluated episodes
        k_fraction: Fraction of worst seeds (e.g., 0.1 = bottom 10%)

    Returns:
        Average success rate of worst seeds
    """
    seed_srs = compute_per_seed_success_rates(episodes)
    if not seed_srs:
        return 0.0

    sorted_srs = sorted(seed_srs.values())
    k = max(1, int(len(sorted_srs) * k_fraction))

    return float(np.mean(sorted_srs[:k]))


def compute_per_perturbation_type_sr(
    episodes: list[Episode],
) -> dict[str, float]:
    """Success rate broken down by perturbation type."""
    type_results: dict[str, list[float]] = defaultdict(list)

    for ep in episodes:
        ptype = ep.perturbation_type.value
        type_results[ptype].append(float(ep.success))

    return {
        ptype: np.mean(results) for ptype, results in type_results.items()
    }


def compute_robustness_gap(
    clean_episodes: list[Episode],
    noisy_episodes: list[Episode],
) -> float:
    """Compute the gap between clean and noisy success rates.

    A smaller gap indicates better robustness.
    """
    clean_sr = np.mean([float(ep.success) for ep in clean_episodes])
    noisy_sr = np.mean([float(ep.success) for ep in noisy_episodes])
    return clean_sr - noisy_sr


def compute_success_cost_frontier(
    episodes_by_method: dict[str, list[Episode]],
) -> list[dict[str, Any]]:
    """Compute the success-rate vs cost frontier across methods.

    Returns list of (method, success_rate, avg_cost) points for plotting.
    """
    frontier = []
    for method, episodes in episodes_by_method.items():
        sr = np.mean([float(ep.success) for ep in episodes])
        cost = np.mean([ep.total_cost for ep in episodes])
        llm_frac = np.mean([
            ep.num_llm_fallbacks / max(ep.num_steps, 1) for ep in episodes
        ])
        frontier.append({
            "method": method,
            "success_rate": sr,
            "avg_cost": cost,
            "llm_call_fraction": llm_frac,
        })

    return sorted(frontier, key=lambda x: x["avg_cost"])
