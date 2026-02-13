"""
Core evaluation metrics for R2V-Agent.

Primary metrics:
- Robust success rate: min_z SR(z) or CVaR_α(failure) over perturbation seeds
- Average success rate on clean + perturbed tasks

Secondary metrics:
- Cost: tokens, # tool calls, # LLM fallbacks, $ proxy
- Latency: wall-clock or model-call count
- Router calibration: ECE / Brier
- Safety failure rate: executed injected instructions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from r2v.data.trajectory import Episode, PerturbationType
from r2v.evaluation.robustness import (
    compute_cvar_failure,
    compute_worst_seed_sr,
    compute_bottom_k_sr,
)
from r2v.evaluation.calibration import compute_ece, compute_brier
from r2v.evaluation.statistical import (
    bootstrap_ci,
    paired_mcnemar_test,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """A single metric with confidence interval."""
    name: str
    value: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_samples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"


@dataclass
class EvaluationResult:
    """Complete evaluation results for one method on one setting."""
    method_name: str
    benchmark: str
    setting: str               # "clean" or "noisy"
    primary_metrics: dict[str, MetricResult] = field(default_factory=dict)
    secondary_metrics: dict[str, MetricResult] = field(default_factory=dict)
    per_task_results: list[dict] = field(default_factory=list)
    per_seed_results: dict[int, float] = field(default_factory=dict)
    ablation_tag: str = ""     # For ablation studies

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "benchmark": self.benchmark,
            "setting": self.setting,
            "ablation_tag": self.ablation_tag,
            "primary_metrics": {
                k: v.to_dict() for k, v in self.primary_metrics.items()
            },
            "secondary_metrics": {
                k: v.to_dict() for k, v in self.secondary_metrics.items()
            },
            "per_task_results": self.per_task_results,
            "per_seed_results": self.per_seed_results,
        }

    @property
    def summary(self) -> str:
        """One-line summary for logging."""
        primary = " | ".join(
            f"{k}={v.value:.3f}" for k, v in self.primary_metrics.items()
        )
        return f"[{self.method_name}/{self.setting}] {primary}"


class R2VEvaluator:
    """Comprehensive evaluation pipeline for R2V-Agent experiments."""

    def __init__(
        self,
        config: dict[str, Any] = None,
        cvar_alpha: float = 0.2,
        num_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ):
        self.config = config or {}
        self.cvar_alpha = cvar_alpha
        self.num_bootstrap = num_bootstrap
        self.confidence_level = confidence_level

    def evaluate(
        self,
        episodes: list[Episode],
        method_name: str,
        benchmark: str = "webarena",
        setting: str = "noisy",
        ablation_tag: str = "",
    ) -> EvaluationResult:
        """Run full evaluation on a set of episodes.

        Args:
            episodes: List of completed episodes
            method_name: Name of the method being evaluated
            benchmark: Benchmark name
            setting: "clean" or "noisy"
            ablation_tag: Optional tag for ablation studies

        Returns:
            Complete EvaluationResult with all metrics
        """
        result = EvaluationResult(
            method_name=method_name,
            benchmark=benchmark,
            setting=setting,
            ablation_tag=ablation_tag,
        )

        successes = np.array([float(ep.success) for ep in episodes])
        seeds = np.array([ep.perturbation_seed or 0 for ep in episodes])

        # ============================================================
        # Primary metrics
        # ============================================================

        # 1. Average success rate with CI
        avg_sr, ci_lo, ci_hi = bootstrap_ci(
            successes, np.mean, self.num_bootstrap, self.confidence_level
        )
        result.primary_metrics["avg_success_rate"] = MetricResult(
            name="Average Success Rate",
            value=avg_sr, ci_lower=ci_lo, ci_upper=ci_hi,
            n_samples=len(successes),
        )

        # 2. Robust success rate (worst-seed)
        seed_srs = compute_worst_seed_sr(episodes)
        result.per_seed_results = seed_srs
        if seed_srs:
            worst_sr = min(seed_srs.values())
            result.primary_metrics["worst_seed_sr"] = MetricResult(
                name="Worst-Seed Success Rate",
                value=worst_sr,
                n_samples=len(seed_srs),
                metadata={"per_seed": seed_srs},
            )

        # 3. CVaR failure rate
        if len(set(seeds)) > 1:
            cvar_fail = compute_cvar_failure(episodes, self.cvar_alpha)
            result.primary_metrics["cvar_failure"] = MetricResult(
                name=f"CVaR_{self.cvar_alpha} Failure Rate",
                value=cvar_fail,
                n_samples=len(successes),
            )

        # 4. Bottom-10% seed success rate
        if seed_srs:
            bottom_sr = compute_bottom_k_sr(episodes, k_fraction=0.1)
            result.primary_metrics["bottom_10pct_sr"] = MetricResult(
                name="Bottom-10% Seed SR",
                value=bottom_sr,
            )

        # ============================================================
        # Secondary metrics
        # ============================================================

        # 5. Cost metrics
        costs = [ep.total_cost for ep in episodes]
        llm_calls = [ep.num_llm_fallbacks for ep in episodes]

        result.secondary_metrics["avg_cost"] = MetricResult(
            name="Average Cost ($)",
            value=np.mean(costs),
            ci_lower=np.percentile(costs, 2.5),
            ci_upper=np.percentile(costs, 97.5),
        )
        result.secondary_metrics["avg_llm_fallbacks"] = MetricResult(
            name="Avg LLM Fallbacks",
            value=np.mean(llm_calls),
        )
        result.secondary_metrics["total_llm_calls"] = MetricResult(
            name="Total LLM Calls",
            value=sum(llm_calls),
        )

        # 6. Latency
        wall_times = [ep.wall_time_seconds for ep in episodes]
        result.secondary_metrics["avg_latency"] = MetricResult(
            name="Avg Latency (s)",
            value=np.mean(wall_times),
        )

        # 7. Steps per episode
        steps = [ep.num_steps for ep in episodes]
        result.secondary_metrics["avg_steps"] = MetricResult(
            name="Avg Steps/Episode",
            value=np.mean(steps),
        )

        # 8. Safety failure rate
        safety_failures = []
        for ep in episodes:
            for step in ep.steps:
                if step.label and step.label.safety_violation:
                    safety_failures.append(1.0)
                    break
            else:
                safety_failures.append(0.0)

        result.secondary_metrics["safety_failure_rate"] = MetricResult(
            name="Safety Failure Rate",
            value=np.mean(safety_failures) if safety_failures else 0.0,
            n_samples=len(safety_failures),
        )

        # 9. Self-correction stats
        corrections = [ep.num_self_corrections for ep in episodes]
        result.secondary_metrics["avg_self_corrections"] = MetricResult(
            name="Avg Self-Corrections",
            value=np.mean(corrections) if corrections else 0.0,
        )

        # ============================================================
        # Per-task results
        # ============================================================
        for ep in episodes:
            result.per_task_results.append({
                "episode_id": ep.episode_id,
                "task_id": ep.metadata.task_id,
                "template_id": ep.metadata.template_id,
                "success": ep.success,
                "num_steps": ep.num_steps,
                "cost": ep.total_cost,
                "llm_fallbacks": ep.num_llm_fallbacks,
                "self_corrections": ep.num_self_corrections,
                "wall_time": ep.wall_time_seconds,
                "perturbation_seed": ep.perturbation_seed,
                "perturbation_type": ep.perturbation_type.value,
            })

        logger.info(result.summary)
        return result

    def compare_methods(
        self,
        result_a: EvaluationResult,
        result_b: EvaluationResult,
        significance: float = 0.05,
    ) -> dict[str, Any]:
        """Statistical comparison between two methods.

        Uses paired McNemar test on per-task success.
        """
        tasks_a = {r["task_id"]: r["success"] for r in result_a.per_task_results}
        tasks_b = {r["task_id"]: r["success"] for r in result_b.per_task_results}

        common_tasks = set(tasks_a.keys()) & set(tasks_b.keys())
        if not common_tasks:
            return {"error": "No common tasks for comparison"}

        successes_a = np.array([tasks_a[t] for t in sorted(common_tasks)])
        successes_b = np.array([tasks_b[t] for t in sorted(common_tasks)])

        stat, p_value = paired_mcnemar_test(successes_a, successes_b)

        return {
            "method_a": result_a.method_name,
            "method_b": result_b.method_name,
            "metric": "success_rate",
            "sr_a": successes_a.mean(),
            "sr_b": successes_b.mean(),
            "mcnemar_statistic": stat,
            "p_value": p_value,
            "significant": p_value < significance,
            "num_common_tasks": len(common_tasks),
        }
