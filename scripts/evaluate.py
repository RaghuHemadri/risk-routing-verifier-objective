#!/usr/bin/env python3
"""
Evaluate trained R2V-Agent and baselines on benchmarks.

Usage:
    python scripts/evaluate.py \
        --config configs/webarena/noisy.yaml \
        --policy-path outputs/policy/webarena_noisy/final \
        --router-path outputs/router/webarena/router_final.pt \
        --output results/webarena_noisy \
        --seeds 1 2 3 4 5 \
        --methods r2v slm_only llm_only entropy_router

Runs evaluation across seeds and conditions, producing:
- Structured JSON results (for LLM paper writing)
- CSV tables (for plotting)
- LaTeX tables (for paper)
- Statistical comparisons with CIs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.evaluation.calibration import compute_calibration_metrics
from r2v.evaluation.metrics import R2VEvaluator, compare_methods
from r2v.evaluation.robustness import (
    compute_bottom_k_sr,
    compute_cvar_failure,
    compute_per_seed_success_rates,
    compute_robustness_gap,
    compute_worst_seed_sr,
)
from r2v.evaluation.statistical import bootstrap_ci
from r2v.utils.config import config_to_dict, load_config, save_config
from r2v.utils.logging import JSONLLogger, setup_logging
from r2v.utils.results import (
    ComparisonResult,
    EvalResult,
    ResultsBundle,
    ResultsManager,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate R2V-Agent")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--router-path", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--methods", nargs="+",
                       default=["r2v", "slm_only", "llm_only", "entropy_router"])
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def run_evaluation_for_method(
    method: str,
    cfg,
    seed: int,
    num_episodes: int,
    policy_path: str | None,
    router_path: str | None,
    logger,
) -> dict:
    """Run evaluation for a single method/seed combination.

    NOTE: This is the integration point with the actual benchmark env.
    Replace the placeholder with actual environment interaction.
    """
    logger.info(f"  Evaluating {method} (seed={seed})...")

    # ── Placeholder: integrate with actual benchmark env ──
    # For full implementation:
    #   1. Load the agent (R2VAgent, SLMOnlyAgent, etc.) with the trained checkpoints
    #   2. Run agent.run_episode(env) for each task
    #   3. Collect per-task results

    # Placeholder results (replace with actual evaluation)
    rng = np.random.RandomState(seed * 1000 + hash(method) % 1000)

    # Simulated per-task results
    successes = rng.binomial(1, 0.7, size=num_episodes)
    costs = rng.uniform(0.5, 2.0, size=num_episodes)
    latencies = rng.uniform(1.0, 10.0, size=num_episodes)

    # Router-specific metrics (simulated)
    router_decisions = rng.binomial(1, 0.3, size=num_episodes) if method == "r2v" else None
    verifier_scores = rng.uniform(0.0, 1.0, size=num_episodes)

    return {
        "successes": successes,
        "costs": costs,
        "latencies": latencies,
        "router_decisions": router_decisions,
        "verifier_scores": verifier_scores,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    benchmark = cfg.get("data", {}).get("benchmark", "unknown")
    condition = "noisy" if cfg.get("perturbations", {}).get("tool_flakiness", {}).get("enabled", False) else "clean"
    num_episodes = args.num_episodes or cfg.get("evaluation", {}).get("num_episodes", 100)

    jsonl_log = JSONLLogger(output_dir / "evaluation_log.jsonl")
    results_manager = ResultsManager(output_dir / "structured_results")

    bundle = ResultsBundle(
        experiment_name=f"eval_{benchmark}_{condition}",
        config=config_to_dict(cfg),
    )

    # ── Run evaluation for all methods × seeds ──
    all_results = {}  # method → seed → results_dict
    for method in args.methods:
        logger.info(f"=== Evaluating: {method} ===")
        all_results[method] = {}

        for seed in args.seeds:
            results = run_evaluation_for_method(
                method=method,
                cfg=cfg,
                seed=seed,
                num_episodes=num_episodes,
                policy_path=args.policy_path,
                router_path=args.router_path,
                logger=logger,
            )
            all_results[method][seed] = results

            # Compute metrics for this seed
            successes = results["successes"]
            sr, sr_lo, sr_hi = bootstrap_ci(successes)

            eval_result = EvalResult(
                method=method,
                benchmark=benchmark,
                condition=condition,
                seed=seed,
                success_rate=sr,
                success_rate_ci=(sr_lo, sr_hi),
                avg_cost=float(np.mean(results["costs"])),
                avg_latency=float(np.mean(results["latencies"])),
                llm_call_rate=float(np.mean(results["router_decisions"])) if results["router_decisions"] is not None else None,
            )
            bundle.eval_results.append(eval_result)

            jsonl_log.log_evaluation({
                "method": method,
                "seed": seed,
                "success_rate": sr,
                "ci": (sr_lo, sr_hi),
                "avg_cost": float(np.mean(results["costs"])),
            })

    # ── Compute robustness metrics ──
    logger.info("Computing robustness metrics...")
    for method in args.methods:
        per_seed_srs = {}
        for seed, res in all_results[method].items():
            per_seed_srs[seed] = float(np.mean(res["successes"]))

        worst = min(per_seed_srs.values())
        avg = np.mean(list(per_seed_srs.values()))

        # Update eval results with robustness metrics
        for er in bundle.eval_results:
            if er.method == method:
                er.worst_seed_sr = worst
                er.robustness_gap = float(avg - worst)

    # ── Statistical comparisons ──
    logger.info("Computing statistical comparisons...")
    if "r2v" in args.methods:
        for other in args.methods:
            if other == "r2v":
                continue

            # Aggregate successes across seeds for comparison
            r2v_successes = np.concatenate([
                all_results["r2v"][s]["successes"] for s in args.seeds
            ])
            other_successes = np.concatenate([
                all_results[other][s]["successes"] for s in args.seeds
            ])

            comparison = compare_methods(
                "r2v", other,
                r2v_successes, other_successes,
                benchmark=benchmark,
                condition=condition,
            )

            bundle.comparisons.append(ComparisonResult(
                method_a="r2v",
                method_b=other,
                benchmark=benchmark,
                condition=condition,
                metric="success_rate",
                value_a=float(np.mean(r2v_successes)),
                value_b=float(np.mean(other_successes)),
                difference=float(np.mean(r2v_successes) - np.mean(other_successes)),
                ci_lower=comparison.get("ci_lower", 0.0),
                ci_upper=comparison.get("ci_upper", 0.0),
                p_value=comparison.get("p_value", 1.0),
                significant=comparison.get("significant", False),
            ))

    # ── Save all results ──
    bundle_path = results_manager.save_bundle(bundle)
    logger.info(f"Results bundle saved: {bundle_path}")

    # Generate output tables
    csv_path = results_manager.generate_main_table_csv()
    logger.info(f"Main table CSV: {csv_path}")

    comp_csv = results_manager.generate_comparison_csv()
    logger.info(f"Comparisons CSV: {comp_csv}")

    llm_path = results_manager.generate_llm_summary()
    logger.info(f"LLM summary: {llm_path}")

    latex_path = results_manager.generate_latex_table()
    logger.info(f"LaTeX table: {latex_path}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
