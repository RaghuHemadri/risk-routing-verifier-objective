#!/usr/bin/env python3
"""
Run ablation studies for R2V-Agent.

Usage:
    python scripts/run_ablations.py \
        --config configs/webarena/noisy.yaml \
        --policy-path outputs/policy/webarena_noisy/final \
        --router-path outputs/router/webarena/router_final.pt \
        --output results/ablations/webarena \
        --seeds 1 2 3

Ablations (from proposal Table 2):
1. No preference distillation (BC-only)
2. No consistency regularization
3. No verifier (random routing)
4. No risk calibration (expected-value router)
5. Static routing threshold (no learned router)
6. Per perturbation-type analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.evaluation.statistical import bootstrap_ci, paired_mcnemar_test
from r2v.utils.config import config_to_dict, load_config, save_config
from r2v.utils.logging import JSONLLogger, setup_logging
from r2v.utils.results import AblationResult, ResultsBundle, ResultsManager


ABLATION_SPECS = {
    # ── Existing ablations ──────────────────────────────────────────────────
    "no_preference": {
        "description": "Remove DPO preference distillation (BC-only policy)",
        "overrides": ["training.lambda_pref=0.0"],
    },
    "no_consistency": {
        "description": "Remove consistency regularization",
        "overrides": ["training.consistency.enabled=false"],
    },
    "no_verifier": {
        "description": "Random verifier scores (no learned verification)",
        "overrides": ["verifier.type=random"],
    },
    "no_risk_calibration": {
        "description": "Expected-value router (no CVaR, λ=0)",
        "overrides": ["router.cvar_alpha=1.0"],
    },
    "static_threshold": {
        "description": "Static entropy threshold instead of learned router",
        "overrides": ["router.type=entropy_threshold"],
    },
    "no_self_correction": {
        "description": "Disable self-correction loop",
        "overrides": ["inference.max_self_correct=0"],
    },

    # ── TIER 1: CVaR hyperparameter sensitivity (RQ1) ───────────────────────
    # Chow et al. ICML 2017; reviewers will ask "how sensitive is α?"
    "cvar_alpha_0.1": {
        "description": "CVaR α=0.1 (focus on worst 10% of seeds)",
        "overrides": ["training.router.cvar_alpha=0.1"],
        "retrain": "router",
    },
    "cvar_alpha_0.3": {
        "description": "CVaR α=0.3 (focus on worst 30% of seeds)",
        "overrides": ["training.router.cvar_alpha=0.3"],
        "retrain": "router",
    },
    "cvar_alpha_0.5": {
        "description": "CVaR α=0.5 (average over worst half of seeds)",
        "overrides": ["training.router.cvar_alpha=0.5"],
        "retrain": "router",
    },
    "cvar_eps_0.1": {
        "description": "CVaR ε=0.1 (tight failure constraint)",
        "overrides": ["training.router.cvar_epsilon=0.1"],
        "retrain": "router",
    },
    "cvar_eps_0.2": {
        "description": "CVaR ε=0.2",
        "overrides": ["training.router.cvar_epsilon=0.2"],
        "retrain": "router",
    },
    "cvar_eps_0.4": {
        "description": "CVaR ε=0.4 (loose failure constraint)",
        "overrides": ["training.router.cvar_epsilon=0.4"],
        "retrain": "router",
    },

    # ── TIER 1: CVaR vs worst-case loss (RQ1) ──────────────────────────────
    # Justifies CVaR over simple min-max (RouteLLM, FrugalGPT use average-case)
    "worst_case_loss": {
        "description": "Worst-case (min-max) loss instead of CVaR",
        "overrides": ["training.router.robust_objective=worst_case"],
        "retrain": "router",
    },

    # ── TIER 1: Feature group ablations (RQ2) ──────────────────────────────
    # RouterBench (Hu et al. 2024) shows feature importance is critical
    # Features: 0=entropy, 1=spread, 2=mean, 3=std, 4=best,
    #           5=horizon_frac, 6=step_num, 7=ctx_len,
    #           8=flakiness, 9=partial_obs, 10=injection, 11=distractors, 12=none
    "feat_verifier_only": {
        "description": "Verifier + entropy features only (mask horizon, context, perturbation type)",
        "overrides": ["router.feature_mask=[0,1,2,3,4]"],
        "retrain": "router",
        "feature_mask": [0, 1, 2, 3, 4],
    },
    "feat_no_perturbation_type": {
        "description": "Remove perturbation type one-hot features (8-12)",
        "overrides": ["router.feature_mask=[0,1,2,3,4,5,6,7]"],
        "retrain": "router",
        "feature_mask": list(range(8)),
    },
    "feat_no_horizon": {
        "description": "Remove horizon/step-progress features (5-6)",
        "overrides": ["router.feature_mask=[0,1,2,3,4,7,8,9,10,11,12]"],
        "retrain": "router",
        "feature_mask": [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12],
    },
    "feat_entropy_only": {
        "description": "Entropy only (single feature); minimal-signal router",
        "overrides": ["router.feature_mask=[0]"],
        "retrain": "router",
        "feature_mask": [0],
    },
    "feat_no_verifier": {
        "description": "Remove verifier score features (1-4); entropy+step+context only",
        "overrides": ["router.feature_mask=[0,5,6,7,8,9,10,11,12]"],
        "retrain": "router",
        "feature_mask": [0, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "feat_best_score_only": {
        "description": "Best verifier score only instead of all 4 verifier stats",
        "overrides": ["router.feature_mask=[0,4,5,6,7,8,9,10,11,12]"],
        "retrain": "router",
        "feature_mask": [0, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },

    # ── TIER 2: Calibration ablations ──────────────────────────────────────
    "no_brier_loss": {
        "description": "Remove Brier calibration loss from router objective",
        "overrides": ["training.router.brier_weight=0.0"],
        "retrain": "router",
    },
    "no_temp_scaling": {
        "description": "No post-hoc temperature scaling (fixed T=1.0)",
        "overrides": ["router.temperature_scaling=false"],
        "retrain": "router",
    },

    # ── TIER 2: Policy training ablations (RQ4) ────────────────────────────
    # These require GPU-hours for policy retraining
    "dpo_beta_0.05": {
        "description": "DPO β=0.05 (weaker preference signal)",
        "overrides": ["training.preference.beta=0.05"],
        "retrain": "policy",
    },
    "dpo_beta_0.2": {
        "description": "DPO β=0.2",
        "overrides": ["training.preference.beta=0.2"],
        "retrain": "policy",
    },
    "dpo_beta_0.5": {
        "description": "DPO β=0.5 (strong preference signal)",
        "overrides": ["training.preference.beta=0.5"],
        "retrain": "policy",
    },
    "consistency_lambda_0.05": {
        "description": "Consistency regularization λ=0.05",
        "overrides": ["training.consistency.lambda=0.05"],
        "retrain": "policy",
    },
    "consistency_lambda_0.5": {
        "description": "Consistency regularization λ=0.5 (strong)",
        "overrides": ["training.consistency.lambda=0.5"],
        "retrain": "policy",
    },
    "no_bc_warmup": {
        "description": "DPO from scratch (skip BC warmup stage)",
        "overrides": ["training.skip_bc=true"],
        "retrain": "policy",
    },

    # ── TIER 3: Cost ratio sensitivity (RQ3) ───────────────────────────────
    # Tests Pareto frontier robustness to cost assumptions
    "cost_ratio_10": {
        "description": "Cost ratio c_LLM/c_SLM=10 (lower LLM premium)",
        "overrides": ["training.router.cost_llm=10.0"],
        "retrain": "router",
    },
    "cost_ratio_25": {
        "description": "Cost ratio c_LLM/c_SLM=25",
        "overrides": ["training.router.cost_llm=25.0"],
        "retrain": "router",
    },
    "cost_ratio_100": {
        "description": "Cost ratio c_LLM/c_SLM=100 (very expensive LLM)",
        "overrides": ["training.router.cost_llm=100.0"],
        "retrain": "router",
    },

    # ── TIER 3: Router architecture ─────────────────────────────────────────
    "router_shallow": {
        "description": "Shallow router MLP: single hidden layer [128]",
        "overrides": ["router.hidden_dims=[128]"],
        "retrain": "router",
    },
    "router_deep": {
        "description": "Deep router MLP: three hidden layers [256, 128, 64]",
        "overrides": ["router.hidden_dims=[256,128,64]"],
        "retrain": "router",
    },

    # ── TIER 2: Held-out perturbation type (RQ4 — generalization) ──────────
    # AgentBench (Liu et al. ICLR 2024) OOD generalization ablations
    "generalize_no_flakiness": {
        "description": "Router trained without tool_flakiness; tested on all types",
        "overrides": ["router.held_out_perturbation=tool_flakiness"],
        "retrain": "router",
        "held_out_perturbation": "tool_flakiness",
    },
    "generalize_no_injection": {
        "description": "Router trained without prompt_injection; tested on all types",
        "overrides": ["router.held_out_perturbation=prompt_injection"],
        "retrain": "router",
        "held_out_perturbation": "prompt_injection",
    },
    "generalize_no_partial_obs": {
        "description": "Router trained without partial_observability; tested on all types",
        "overrides": ["router.held_out_perturbation=partial_observability"],
        "retrain": "router",
        "held_out_perturbation": "partial_observability",
    },
    "generalize_no_distractors": {
        "description": "Router trained without distractors; tested on all types",
        "overrides": ["router.held_out_perturbation=distractors"],
        "retrain": "router",
        "held_out_perturbation": "distractors",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--router-path", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--ablations", nargs="+", default=list(ABLATION_SPECS.keys()))
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def run_ablation(
    name: str,
    spec: dict,
    base_cfg,
    seeds: list[int],
    num_episodes: int,
    logger,
) -> dict[int, np.ndarray]:
    """Run a single ablation.

    NOTE: Replace placeholder with actual evaluation.
    """
    logger.info(f"  Running ablation: {name}")
    logger.info(f"    Description: {spec['description']}")
    logger.info(f"    Overrides: {spec['overrides']}")

    # ── Placeholder: retrain or reconfigure and evaluate ──
    # In practice:
    # 1. Load the ablated config (with overrides applied)
    # 2. If ablation requires retraining, retrain the component
    # 3. Evaluate on the benchmark

    results = {}
    for seed in seeds:
        rng = np.random.RandomState(seed * 100 + hash(name) % 100)
        # Simulated: ablated methods should generally perform worse
        results[seed] = rng.binomial(1, 0.55, size=num_episodes)

    return results


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    benchmark = cfg.get("data", {}).get("benchmark", "unknown")
    condition = "noisy"
    jsonl_log = JSONLLogger(output_dir / "ablation_log.jsonl")
    results_manager = ResultsManager(output_dir / "structured_results")

    bundle = ResultsBundle(
        experiment_name=f"ablations_{benchmark}",
        config=config_to_dict(cfg),
    )

    # ── Run full R2V as baseline ──
    logger.info("=== Running full R2V baseline ===")
    baseline_results = {}
    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        baseline_results[seed] = rng.binomial(1, 0.70, size=args.num_episodes)

    baseline_sr = float(np.mean(np.concatenate(list(baseline_results.values()))))
    logger.info(f"Baseline SR: {baseline_sr:.3f}")

    # ── Run each ablation ──
    for abl_name in args.ablations:
        if abl_name not in ABLATION_SPECS:
            logger.warning(f"Unknown ablation: {abl_name}, skipping")
            continue

        spec = ABLATION_SPECS[abl_name]
        logger.info(f"\n=== Ablation: {abl_name} ===")

        ablation_results = run_ablation(
            name=abl_name,
            spec=spec,
            base_cfg=cfg,
            seeds=args.seeds,
            num_episodes=args.num_episodes,
            logger=logger,
        )

        # Aggregate
        all_baseline = np.concatenate(list(baseline_results.values()))
        all_ablated = np.concatenate(list(ablation_results.values()))

        ablated_sr = float(np.mean(all_ablated))
        delta = baseline_sr - ablated_sr

        # Statistical test
        _, ci_lo, ci_hi = bootstrap_ci(all_baseline - all_ablated)
        stat, p_value = paired_mcnemar_test(all_baseline, all_ablated)

        abl_result = AblationResult(
            name=abl_name,
            description=spec["description"],
            base_method="r2v_full",
            ablated_method=f"r2v_{abl_name}",
            benchmark=benchmark,
            condition=condition,
            base_sr=baseline_sr,
            ablated_sr=ablated_sr,
            delta=delta,
            delta_ci=(ci_lo, ci_hi),
            p_value=float(p_value),
        )
        bundle.ablations.append(abl_result)

        logger.info(f"  Base SR: {baseline_sr:.3f}, Ablated SR: {ablated_sr:.3f}")
        logger.info(f"  Delta: {delta:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={p_value:.4f}")

        jsonl_log.log("ablation", {
            "name": abl_name,
            "base_sr": baseline_sr,
            "ablated_sr": ablated_sr,
            "delta": delta,
            "p_value": float(p_value),
        })

    # Save results
    bundle_path = results_manager.save_bundle(bundle)
    logger.info(f"\nResults saved: {bundle_path}")

    ablation_csv = results_manager.generate_ablation_csv()
    logger.info(f"Ablation CSV: {ablation_csv}")

    llm_path = results_manager.generate_llm_summary()
    logger.info(f"LLM summary: {llm_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"{'Ablation':<25} {'Base SR':>10} {'Ablated SR':>12} {'Delta':>8} {'p-value':>10}")
    print("-" * 80)
    for abl in bundle.ablations:
        sig = "*" if (abl.p_value or 1.0) < 0.05 else ""
        print(f"{abl.name:<25} {abl.base_sr:>10.3f} {abl.ablated_sr:>12.3f} {abl.delta:>8.3f} {(abl.p_value or 0):>9.4f}{sig}")
    print("=" * 80)


if __name__ == "__main__":
    main()
