#!/usr/bin/env python3
"""
Train the risk-calibrated router with Lagrangian CVaR objective.

Usage:
    python scripts/train_router.py \
        --config configs/webarena/noisy.yaml \
        --output outputs/router/webarena \
        --features data/router_features/webarena.jsonl

The router decides when to escalate from SLM to teacher LLM
based on calibrated confidence scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.datasets import RouterDataset, RouterExample
from r2v.data.splits import load_and_split
from r2v.models.router import Router
from r2v.training.router_trainer import RouterTrainer
from r2v.utils.config import config_to_dict, load_config, save_config
from r2v.utils.logging import JSONLLogger, init_wandb, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train risk-calibrated router")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--features", type=str, required=True, help="Router features JSONL")
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Trajectory JSONL for task-level split (episode_id → task_id)")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def load_router_features(
    path: str,
    allowed_episode_ids: set[str] | None = None,
) -> list[RouterExample]:
    """Load router training features from JSONL into RouterExample objects.

    Each line: {"features": [...], "slm_success": 0/1, "cost": 1.0,
                "episode_id": ..., "step_idx": ...}

    Parameters
    ----------
    path
        JSONL file with per-step router features.
    allowed_episode_ids
        If provided, only include records whose episode_id is in this set.
    """
    examples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if allowed_episode_ids is not None:
                eid = record.get("episode_id")
                if eid is not None and eid not in allowed_episode_ids:
                    skipped += 1
                    continue
            slm_success = float(record.get("slm_success", 0))
            examples.append(RouterExample(
                features=record["features"],
                label=1.0 - slm_success,
                success=slm_success,
                perturbation_seed=record.get("perturbation_seed", 0),
                cost=record.get("cost", 1.0),
            ))
    if skipped:
        logging.getLogger(__name__).info(f"Skipped {skipped} records outside split")
    return examples


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    jsonl_log = JSONLLogger(output_dir / "training_log.jsonl")
    jsonl_log.log_config(config_to_dict(cfg))

    init_wandb(
        project=cfg.get("logging", {}).get("project", "r2v-agent"),
        name=f"router_{cfg.get('data', {}).get('benchmark', 'unknown')}",
        config=config_to_dict(cfg),
        tags=["router"],
        mode=cfg.get("logging", {}).get("wandb_mode", "online"),
    )

    # Build task-level split from trajectories (if provided)
    train_eids: set[str] | None = None
    val_eids: set[str] | None = None
    if args.trajectories and Path(args.trajectories).exists():
        max_perturbations = int(cfg.get("data", {}).get("max_perturbations_per_task", 2))
        splits = load_and_split(
            args.trajectories,
            max_perturbations_per_task=max_perturbations,
            seed=int(cfg.get("project", {}).get("seed", 42)),
        )
        train_eids = {ep.episode_id for ep in splits["train"]}
        val_eids = {ep.episode_id for ep in splits["val"]}
        logger.info(
            f"Task-level split: {len(train_eids)} train, "
            f"{len(val_eids)} val, {len(splits['test'])} test episode IDs"
        )

    train_examples = load_router_features(args.features, allowed_episode_ids=train_eids)
    val_examples = load_router_features(args.features, allowed_episode_ids=val_eids)
    logger.info(f"Router features: {len(train_examples)} train, {len(val_examples)} val samples")

    all_examples = train_examples + val_examples
    successes = [ex.success for ex in all_examples]
    logger.info(f"  SLM success rate: {sum(successes) / len(successes):.3f}")
    logger.info(f"  Mean cost: {sum(ex.cost for ex in all_examples) / len(all_examples):.3f}")

    train_dataset = RouterDataset(train_examples)
    # Val/test must use training-set normalization stats to avoid leakage.
    val_dataset = RouterDataset(val_examples, mean=train_dataset.mean, std=train_dataset.std)

    # Create router — Router takes a config dict, not keyword args.
    # Override input_features to match actual feature dimensionality from data.
    rcfg = OmegaConf.to_container(cfg.get("router", {}), resolve=True)
    input_dim = len(all_examples[0].features)

    # Build a config that tells Router the exact input dim
    router_config = dict(rcfg)
    # Override _compute_input_dim by setting policy_hidden_dim so total = input_dim
    # Simplest: just set the feature flags to False and use policy_hidden_dim as the dim
    router_config["input_features"] = {
        "verifier_score": False,
        "entropy": False,
        "step_number": False,
        "token_count": False,
        "policy_hidden_dim": input_dim - 1,  # +1 for the always-included risk_score
    }
    router = Router(router_config)
    logger.info(f"Router input_dim={input_dim} (from data)")

    # Train
    train_cfg = OmegaConf.to_container(cfg.get("training", {}).get("router", {}), resolve=True)
    trainer = RouterTrainer(
        router=router,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=train_cfg,
        output_dir=str(output_dir),
    )

    trainer.train()

    # RouterTrainer already performs post-hoc calibration on eval data.
    # Re-calibrating here can drift temperature and, if labels are mismatched,
    # push to boundary values (e.g., T=10.0).
    calibrated_temperature = float(router.temperature.item())
    logger.info(f"Using trainer-calibrated temperature: {calibrated_temperature:.3f}")

    # Save
    torch.save({
        "router_state_dict": router.state_dict(),
        "temperature": calibrated_temperature,
        "input_dim": input_dim,
        "config": config_to_dict(cfg),
        "feature_mean": train_dataset.mean.tolist(),
        "feature_std": train_dataset.std.tolist(),
    }, output_dir / "router_final.pt")

    logger.info(f"Router saved to {output_dir / 'router_final.pt'}")
    jsonl_log.log("training_complete", {"output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
