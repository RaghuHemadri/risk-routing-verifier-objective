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
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.datasets import RouterDataset, RouterExample
from r2v.models.router import Router, TemperatureScaling
from r2v.training.router_trainer import RouterTrainer
from r2v.utils.config import config_to_dict, load_config, save_config
from r2v.utils.logging import JSONLLogger, init_wandb, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train risk-calibrated router")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--features", type=str, required=True, help="Router features JSONL")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def load_router_features(path: str) -> list[RouterExample]:
    """Load router training features from JSONL into RouterExample objects.

    Each line: {"features": [...], "slm_success": 0/1, "cost": 1.0,
                "episode_id": ..., "step_idx": ...}

    The router needs:
    - features: 13-dim feature vector
    - label: routing decision (1.0 = should fallback to LLM = SLM failed)
    - success: episode-level SLM success indicator (for CVaR)
    - perturbation_seed: for per-seed CVaR (default 0 if not present)
    - cost: step cost
    """
    examples = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            slm_success = float(record.get("slm_success", 0))
            examples.append(RouterExample(
                features=record["features"],
                label=1.0 - slm_success,  # label=1 means "should have used LLM"
                success=slm_success,
                perturbation_seed=record.get("perturbation_seed", 0),
                cost=record.get("cost", 1.0),
            ))
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

    # Load features into RouterExample objects (dict-style __getitem__)
    examples = load_router_features(args.features)
    logger.info(f"Loaded {len(examples)} router training samples")

    successes = [ex.success for ex in examples]
    logger.info(f"  SLM success rate: {sum(successes) / len(successes):.3f}")
    logger.info(f"  Mean cost: {sum(ex.cost for ex in examples) / len(examples):.3f}")

    # Split into train/val
    n = len(examples)
    val_size = max(1, int(n * 0.15))
    perm = torch.randperm(n).tolist()
    train_examples = [examples[i] for i in perm[val_size:]]
    val_examples = [examples[i] for i in perm[:val_size]]

    train_dataset = RouterDataset(train_examples)
    val_dataset = RouterDataset(val_examples)

    # Create router — Router takes a config dict, not keyword args.
    # Override input_features to match actual feature dimensionality from data.
    rcfg = OmegaConf.to_container(cfg.get("router", {}), resolve=True)
    input_dim = len(examples[0].features)

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

    # Post-hoc temperature scaling
    logger.info("Applying temperature scaling...")
    temp_scaler = TemperatureScaling()

    router.eval()
    device = next(router.parameters()).device
    with torch.no_grad():
        val_feats = torch.tensor(
            [ex.features for ex in val_examples], dtype=torch.float32
        ).to(device)
        val_labels_np = np.array([ex.success for ex in val_examples])
        # Router.mlp gives raw logits before temperature/sigmoid
        val_logits = router.mlp(val_feats).squeeze(-1).cpu().numpy()

    temp_scaler.fit(val_logits, val_labels_np)
    logger.info(f"Optimal temperature: {temp_scaler.temperature:.3f}")

    # Save
    torch.save({
        "router_state_dict": router.state_dict(),
        "temperature": temp_scaler.temperature,
        "input_dim": input_dim,
        "config": config_to_dict(cfg),
    }, output_dir / "router_final.pt")

    logger.info(f"Router saved to {output_dir / 'router_final.pt'}")
    jsonl_log.log("training_complete", {"output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
