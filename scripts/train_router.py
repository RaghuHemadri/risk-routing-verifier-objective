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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def load_router_features(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load router training features from JSONL.

    Each line: {"features": [...], "slm_success": 0/1, "cost": 1.0}

    Returns:
        features, labels (SLM success), costs
    """
    features, labels, costs = [], [], []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            features.append(record["features"])
            labels.append(record["slm_success"])
            costs.append(record.get("cost", 1.0))

    return np.array(features), np.array(labels), np.array(costs)


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

    # Load features
    features, labels, costs = load_router_features(args.features)
    logger.info(f"Loaded {len(features)} router training samples")
    logger.info(f"  SLM success rate: {labels.mean():.3f}")
    logger.info(f"  Mean cost: {costs.mean():.3f}")

    # Convert to tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    c = torch.tensor(costs, dtype=torch.float32)

    # Split into train/val
    n = len(X)
    val_size = max(1, int(n * 0.15))
    perm = torch.randperm(n)
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]

    train_dataset = torch.utils.data.TensorDataset(X[train_idx], y[train_idx], c[train_idx])
    val_dataset = torch.utils.data.TensorDataset(X[val_idx], y[val_idx], c[val_idx])

    # Create router
    rcfg = cfg.get("router", {})
    input_dim = features.shape[1]
    router = Router(
        input_dim=input_dim,
        hidden_dims=[rcfg.get("hidden_dim", 128)] * rcfg.get("num_layers", 2),
        dropout=rcfg.get("dropout", 0.1),
    )

    # Train
    trainer = RouterTrainer(
        model=router,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=str(output_dir),
        num_epochs=rcfg.get("num_epochs", 50),
        batch_size=rcfg.get("batch_size", 256),
        learning_rate=rcfg.get("lr", 1e-3),
        dual_learning_rate=rcfg.get("dual_lr", 1e-2),
        cvar_alpha=rcfg.get("cvar_alpha", 0.3),
        cvar_epsilon=rcfg.get("cvar_epsilon", 0.3),
    )

    trainer.train()

    # Post-hoc temperature scaling
    logger.info("Applying temperature scaling...")
    temp_scaler = TemperatureScaling()

    router.eval()
    with torch.no_grad():
        val_features = X[val_idx]
        val_labels = y[val_idx]
        val_logits = router.get_logits(val_features)

    temp_scaler.fit(val_logits, val_labels)
    logger.info(f"Optimal temperature: {temp_scaler.temperature.item():.3f}")

    # Save
    torch.save({
        "router_state_dict": router.state_dict(),
        "temperature": temp_scaler.temperature.item(),
        "input_dim": input_dim,
        "config": config_to_dict(cfg),
    }, output_dir / "router_final.pt")

    logger.info(f"Router saved to {output_dir / 'router_final.pt'}")
    jsonl_log.log("training_complete", {"output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
