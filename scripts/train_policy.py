#!/usr/bin/env python3
"""
Train the SLM policy with BC + DPO preference + consistency objectives.

Usage:
    python scripts/train_policy.py \
        --config configs/webarena/noisy.yaml \
        --output outputs/policy/webarena_noisy \
        --overrides training.num_epochs=3 training.batch_size=4

This is the main policy training script that combines:
1. Behavior Cloning (BC) on teacher demonstrations
2. DPO Preference distillation (optional, after candidate generation)
3. Consistency regularization (optional)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.datasets import BCDataset, ConsistencyDataset, PreferenceDataset
from r2v.data.trajectory import TrajectoryStore
from r2v.models.policy import PolicyModel
from r2v.training.bc_trainer import BCTrainer
from r2v.training.consistency import ConsistencyRegularizer
from r2v.training.preference_trainer import PreferenceTrainer
from r2v.utils.config import config_to_dict, load_config
from r2v.utils.logging import JSONLLogger, init_wandb, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train SLM policy")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--stage", choices=["bc", "preference", "all"], default="all")
    parser.add_argument("--trajectories", type=str, help="Path to trajectory JSONL")
    parser.add_argument("--preference-data", type=str, help="Path to preference pairs JSONL")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    from r2v.utils.config import save_config
    save_config(cfg, output_dir / "config.yaml")

    # Logging
    jsonl_log = JSONLLogger(output_dir / "training_log.jsonl")
    jsonl_log.log_config(config_to_dict(cfg))

    wandb_mode = cfg.get("logging", {}).get("wandb_mode", "online")
    init_wandb(
        project=cfg.get("logging", {}).get("project", "r2v-agent"),
        name=f"policy_{cfg.get('data', {}).get('benchmark', 'unknown')}",
        config=config_to_dict(cfg),
        tags=["policy", cfg.get("data", {}).get("benchmark", "unknown")],
        mode=wandb_mode,
    )

    # Load policy model
    logger.info("Loading policy model...")
    policy = PolicyModel(
        model_name=cfg.policy.model_name,
        lora_r=cfg.policy.get("lora_r", 64),
        lora_alpha=cfg.policy.get("lora_alpha", 128),
        lora_dropout=cfg.policy.get("lora_dropout", 0.05),
        load_in_4bit=cfg.policy.get("load_in_4bit", True),
    )

    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        policy.load(args.resume)

    # ── Stage 1: Behavior Cloning ──
    if args.stage in ("bc", "all"):
        logger.info("=== Stage 1: Behavior Cloning ===")

        traj_path = args.trajectories or cfg.get("data", {}).get("trajectory_file", "")
        if not traj_path:
            logger.error("No trajectory file specified. Use --trajectories or config data.trajectory_file")
            sys.exit(1)

        store = TrajectoryStore(traj_path)
        episodes = store.load_all()
        logger.info(f"Loaded {len(episodes)} episodes for BC training")

        bc_dataset = BCDataset(episodes, policy.tokenizer, max_length=cfg.training.get("max_seq_len", 2048))

        # Split train/val
        val_frac = cfg.training.get("val_fraction", 0.1)
        val_size = max(1, int(len(bc_dataset) * val_frac))
        train_size = len(bc_dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(bc_dataset, [train_size, val_size])

        bc_trainer = BCTrainer(
            model=policy,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=str(output_dir / "bc"),
            num_epochs=cfg.training.get("num_epochs", 3),
            batch_size=cfg.training.get("batch_size", 4),
            learning_rate=cfg.training.get("lr", 2e-5),
            gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 4),
            max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
            warmup_ratio=cfg.training.get("warmup_ratio", 0.1),
            eval_steps=cfg.training.get("eval_steps", 100),
            save_steps=cfg.training.get("save_steps", 500),
        )

        bc_trainer.train()
        logger.info("BC training complete")

    # ── Stage 2: DPO Preference Training ──
    if args.stage in ("preference", "all"):
        pref_path = args.preference_data or cfg.get("data", {}).get("preference_file", "")
        if not pref_path:
            logger.warning("No preference data specified, skipping DPO stage")
        else:
            logger.info("=== Stage 2: DPO Preference Training ===")

            # Load preference pairs (expects JSONL with chosen/rejected)
            import json
            pairs = []
            with open(pref_path) as f:
                for line in f:
                    pairs.append(json.loads(line))

            pref_dataset = PreferenceDataset(
                pairs, policy.tokenizer, max_length=cfg.training.get("max_seq_len", 2048)
            )

            pref_trainer = PreferenceTrainer(
                model=policy,
                train_dataset=pref_dataset,
                output_dir=str(output_dir / "preference"),
                num_epochs=cfg.training.get("preference_epochs", 1),
                batch_size=cfg.training.get("batch_size", 4),
                learning_rate=cfg.training.get("preference_lr", 5e-6),
                beta=cfg.training.get("dpo_beta", 0.1),
            )

            pref_trainer.train()
            logger.info("DPO preference training complete")

    # Save final model
    final_path = output_dir / "final"
    policy.save(str(final_path))
    logger.info(f"Final policy saved to {final_path}")

    jsonl_log.log("training_complete", {
        "output_dir": str(output_dir),
        "final_model": str(final_path),
    })


if __name__ == "__main__":
    main()
