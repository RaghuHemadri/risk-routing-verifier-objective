#!/usr/bin/env python3
"""
Train the SLM policy with BC + DPO preference objectives.

Usage:
    python scripts/train_policy.py \
        --config configs/gaia/clean.yaml \
        --output outputs/policy/gaia_clean \
        --trajectories data/trajectories/gaia_clean/trajectories.jsonl \
        --overrides training.bc.epochs=3

This is the main policy training script that combines:
1. Behavior Cloning (BC) on teacher demonstrations
2. DPO Preference distillation (optional, after candidate generation)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.datasets import BCDataset, PreferenceDataset
from r2v.data.trajectory import TrajectoryStore
from r2v.models.policy import PolicyModel
from r2v.training.bc_trainer import BCTrainer
from r2v.training.preference_trainer import PreferenceTrainer
from r2v.utils.config import config_to_dict, load_config, save_config
from r2v.utils.logging import JSONLLogger, init_wandb, setup_logging


def is_accelerate_state_checkpoint(path: str) -> bool:
    """Heuristically detect an Accelerate save_state checkpoint directory."""
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False
    state_markers = [
        "optimizer.bin", "optimizer.pt", "scheduler.bin", "scheduler.pt",
        "random_states_0.pkl", "pytorch_model.bin", "model.safetensors",
    ]
    return any((p / m).exists() for m in state_markers)


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
    save_config(cfg, output_dir / "config.yaml")

    # Logging
    jsonl_log = JSONLLogger(output_dir / "training_log.jsonl")
    jsonl_log.log_config(config_to_dict(cfg))

    log_cfg = cfg.get("logging", {})
    init_wandb(
        project=log_cfg.get("wandb_project", "r2v-agent"),
        name=f"policy_{cfg.get('data', {}).get('benchmark', 'unknown')}",
        config=config_to_dict(cfg),
        tags=["policy", cfg.get("data", {}).get("benchmark", "unknown")],
        mode=log_cfg.get("wandb_mode", "disabled"),
    )

    # ── Load policy model ──
    logger.info("Loading policy model...")
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    policy = PolicyModel(policy_cfg)

    resume_bc_state_path = None
    # Resume if specified
    if args.resume:
        if args.stage in ("bc", "all") and is_accelerate_state_checkpoint(args.resume):
            resume_bc_state_path = args.resume
            logger.info(
                f"Detected Accelerate BC state checkpoint at {args.resume}; "
                "will restore trainer state during BC training"
            )
        else:
            logger.info(f"Resuming model weights from {args.resume}")
            policy.load(args.resume)

    # ── Stage 1: Behavior Cloning ──
    if args.stage in ("bc", "all"):
        logger.info("=== Stage 1: Behavior Cloning ===")

        traj_path = args.trajectories or cfg.get("data", {}).get("trajectory_file", "")
        if not traj_path:
            logger.error("No trajectory file specified. Use --trajectories or config data.trajectory_file")
            sys.exit(1)

        # Verify trajectory file exists
        if not Path(traj_path).exists():
            logger.error(f"Trajectory file not found: {traj_path}")
            sys.exit(1)

        n_episodes = TrajectoryStore(traj_path).count()
        logger.info(f"Found {n_episodes} episodes in {traj_path}")

        # BCDataset loads from the JSONL path directly
        max_seq_len = cfg.policy.get("max_seq_len", 4096)
        bc_dataset = BCDataset(traj_path, policy.tokenizer, max_seq_len=max_seq_len)
        logger.info(f"BCDataset: {len(bc_dataset)} examples (from successful episodes)")

        # Train/val split
        val_frac = cfg.get("data", {}).get("train_ratio", 0.8)
        val_frac = 1.0 - val_frac  # convert train_ratio to val_fraction
        val_size = max(1, int(len(bc_dataset) * val_frac))
        train_size = len(bc_dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(bc_dataset, [train_size, val_size])
        logger.info(f"Split: {train_size} train, {val_size} val")

        # BCTrainer reads hyperparams from config dict
        bc_cfg = OmegaConf.to_container(cfg.training.bc, resolve=True)
        bc_trainer = BCTrainer(
            policy=policy,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            config=bc_cfg,
            output_dir=str(output_dir / "bc"),
            collate_fn=BCDataset.collate_fn,
            resume_state_path=resume_bc_state_path,
        )

        bc_trainer.train()
        logger.info("BC training complete")

    # ── Stage 2: DPO Preference Training ──
    if args.stage in ("preference", "all"):
        pref_path = args.preference_data or cfg.get("data", {}).get("preference_file", "")
        if not pref_path or not Path(pref_path).exists():
            logger.warning("No preference data found, skipping DPO stage")
        else:
            logger.info("=== Stage 2: DPO Preference Training ===")

            max_seq_len = cfg.policy.get("max_seq_len", 4096)
            pref_cfg = OmegaConf.to_container(cfg.training.preference, resolve=True)
            min_score_gap = float(pref_cfg.get("min_score_gap", 0.1))
            pref_dataset = PreferenceDataset(
                pref_path,
                policy.tokenizer,
                max_seq_len=max_seq_len,
                min_score_gap=min_score_gap,
            )
            logger.info(f"PreferenceDataset: {len(pref_dataset)} pairs")
            if hasattr(pref_dataset, "stats"):
                logger.info(f"PreferenceDataset stats: {pref_dataset.stats}")

            if len(pref_dataset) == 0:
                logger.error(
                    "No usable preference pairs loaded. "
                    "Likely causes: chosen==rejected for most rows, or score-gap filter too strict. "
                    f"Current min_score_gap={min_score_gap}."
                )
                logger.error(
                    "Try either regenerating candidates with distinct chosen/rejected pairs, "
                    "or temporarily lowering training.preference.min_score_gap (e.g., 0.0)."
                )
                sys.exit(1)

            pref_trainer = PreferenceTrainer(
                policy=policy,
                train_dataset=pref_dataset,
                config=pref_cfg,
                output_dir=str(output_dir / "preference"),
                collate_fn=PreferenceDataset.collate_fn,
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
