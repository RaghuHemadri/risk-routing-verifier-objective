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
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.datasets import BCDataset, PreferenceDataset
from r2v.data.splits import load_and_split
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
    parser.add_argument(
        "--data-fraction", type=float, default=1.0,
        help="Fraction of training tasks to use (0.0–1.0). Subsamples at the task level so all episodes for selected tasks are kept.",
    )
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

        if not Path(traj_path).exists():
            logger.error(f"Trajectory file not found: {traj_path}")
            sys.exit(1)

        max_perturbations = int(cfg.get("data", {}).get("max_perturbations_per_task", 2))
        splits = load_and_split(
            traj_path,
            max_perturbations_per_task=max_perturbations,
            seed=int(cfg.get("project", {}).get("seed", 42)),
        )
        logger.info(
            f"Task-level split: {len(splits['train'])} train, "
            f"{len(splits['val'])} val, {len(splits['test'])} test episodes"
        )

        train_episodes = splits["train"]
        if args.data_fraction < 1.0:
            rng = random.Random(int(cfg.get("project", {}).get("seed", 42)))
            task_to_eps: dict[str, list] = {}
            for ep in train_episodes:
                task_to_eps.setdefault(ep.metadata.task_id, []).append(ep)
            all_task_ids = sorted(task_to_eps.keys())
            n_tasks_keep = max(1, int(len(all_task_ids) * args.data_fraction))
            selected_tasks = set(rng.sample(all_task_ids, n_tasks_keep))
            train_episodes = [ep for tid in selected_tasks for ep in task_to_eps[tid]]
            n_success = sum(1 for ep in train_episodes if ep.success)
            logger.info(
                f"BC task subsampling (fraction={args.data_fraction}): "
                f"{n_tasks_keep}/{len(all_task_ids)} tasks, "
                f"{len(train_episodes)}/{len(splits['train'])} episodes "
                f"({n_success} success, {len(train_episodes) - n_success} failure)"
            )

        max_seq_len = cfg.policy.get("max_seq_len", 4096)
        train_ds = BCDataset(tokenizer=policy.tokenizer, max_seq_len=max_seq_len, episodes=train_episodes)
        val_ds = BCDataset(tokenizer=policy.tokenizer, max_seq_len=max_seq_len, episodes=splits["val"])
        logger.info(f"BCDataset: {len(train_ds)} train, {len(val_ds)} val examples (from successful episodes)")

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
            if args.stage == "all":
                logger.info("DPO initialization: using in-memory policy state after Stage 1 BC")
            elif args.resume:
                logger.info(f"DPO initialization: using weights loaded from --resume ({args.resume})")
            else:
                logger.info("DPO initialization: using base model weights (no BC checkpoint loaded)")

            # Build train-split episode_id allowlist from trajectories
            traj_path = args.trajectories or cfg.get("data", {}).get("trajectory_file", "")
            train_episode_ids: set[str] | None = None
            val_episode_ids: set[str] | None = None
            if traj_path and Path(traj_path).exists():
                max_perturbations = int(cfg.get("data", {}).get("max_perturbations_per_task", 2))
                pref_splits = load_and_split(
                    traj_path,
                    max_perturbations_per_task=max_perturbations,
                    seed=int(cfg.get("project", {}).get("seed", 42)),
                )
                train_episode_ids = {ep.episode_id for ep in pref_splits["train"]}
                val_episode_ids = {ep.episode_id for ep in pref_splits["val"]}
                logger.info(
                    f"DPO split filter: {len(train_episode_ids)} train, "
                    f"{len(val_episode_ids)} val episode IDs"
                )

            max_seq_len = cfg.policy.get("max_seq_len", 4096)
            pref_cfg = OmegaConf.to_container(cfg.training.preference, resolve=True)
            min_score_gap = float(pref_cfg.get("min_score_gap", 0.1))
            pref_train_dataset = PreferenceDataset(
                pref_path,
                policy.tokenizer,
                max_seq_len=max_seq_len,
                min_score_gap=min_score_gap,
                allowed_episode_ids=train_episode_ids,
            )
            pref_val_dataset = PreferenceDataset(
                pref_path,
                policy.tokenizer,
                max_seq_len=max_seq_len,
                min_score_gap=min_score_gap,
                allowed_episode_ids=val_episode_ids,
            ) if val_episode_ids else None
            logger.info(f"PreferenceDataset: {len(pref_train_dataset)} train pairs")
            if pref_val_dataset is not None:
                logger.info(f"PreferenceDataset: {len(pref_val_dataset)} val pairs")
            if hasattr(pref_train_dataset, "stats"):
                logger.info(f"PreferenceDataset stats: {pref_train_dataset.stats}")

            if len(pref_train_dataset) == 0:
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
                train_dataset=pref_train_dataset,
                eval_dataset=pref_val_dataset,
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
