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
import re
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
    parser.add_argument("--train-data", type=str, default=None,
                        help="Pre-split BC train JSONL (static splitting)")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Pre-split BC val JSONL (static splitting)")
    parser.add_argument("--preference-data", type=str, help="Path to preference pairs JSONL (single file, needs runtime filtering)")
    parser.add_argument("--pref-train-data", type=str, default=None,
                        help="Pre-split preference train JSONL (static splitting)")
    parser.add_argument("--pref-val-data", type=str, default=None,
                        help="Pre-split preference val JSONL (static splitting)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--data-fraction", type=float, default=1.0,
        help="(Deprecated — use --data-fraction on save_static_splits.py instead.) "
             "Ignored when --train-data is provided.",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()

def model_type_tag(model_name: str) -> str:
    """Create a filesystem-safe short model tag for checkpoint directories."""
    # Keep the right-most segment so HF org prefixes do not bloat folder names.
    short_name = str(model_name).split("/")[-1].strip().lower()
    tag = re.sub(r"[^a-z0-9._-]+", "-", short_name)
    tag = re.sub(r"-+", "-", tag).strip("-._")
    return tag or "unknown-model"

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

        use_static_bc = args.train_data and args.val_data
        if use_static_bc:
            logger.info("Loading static pre-split BC data files")
            splits = {
                "train": TrajectoryStore(args.train_data).load_episodes(),
                "val": TrajectoryStore(args.val_data).load_episodes(),
            }
        else:
            traj_path = args.trajectories or cfg.get("data", {}).get("trajectory_file", "")
            if not traj_path:
                logger.error("No trajectory file specified. Use --trajectories/--train-data or config data.trajectory_file")
                sys.exit(1)

            if not Path(traj_path).exists():
                logger.error(f"Trajectory file not found: {traj_path}")
                sys.exit(1)

            max_perturbations = int(cfg.get("data", {}).get("max_perturbations_per_task", 2))
            splits = load_and_split(
                traj_path,
                max_perturbations_per_task=max_perturbations,
                seed=int(cfg.get("project", {}).get("seed", 42)),
                success_only=True,
            )

        train_episodes = splits["train"]
        n_train_tasks = len({ep.metadata.task_id for ep in train_episodes})
        n_success = sum(1 for ep in train_episodes if ep.success)
        logger.info(
            f"BC data: {len(train_episodes)} train episodes across {n_train_tasks} tasks "
            f"({n_success} success, {len(train_episodes) - n_success} failure), "
            f"{len(splits['val'])} val episodes"
        )

        max_seq_len = cfg.policy.get("max_seq_len", 4096)
        train_ds = BCDataset(tokenizer=policy.tokenizer, max_seq_len=max_seq_len, episodes=train_episodes)
        val_ds = BCDataset(tokenizer=policy.tokenizer, max_seq_len=max_seq_len, episodes=splits["val"])
        logger.info(f"BCDataset: {len(train_ds)} train, {len(val_ds)} val examples (from successful episodes)")

        bc_cfg = OmegaConf.to_container(cfg.training.bc, resolve=True)
        model_name = cfg.get("policy", {}).get("model_name", "unknown-model")
        bc_output_dir = output_dir / f"bc_{model_type_tag(model_name)}"
        logger.info(f"BC checkpoint directory: {bc_output_dir}")
        bc_trainer = BCTrainer(
            policy=policy,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            config=bc_cfg,
            output_dir=str(bc_output_dir),
            collate_fn=BCDataset.collate_fn,
            resume_state_path=resume_bc_state_path,
        )

        bc_trainer.train()
        logger.info("BC training complete")

    # ── Stage 2: DPO Preference Training ──
    if args.stage in ("preference", "all"):
        use_static_pref = args.pref_train_data and Path(args.pref_train_data).exists()
        pref_path = args.preference_data or cfg.get("data", {}).get("preference_file", "")

        if not use_static_pref and (not pref_path or not Path(pref_path).exists()):
            logger.warning("No preference data found, skipping DPO stage")
        else:
            logger.info("=== Stage 2: DPO Preference Training ===")
            if args.stage == "all":
                logger.info("DPO initialization: using in-memory policy state after Stage 1 BC")
            elif args.resume:
                logger.info(f"DPO initialization: using weights loaded from --resume ({args.resume})")
            else:
                logger.info("DPO initialization: using base model weights (no BC checkpoint loaded)")

            max_seq_len = cfg.policy.get("max_seq_len", 4096)
            pref_cfg = OmegaConf.to_container(cfg.training.preference, resolve=True)
            if hasattr(cfg.training, "consistency"):
                pref_cfg["consistency"] = OmegaConf.to_container(cfg.training.consistency, resolve=True)
            min_score_gap = float(pref_cfg.get("min_score_gap", 0.1))

            if use_static_pref:
                logger.info("Loading static pre-split preference data files")
                logger.info(f"  train: {args.pref_train_data}")
                pref_train_dataset = PreferenceDataset(
                    args.pref_train_data,
                    policy.tokenizer,
                    max_seq_len=max_seq_len,
                    min_score_gap=min_score_gap,
                )
                pref_val_dataset = None
                if args.pref_val_data and Path(args.pref_val_data).exists():
                    logger.info(f"  val:   {args.pref_val_data}")
                    pref_val_dataset = PreferenceDataset(
                        args.pref_val_data,
                        policy.tokenizer,
                        max_seq_len=max_seq_len,
                        min_score_gap=min_score_gap,
                    )
            else:
                logger.info("Using runtime episode-ID filtering on %s", pref_path)
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
