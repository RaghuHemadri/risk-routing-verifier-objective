#!/usr/bin/env python3
"""
Train the verifier model (LLM-judge distillation or direct training).

Usage:
    python scripts/train_verifier.py \
        --config configs/gaia/clean.yaml \
        --output outputs/verifier/gaia \
        --trajectories data/trajectories/gaia_clean/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.datasets import VerifierDataset
from r2v.data.labeling import create_labeler
from r2v.data.splits import load_and_split
from r2v.data.trajectory import TrajectoryStore
from r2v.models.verifier import create_verifier
from r2v.training.verifier_trainer import VerifierTrainer
from r2v.utils.config import config_to_dict, load_config, save_config
from r2v.utils.logging import JSONLLogger, init_wandb, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train verifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Full trajectories JSONL (triggers dynamic splitting)")
    parser.add_argument("--train-data", type=str, default=None,
                        help="Pre-split train JSONL (static splitting)")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Pre-split val JSONL (static splitting)")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Pre-split test JSONL (static splitting)")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    jsonl_log = JSONLLogger(output_dir / "training_log.jsonl")
    jsonl_log.log_config(config_to_dict(cfg))

    log_cfg = cfg.get("logging", {})
    init_wandb(
        project=log_cfg.get("wandb_project", "r2v-agent"),
        name=f"verifier_{cfg.get('data', {}).get('benchmark', 'unknown')}",
        config=config_to_dict(cfg),
        tags=["verifier"],
        mode=log_cfg.get("wandb_mode", "disabled"),
    )

    # ── Load trajectories (static pre-split files or dynamic splitting) ──
    use_static = args.train_data and args.val_data
    if use_static:
        logger.info("Loading static pre-split data files")
        splits = {
            "train": TrajectoryStore(args.train_data).load_episodes(),
            "val": TrajectoryStore(args.val_data).load_episodes(),
            "test": TrajectoryStore(args.test_data).load_episodes() if args.test_data else [],
        }
    else:
        if not args.trajectories:
            raise ValueError("Provide either --trajectories or --train-data/--val-data")
        max_perturbations = int(cfg.get("data", {}).get("max_perturbations_per_task", 2))
        splits = load_and_split(
            args.trajectories,
            max_perturbations_per_task=max_perturbations,
            seed=int(cfg.get("project", {}).get("seed", 42)),
        )
    all_episodes = splits["train"] + splits["val"] + splits["test"]
    logger.info(
        f"Task-level split: {len(splits['train'])} train, "
        f"{len(splits['val'])} val, {len(splits['test'])} test episodes"
    )

    for sname, seps in splits.items():
        if not seps:
            continue
        n_s = sum(1 for ep in seps if ep.success)
        n_f = len(seps) - n_s
        task_ids = {ep.metadata.task_id for ep in seps}
        logger.info(
            "  Verifier %-5s: %d episodes (%d success / %d failure, %.1f%% success) "
            "across %d tasks",
            sname, len(seps), n_s, n_f,
            n_s / len(seps) * 100, len(task_ids),
        )

    # Apply step-level labeling
    benchmark = cfg.get("data", {}).get("benchmark", "gaia")
    data_cfg = OmegaConf.to_container(cfg.get("data", {}), resolve=True)
    labeler = create_labeler(benchmark, config=data_cfg)
    for ep in all_episodes:
        labeler.label_episode(ep)
    logger.info("Step-level labeling complete")

    # Save labeled episodes for reproducibility
    labeled_path = output_dir / "labeled_trajectories.jsonl"
    labeled_store = TrajectoryStore(labeled_path)
    labeled_store.save_episodes(all_episodes)
    logger.info(f"Saved {len(all_episodes)} labeled episodes to {labeled_path}")

    # ── Create verifier model ──
    vcfg = OmegaConf.to_container(cfg.get("verifier", {}), resolve=True)
    mode = vcfg.get("mode", "llm_judge")

    if mode == "trained":
        logger.info("=== Training verifier model ===")
        verifier = create_verifier(vcfg)

        max_seq_len = cfg.policy.get("max_seq_len", 4096)
        train_ds = VerifierDataset(
            tokenizer=verifier.tokenizer, max_seq_len=max_seq_len,
            episodes=splits["train"],
        )
        val_ds = VerifierDataset(
            tokenizer=verifier.tokenizer, max_seq_len=max_seq_len,
            episodes=splits["val"],
        )
        logger.info(f"VerifierDataset: {len(train_ds)} train, {len(val_ds)} val examples")

        v_train_cfg = OmegaConf.to_container(cfg.training.verifier, resolve=True)
        trainer = VerifierTrainer(
            verifier=verifier,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            config=v_train_cfg,
            output_dir=str(output_dir),
        )

        trainer.train()
        logger.info("Verifier training complete")
    else:
        logger.info("Using LLM-judge verifier (no training needed)")
        logger.info("Verifier will be initialized at inference time from config")

    jsonl_log.log("training_complete", {"output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
