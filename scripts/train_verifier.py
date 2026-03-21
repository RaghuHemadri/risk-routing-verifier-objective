#!/usr/bin/env python3
"""
Train the verifier model (LLM-judge distillation or direct training).

Usage:
    python scripts/train_verifier.py \
        --config configs/humaneval/clean.yaml \
        --output outputs/verifier/humaneval \
        --trajectories data/trajectories/humaneval_clean/trajectories.jsonl
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
from r2v.data.trajectory import TrajectoryStore
from r2v.models.verifier import create_verifier
from r2v.training.verifier_trainer import VerifierTrainer
from r2v.utils.config import config_to_dict, load_config, save_config
from r2v.utils.logging import JSONLLogger, init_wandb, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train verifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--trajectories", type=str, required=True)
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

    # ── Load and label trajectories ──
    store = TrajectoryStore(args.trajectories)
    episodes = store.load_episodes()
    logger.info(f"Loaded {len(episodes)} episodes from {args.trajectories}")

    # Apply step-level labeling
    benchmark = cfg.get("benchmark", "humaneval")
    data_cfg = OmegaConf.to_container(cfg.get("data", {}), resolve=True)
    labeler = create_labeler(benchmark, config=data_cfg)
    for ep in episodes:
        labeler.label_episode(ep)
    logger.info("Step-level labeling complete")

    # Save labeled episodes to a temp file so VerifierDataset can reload them
    labeled_path = output_dir / "labeled_trajectories.jsonl"
    labeled_store = TrajectoryStore(labeled_path)
    labeled_store.save_episodes(episodes)
    logger.info(f"Saved {len(episodes)} labeled episodes to {labeled_path}")

    # ── Create verifier model ──
    vcfg = OmegaConf.to_container(cfg.get("verifier", {}), resolve=True)
    mode = vcfg.get("mode", "llm_judge")

    if mode == "trained":
        logger.info("=== Training verifier model ===")
        verifier = create_verifier(vcfg)

        # Build dataset from labeled trajectories
        max_seq_len = cfg.policy.get("max_seq_len", 4096)
        dataset = VerifierDataset(
            str(labeled_path), verifier.tokenizer, max_seq_len=max_seq_len,
        )
        logger.info(f"VerifierDataset: {len(dataset)} examples")

        # Train/val split
        val_frac = 1.0 - cfg.get("data", {}).get("train_ratio", 0.8)
        val_size = max(1, int(len(dataset) * val_frac))
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        logger.info(f"Split: {train_size} train, {val_size} val")

        # Train
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
