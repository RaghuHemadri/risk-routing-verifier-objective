#!/usr/bin/env python3
"""
Train the verifier model (LLM-judge distillation or direct training).

Usage:
    python scripts/train_verifier.py \
        --config configs/webarena/noisy.yaml \
        --output outputs/verifier/webarena \
        --trajectories data/trajectories/webarena_teacher/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

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

    init_wandb(
        project=cfg.get("logging", {}).get("project", "r2v-agent"),
        name=f"verifier_{cfg.get('data', {}).get('benchmark', 'unknown')}",
        config=config_to_dict(cfg),
        tags=["verifier"],
        mode=cfg.get("logging", {}).get("wandb_mode", "online"),
    )

    # Load trajectories and label them
    store = TrajectoryStore(args.trajectories)
    episodes = store.load_all()
    logger.info(f"Loaded {len(episodes)} episodes")

    # Apply labeling
    benchmark = cfg.get("data", {}).get("benchmark", "webarena")
    labeler = create_labeler(benchmark, discount=cfg.get("data", {}).get("temporal_discount", 0.95))
    for ep in episodes:
        labeler.label_episode(ep)

    # Create verifier
    vcfg = cfg.get("verifier", {})
    if vcfg.get("type", "llm_judge") == "trained":
        from r2v.models.verifier import TrainedVerifier
        from transformers import AutoTokenizer, AutoModel

        backbone_name = vcfg.get("backbone", cfg.policy.model_name)
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        verifier = TrainedVerifier(
            backbone_name=backbone_name,
            hidden_dim=vcfg.get("hidden_dim", 256),
        )

        # Create dataset
        dataset = VerifierDataset(episodes, tokenizer, max_length=cfg.training.get("max_seq_len", 2048))

        val_frac = cfg.training.get("val_fraction", 0.1)
        val_size = max(1, int(len(dataset) * val_frac))
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        trainer = VerifierTrainer(
            model=verifier,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=str(output_dir),
            num_epochs=cfg.training.get("verifier_epochs", 5),
            batch_size=cfg.training.get("verifier_batch_size", 8),
            learning_rate=cfg.training.get("verifier_lr", 1e-4),
        )

        trainer.train()
        logger.info("Verifier training complete")
    else:
        logger.info("Using LLM-judge verifier (no training needed)")
        logger.info("Verifier will be initialized at inference time from config")

    jsonl_log.log("training_complete", {"output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
