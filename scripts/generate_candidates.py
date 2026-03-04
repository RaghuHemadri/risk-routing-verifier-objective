#!/usr/bin/env python3
"""
Generate candidate actions for DPO preference training.

Usage:
    python scripts/generate_candidates.py \
        --config configs/webarena/noisy.yaml \
        --policy-path outputs/policy/webarena_noisy/bc/final \
        --trajectories data/trajectories/webarena_teacher/trajectories.jsonl \
        --output data/candidates/webarena.jsonl \
        --K 5

For each step in teacher trajectories, samples K candidates from the
SLM policy and scores them with the verifier to create preference pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from r2v.data.trajectory import TrajectoryStore
from r2v.models.policy import PolicyModel
from r2v.models.verifier import create_verifier
from r2v.utils.config import config_to_dict, load_config
from r2v.utils.logging import JSONLLogger, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate candidate actions")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--verifier-path", type=str, default=None,
                        help="Path to trained verifier weights (verifier.pt)")
    parser.add_argument("--trajectories", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--K", type=int, default=5, help="Number of candidates per step")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_log = JSONLLogger(output_path.parent / "candidates_log.jsonl")

    # Load policy
    logger.info(f"Loading policy from {args.policy_path}")
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    policy = PolicyModel(policy_cfg)
    policy.load(args.policy_path)
    policy.model.eval()

    # Load verifier
    logger.info("Loading verifier...")
    vcfg = OmegaConf.to_container(cfg.get("verifier", {}), resolve=True)
    verifier = create_verifier(vcfg)

    # Load trained weights if provided
    if args.verifier_path:
        import torch as _torch
        state_dict = _torch.load(args.verifier_path, map_location="cpu")
        verifier.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded trained verifier weights from {args.verifier_path}")

    # Move verifier to GPU if available
    import torch as _torch
    if hasattr(verifier, 'to') and _torch.cuda.is_available():
        verifier = verifier.to("cuda")
    if hasattr(verifier, 'eval'):
        verifier.eval()

    # Load trajectories
    store = TrajectoryStore(args.trajectories)
    episodes = store.load_episodes()
    logger.info(f"Loaded {len(episodes)} episodes, generating K={args.K} candidates per step")

    total_pairs = 0
    with open(output_path, "w") as f:
        for ep_idx, episode in enumerate(episodes):
            if ep_idx % 100 == 0:
                logger.info(f"Processing episode {ep_idx}/{len(episodes)}")

            goal = episode.metadata.goal if episode.metadata else ""
            context = ""
            for step_idx, step in enumerate(episode.steps):
                context += step.observation.text + "\n"

                # Generate K candidates from the BC policy
                candidates = policy.generate_candidates(
                    context=context,
                    num_candidates=args.K,
                    temperature=cfg.get("inference", {}).get("temperature", 0.7),
                    max_new_tokens=cfg.get("inference", {}).get("max_tokens", 512),
                )

                # Score each candidate with verifier
                scored = []
                for cand in candidates:
                    score = verifier.score(
                        context=context,
                        action=cand["text"],
                        goal=goal,
                    )
                    scored.append({
                        "action": cand["text"],
                        "log_prob": cand["log_prob"],
                        "verifier_score": score,
                    })

                # Sort by verifier score → create preference pairs
                scored.sort(key=lambda x: x["verifier_score"], reverse=True)

                if len(scored) >= 2:
                    # Best vs worst as preference pair
                    pair = {
                        "context": context,
                        "chosen": scored[0]["action"],
                        "rejected": scored[-1]["action"],
                        "chosen_score": scored[0]["verifier_score"],
                        "rejected_score": scored[-1]["verifier_score"],
                        "episode_id": episode.episode_id,
                        "step_idx": step_idx,
                        "all_candidates": scored,
                    }
                    f.write(json.dumps(pair) + "\n")
                    total_pairs += 1

                context += step.action.raw_text + "\n"

    logger.info(f"Generated {total_pairs} preference pairs → {output_path}")
    jsonl_log.log("summary", {
        "total_pairs": total_pairs,
        "num_episodes": len(episodes),
        "K": args.K,
    })


if __name__ == "__main__":
    main()
