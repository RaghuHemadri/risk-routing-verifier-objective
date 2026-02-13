#!/usr/bin/env python3
"""
Generate router training features from collected trajectories.

Usage:
    python scripts/generate_router_features.py \
        --config configs/webarena/noisy.yaml \
        --policy-path outputs/policy/webarena_noisy/final \
        --trajectories data/trajectories/webarena_noisy/trajectories.jsonl \
        --output data/router_features/webarena.jsonl

Features extracted per decision point:
- SLM entropy (H(π_θ))
- Verifier score spread (max - min over K candidates)
- Step count / horizon fraction
- Perturbation-type indicator (one-hot)
- Context length
- Action log-probability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.trajectory import PerturbationType, TrajectoryStore
from r2v.models.policy import PolicyModel
from r2v.models.verifier import create_verifier
from r2v.utils.config import load_config
from r2v.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate router features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--trajectories", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def extract_features(
    policy: PolicyModel,
    verifier,
    context: str,
    step_idx: int,
    max_steps: int,
    perturbation_type: str | None,
    K: int = 5,
    cfg=None,
) -> list[float]:
    """Extract feature vector for router decision.

    Returns fixed-size feature vector.
    """
    features = []

    # 1. SLM entropy
    try:
        entropy = policy.compute_entropy(context)
        features.append(float(entropy))
    except Exception:
        features.append(0.0)

    # 2. Generate candidates and compute score spread
    try:
        candidates = policy.generate_candidates(
            context=context,
            K=K,
            temperature=cfg.get("inference", {}).get("temperature", 0.7) if cfg else 0.7,
            max_new_tokens=cfg.get("inference", {}).get("max_tokens", 512) if cfg else 512,
        )
        scores = []
        for cand_text, _ in candidates:
            score = verifier.score_action(context=context, action=cand_text)
            scores.append(score)

        features.append(max(scores) - min(scores))  # Score spread
        features.append(float(np.mean(scores)))       # Mean score
        features.append(float(np.std(scores)))         # Score std
        features.append(max(scores))                   # Best score
    except Exception:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # 3. Step count features
    features.append(step_idx / max(max_steps, 1))  # Horizon fraction
    features.append(float(step_idx))                 # Absolute step

    # 4. Context length (normalized)
    features.append(len(context) / 10000.0)

    # 5. Perturbation type (one-hot, 5 categories: none + 4 types)
    pert_types = [None, "tool_flakiness", "partial_observability", "prompt_injection", "distractors"]
    one_hot = [0.0] * len(pert_types)
    if perturbation_type in pert_types:
        one_hot[pert_types.index(perturbation_type)] = 1.0
    else:
        one_hot[0] = 1.0  # Default to "none"
    features.extend(one_hot)

    return features  # Total: 1 + 4 + 2 + 1 + 5 = 13


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load models
    logger.info("Loading policy...")
    policy = PolicyModel(
        model_name=cfg.policy.model_name,
        lora_r=cfg.policy.get("lora_r", 64),
        lora_alpha=cfg.policy.get("lora_alpha", 128),
        load_in_4bit=cfg.policy.get("load_in_4bit", True),
    )
    policy.load(args.policy_path)

    logger.info("Loading verifier...")
    verifier = create_verifier(cfg.get("verifier", {}))

    # Load trajectories
    store = TrajectoryStore(args.trajectories)
    episodes = store.load_all()
    logger.info(f"Loaded {len(episodes)} episodes")

    max_steps = cfg.get("inference", {}).get("step_limit", 15)
    total_features = 0

    with open(output_path, "w") as f:
        for ep_idx, episode in enumerate(episodes):
            if ep_idx % 10 == 0:
                logger.info(f"Processing episode {ep_idx}/{len(episodes)}")

            context = ""
            for step_idx, step in enumerate(episode.steps):
                context += step.observation.text + "\n"

                pert_type = None
                if hasattr(step.observation, "perturbation_type") and step.observation.perturbation_type:
                    pert_type = step.observation.perturbation_type

                features = extract_features(
                    policy=policy,
                    verifier=verifier,
                    context=context,
                    step_idx=step_idx,
                    max_steps=max_steps,
                    perturbation_type=pert_type,
                    K=args.K,
                    cfg=cfg,
                )

                # Label: did SLM succeed from this point?
                slm_success = 1.0 if episode.success else 0.0

                # Cost: proportion of LLM calls in episode
                from r2v.data.trajectory import ActionSource
                llm_steps = sum(
                    1 for s in episode.steps if s.action.source == ActionSource.TEACHER
                )
                cost = llm_steps / max(len(episode.steps), 1)

                record = {
                    "features": features,
                    "slm_success": slm_success,
                    "cost": cost,
                    "episode_id": episode.episode_id,
                    "step_idx": step_idx,
                }
                f.write(json.dumps(record) + "\n")
                total_features += 1

                context += step.action.text + "\n"

    logger.info(f"Generated {total_features} feature vectors → {output_path}")


if __name__ == "__main__":
    main()
