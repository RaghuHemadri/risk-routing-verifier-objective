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
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from r2v.data.trajectory import ActionSource, PerturbationType, TrajectoryStore
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
    goal: str = "",
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
        inf_cfg = cfg.get("inference", {}) if cfg else {}
        candidates = policy.generate_candidates(
            context=context,
            num_candidates=K,
            temperature=float(inf_cfg.get("temperature", 0.7)) if inf_cfg else 0.7,
            max_new_tokens=int(inf_cfg.get("max_tokens", 512)) if inf_cfg else 512,
        )
        cand_texts = [c["text"] for c in candidates]
        scores = verifier.score_candidates(
            context=context, candidates=cand_texts, goal=goal,
        )

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
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    policy = PolicyModel(policy_cfg)
    policy.load(args.policy_path)
    policy.model.eval()

    # Disable gradient checkpointing for inference speed
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    policy.model.config.use_cache = True

    logger.info("Loading verifier...")
    vcfg = OmegaConf.to_container(cfg.get("verifier", {}), resolve=True)
    verifier = create_verifier(vcfg)
    if hasattr(verifier, "eval"):
        verifier.eval()
    if torch.cuda.is_available() and hasattr(verifier, "to"):
        verifier = verifier.to("cuda")

    # Load trajectories
    store = TrajectoryStore(args.trajectories)
    episodes = store.load_episodes()
    logger.info(f"Loaded {len(episodes)} episodes")

    max_steps = cfg.get("inference", {}).get("step_limit", 15) if cfg.get("inference") else 15
    total_features = 0
    t0 = time.time()

    with open(output_path, "w") as f:
        for ep_idx, episode in enumerate(episodes):
            if ep_idx % 10 == 0:
                elapsed = time.time() - t0
                rate = ep_idx / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Processing episode {ep_idx}/{len(episodes)} "
                    f"({rate:.2f} ep/s)"
                )

            goal = episode.metadata.goal if episode.metadata else ""
            context = ""
            for step_idx, step in enumerate(episode.steps):
                context += step.observation.raw_text + "\n"

                pert_type = None
                if step.perturbation_type and step.perturbation_type != PerturbationType.NONE:
                    pert_type = step.perturbation_type.value

                features = extract_features(
                    policy=policy,
                    verifier=verifier,
                    context=context,
                    step_idx=step_idx,
                    max_steps=max_steps,
                    perturbation_type=pert_type,
                    goal=goal,
                    K=args.K,
                    cfg=cfg,
                )

                # Label: did SLM succeed from this point?
                slm_success = 1.0 if episode.success else 0.0

                # Cost: proportion of LLM calls in episode
                llm_steps = sum(
                    1 for s in episode.steps if s.action_source == ActionSource.TEACHER
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

                context += step.action.raw_text + "\n"

    elapsed = time.time() - t0
    logger.info(
        f"Generated {total_features} feature vectors → {output_path} "
        f"in {elapsed / 60:.1f}min"
    )


if __name__ == "__main__":
    main()
