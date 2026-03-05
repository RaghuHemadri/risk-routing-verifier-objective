#!/usr/bin/env python3
"""Generate candidate actions for DPO preference training.

Usage (single GPU):
    python scripts/generate_candidates.py \
        --config configs/swebench/noisy.yaml \
        --policy-path outputs/policy/swebench_noisy/final \
        --verifier-path outputs/verifier/swebench_noisy/final/verifier.pt \
        --trajectories data/trajectories/swebench_noisy/trajectories.jsonl \
        --output data/candidates/swebench_noisy.jsonl \
        --K 5

Usage (multi-GPU via launch_candidates.sh — 4 GPUs):
    bash scripts/launch_candidates.sh 4 \
        --config configs/swebench/noisy.yaml \
        --policy-path outputs/policy/swebench_noisy/final \
        --verifier-path outputs/verifier/swebench_noisy/final/verifier.pt \
        --trajectories data/trajectories/swebench_noisy/trajectories.jsonl \
        --output data/candidates/swebench_noisy.jsonl \
        --K 5

Merge shards after all finish:
    python scripts/generate_candidates.py --merge \
        --output data/candidates/swebench_noisy.jsonl

For each step in teacher trajectories, samples K candidates from the
SLM policy and scores them with the verifier to create preference pairs.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
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
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--verifier-path", type=str, default=None,
                        help="Path to trained verifier weights (verifier.pt)")
    parser.add_argument("--trajectories", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--K", type=int, default=5, help="Number of candidates per step")
    # Multi-GPU sharding
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This worker's shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--merge", action="store_true",
                        help="Merge shard outputs instead of generating")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def merge_shards(output_path: str, logger):
    """Merge all shard files into the final output."""
    output = Path(output_path)
    shard_pattern = str(output.parent / f"{output.stem}.shard_*{output.suffix}")
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        logger.error(f"No shard files found matching {shard_pattern}")
        sys.exit(1)

    total = 0
    with open(output, "w") as out_f:
        for sf in shard_files:
            with open(sf) as in_f:
                for line in in_f:
                    out_f.write(line)
                    total += 1
            logger.info(f"  Merged {sf}")

    logger.info(f"Merged {len(shard_files)} shards → {output} ({total} pairs)")


def main():
    args = parse_args()
    logger = setup_logging(level="INFO")

    # --- Merge mode ---
    if args.merge:
        merge_shards(args.output, logger)
        return

    # --- Generation mode: require config, policy-path, trajectories ---
    if not args.config or not args.policy_path or not args.trajectories:
        logger.error("--config, --policy-path, and --trajectories are required "
                      "for generation mode (omit --merge)")
        sys.exit(1)

    cfg = load_config(args.config, args.overrides)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine shard output path
    if args.num_shards > 1:
        shard_output = (output_path.parent
                        / f"{output_path.stem}.shard_{args.shard_id:03d}{output_path.suffix}")
        logger.info(f"Shard {args.shard_id}/{args.num_shards} → {shard_output}")
    else:
        shard_output = output_path

    jsonl_log = JSONLLogger(
        output_path.parent / f"candidates_log_shard{args.shard_id}.jsonl"
    )

    # Load policy
    logger.info(f"Loading policy from {args.policy_path}")
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    policy = PolicyModel(policy_cfg)
    policy.load(args.policy_path)
    policy.model.eval()

    # Load verifier — force mode=trained when --verifier-path is given
    logger.info("Loading verifier...")
    vcfg = OmegaConf.to_container(cfg.get("verifier", {}), resolve=True)
    if args.verifier_path:
        vcfg["mode"] = "trained"
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

    # Load trajectories and select this shard's slice
    store = TrajectoryStore(args.trajectories)
    all_episodes = store.load_episodes()
    episodes = all_episodes[args.shard_id::args.num_shards]
    logger.info(
        f"Loaded {len(all_episodes)} total episodes, "
        f"shard {args.shard_id} processing {len(episodes)} episodes, "
        f"generating K={args.K} candidates per step"
    )

    total_pairs = 0
    t0 = time.time()
    with open(shard_output, "w") as f:
        for ep_idx, episode in enumerate(episodes):
            if ep_idx % 50 == 0:
                elapsed = time.time() - t0
                rate = ep_idx / elapsed if elapsed > 0 else 0
                eta = (len(episodes) - ep_idx) / rate if rate > 0 else float('inf')
                logger.info(
                    f"[shard {args.shard_id}] Episode {ep_idx}/{len(episodes)} "
                    f"({total_pairs} pairs, {rate:.1f} ep/s, "
                    f"ETA {eta / 60:.0f}min)"
                )

            goal = episode.metadata.goal if episode.metadata else ""
            context = ""
            for step_idx, step in enumerate(episode.steps):
                context += step.observation.raw_text + "\n"

                # Generate K candidates (batched: one forward pass)
                candidates = policy.generate_candidates(
                    context=context,
                    num_candidates=args.K,
                    temperature=cfg.get("inference", {}).get("temperature", 0.7),
                    max_new_tokens=cfg.get("inference", {}).get("max_tokens", 512),
                )

                # Score all candidates in one batched verifier call
                cand_texts = [c["text"] for c in candidates]
                scores = verifier.score_candidates(
                    context=context,
                    candidates=cand_texts,
                    goal=goal,
                )

                scored = [
                    {
                        "action": cand["text"],
                        "log_prob": cand["log_prob"],
                        "verifier_score": sc,
                    }
                    for cand, sc in zip(candidates, scores)
                ]

                # Sort by verifier score → create preference pairs
                scored.sort(key=lambda x: x["verifier_score"], reverse=True)

                if len(scored) >= 2:
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
                    f.flush()
                    total_pairs += 1

                context += step.action.raw_text + "\n"

    elapsed = time.time() - t0
    logger.info(
        f"[shard {args.shard_id}] Done: {total_pairs} pairs from "
        f"{len(episodes)} episodes in {elapsed / 60:.1f}min → {shard_output}"
    )
    jsonl_log.log("summary", {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "total_pairs": total_pairs,
        "num_episodes": len(episodes),
        "K": args.K,
        "elapsed_seconds": elapsed,
    })


if __name__ == "__main__":
    main()
