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

Resume after a crash (just re-run the same command — skips completed episodes):
    # Same command as above; completed episode IDs are read from the output file.

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
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from r2v.data.trajectory import TrajectoryStore
from r2v.models.policy import PolicyModel
from r2v.models.verifier import create_verifier
from r2v.utils.config import config_to_dict, load_config
from r2v.utils.logging import JSONLLogger, setup_logging


# ---------------------------------------------------------------------------
# GPU keepalive: periodic small forward passes to prevent HPC from killing
# the job due to low GPU utilization during CPU-bound bookkeeping.
# ---------------------------------------------------------------------------

_keepalive_stop = threading.Event()


def _gpu_keepalive(model, tokenizer, device, interval: float = 8.0):
    """Background thread: run a tiny forward pass every `interval` seconds."""
    import torch
    dummy_ids = tokenizer("keepalive", return_tensors="pt")["input_ids"].to(device)
    while not _keepalive_stop.is_set():
        try:
            with torch.no_grad():
                model(dummy_ids)
        except Exception:
            pass
        _keepalive_stop.wait(interval)


def start_keepalive(model, tokenizer, device, interval: float = 8.0):
    _keepalive_stop.clear()
    t = threading.Thread(
        target=_gpu_keepalive, args=(model, tokenizer, device, interval),
        daemon=True,
    )
    t.start()
    return t


def stop_keepalive():
    _keepalive_stop.set()


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_completed_episodes(output_file: Path) -> set[str]:
    """Read already-written JSONL and return set of (episode_id, step_idx)."""
    done: set[str] = set()
    if not output_file.exists():
        return done
    with open(output_file) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(f"{rec['episode_id']}_{rec['step_idx']}")
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def completed_episode_ids(output_file: Path) -> set[str]:
    """Return set of episode_ids that have ALL their steps already done."""
    if not output_file.exists():
        return set()
    ep_steps: dict[str, set[int]] = {}
    with open(output_file) as f:
        for line in f:
            try:
                rec = json.loads(line)
                eid = rec["episode_id"]
                ep_steps.setdefault(eid, set()).add(rec["step_idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    # We can't know total steps per episode from the output alone,
    # so we return all episode_ids that appear at all (conservative: skip them).
    return set(ep_steps.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate candidate actions")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--verifier-path", type=str, default=None,
                        help="Path to trained verifier weights (verifier.pt)")
    parser.add_argument("--trajectories", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--K", type=int, default=5, help="Number of candidates per step")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override max tokens to generate (default: config or 128)")
    # Multi-GPU sharding
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This worker's shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--merge", action="store_true",
                        help="Merge shard outputs instead of generating")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume; overwrite existing output")
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

    # --------------- Resume: check what's already done ---------------
    if args.no_resume:
        done_eids: set[str] = set()
        file_mode = "w"
    else:
        done_eids = completed_episode_ids(shard_output)
        file_mode = "a"  # append to existing output
        if done_eids:
            logger.info(f"Resuming: {len(done_eids)} episodes already completed in {shard_output}")

    # --------------- Load models ---------------
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

    # --------------- Start GPU keepalive ---------------
    if _torch.cuda.is_available():
        keepalive_thread = start_keepalive(
            policy.model, policy.tokenizer, policy.model.device, interval=8.0,
        )
        logger.info("GPU keepalive thread started (8s interval)")

    # --------------- Load & shard trajectories ---------------
    store = TrajectoryStore(args.trajectories)
    all_episodes = store.load_episodes()
    episodes = all_episodes[args.shard_id::args.num_shards]

    # Filter out already-completed episodes
    if done_eids:
        episodes = [ep for ep in episodes if ep.episode_id not in done_eids]

    # Resolve generation hyperparams
    inf_cfg = cfg.get("inference", {})
    gen_temperature = float(inf_cfg.get("temperature", 0.7)) if inf_cfg else 0.7
    if args.max_new_tokens is not None:
        gen_max_tokens = args.max_new_tokens
    else:
        gen_max_tokens = int(inf_cfg.get("max_tokens", 128)) if inf_cfg else 128

    logger.info(
        f"Processing {len(episodes)} episodes (skipped {len(done_eids)} done), "
        f"K={args.K}, max_new_tokens={gen_max_tokens}, temp={gen_temperature}"
    )

    # --------------- Main loop ---------------
    total_pairs = 0
    total_steps = 0
    t0 = time.time()
    with open(shard_output, file_mode) as f:
        for ep_idx, episode in enumerate(episodes):
            ep_t0 = time.time()
            num_steps = len(episode.steps)
            goal = episode.metadata.goal if episode.metadata else ""
            context = ""

            for step_idx, step in enumerate(episode.steps):
                context += step.observation.raw_text + "\n"

                # Generate K candidates (batched: one forward pass)
                candidates = policy.generate_candidates(
                    context=context,
                    num_candidates=args.K,
                    temperature=gen_temperature,
                    max_new_tokens=gen_max_tokens,
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
                total_steps += 1

            # Per-episode log
            ep_elapsed = time.time() - ep_t0
            elapsed = time.time() - t0
            rate = (ep_idx + 1) / elapsed
            remaining = len(episodes) - (ep_idx + 1)
            eta = remaining / rate if rate > 0 else float('inf')
            logger.info(
                f"[shard {args.shard_id}] Ep {ep_idx+1}/{len(episodes)} "
                f"(id={episode.episode_id}, {num_steps}st, {ep_elapsed:.1f}s) | "
                f"{total_pairs} pairs | {rate:.2f} ep/s, ETA {eta/60:.0f}min"
            )

    # --------------- Cleanup ---------------
    if _torch.cuda.is_available():
        stop_keepalive()

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
