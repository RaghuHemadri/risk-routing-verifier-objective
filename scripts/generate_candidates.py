#!/usr/bin/env python3
"""Generate candidate actions for DPO preference training.

Usage (single GPU):
    python scripts/generate_candidates.py \
        --config configs/gaia/noisy.yaml \
        --policy-path outputs/policy/gaia_noisy/final \
        --verifier-path outputs/verifier/gaia_noisy/final/verifier.pt \
        --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
        --output data/candidates/gaia_noisy.jsonl \
        --K 5

Usage (multi-GPU via launch_candidates.sh — 4 GPUs):
    bash scripts/launch_candidates.sh 4 \
        --config configs/gaia/noisy.yaml \
        --policy-path outputs/policy/gaia_noisy/final \
        --verifier-path outputs/verifier/gaia_noisy/final/verifier.pt \
        --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
        --output data/candidates/gaia_noisy.jsonl \
        --K 5

Resume after a crash (just re-run the same command — skips completed episodes):
    # Same command as above; completed episode IDs are read from the output file.

Merge shards after all finish:
    python scripts/generate_candidates.py --merge \
        --output data/candidates/gaia_noisy.jsonl

For each step in teacher trajectories, samples K candidates from the
SLM policy and scores them with the verifier to create preference pairs.
"""

from __future__ import annotations

import argparse
import glob
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from r2v.data.trajectory import TrajectoryStore
from r2v.models.policy import PolicyModel
from r2v.models.verifier import create_verifier
from r2v.utils.config import load_config
from r2v.utils.logging import JSONLLogger, setup_logging

import torch.nn.functional as F


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

# ---------------------------------------------------------------------------
# Prefetch iterator: tokenises the next batch on a CPU thread while the
# GPU processes the current batch.  Mirrors the pattern used in
# generate_router_features.py for maximum GPU utilisation.
# ---------------------------------------------------------------------------

class _PrefetchIter:
    """Yields ``(batch_items, gen_tokens)`` with one batch pre-tokenised
    on a daemon thread ahead of consumption.

    gen_tokens are **left-padded** so that ``model.generate()`` appends
    new tokens on the right for every sequence in the batch.

    The background thread only touches the *tokenizer* (CPU-only);
    all GPU work stays on the main thread, so there is no device
    contention.
    """

    def __init__(self, items, batch_size, tokenizer, max_seq_len, max_new_tokens):
        self._items = items
        self._bs = batch_size
        self._tok = tokenizer
        self._max_seq = max_seq_len
        self._max_new = max_new_tokens
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self):
        for start in range(0, len(self._items), self._bs):
            batch = self._items[start : start + self._bs]
            prompts = [f"{it['context']}\nAction:" for it in batch]

            # Left-padding for generation (new tokens appended on the right)
            orig_side = self._tok.padding_side
            self._tok.padding_side = "left"
            gen_tok = self._tok(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_seq - self._max_new,
                padding=True,
            )
            self._tok.padding_side = orig_side

            self._q.put((batch, gen_tok))
        self._q.put(None)  # sentinel

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            yield item


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
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of steps to batch together for GPU inference")
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

    # Disable gradient checkpointing for inference — it forces use_cache=False,
    # meaning generate() recomputes the full prefix at every decoding step.
    # Re-enabling KV cache gives a major speedup.
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    policy.model.config.use_cache = True
    logger.info("KV cache enabled (gradient checkpointing disabled for inference)")

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
        gen_max_tokens = int(inf_cfg.get("max_tokens", 256)) if inf_cfg else 256

    # --------------- Pre-collect all step items ---------------
    logger.info("Pre-collecting step items...")
    all_items: list[dict] = []
    for episode in episodes:
        goal = episode.metadata.goal if episode.metadata else ""
        context = ""
        for step_idx, step in enumerate(episode.steps):
            context += step.observation.raw_text + "\n"
            all_items.append({
                "context": context,
                "step_idx": step_idx,
                "goal": goal,
                "episode_id": episode.episode_id,
            })
            context += step.action.raw_text + "\n"

    batch_size = args.batch_size
    total_batches = (len(all_items) + batch_size - 1) // batch_size
    logger.info(
        f"Processing {len(all_items)} steps from {len(episodes)} episodes "
        f"(skipped {len(done_eids)} done), "
        f"K={args.K}, batch_size={batch_size}, "
        f"max_new_tokens={gen_max_tokens}, temp={gen_temperature}"
    )

    # --------------- Prefetch iterator ---------------
    prefetcher = _PrefetchIter(
        all_items, batch_size,
        policy.tokenizer, policy.max_seq_len, gen_max_tokens,
    )

    # --------------- Batched main loop ---------------
    total_pairs = 0
    t0 = time.time()
    device = policy.model.device
    pad_id = policy.tokenizer.pad_token_id

    with open(shard_output, file_mode) as f:
        for batch_idx, (batch, gen_tok) in enumerate(prefetcher):
            batch_t0 = time.time()
            B = len(batch)

            # ---------- 1. Batched candidate generation ---------------
            try:
                gen_ids = gen_tok["input_ids"].to(device)
                gen_mask = gen_tok["attention_mask"].to(device)
                with _torch.no_grad():
                    gen_out = policy.model.generate(
                        input_ids=gen_ids,
                        attention_mask=gen_mask,
                        max_new_tokens=gen_max_tokens,
                        num_return_sequences=args.K,
                        temperature=gen_temperature,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=pad_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                prompt_len = gen_ids.shape[1]
                all_candidates: list[list[dict]] = []
                for b in range(B):
                    cands = []
                    for k in range(args.K):
                        idx = b * args.K + k
                        gen_tokens = gen_out.sequences[idx, prompt_len:]
                        gen_tokens = gen_tokens[gen_tokens != pad_id]
                        text = policy.tokenizer.decode(
                            gen_tokens, skip_special_tokens=True
                        ).strip()

                        lp = 0.0
                        if gen_out.scores:
                            for t, sc in enumerate(gen_out.scores):
                                if t >= len(gen_tokens):
                                    break
                                lp += F.log_softmax(
                                    sc[idx], dim=-1
                                )[gen_tokens[t]].item()

                        cands.append({
                            "text": text,
                            "log_prob": lp,
                            "num_tokens": len(gen_tokens),
                        })
                    all_candidates.append(cands)
            except Exception as exc:
                logger.warning(f"Generation failed for batch {batch_idx}: {exc}")
                all_candidates = [
                    [{"text": "", "log_prob": 0.0, "num_tokens": 0}] * args.K
                ] * B

            # ---------- 2. Batched verifier scoring -------------------
            flat_ctx, flat_cand, flat_goal = [], [], []
            for b in range(B):
                for c in all_candidates[b]:
                    flat_ctx.append(batch[b]["context"])
                    flat_cand.append(c["text"])
                    flat_goal.append(batch[b]["goal"])

            try:
                flat_scores = verifier.score_batch(flat_ctx, flat_cand, flat_goal)
            except Exception:
                flat_scores = [0.5] * len(flat_ctx)

            all_scores = [
                flat_scores[b * args.K : (b + 1) * args.K] for b in range(B)
            ]

            # ---------- 3. Create preference pairs & write ------------
            for i, item in enumerate(batch):
                scored = [
                    {
                        "action": c["text"],
                        "log_prob": c["log_prob"],
                        "verifier_score": sc,
                    }
                    for c, sc in zip(all_candidates[i], all_scores[i])
                ]
                scored.sort(key=lambda x: x["verifier_score"], reverse=True)

                if len(scored) >= 2:
                    pair = {
                        "context": item["context"],
                        "chosen": scored[0]["action"],
                        "rejected": scored[-1]["action"],
                        "chosen_score": scored[0]["verifier_score"],
                        "rejected_score": scored[-1]["verifier_score"],
                        "episode_id": item["episode_id"],
                        "step_idx": item["step_idx"],
                        "all_candidates": scored,
                    }
                    f.write(json.dumps(pair) + "\n")
                    total_pairs += 1

            # ---------- 4. Progress logging ---------------------------
            batch_elapsed = time.time() - batch_t0
            elapsed = time.time() - t0
            steps_done = min((batch_idx + 1) * batch_size, len(all_items))
            rate = steps_done / elapsed if elapsed > 0 else 0
            remaining = len(all_items) - steps_done
            eta = remaining / rate if rate > 0 else float("inf")
            pct = 100.0 * steps_done / len(all_items) if all_items else 100.0

            logger.info(
                f"[shard {args.shard_id}] "
                f"Batch {batch_idx + 1}/{total_batches} "
                f"({pct:5.1f}%)  |  "
                f"{total_pairs} pairs  |  "
                f"{rate:.1f} steps/s  |  "
                f"batch {batch_elapsed:.2f}s  |  "
                f"ETA {eta / 60:.1f}min"
            )
            f.flush()

    # --------------- Cleanup ---------------
    if _torch.cuda.is_available():
        stop_keepalive()

    elapsed = time.time() - t0
    logger.info(
        f"[shard {args.shard_id}] Done: {total_pairs} pairs from "
        f"{len(episodes)} episodes ({len(all_items)} steps) in {elapsed / 60:.1f}min "
        f"({len(all_items) / max(elapsed, 1e-6):.1f} steps/s) → {shard_output}"
    )
    jsonl_log.log("summary", {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "total_pairs": total_pairs,
        "num_episodes": len(episodes),
        "K": args.K,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
    })


if __name__ == "__main__":
    main()
