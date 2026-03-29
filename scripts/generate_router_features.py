#!/usr/bin/env python3
"""
Generate router training features from collected trajectories.

Single-GPU usage:
    python scripts/generate_router_features.py \
        --config configs/gaia/noisy.yaml \
        --policy-path outputs/policy/gaia_noisy/final \
        --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
        --output data/router_features/gaia.jsonl

Multi-GPU usage (via launch script — shards episodes across GPUs):
    bash scripts/launch_router_features.sh 4 \
        --config configs/gaia/noisy.yaml \
        --policy-path outputs/policy/gaia_noisy/final \
        --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
        --output data/router_features/gaia.jsonl

Merge shards after all GPUs finish:
    python scripts/generate_router_features.py --merge \
        --output data/router_features/gaia.jsonl

Resume after a crash (re-run the same command — skips completed steps):
    # Same command; already-written episode_ids are detected automatically.

Features extracted per decision point (24-dim):
- SLM entropy (H(π_θ))
- Verifier score spread, mean, std, best, worst (over K candidates)
- Action log-probability best, mean, std (over K candidates)
- Candidate consistency (fraction sharing the same leading token)
- Semantic entropy over K candidates (entropy over first-token cluster distribution)
- Step count / horizon fraction
- Context length
- Goal length (task complexity proxy)
- Benchmark one-hot (gaia / alfworld / humaneval / webarena, 4 dims)
- Perturbation-type indicator (one-hot, 5 dims)
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

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from r2v.data.trajectory import ActionSource, PerturbationType, TrajectoryStore
from r2v.models.policy import PolicyModel
from r2v.models.verifier import create_verifier
from r2v.utils.config import load_config
from r2v.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate router features")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--trajectories", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Micro-batch size for GPU batching. Increase for more GPU "
             "utilisation; decrease if you hit OOM (default: 4).",
    )
    # Multi-GPU sharding
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This worker's shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel workers / GPUs")
    parser.add_argument("--merge", action="store_true",
                        help="Merge shard outputs instead of generating")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume; overwrite existing output")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


# ================================================================
# GPU keepalive: periodic tiny forward passes to prevent HPC from
# killing the job due to low GPU utilization during CPU-bound phases.
# ================================================================

_keepalive_stop = threading.Event()


def _gpu_keepalive(model, tokenizer, device, interval: float = 8.0):
    """Background thread: run a tiny forward pass every *interval* seconds."""
    dummy_ids = tokenizer("keepalive", return_tensors="pt")["input_ids"].to(device)
    while not _keepalive_stop.is_set():
        try:
            with torch.no_grad():
                model(dummy_ids)
        except Exception:
            pass
        _keepalive_stop.wait(interval)


def _start_keepalive(model, tokenizer, device, interval: float = 8.0):
    _keepalive_stop.clear()
    t = threading.Thread(
        target=_gpu_keepalive, args=(model, tokenizer, device, interval),
        daemon=True,
    )
    t.start()
    return t


def _stop_keepalive():
    _keepalive_stop.set()


# ================================================================
# Resume helpers
# ================================================================

def _completed_episode_ids(output_file: Path) -> set[str]:
    """Return set of episode_ids already present in *output_file*."""
    if not output_file.exists():
        return set()
    eids: set[str] = set()
    with open(output_file) as f:
        for line in f:
            try:
                eids.add(json.loads(line)["episode_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return eids


# ================================================================
# Merge shards
# ================================================================

def _merge_shards(output_path: str, logger):
    """Combine all shard JSONL files into a single output."""
    out = Path(output_path)
    pattern = str(out.parent / f"{out.stem}.shard_*{out.suffix}")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        logger.error(f"No shard files matching {pattern}")
        sys.exit(1)

    total = 0
    with open(out, "w") as fout:
        for sf in shard_files:
            with open(sf) as fin:
                for line in fin:
                    fout.write(line)
                    total += 1
            logger.info(f"  Merged {sf}")
    logger.info(f"Merged {len(shard_files)} shards -> {out} ({total} records)")


# ---- Perturbation one-hot encoding ------------------------------------
_PERT_TYPES = [
    None, "tool_flakiness", "partial_observability",
    "prompt_injection", "distractors",
]

# ---- Benchmark one-hot encoding ---------------------------------------
_BENCHMARKS = ["gaia", "alfworld", "humaneval", "webarena"]


def _detect_benchmark(config_path: str | None) -> str | None:
    """Infer benchmark name from the config file path."""
    if not config_path:
        return None
    for bench in _BENCHMARKS:
        if bench in config_path.lower():
            return bench
    return None


def _benchmark_onehot(benchmark: str | None) -> list[float]:
    """Encode benchmark as a 4-dim one-hot vector."""
    one_hot = [0.0] * len(_BENCHMARKS)
    if benchmark in _BENCHMARKS:
        one_hot[_BENCHMARKS.index(benchmark)] = 1.0
    return one_hot


def _candidate_consistency(texts: list[str]) -> float:
    """Fraction of K candidates sharing the most common leading token."""
    from collections import Counter
    if not texts:
        return 1.0
    first_tokens = [t.split()[0].lower() if t.strip() else "" for t in texts]
    most_common_count = Counter(first_tokens).most_common(1)[0][1]
    return most_common_count / len(first_tokens)


def _semantic_entropy(texts: list[str]) -> float:
    """Shannon entropy over first-token cluster distribution of K candidates.

    Complements candidate_consistency: consistency gives the mode fraction,
    semantic_entropy captures spread across *all* clusters.
    e.g. [print, print, print, return, def] → H = -3/5*log(3/5) - 1/5*log(1/5)*2
    """
    from collections import Counter
    if not texts:
        return 0.0
    first_tokens = [t.split()[0].lower() if t.strip() else "" for t in texts]
    counts = Counter(first_tokens)
    total = len(first_tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)
    return float(entropy)


def _perturbation_onehot(pert_type: str | None) -> list[float]:
    """Encode perturbation type as a 5-dim one-hot vector."""
    one_hot = [0.0] * len(_PERT_TYPES)
    if pert_type in _PERT_TYPES:
        one_hot[_PERT_TYPES.index(pert_type)] = 1.0
    else:
        one_hot[0] = 1.0
    return one_hot


def _assemble_features(
    entropy: float,
    scores: list[float],
    log_probs: list[float],
    step_idx: int,
    max_steps: int,
    context_len: int,
    pert_type: str | None,
    candidate_texts: list[str],
    benchmark: str | None,
    goal: str,
) -> list[float]:
    """Build the 24-dim feature vector from pre-computed components.

    Dims:
      0        entropy
      1-5      verifier scores: spread, mean, std, best, worst
      6-8      log-prob stats: best, mean, std
      9        candidate consistency (fraction with same leading token)
      10       semantic entropy (entropy over first-token cluster distribution)
      11       horizon fraction (step_idx / max_steps)
      12       step number (absolute)
      13       normalized context length
      14       goal length (task complexity proxy)
      15-18    benchmark one-hot (gaia, alfworld, humaneval, webarena)
      19-23    perturbation one-hot (5 dims)
    """
    features: list[float] = []
    # 1. SLM entropy
    features.append(entropy)
    # 2. Verifier candidate scores (5 stats)
    features.append(max(scores) - min(scores))    # spread
    features.append(float(np.mean(scores)))        # mean
    features.append(float(np.std(scores)))         # std
    features.append(max(scores))                   # best
    features.append(min(scores))                   # worst
    # 3. Action log-probability stats
    features.append(max(log_probs))                # log_prob_best
    features.append(float(np.mean(log_probs)))     # log_prob_mean
    features.append(float(np.std(log_probs)))      # log_prob_std
    # 4. Candidate diversity signals
    features.append(_candidate_consistency(candidate_texts))
    features.append(_semantic_entropy(candidate_texts))
    # 5. Step features
    features.append(step_idx / max(max_steps, 1))  # horizon fraction
    features.append(float(step_idx))               # absolute step
    # 6. Context / task complexity
    features.append(context_len / 10000.0)         # normalized context length
    features.append(len(goal) / 1000.0)            # goal length (task complexity proxy)
    # 7. Benchmark one-hot (4 dims)
    features.extend(_benchmark_onehot(benchmark))
    # 8. Perturbation one-hot (5 dims)
    features.extend(_perturbation_onehot(pert_type))
    return features  # length 24


# ================================================================
# Prefetch iterator: tokenises the next batch on a CPU thread
# while the GPU processes the current batch.
# ================================================================

class _PrefetchIter:
    """Yields ``(batch_items, entropy_tokens, gen_tokens)`` with one
    batch pre-tokenised on a daemon thread ahead of consumption.

    * entropy_tokens — right-padded (default), used for the single
      forward pass that computes next-token entropy.
    * gen_tokens — **left-padded**, so that ``model.generate()`` appends
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

    # ---- background producer ------------------------------------------
    def _produce(self):
        for start in range(0, len(self._items), self._bs):
            batch = self._items[start : start + self._bs]
            prompts = [f"{it['context']}\nAction:" for it in batch]

            # Entropy forward pass — right-padding (default)
            entropy_tok = self._tok(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_seq,
                padding=True,
            )

            # Generation — left-padding so new tokens are right-aligned
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

            self._q.put((batch, entropy_tok, gen_tok))
        self._q.put(None)  # sentinel

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            yield item


def main():
    args = parse_args()
    logger = setup_logging(level="INFO")

    # ---- Merge mode --------------------------------------------------
    if args.merge:
        _merge_shards(args.output, logger)
        return

    # ---- Generation mode: require config, policy-path, trajectories --
    if not args.config or not args.policy_path or not args.trajectories:
        logger.error(
            "--config, --policy-path, and --trajectories are required "
            "for generation mode (omit --merge)"
        )
        sys.exit(1)

    cfg = load_config(args.config, args.overrides)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine shard output path
    if args.num_shards > 1:
        shard_output = (
            output_path.parent
            / f"{output_path.stem}.shard_{args.shard_id:03d}{output_path.suffix}"
        )
        logger.info(
            f"[shard {args.shard_id}/{args.num_shards}] "
            f"Writing to {shard_output}"
        )
    else:
        shard_output = output_path

    # ---- Resume: detect already-completed episodes -------------------
    if args.no_resume:
        done_eids: set[str] = set()
        file_mode = "w"
    else:
        done_eids = _completed_episode_ids(shard_output)
        file_mode = "a"
        if done_eids:
            logger.info(
                f"Resuming: {len(done_eids)} episodes already in {shard_output}"
            )

    # ---- Load models -------------------------------------------------
    logger.info("Loading policy...")
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    policy = PolicyModel(policy_cfg)
    policy.load(args.policy_path, for_inference=True)
    if torch.cuda.is_available():
        policy.model = policy.model.to("cuda")
        logger.info(f"Policy moved to GPU: {next(policy.model.parameters()).device}")
    policy.model.eval()

    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    policy.model.config.use_cache = True
    logger.info("KV cache enabled (gradient checkpointing disabled for inference)")

    logger.info("Loading verifier...")
    vcfg = OmegaConf.to_container(cfg.get("verifier", {}), resolve=True)
    verifier = create_verifier(vcfg)
    if hasattr(verifier, "eval"):
        verifier.eval()
    if torch.cuda.is_available():
        if hasattr(verifier, "to"):
            verifier = verifier.to("cuda")
            logger.info("Verifier moved to GPU")
        else:
            raise RuntimeError(
                "CUDA is available but verifier has no .to() method; "
                "cannot guarantee GPU execution for launch_router_features."
            )

    # ---- Start GPU keepalive -----------------------------------------
    if torch.cuda.is_available():
        _start_keepalive(
            policy.model, policy.tokenizer, policy.model.device, interval=8.0,
        )
        logger.info("GPU keepalive thread started (8 s interval)")

    # ---- Load & shard trajectories -----------------------------------
    store = TrajectoryStore(args.trajectories)
    all_episodes = store.load_episodes()
    episodes = all_episodes[args.shard_id :: args.num_shards]

    # Filter out already-completed episodes on resume
    if done_eids:
        episodes = [ep for ep in episodes if ep.episode_id not in done_eids]

    logger.info(
        f"[shard {args.shard_id}] {len(episodes)} episodes to process "
        f"(total {len(all_episodes)}, skipped {len(done_eids)} done)"
    )

    benchmark = _detect_benchmark(args.config)
    logger.info(f"Benchmark detected from config path: {benchmark!r}")

    max_steps = (
        cfg.get("inference", {}).get("step_limit", 15)
        if cfg.get("inference") else 15
    )
    inf_cfg = (
        OmegaConf.to_container(cfg.get("inference", {}), resolve=True)
        if cfg.get("inference") else {}
    )
    gen_temperature = float(inf_cfg.get("temperature", 0.7)) if inf_cfg else 0.7
    gen_max_tokens  = int(inf_cfg.get("max_tokens", 512))   if inf_cfg else 512
    K = args.K

    # ---- Pre-collect every (context, metadata) pair ------------------
    logger.info("Pre-collecting step items...")
    all_items: list[dict] = []
    for episode in episodes:
        goal = episode.metadata.goal if episode.metadata else ""
        slm_success = 1.0 if episode.success else 0.0
        llm_steps = sum(
            1 for s in episode.steps
            if s.action_source == ActionSource.TEACHER
        )
        cost = llm_steps / max(len(episode.steps), 1)

        context = ""
        for step_idx, step in enumerate(episode.steps):
            context += step.observation.raw_text + "\n"

            pert_type = None
            if (step.perturbation_type
                    and step.perturbation_type != PerturbationType.NONE):
                pert_type = step.perturbation_type.value

            all_items.append({
                "context": context,
                "step_idx": step_idx,
                "goal": goal,
                "pert_type": pert_type,
                "slm_success": slm_success,
                "cost": cost,
                "episode_id": episode.episode_id,
                "benchmark": benchmark,
            })

            context += step.action.raw_text + "\n"

    total_batches = (len(all_items) + args.batch_size - 1) // args.batch_size
    logger.info(
        f"[shard {args.shard_id}] Collected {len(all_items)} step items "
        f"from {len(episodes)} episodes -> {total_batches} batches "
        f"(batch_size={args.batch_size})"
    )

    # ---- Batched processing ------------------------------------------
    t0 = time.time()
    total_features = 0
    device = policy.model.device
    pad_id = policy.tokenizer.pad_token_id

    prefetcher = _PrefetchIter(
        all_items, args.batch_size,
        policy.tokenizer, policy.max_seq_len, gen_max_tokens,
    )

    with open(shard_output, file_mode) as f:
        for batch_idx, (batch, entropy_tok, gen_tok) in enumerate(prefetcher):
            batch_t0 = time.time()
            B = len(batch)

            # ---------- 1. Batched entropy ----------------------------
            try:
                ent_ids  = entropy_tok["input_ids"].to(device)
                ent_mask = entropy_tok["attention_mask"].to(device)
                with torch.no_grad():
                    ent_out = policy.model(
                        input_ids=ent_ids, attention_mask=ent_mask,
                    )
                seq_lens = ent_mask.sum(dim=1) - 1
                last_logits = ent_out.logits[
                    torch.arange(B, device=device), seq_lens
                ]
                entropies = (
                    torch.distributions.Categorical(logits=last_logits)
                    .entropy()
                    .tolist()
                )
            except Exception:
                entropies = [0.0] * B

            # ---------- 2. Batched candidate generation ---------------
            try:
                gen_ids  = gen_tok["input_ids"].to(device)
                gen_mask = gen_tok["attention_mask"].to(device)
                with torch.no_grad():
                    gen_out = policy.model.generate(
                        input_ids=gen_ids,
                        attention_mask=gen_mask,
                        max_new_tokens=gen_max_tokens,
                        num_return_sequences=K,
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
                    for k in range(K):
                        idx = b * K + k
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
            except Exception:
                all_candidates = [
                    [{"text": "", "log_prob": 0.0, "num_tokens": 0}] * K
                ] * B

            # ---------- 3. Batched verifier scoring -------------------
            flat_ctx, flat_cand, flat_goal = [], [], []
            for b in range(B):
                for c in all_candidates[b]:
                    flat_ctx.append(batch[b]["context"])
                    flat_cand.append(c["text"])
                    flat_goal.append(batch[b]["goal"])

            try:
                flat_scores = verifier.score_batch(
                    flat_ctx, flat_cand, flat_goal,
                )
            except Exception:
                flat_scores = [0.5] * len(flat_ctx)

            all_scores = [
                flat_scores[b * K : (b + 1) * K] for b in range(B)
            ]

            # ---------- 4. Assemble features & write ------------------
            for i, item in enumerate(batch):
                features = _assemble_features(
                    entropy=entropies[i],
                    scores=all_scores[i],
                    log_probs=[c["log_prob"] for c in all_candidates[i]],
                    step_idx=item["step_idx"],
                    max_steps=max_steps,
                    context_len=len(item["context"]),
                    pert_type=item["pert_type"],
                    candidate_texts=[c["text"] for c in all_candidates[i]],
                    benchmark=item["benchmark"],
                    goal=item["goal"],
                )
                record = {
                    "features": features,
                    "slm_success": item["slm_success"],
                    "cost": item["cost"],
                    "episode_id": item["episode_id"],
                    "step_idx": item["step_idx"],
                }
                f.write(json.dumps(record) + "\n")
                total_features += 1

            # ---------- 5. Progress logging ---------------------------
            batch_elapsed = time.time() - batch_t0
            elapsed = time.time() - t0
            steps_done = total_features
            rate = steps_done / elapsed if elapsed > 0 else 0
            remaining = len(all_items) - steps_done
            eta = remaining / rate if rate > 0 else float("inf")
            pct = 100.0 * steps_done / len(all_items) if all_items else 100.0

            logger.info(
                f"[shard {args.shard_id}] "
                f"Batch {batch_idx + 1}/{total_batches} "
                f"({pct:5.1f}%)  |  "
                f"{steps_done}/{len(all_items)} steps  |  "
                f"{rate:.1f} steps/s  |  "
                f"batch {batch_elapsed:.2f}s  |  "
                f"ETA {eta / 60:.1f}min"
            )

            # Flush every batch so progress is visible in log files
            f.flush()

    # ---- Cleanup -----------------------------------------------------
    if torch.cuda.is_available():
        _stop_keepalive()

    elapsed = time.time() - t0
    logger.info(
        f"[shard {args.shard_id}] Done: {total_features} feature vectors "
        f"from {len(episodes)} episodes in {elapsed / 60:.1f}min "
        f"({total_features / max(elapsed, 1e-6):.1f} steps/s) "
        f"-> {shard_output}"
    )


if __name__ == "__main__":
    main()
