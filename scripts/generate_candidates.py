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
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import glob
import json
import math
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
from r2v.utils.async_writer import AsyncJSONLWriter
from r2v.utils.config import load_config
from r2v.utils.logging import JSONLLogger, setup_logging

import torch.nn.functional as F


_keepalive_stop = threading.Event()


def _gpu_keepalive(device, interval: float):
    import torch
    # Tiny periodic matmul to keep GPU from appearing idle to cluster watchdogs.
    x = torch.randn((128, 128), device=device, dtype=torch.float16)
    while not _keepalive_stop.is_set():
        try:
            _ = x @ x
        except Exception:
            pass
        _keepalive_stop.wait(interval)


def start_gpu_keepalive(device, interval: float):
    _keepalive_stop.clear()
    t = threading.Thread(target=_gpu_keepalive, args=(device, interval), daemon=True)
    t.start()
    return t


def stop_gpu_keepalive():
    _keepalive_stop.set()


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

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

    def __init__(
        self,
        item_iter,
        batch_size,
        tokenizer,
        max_seq_len,
        max_new_tokens,
        tokenize_cache_size=256,
        prefetch_depth=2,
        logger=None,
        log_every_batches=10,
    ):
        self._item_iter = item_iter
        self._bs = batch_size
        self._tok = tokenizer
        self._max_seq = max_seq_len
        self._max_new = max_new_tokens
        self._logger = logger
        self._log_every_batches = max(1, int(log_every_batches))
        self._cache_size = max(0, tokenize_cache_size)
        self._tok_cache: OrderedDict[tuple[str, ...], dict] = OrderedDict()
        self._q: queue.Queue = queue.Queue(maxsize=max(1, prefetch_depth))
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self):
        try:
            batch = []
            produced = 0
            for item in self._item_iter:
                batch.append(item)
                if len(batch) < self._bs:
                    continue
                self._q.put((batch, self._tokenize_batch(batch)))
                produced += 1
                if self._logger and (produced == 1 or produced % self._log_every_batches == 0):
                    self._logger.info(
                        f"Prefetch prepared {produced} batch(es) "
                        f"(queue={self._q.qsize()})"
                    )
                batch = []

            if batch:
                self._q.put((batch, self._tokenize_batch(batch)))
                produced += 1
                if self._logger:
                    self._logger.info(
                        f"Prefetch prepared final partial batch "
                        f"(total={produced}, queue={self._q.qsize()})"
                    )

            self._q.put(None)  # sentinel
            if self._logger:
                self._logger.info("Prefetch producer finished")
        except Exception as exc:  # pragma: no cover
            self._q.put(exc)

    def _tokenize_batch(self, batch):
        prompts = [f"{it['context']}\nAction:" for it in batch]
        cache_key = tuple(prompts)

        if self._cache_size > 0:
            cached = self._tok_cache.get(cache_key)
            if cached is not None:
                self._tok_cache.move_to_end(cache_key)
                return cached

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

        if self._cache_size > 0:
            self._tok_cache[cache_key] = gen_tok
            while len(self._tok_cache) > self._cache_size:
                self._tok_cache.popitem(last=False)

        return gen_tok

    def __iter__(self):
        while True:
            item = self._q.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                break
            yield item


def iter_step_items(episodes):
    """Yield step items lazily to avoid large pre-collection stalls/memory."""
    for episode in episodes:
        goal = episode.metadata.goal if episode.metadata else ""
        context = ""
        for step_idx, step in enumerate(episode.steps):
            context += step.observation.raw_text + "\n"
            yield {
                "context": context,
                "step_idx": step_idx,
                "goal": goal,
                "episode_id": episode.episode_id,
            }
            context += step.action.raw_text + "\n"


def iter_sharded_episodes(store: TrajectoryStore, shard_id: int, num_shards: int, done_eids: set[str]):
    """Stream episodes for one shard directly from JSONL to keep RAM bounded."""
    for idx, ep in enumerate(store.iter_episodes()):
        if idx % num_shards != shard_id:
            continue
        if ep.episode_id in done_eids:
            continue
        yield ep


def count_sharded_steps(store: TrajectoryStore, shard_id: int, num_shards: int, done_eids: set[str]) -> tuple[int, int]:
    """Count (steps, episodes) for this shard via a lightweight streaming pass."""
    total_steps = 0
    total_eps = 0
    for idx, ep in enumerate(store.iter_episodes()):
        if idx % num_shards != shard_id:
            continue
        if ep.episode_id in done_eids:
            continue
        total_eps += 1
        total_steps += len(ep.steps)
    return total_steps, total_eps


def count_sharded_steps_with_logging(
    store: TrajectoryStore,
    shard_id: int,
    num_shards: int,
    done_eids: set[str],
    logger,
    log_every_episodes: int = 200,
) -> tuple[int, int]:
    """Streaming count with progress logs so long scans don't look stalled."""
    total_steps = 0
    total_eps = 0
    scanned = 0
    t0 = time.time()
    every = max(1, log_every_episodes)

    for idx, ep in enumerate(store.iter_episodes()):
        scanned += 1
        if idx % num_shards != shard_id:
            continue
        if ep.episode_id in done_eids:
            continue
        total_eps += 1
        total_steps += len(ep.steps)

        if total_eps % every == 0:
            elapsed = max(time.time() - t0, 1e-6)
            logger.info(
                f"Shard scan progress: kept={total_eps} episodes, "
                f"steps={total_steps}, scanned={scanned}, "
                f"rate={scanned / elapsed:.1f} eps/s"
            )

    elapsed = max(time.time() - t0, 1e-6)
    logger.info(
        f"Shard scan complete: kept={total_eps} episodes, "
        f"steps={total_steps}, scanned={scanned}, "
        f"elapsed={elapsed:.1f}s"
    )
    return total_steps, total_eps


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
    parser.add_argument(
        "--tokenize-cache-size",
        type=int,
        default=256,
        help="Number of tokenized prompt batches to cache in prefetch thread",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=4,
        help="Number of prefetched tokenized batches kept on CPU",
    )
    parser.add_argument(
        "--gpu-keepalive-interval",
        type=float,
        default=1.5,
        help="Seconds between tiny GPU keepalive kernels (0 to disable)",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=3,
        help="Max number of in-flight scorer/write batches (must be >=1)",
    )
    parser.add_argument(
        "--gen-micro-batch-size",
        type=int,
        default=0,
        help=(
            "Generation micro-batch size per shard batch (0 = auto/full). "
            "Lower values reduce CUDA OOM risk."
        ),
    )
    parser.add_argument(
        "--store-all-candidates",
        action="store_true",
        help=(
            "Store full sorted candidate list in output JSONL. "
            "Disabled by default for faster generation and smaller files."
        ),
    )
    parser.add_argument(
        "--compute-logprobs",
        action="store_true",
        help="Compute candidate log-probabilities (slower; not needed for DPO pairs)",
    )
    # Multi-GPU sharding
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This worker's shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--merge", action="store_true",
                        help="Merge shard outputs instead of generating")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume; overwrite existing output")
    parser.add_argument(
        "--write-queue-size",
        type=int,
        default=50000,
        help="Max pending JSON records buffered by async writer (0 = unbounded)",
    )
    parser.add_argument(
        "--write-flush-every",
        type=int,
        default=1024,
        help="Async writer flush interval in records",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument(
        "--scan-log-every-episodes",
        type=int,
        default=200,
        help="Log shard-scan progress every N kept episodes",
    )
    parser.add_argument(
        "--prefetch-log-every-batches",
        type=int,
        default=10,
        help="Log tokenization prefetch progress every N batches",
    )
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


def _score_and_write_batch(
    verifier,
    writer: AsyncJSONLWriter,
    batch: list[dict],
    all_candidates: list[list[dict]],
    store_all_candidates: bool,
    logger=None,
) -> int:
    """Score one generated batch and enqueue preference pairs for async JSON write."""
    B = len(batch)
    K = len(all_candidates[0]) if B > 0 and all_candidates[0] else 0

    flat_ctx: list[str] = []
    flat_cand: list[str] = []
    flat_goal: list[str] = []
    for b in range(B):
        for c in all_candidates[b]:
            flat_ctx.append(batch[b]["context"])
            flat_cand.append(c["text"])
            flat_goal.append(batch[b]["goal"])

    try:
        flat_scores = verifier.score_batch(flat_ctx, flat_cand, flat_goal)
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Verifier scoring failed for a batch; using 0.5 fallback. Error: {exc}")
        flat_scores = [0.5] * len(flat_ctx)

    all_scores = [
        flat_scores[b * K : (b + 1) * K] for b in range(B)
    ]

    written = 0
    for i, item in enumerate(batch):
        scored = [
            {
                "action": c["text"],
                "log_prob": c["log_prob"],
                "verifier_score": sc,
            }
            for c, sc in zip(all_candidates[i], all_scores[i])
        ]

        if len(scored) >= 2:
            # Pick best normally.
            best = max(scored, key=lambda x: x["verifier_score"])

            # For rejected, prefer the lowest-scored candidate with an action text
            # different from chosen. This avoids degenerate chosen==rejected pairs
            # when scores tie or verifier falls back to constant values.
            ranked_worst_first = sorted(scored, key=lambda x: x["verifier_score"])
            worst = None
            for cand in ranked_worst_first:
                if cand["action"] != best["action"]:
                    worst = cand
                    break

            # If all candidate texts are identical, skip this step pair.
            if worst is None:
                continue

            pair = {
                "context": item["context"],
                "chosen": best["action"],
                "rejected": worst["action"],
                "chosen_score": best["verifier_score"],
                "rejected_score": worst["verifier_score"],
                "episode_id": item["episode_id"],
                "step_idx": item["step_idx"],
            }
            if store_all_candidates:
                pair["all_candidates"] = sorted(
                    scored, key=lambda x: x["verifier_score"], reverse=True
                )
            writer.write(pair)
            written += 1

    return written


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "out of memory" in msg


def _generate_candidates_adaptive(
    policy,
    gen_ids,
    gen_mask,
    *,
    K: int,
    gen_max_tokens: int,
    gen_temperature: float,
    pad_id,
    compute_logprobs: bool,
    logger,
    shard_id: int,
    initial_micro_batch_size: int = 0,
):
    """Generate candidates with adaptive micro-batching on CUDA OOM.

    Returns
    -------
    all_candidates : list[list[dict]]
        Per-item candidate lists, length B x K.
    """
    import torch as _torch

    B = gen_ids.shape[0]
    all_candidates: list[list[dict]] = []

    if initial_micro_batch_size <= 0:
        micro_bs = B
    else:
        micro_bs = max(1, min(initial_micro_batch_size, B))

    start = 0
    while start < B:
        cur_bs = min(micro_bs, B - start)
        chunk_done = False

        while not chunk_done:
            try:
                ids = gen_ids[start : start + cur_bs]
                mask = gen_mask[start : start + cur_bs]
                with _torch.inference_mode():
                    gen_out = policy.model.generate(
                        input_ids=ids,
                        attention_mask=mask,
                        max_new_tokens=gen_max_tokens,
                        num_return_sequences=K,
                        temperature=gen_temperature,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=pad_id,
                        return_dict_in_generate=True,
                        output_scores=compute_logprobs,
                    )

                prompt_len = ids.shape[1]
                gen_only = gen_out.sequences[:, prompt_len:]
                decoded_texts = [
                    t.strip()
                    for t in policy.tokenizer.batch_decode(
                        gen_only, skip_special_tokens=True
                    )
                ]
                if pad_id is None:
                    token_lens = [int(x.numel()) for x in gen_only]
                else:
                    token_lens = (gen_only != pad_id).sum(dim=1).tolist()

                for b in range(cur_bs):
                    cands = []
                    for k in range(K):
                        idx = b * K + k
                        gen_tokens = gen_only[idx]
                        lp = 0.0
                        if compute_logprobs and gen_out.scores:
                            if pad_id is not None:
                                gen_tokens = gen_tokens[gen_tokens != pad_id]
                            for t, sc in enumerate(gen_out.scores):
                                if t >= len(gen_tokens):
                                    break
                                lp += F.log_softmax(sc[idx], dim=-1)[gen_tokens[t]].item()

                        cands.append(
                            {
                                "text": decoded_texts[idx],
                                "log_prob": lp,
                                "num_tokens": token_lens[idx],
                            }
                        )
                    all_candidates.append(cands)

                start += cur_bs
                chunk_done = True
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()

                if cur_bs == 1:
                    logger.warning(
                        f"[shard {shard_id}] OOM at micro-batch=1; using empty candidates for one item"
                    )
                    all_candidates.append(
                        [{"text": "", "log_prob": 0.0, "num_tokens": 0}] * K
                    )
                    start += 1
                    chunk_done = True
                else:
                    new_bs = max(1, cur_bs // 2)
                    logger.warning(
                        f"[shard {shard_id}] CUDA OOM at micro-batch={cur_bs}; retrying with {new_bs}"
                    )
                    cur_bs = new_bs

    return all_candidates


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
    import torch as _torch
    if _torch.cuda.is_available():
        try:
            policy.model = policy.model.to("cuda")
            logger.info("Policy moved to CUDA for inference")
        except Exception as exc:
            logger.warning(f"Could not move policy to CUDA explicitly: {exc}")

    # For inference throughput, ensure KV cache is enabled.
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    if hasattr(policy.model, "config"):
        policy.model.config.use_cache = True
    policy.model.eval()

    # Load verifier — force mode=trained when --verifier-path is given
    logger.info("Loading verifier...")
    vcfg = OmegaConf.to_container(cfg.get("verifier", {}), resolve=True)
    if args.verifier_path:
        vcfg["mode"] = "trained"
    verifier = create_verifier(vcfg)

    # Load trained weights if provided
    if args.verifier_path:
        state_dict = _torch.load(args.verifier_path, map_location="cpu")
        verifier.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded trained verifier weights from {args.verifier_path}")

    # Move verifier to GPU if available
    if hasattr(verifier, 'to') and _torch.cuda.is_available():
        try:
            verifier = verifier.to(device="cuda", dtype=_torch.float16)
            logger.info("Verifier moved to CUDA fp16 for inference")
        except TypeError:
            verifier = verifier.to("cuda")
            if hasattr(verifier, "half"):
                verifier = verifier.half()
                logger.info("Verifier moved to CUDA and cast to fp16")
    if hasattr(verifier, 'eval'):
        verifier.eval()

    if _torch.cuda.is_available() and args.gpu_keepalive_interval > 0:
        start_gpu_keepalive(policy.model.device, args.gpu_keepalive_interval)
        logger.info(
            "GPU keepalive enabled "
            f"(interval={args.gpu_keepalive_interval:.2f}s)"
        )

    # --------------- Load & shard trajectories (streamed) ---------------
    store = TrajectoryStore(args.trajectories)

    # Resolve generation hyperparams
    inf_cfg = cfg.get("inference", {})
    gen_temperature = float(inf_cfg.get("temperature", 0.7)) if inf_cfg else 0.7
    if args.max_new_tokens is not None:
        gen_max_tokens = args.max_new_tokens
    else:
        gen_max_tokens = int(inf_cfg.get("max_tokens", 256)) if inf_cfg else 256

    # --------------- Stream step items (no giant pre-collect stall) ---------------
    batch_size = args.batch_size
    logger.info("Starting shard scan (streaming pass for step/episode counts)...")
    total_steps, total_episodes = count_sharded_steps_with_logging(
        store,
        args.shard_id,
        args.num_shards,
        done_eids,
        logger,
        log_every_episodes=args.scan_log_every_episodes,
    )
    total_batches = math.ceil(total_steps / batch_size) if total_steps else 0
    logger.info(
        f"Processing {total_steps} steps from {total_episodes} episodes "
        f"(skipped {len(done_eids)} done), "
        f"K={args.K}, batch_size={batch_size}, "
        f"max_new_tokens={gen_max_tokens}, temp={gen_temperature}, "
        f"compute_logprobs={args.compute_logprobs}"
    )

    # --------------- Prefetch iterator ---------------
    prefetcher = _PrefetchIter(
        iter_step_items(iter_sharded_episodes(store, args.shard_id, args.num_shards, done_eids)), batch_size,
        policy.tokenizer, policy.max_seq_len, gen_max_tokens,
        tokenize_cache_size=args.tokenize_cache_size,
        prefetch_depth=args.prefetch_depth,
        logger=logger,
        log_every_batches=args.prefetch_log_every_batches,
    )

    # --------------- Batched main loop ---------------
    total_pairs = 0
    t0 = time.time()
    device = policy.model.device
    pad_id = policy.tokenizer.pad_token_id

    writer = AsyncJSONLWriter(
        shard_output,
        mode=file_mode,
        maxsize=max(0, args.write_queue_size),
        flush_every=max(1, args.write_flush_every),
    ).start()

    pipeline_depth = max(1, args.pipeline_depth)
    executor = ThreadPoolExecutor(max_workers=pipeline_depth)
    pending_futures: list[Future[int]] = []

    try:
        for batch_idx, (batch, gen_tok) in enumerate(prefetcher):
            batch_t0 = time.time()
            gen_t0 = time.time()
            B = len(batch)

            # ---------- 1. Batched candidate generation ---------------
            try:
                gen_ids = gen_tok["input_ids"].to(device)
                gen_mask = gen_tok["attention_mask"].to(device)
                all_candidates = _generate_candidates_adaptive(
                    policy,
                    gen_ids,
                    gen_mask,
                    K=args.K,
                    gen_max_tokens=gen_max_tokens,
                    gen_temperature=gen_temperature,
                    pad_id=pad_id,
                    compute_logprobs=args.compute_logprobs,
                    logger=logger,
                    shard_id=args.shard_id,
                    initial_micro_batch_size=args.gen_micro_batch_size,
                )
                gen_elapsed = time.time() - gen_t0
            except Exception as exc:
                logger.warning(f"Generation failed for batch {batch_idx}: {exc}")
                all_candidates = [
                    [{"text": "", "log_prob": 0.0, "num_tokens": 0}] * args.K
                ] * B
                gen_elapsed = time.time() - gen_t0

            # ---------- 2. Pipeline verifier scoring + write ----------
            score_submit_t0 = time.time()
            pending_futures.append(executor.submit(
                _score_and_write_batch,
                verifier,
                writer,
                batch,
                all_candidates,
                args.store_all_candidates,
                logger,
            ))
            submit_elapsed = time.time() - score_submit_t0

            if len(pending_futures) >= pipeline_depth:
                wait_t0 = time.time()
                total_pairs += pending_futures.pop(0).result()
                wait_elapsed = time.time() - wait_t0
            else:
                wait_elapsed = 0.0

            # ---------- 4. Progress logging ---------------------------
            batch_elapsed = time.time() - batch_t0
            elapsed = time.time() - t0
            steps_done = min((batch_idx + 1) * batch_size, total_steps)
            rate = steps_done / elapsed if elapsed > 0 else 0
            remaining = total_steps - steps_done
            eta = remaining / rate if rate > 0 else float("inf")
            pct = 100.0 * steps_done / total_steps if total_steps else 100.0

            logger.info(
                f"[shard {args.shard_id}] "
                f"Batch {batch_idx + 1}/{total_batches} "
                f"({pct:5.1f}%)  |  "
                f"{total_pairs} pairs  |  "
                f"{rate:.1f} steps/s  |  "
                f"gen {gen_elapsed:.2f}s  |  "
                f"submit {submit_elapsed:.3f}s  |  "
                f"wait {wait_elapsed:.2f}s  |  "
                f"batch {batch_elapsed:.2f}s  |  "
                f"write_q={writer.pending}  |  "
                f"ETA {eta / 60:.1f}min"
            )

        for fut in pending_futures:
            total_pairs += fut.result()
    finally:
        stop_gpu_keepalive()
        executor.shutdown(wait=False)
        # Ensure pending writes are fully persisted before process exit.
        writer.close()

    # --------------- Cleanup ---------------
    elapsed = time.time() - t0
    logger.info(
        f"[shard {args.shard_id}] Done: {total_pairs} pairs from "
        f"{total_episodes} episodes ({total_steps} steps) in {elapsed / 60:.1f}min "
        f"({total_steps / max(elapsed, 1e-6):.1f} steps/s) → {shard_output}"
    )
    jsonl_log.log("summary", {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "total_pairs": total_pairs,
        "num_episodes": total_episodes,
        "K": args.K,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
    })


if __name__ == "__main__":
    main()
