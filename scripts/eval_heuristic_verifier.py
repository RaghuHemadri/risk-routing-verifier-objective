"""
Evaluate the HeuristicVerifier on HumanEval and TextWorld splits.

Metrics:
  - Episode-level AUROC: aggregate step scores → predict episode success
  - Spearman correlation between step score and environment reward (TextWorld)
  - Score distribution by episode outcome (success/failure)
  - Step-level calibration: mean score per action type
  - Reward-hacking detection rate (false positive / negative)

Usage:
    python scripts/eval_heuristic_verifier.py \
        --humaneval-train data/trajectories/humaneval_noisy/verifier_train.jsonl \
        --humaneval-val   data/trajectories/humaneval_noisy/verifier_val.jsonl \
        --humaneval-test  data/trajectories/humaneval_noisy/verifier_test.jsonl \
        --textworld-train data/trajectories/textworld_train.jsonl \
        --textworld-val   data/trajectories/textworld_val.jsonl \
        --textworld-test  data/trajectories/textworld_test.jsonl \
        --no-run-code      # skip subprocess execution (fast mode)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── Optional scipy for stats ──────────────────────────────────
try:
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warn] scipy not available — skipping correlation metrics")

# ── Add project root to path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r2v.models.heuristic_verifier import HeuristicVerifier, _detect_benchmark

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────

def load_episodes(path: str) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def episode_goal(ep: dict) -> str:
    return ep.get("metadata", {}).get("goal", "")


def episode_benchmark(ep: dict) -> str:
    bm = ep.get("metadata", {}).get("benchmark", "")
    return bm if bm else _detect_benchmark("", episode_goal(ep))


def build_context(steps: list[dict], up_to_idx: int) -> str:
    """
    Linearise steps into context string for scoring steps[up_to_idx].

    Includes:
      - For each past step (0..up_to_idx-1): observation + action
      - For the current step (up_to_idx):    observation only
        (this is crucial — the agent sees the observation BEFORE choosing its action,
         so e.g. the test result observation is visible when scoring 'submit')

    Truncation strategy:
      - Older observations/actions get 400 chars (enough for game obs)
      - The most recent past observation gets 1500 chars (may contain code)
      - The current step's observation gets 3000 chars (test output + full code)
      - write_code actions get 2000 chars (full code)
    """
    parts = []
    n = up_to_idx + 1
    for i, s in enumerate(steps[:n]):
        obs = s.get("observation", {})
        obs_text = obs.get("raw_text", "") if isinstance(obs, dict) else str(obs)
        act = s.get("action", {})
        act_text = act.get("raw_text", "") if isinstance(act, dict) else str(act)

        is_current = (i == up_to_idx)
        is_recent  = (i == up_to_idx - 1)

        if obs_text:
            if is_current:
                obs_limit = 3000
            elif is_recent:
                obs_limit = 1500
            else:
                obs_limit = 400
            parts.append(f"Observation: {obs_text[:obs_limit]}")

        if not is_current and act_text:
            # Larger limit for write_code actions to preserve full code
            act_lower = act_text.strip().lower()
            if act_lower.startswith("write_code") or act_lower.startswith("def "):
                act_limit = 2000
            else:
                act_limit = 200
            parts.append(f"Action: {act_text[:act_limit]}")
    return "\n".join(parts)


def collect_step_records(
    episodes: list[dict],
    verifier: HeuristicVerifier,
    benchmark: str,
    max_episodes: int = 0,
    show_progress: bool = True,
) -> list[dict]:
    """
    Score every step in episodes and return flat list of records.
    Each record: {
        episode_id, step_idx, score, reward, episode_success,
        action_type, benchmark, perturbation_type
    }
    """
    records = []
    total = min(len(episodes), max_episodes) if max_episodes else len(episodes)
    t0 = time.time()

    for i, ep in enumerate(episodes[:total]):
        if show_progress and (i % 100 == 0 or i == total - 1):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            eta = (total - i - 1) / max(rate, 0.001)
            print(f"  [{benchmark}] {i+1}/{total} episodes  "
                  f"({rate:.1f} ep/s, ETA {eta:.0f}s)", end="\r", flush=True)

        ep_id = ep.get("episode_id", f"ep_{i}")
        success = bool(ep.get("success", False))
        goal = episode_goal(ep)
        steps = ep.get("steps", [])
        perturb = ep.get("perturbation_type", "none")

        for j, step in enumerate(steps):
            context = build_context(steps, j)
            act = step.get("action", {})
            act_text = act.get("raw_text", "") if isinstance(act, dict) else str(act)
            reward = float(step.get("reward", 0.0))

            score = verifier.score(context, act_text, goal)

            # Action type label
            act_lower = act_text.strip().lower()
            if act_lower.startswith("write_code") or act_lower.startswith("def "):
                atype = "write_code"
            elif act_lower.startswith("test"):
                atype = "test"
            elif act_lower == "submit":
                atype = "submit"
            elif act_lower.startswith("go"):
                atype = "go"
            elif act_lower in ("look", "inventory", "examine"):
                atype = "idle"
            else:
                atype = "other"

            records.append({
                "episode_id": ep_id,
                "step_idx": j,
                "score": score,
                "reward": reward,
                "episode_success": success,
                "action_type": atype,
                "benchmark": benchmark,
                "perturbation_type": perturb,
            })

    if show_progress:
        print()  # newline after progress

    return records


# ──────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────

def auroc(y_true: list[float], y_score: list[float]) -> float:
    """Manual AUROC via trapezoidal rule (avoids sklearn dependency)."""
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    P = sum(y_true)
    N = len(y_true) - P
    if P == 0 or N == 0:
        return float("nan")
    tp, fp = 0, 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += prev_tp  # trapezoidal area
        prev_fp = fp
        prev_tp = tp
    return auc / (P * N) if P * N > 0 else float("nan")


def brier_score(y_true: list[float], y_pred: list[float]) -> float:
    return float(np.mean([(p - t) ** 2 for p, t in zip(y_pred, y_true)]))


def ece_score(y_true: list[float], y_pred: list[float], n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) — lower is better.

    Partitions predictions into equal-width bins and measures the weighted
    mean absolute deviation between mean confidence and fraction of positives.

    ECE = sum_b (|bin_b| / n) * |conf_b - acc_b|

    Returns NaN if fewer than 2 samples.
    """
    if len(y_true) < 2:
        return float("nan")
    n = len(y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        idx = [i for i, p in enumerate(y_pred) if lo <= p < hi]
        if not idx:
            continue
        # last bin: include 1.0
        if hi == 1.0:
            idx = [i for i, p in enumerate(y_pred) if lo <= p <= hi]
        if not idx:
            continue
        conf = float(np.mean([y_pred[i] for i in idx]))
        acc  = float(np.mean([y_true[i] for i in idx]))
        ece += (len(idx) / n) * abs(conf - acc)
    return float(ece)


def reliability_diagram_data(
    y_true: list[float], y_pred: list[float], n_bins: int = 10
) -> list[dict]:
    """Return per-bin calibration data for plotting reliability diagrams."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        idx = [i for i, p in enumerate(y_pred) if lo <= p <= hi]
        if not idx:
            continue
        rows.append({
            "bin_lo": round(float(lo), 2),
            "bin_hi": round(float(hi), 2),
            "n": len(idx),
            "mean_conf": round(float(np.mean([y_pred[i] for i in idx])), 4),
            "mean_acc":  round(float(np.mean([y_true[i] for i in idx])), 4),
        })
    return rows


def aggregate_episode_score(
    records: list[dict],
    agg: str = "mean",
) -> dict[str, dict]:
    """Aggregate step scores per episode → {ep_id: {score, success}}."""
    ep_scores: dict[str, list[float]] = defaultdict(list)
    ep_success: dict[str, bool] = {}
    for r in records:
        ep_scores[r["episode_id"]].append(r["score"])
        ep_success[r["episode_id"]] = r["episode_success"]

    result = {}
    for ep_id, scores in ep_scores.items():
        arr = np.array(scores)
        if agg == "mean":
            agg_score = float(arr.mean())
        elif agg == "min":
            agg_score = float(arr.min())
        elif agg == "last":
            agg_score = float(arr[-1])
        elif agg == "max":
            agg_score = float(arr.max())
        else:
            agg_score = float(arr.mean())
        result[ep_id] = {"score": agg_score, "success": ep_success[ep_id]}
    return result


def compute_metrics(records: list[dict], split_name: str, benchmark: str) -> dict:
    """Compute all metrics for a given set of step records."""
    if not records:
        return {"split": split_name, "benchmark": benchmark, "n_steps": 0}

    scores = [r["score"] for r in records]
    rewards = [r["reward"] for r in records]
    successes = [float(r["episode_success"]) for r in records]
    n_eps = len({r["episode_id"] for r in records})

    # ── Episode-level AUROC ────────────────────────────────────
    ep_data = aggregate_episode_score(records, agg="mean")
    ep_scores = [v["score"] for v in ep_data.values()]
    ep_labels = [float(v["success"]) for v in ep_data.values()]
    ep_auroc = auroc(ep_labels, ep_scores)
    ep_brier = brier_score(ep_labels, ep_scores)
    ep_ece = ece_score(ep_labels, ep_scores)
    ep_reliability = reliability_diagram_data(ep_labels, ep_scores)

    # Also try last-step and min aggregations
    ep_last = aggregate_episode_score(records, agg="last")
    ep_last_scores = [v["score"] for v in ep_last.values()]
    ep_last_labels = [float(v["success"]) for v in ep_last.values()]
    ep_auroc_last = auroc(ep_last_labels, ep_last_scores)

    # ── Step-level: score vs reward (TextWorld) ────────────────
    nonzero_rewards = [(s, r) for s, r in zip(scores, rewards) if r > 0 or True]
    spearman_reward = float("nan")
    if HAS_SCIPY and len(nonzero_rewards) >= 10:
        s_arr = [x[0] for x in nonzero_rewards]
        r_arr = [x[1] for x in nonzero_rewards]
        try:
            corr, _ = spearmanr(s_arr, r_arr)
            spearman_reward = float(corr)
        except Exception:
            pass

    # ── Score distributions by outcome ────────────────────────
    succ_scores = [r["score"] for r in records if r["episode_success"]]
    fail_scores = [r["score"] for r in records if not r["episode_success"]]

    # ── Score by action type ───────────────────────────────────
    by_atype: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_atype[r["action_type"]].append(r["score"])

    # ── Score by perturbation type ─────────────────────────────
    by_perturb: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_perturb[r["perturbation_type"]].append(r["score"])

    return {
        "split": split_name,
        "benchmark": benchmark,
        "n_episodes": n_eps,
        "n_steps": len(records),
        "success_rate": float(np.mean([r["episode_success"] for r in records])),
        # Episode-level
        "ep_auroc_mean_agg": round(ep_auroc, 4),
        "ep_auroc_last_agg": round(ep_auroc_last, 4),
        "ep_brier": round(ep_brier, 4),
        "ep_ece": round(ep_ece, 4),
        "ep_reliability_diagram": ep_reliability,
        # Step-level distributions
        "step_score_mean": round(float(np.mean(scores)), 4),
        "step_score_std": round(float(np.std(scores)), 4),
        "step_score_success_mean": round(float(np.mean(succ_scores)) if succ_scores else float("nan"), 4),
        "step_score_fail_mean": round(float(np.mean(fail_scores)) if fail_scores else float("nan"), 4),
        "score_gap_success_vs_fail": round(
            (float(np.mean(succ_scores)) if succ_scores else 0)
            - (float(np.mean(fail_scores)) if fail_scores else 0), 4
        ),
        # Correlation with env rewards
        "spearman_vs_reward": round(spearman_reward, 4),
        # By action type
        "score_by_action_type": {
            k: round(float(np.mean(v)), 4) for k, v in sorted(by_atype.items())
        },
        # By perturbation type
        "score_by_perturbation": {
            k: round(float(np.mean(v)), 4) for k, v in sorted(by_perturb.items())
        },
    }


def print_metrics(m: dict) -> None:
    sep = "─" * 55
    print(f"\n{'━'*55}")
    print(f"  {m['benchmark'].upper()} | {m['split'].upper()}")
    print(f"{'━'*55}")
    print(f"  Episodes       : {m.get('n_episodes', '?')}")
    print(f"  Steps          : {m.get('n_steps', '?')}")
    print(f"  Success rate   : {m.get('success_rate', 0):.1%}")
    print(sep)
    print(f"  ep AUROC (mean agg)  : {m.get('ep_auroc_mean_agg', 'n/a')}")
    print(f"  ep AUROC (last step) : {m.get('ep_auroc_last_agg', 'n/a')}")
    print(f"  ep Brier score       : {m.get('ep_brier', 'n/a')}")
    print(f"  ep ECE               : {m.get('ep_ece', 'n/a')}")
    print(sep)
    print(f"  Step score  mean : {m.get('step_score_mean', 'n/a')} ± {m.get('step_score_std', '?')}")
    print(f"  Score gap (succ - fail): {m.get('score_gap_success_vs_fail', 'n/a')}")
    print(f"    success steps  : {m.get('step_score_success_mean', 'n/a')}")
    print(f"    failure steps  : {m.get('step_score_fail_mean', 'n/a')}")
    print(f"  Spearman vs reward: {m.get('spearman_vs_reward', 'n/a')}")
    print(sep)
    print("  Score by action type:")
    for k, v in m.get("score_by_action_type", {}).items():
        print(f"    {k:<18} {v:.4f}")
    print("  Score by perturbation:")
    for k, v in m.get("score_by_perturbation", {}).items():
        print(f"    {k:<30} {v:.4f}")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HeuristicVerifier")
    p.add_argument("--humaneval-train", default="data/trajectories/humaneval_noisy/verifier_train.jsonl")
    p.add_argument("--humaneval-val",   default="data/trajectories/humaneval_noisy/verifier_val.jsonl")
    p.add_argument("--humaneval-test",  default="data/trajectories/humaneval_noisy/verifier_test.jsonl")
    p.add_argument("--textworld-train", default="data/trajectories/textworld_train.jsonl")
    p.add_argument("--textworld-val",   default="data/trajectories/textworld_val.jsonl")
    p.add_argument("--textworld-test",  default="data/trajectories/textworld_test.jsonl")
    p.add_argument("--no-run-code", action="store_true",
                   help="Disable subprocess code execution (faster but lower quality)")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="Limit episodes per split (0=all)")
    p.add_argument("--output", default="results/heuristic_verifier_eval.json",
                   help="JSON output path")
    p.add_argument("--benchmark", choices=["humaneval", "textworld", "both"],
                   default="both")
    return p.parse_args()


def main():
    args = parse_args()
    run_code = not args.no_run_code

    print(f"\n{'='*55}")
    print(f"  Heuristic Verifier Evaluation")
    print(f"  run_code={run_code}")
    print(f"{'='*55}")

    verifier = HeuristicVerifier(run_code=run_code)
    all_metrics = []

    # ── HumanEval ─────────────────────────────────────────────
    if args.benchmark in ("humaneval", "both"):
        he_splits = [
            ("train", args.humaneval_train),
            ("val",   args.humaneval_val),
            ("test",  args.humaneval_test),
        ]
        for split_name, path in he_splits:
            if not Path(path).exists():
                print(f"[skip] {path} not found")
                continue
            print(f"\nLoading HumanEval {split_name} from {path}...")
            episodes = load_episodes(path)
            print(f"  {len(episodes)} episodes loaded")
            records = collect_step_records(
                episodes, verifier, benchmark="humaneval",
                max_episodes=args.max_episodes,
            )
            m = compute_metrics(records, split_name, "humaneval")
            print_metrics(m)
            all_metrics.append(m)

    # ── TextWorld ─────────────────────────────────────────────
    if args.benchmark in ("textworld", "both"):
        tw_splits = [
            ("train", args.textworld_train),
            ("val",   args.textworld_val),
            ("test",  args.textworld_test),
        ]
        for split_name, path in tw_splits:
            if not Path(path).exists():
                print(f"[skip] {path} not found")
                continue
            print(f"\nLoading TextWorld {split_name} from {path}...")
            episodes = load_episodes(path)
            print(f"  {len(episodes)} episodes loaded")
            records = collect_step_records(
                episodes, verifier, benchmark="textworld",
                max_episodes=args.max_episodes,
            )
            m = compute_metrics(records, split_name, "textworld")
            print_metrics(m)
            all_metrics.append(m)

    # ── Save results ──────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n\nResults saved to {out}")


if __name__ == "__main__":
    main()
