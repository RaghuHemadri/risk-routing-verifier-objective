#!/usr/bin/env python3
"""Create a task-disjoint TextWorld train/validation split.

The splitter is designed for trajectory datasets where each task has
multiple episodes (e.g. different seeds). It enforces task independence by
placing all episodes from a task into exactly one split.

Balancing targets:
1) Success rate
2) Toughness score (derived from per-task behavior)
3) Average step count

Usage:
    python scripts/create_textworld_val_split.py \
      --input data/runs/textworld_train_google_gemini-3-flash-preview_v2/trajectories.jsonl \
      --out-train data/trajectories/textworld_train_balanced.jsonl \
      --out-val data/trajectories/textworld_val_balanced.jsonl \
      --report outputs/perturbation_checks/textworld_split_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TaskStats:
    task_id: str
    n_episodes: int
    success_rate: float
    avg_steps: float
    dead_end_rate: float
    unknown_action_rate: float
    look_inventory_ratio: float
    toughness: float


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _compute_task_stats(rows: list[dict[str, Any]]) -> dict[str, TaskStats]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in rows:
        task_id = ep.get("metadata", {}).get("task_id")
        if task_id:
            by_task[task_id].append(ep)

    raw: dict[str, dict[str, float]] = {}
    max_avg_steps = 1.0

    for task_id, eps in by_task.items():
        n_eps = len(eps)
        successes = sum(1 for ep in eps if ep.get("success", False))
        total_steps = 0
        dead_end = 0
        total_actions = 0
        unknown_actions = 0
        look_inventory = 0

        for ep in eps:
            steps = ep.get("steps", [])
            total_steps += len(steps)
            for step in steps:
                total_actions += 1
                obs_text = (step.get("observation", {}) or {}).get("raw_text", "")
                if "That action did not help" in obs_text:
                    dead_end += 1
                action_type = (step.get("action", {}) or {}).get("action_type", "")
                if action_type == "unknown":
                    unknown_actions += 1
                if action_type in {"look", "inventory"}:
                    look_inventory += 1

        avg_steps = total_steps / max(1, n_eps)
        max_avg_steps = max(max_avg_steps, avg_steps)

        raw[task_id] = {
            "n_episodes": float(n_eps),
            "success_rate": successes / max(1, n_eps),
            "avg_steps": avg_steps,
            "dead_end_rate": dead_end / max(1, total_actions),
            "unknown_action_rate": unknown_actions / max(1, total_actions),
            "look_inventory_ratio": look_inventory / max(1, total_actions),
        }

    stats: dict[str, TaskStats] = {}
    for task_id, m in raw.items():
        # Toughness emphasizes actual failure and dead-end behaviors.
        step_norm = m["avg_steps"] / max_avg_steps
        toughness = (
            0.50 * (1.0 - m["success_rate"])
            + 0.20 * m["dead_end_rate"]
            + 0.15 * step_norm
            + 0.10 * m["unknown_action_rate"]
            + 0.05 * m["look_inventory_ratio"]
        )
        stats[task_id] = TaskStats(
            task_id=task_id,
            n_episodes=int(m["n_episodes"]),
            success_rate=m["success_rate"],
            avg_steps=m["avg_steps"],
            dead_end_rate=m["dead_end_rate"],
            unknown_action_rate=m["unknown_action_rate"],
            look_inventory_ratio=m["look_inventory_ratio"],
            toughness=toughness,
        )

    return stats


def _avg(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _quantile_bins(task_ids: list[str], scores: dict[str, float], n_bins: int) -> list[list[str]]:
    ranked = sorted(task_ids, key=lambda t: scores[t])
    bins: list[list[str]] = [[] for _ in range(max(1, n_bins))]
    for i, task_id in enumerate(ranked):
        b = min(len(bins) - 1, int(i * len(bins) / max(1, len(ranked))))
        bins[b].append(task_id)
    return bins


def _evaluate_candidate(
    val_tasks: set[str],
    all_tasks: list[str],
    stats: dict[str, TaskStats],
) -> tuple[float, dict[str, float]]:
    train_tasks = [t for t in all_tasks if t not in val_tasks]
    val_task_list = [t for t in all_tasks if t in val_tasks]

    all_success = _avg([stats[t].success_rate for t in all_tasks])
    all_tough = _avg([stats[t].toughness for t in all_tasks])
    all_steps = _avg([stats[t].avg_steps for t in all_tasks])

    val_success = _avg([stats[t].success_rate for t in val_task_list])
    val_tough = _avg([stats[t].toughness for t in val_task_list])
    val_steps = _avg([stats[t].avg_steps for t in val_task_list])

    train_success = _avg([stats[t].success_rate for t in train_tasks])
    train_tough = _avg([stats[t].toughness for t in train_tasks])
    train_steps = _avg([stats[t].avg_steps for t in train_tasks])

    # Lower is better. Prioritize closeness to global distribution.
    objective = (
        4.0 * abs(val_success - all_success)
        + 3.0 * abs(val_tough - all_tough)
        + 1.5 * abs(val_steps - all_steps) / max(1e-6, all_steps)
        + 0.8 * abs(val_success - train_success)
        + 0.8 * abs(val_tough - train_tough)
    )

    details = {
        "all_success": all_success,
        "all_toughness": all_tough,
        "all_avg_steps": all_steps,
        "val_success": val_success,
        "val_toughness": val_tough,
        "val_avg_steps": val_steps,
        "train_success": train_success,
        "train_toughness": train_tough,
        "train_avg_steps": train_steps,
    }
    return objective, details


def create_balanced_split(
    stats: dict[str, TaskStats],
    val_ratio: float,
    n_bins: int,
    search_iters: int,
    seed: int,
) -> tuple[set[str], dict[str, float], float]:
    rng = random.Random(seed)
    all_tasks = sorted(stats.keys())
    val_count = max(1, int(round(len(all_tasks) * val_ratio)))

    score_map = {t: stats[t].toughness for t in all_tasks}
    bins = _quantile_bins(all_tasks, score_map, n_bins)

    # Assign desired val counts per bin.
    targets = [len(b) * val_count / max(1, len(all_tasks)) for b in bins]
    per_bin = [int(math.floor(x)) for x in targets]
    remaining = val_count - sum(per_bin)
    frac_order = sorted(
        range(len(bins)), key=lambda i: (targets[i] - per_bin[i]), reverse=True
    )
    for i in frac_order[:remaining]:
        per_bin[i] += 1

    best_obj = float("inf")
    best_val: set[str] = set()
    best_details: dict[str, float] = {}

    for _ in range(max(100, search_iters)):
        val_tasks: set[str] = set()
        for b, k in zip(bins, per_bin):
            if not b:
                continue
            k = min(k, len(b))
            val_tasks.update(rng.sample(b, k))

        # Fill / trim in case of edge rounding interactions.
        if len(val_tasks) < val_count:
            rest = [t for t in all_tasks if t not in val_tasks]
            val_tasks.update(rng.sample(rest, val_count - len(val_tasks)))
        elif len(val_tasks) > val_count:
            val_tasks = set(rng.sample(sorted(val_tasks), val_count))

        obj, details = _evaluate_candidate(val_tasks, all_tasks, stats)
        if obj < best_obj:
            best_obj = obj
            best_val = set(val_tasks)
            best_details = details

    return best_val, best_details, best_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create balanced TextWorld validation split")
    parser.add_argument(
        "--input",
        type=str,
        default="data/runs/textworld_train_google_gemini-3-flash-preview_v2/trajectories.jsonl",
        help="Input trajectory JSONL (training source)",
    )
    parser.add_argument(
        "--out-train",
        type=str,
        default="data/trajectories/textworld_train_balanced.jsonl",
        help="Output train JSONL",
    )
    parser.add_argument(
        "--out-val",
        type=str,
        default="data/trajectories/textworld_val_balanced.jsonl",
        help="Output validation JSONL",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="outputs/perturbation_checks/textworld_split_report.json",
        help="Output JSON report path",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--search-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    rows = _read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    stats = _compute_task_stats(rows)
    val_tasks, details, objective = create_balanced_split(
        stats=stats,
        val_ratio=args.val_ratio,
        n_bins=args.bins,
        search_iters=args.search_iters,
        seed=args.seed,
    )

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for ep in rows:
        task_id = ep.get("metadata", {}).get("task_id")
        if task_id in val_tasks:
            val_rows.append(ep)
        else:
            train_rows.append(ep)

    _write_jsonl(Path(args.out_train), train_rows)
    _write_jsonl(Path(args.out_val), val_rows)

    all_tasks = sorted(stats.keys())
    train_tasks = sorted(t for t in all_tasks if t not in val_tasks)
    val_tasks_sorted = sorted(val_tasks)

    report = {
        "input": str(input_path),
        "total_episodes": len(rows),
        "total_tasks": len(all_tasks),
        "train_episodes": len(train_rows),
        "val_episodes": len(val_rows),
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks_sorted),
        "val_ratio": args.val_ratio,
        "bins": args.bins,
        "search_iters": args.search_iters,
        "seed": args.seed,
        "objective": objective,
        "balance": details,
        "val_task_ids": val_tasks_sorted,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Split created successfully")
    print(f"train episodes: {len(train_rows)} | val episodes: {len(val_rows)}")
    print(f"train tasks: {len(train_tasks)} | val tasks: {len(val_tasks_sorted)}")
    print(
        "success(all/train/val): "
        f"{details['all_success']:.4f} / {details['train_success']:.4f} / {details['val_success']:.4f}"
    )
    print(
        "toughness(all/train/val): "
        f"{details['all_toughness']:.4f} / {details['train_toughness']:.4f} / {details['val_toughness']:.4f}"
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
