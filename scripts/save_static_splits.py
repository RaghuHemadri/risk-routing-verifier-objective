#!/usr/bin/env python3
"""
Save deterministic train/val/test splits as separate JSONL files.

Run once to create fixed split files that won't change between experiments.
Splits by task_id (no task appears in multiple splits), with success/failure
tasks distributed proportionally.  Saves a split_manifest.json for the paper.

Usage (verifier — all tasks, all perturbations):
    python scripts/save_static_splits.py \
        --trajectories data/trajectories/humaneval_noisy/trajectories.jsonl \
        --output-dir data/trajectories/humaneval_noisy \
        --prefix verifier \
        --max-perturbations-per-task 9999 \
        --seed 42

Usage (BC — success-only tasks, 2 perturbations per clean episode, 40% of tasks):
    python scripts/save_static_splits.py \
        --trajectories data/trajectories/humaneval_noisy/trajectories.jsonl \
        --output-dir data/trajectories/humaneval_noisy \
        --prefix bc \
        --success-only \
        --max-perturbations-per-task 2 \
        --data-fraction 0.4 \
        --seed 42

Produces (with --prefix bc):
    data/trajectories/humaneval_noisy/bc_train.jsonl
    data/trajectories/humaneval_noisy/bc_val.jsonl
    data/trajectories/humaneval_noisy/bc_test.jsonl
    data/trajectories/humaneval_noisy/bc_split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.splits import SPLIT_RATIOS, build_split_manifest, split_episodes
from r2v.data.trajectory import TrajectoryStore
from r2v.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split trajectories into static train/val/test JSONL files",
    )
    parser.add_argument("--trajectories", type=str, required=True,
                        help="Path to the full trajectories JSONL file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write train/val/test.jsonl "
                             "(defaults to same directory as --trajectories)")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Filename prefix (e.g. 'bc' → bc_train.jsonl)")
    parser.add_argument("--max-perturbations-per-task", type=int, default=None,
                        help="Max perturbed variants per clean episode")
    parser.add_argument("--success-only", action="store_true", default=False,
                        help="Drop failure tasks before splitting (for BC)")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of training tasks to keep (0.0–1.0). "
                             "Subsamples at the task level; val/test are unaffected.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prefixed(name: str, prefix: str | None) -> str:
    return f"{prefix}_{name}" if prefix else name


def main():
    args = parse_args()
    logger = setup_logging(level="INFO")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.trajectories).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    store = TrajectoryStore(args.trajectories)
    episodes = store.load_episodes()
    logger.info("Loaded %d episodes from %s", len(episodes), args.trajectories)

    # Subsample tasks before splitting (e.g. 40% of all tasks for BC)
    if args.data_fraction < 1.0:
        rng = random.Random(args.seed)
        task_to_eps: dict[str, list] = {}
        for ep in episodes:
            task_to_eps.setdefault(ep.metadata.task_id, []).append(ep)
        all_task_ids = sorted(task_to_eps.keys())

        # If success_only, filter to success tasks before subsampling
        if args.success_only:
            all_task_ids = [tid for tid in all_task_ids if task_to_eps[tid][0].success]

        n_keep = max(1, int(len(all_task_ids) * args.data_fraction))
        selected = set(rng.sample(all_task_ids, n_keep))
        episodes = [ep for ep in episodes if ep.metadata.task_id in selected]
        logger.info(
            "data_fraction=%.2f: kept %d/%d tasks (%d episodes)",
            args.data_fraction, n_keep, len(all_task_ids), len(episodes),
        )
        # success_only already applied above, don't double-filter
        success_only_for_split = False
    else:
        success_only_for_split = args.success_only

    splits = split_episodes(
        episodes,
        max_perturbations_per_task=args.max_perturbations_per_task,
        seed=args.seed,
        success_only=success_only_for_split,
    )

    # For BC: drop failure episodes (BCDataset skips them anyway)
    if args.success_only:
        for split_name in splits:
            before = len(splits[split_name])
            splits[split_name] = [ep for ep in splits[split_name] if ep.success]
            dropped = before - len(splits[split_name])
            if dropped:
                logger.info(
                    "Dropped %d failure episodes from %s split (%d → %d)",
                    dropped, split_name, before, len(splits[split_name]),
                )

    for split_name, episodes in splits.items():
        out_path = output_dir / f"{_prefixed(split_name, args.prefix)}.jsonl"
        if out_path.exists():
            out_path.unlink()
        store = TrajectoryStore(out_path)
        store.save_episodes(episodes)

    manifest = build_split_manifest(
        splits,
        seed=args.seed,
        ratios=SPLIT_RATIOS,
        max_perturbations_per_task=args.max_perturbations_per_task,
    )
    manifest["split_config"]["success_only"] = args.success_only
    manifest["split_config"]["data_fraction"] = args.data_fraction
    manifest["split_config"]["prefix"] = args.prefix
    manifest_path = output_dir / f"{_prefixed('split_manifest', args.prefix)}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    label = f" ({args.prefix})" if args.prefix else ""
    logger.info("")
    logger.info("═══ Static Split Summary%s ═══", label)
    if args.success_only:
        logger.info("  success_only=True (failure tasks excluded)")
    g = manifest["global"]
    logger.info(
        "Global: %d episodes, %d tasks, %d success / %d failure (%.1f%% success rate)",
        g["num_episodes"], g["num_tasks"], g["num_success"], g["num_failure"],
        g["success_rate"] * 100,
    )
    if g.get("perturbation_type_counts"):
        logger.info("  Perturbation types: %s", g["perturbation_type_counts"])
    logger.info("")

    for split_name in ("train", "val", "test"):
        s = manifest["splits"][split_name]
        logger.info(
            "  %-5s → %4d episodes across %3d tasks  |  %d success, %d failure (%.1f%%)",
            split_name, s["num_episodes"], s["num_tasks"],
            s["num_success"], s["num_failure"], s["success_rate"] * 100,
        )
        if s.get("perturbation_type_counts"):
            logger.info("          perturbations: %s", s["perturbation_type_counts"])

    logger.info("")
    logger.info("Split manifest -> %s", manifest_path)
    logger.info("Static splits saved to %s", output_dir)


if __name__ == "__main__":
    main()
