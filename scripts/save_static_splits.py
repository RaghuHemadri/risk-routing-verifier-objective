#!/usr/bin/env python3
"""
Save deterministic train/val/test splits as separate JSONL files.

Run once to create fixed split files that won't change between experiments.
Splits by task_id (no task appears in multiple splits), with success/failure
tasks distributed proportionally.  Saves a split_manifest.json for the paper.

Usage:
    python scripts/save_static_splits.py \
        --trajectories data/trajectories/humaneval_noisy/trajectories.jsonl \
        --output-dir data/trajectories/humaneval_noisy \
        --max-perturbations-per-task 9999 \
        --seed 42

Produces:
    data/trajectories/humaneval_noisy/train.jsonl
    data/trajectories/humaneval_noisy/val.jsonl
    data/trajectories/humaneval_noisy/test.jsonl
    data/trajectories/humaneval_noisy/split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.splits import SPLIT_RATIOS, build_split_manifest, load_and_split
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
    parser.add_argument("--max-perturbations-per-task", type=int, default=None,
                        help="Cap perturbation variants per task")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(level="INFO")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.trajectories).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_and_split(
        args.trajectories,
        max_perturbations_per_task=args.max_perturbations_per_task,
        seed=args.seed,
    )

    for split_name, episodes in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
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
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info("")
    logger.info("═══ Static Split Summary ═══")
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
