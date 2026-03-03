#!/usr/bin/env python3
"""
Batch evaluate collected SWE-bench trajectories using the Docker harness.

Usage:
    python scripts/batch_evaluate_swebench.py \
        --trajectories data/runs/<run_id>/trajectories.jsonl \
        --max-workers 4 \
        --timeout 600

This script:
1. Reads collected trajectories from a JSONL file
2. Extracts instance_id + predicted patch for each episode
3. Runs swebench Docker harness on all predictions in batch
4. Writes an updated JSONL with corrected success labels
5. Prints a summary of resolved vs unresolved instances

Prerequisites:
    - pip install swebench
    - Docker running locally
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluate SWE-bench trajectories with Docker harness"
    )
    parser.add_argument(
        "--trajectories", type=str, required=True,
        help="Path to trajectories.jsonl from a collection run"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for updated JSONL. Default: adds _evaluated suffix"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="princeton-nlp/SWE-bench",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Number of parallel Docker containers"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Timeout per instance in seconds"
    )
    parser.add_argument(
        "--run-id", type=str, default="r2v_batch_eval",
        help="Unique run ID for swebench harness"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just show what would be evaluated without running Docker"
    )
    return parser.parse_args()


def extract_patches_from_trajectories(traj_path: str) -> list[dict]:
    """Read trajectories JSONL and extract instance_id + patch for each episode.

    Each trajectory line is a JSON object with fields like:
        episode_id, metadata.task_id, steps[].action.raw_text, success, partial_score

    We extract the final patch from the last step's action.
    """
    episodes = []
    with open(traj_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: Skipping malformed line {line_num}")
                continue

            task_id = ep.get("metadata", {}).get("task_id", "")
            benchmark = ep.get("metadata", {}).get("benchmark", "")

            if benchmark != "swebench":
                continue

            # Extract patch from the last step's action
            steps = ep.get("steps", [])
            if not steps:
                continue

            # The patch is in the last step's action raw_text
            last_action = steps[-1].get("action", {}).get("raw_text", "")
            patch = _extract_patch_from_action(last_action)

            if not patch:
                print(f"  No patch found for {task_id} (episode {ep.get('episode_id', '?')})")
                continue

            episodes.append({
                "line_num": line_num,
                "episode_id": ep.get("episode_id", ""),
                "instance_id": task_id,
                "patch": patch,
                "original_success": ep.get("success", False),
                "original_score": ep.get("partial_score", 0.0),
            })

    return episodes


def _extract_patch_from_action(action_text: str) -> str | None:
    """Extract patch content from an action string.

    Handles:
      - stop [...] wrapper from parse_action_from_response
      - <patch>...</patch> tags
      - ```diff...``` blocks
      - Raw unified diffs
    """
    import re

    # Unwrap stop [...] if present
    match = re.match(r"stop\s*\[(.+)\]", action_text, re.DOTALL | re.IGNORECASE)
    if match:
        action_text = match.group(1)

    # Try <patch> tags
    match = re.search(r"<patch>(.*?)</patch>", action_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ```diff blocks
    match = re.search(r"```(?:diff|patch)?\n(.*?)(?:```|$)", action_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Raw diff detection
    if "--- a/" in action_text or "diff --git" in action_text:
        return action_text.strip()

    # Look for unified diff markers
    match = re.search(r"(---\s+a/.+?\n\+\+\+\s+b/.+?\n@@.+)", action_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def run_batch_evaluation(
    predictions: list[dict],
    dataset_name: str,
    split: str,
    run_id: str,
    max_workers: int,
    timeout: int,
) -> dict[str, bool]:
    """Run swebench Docker harness on a batch of predictions.

    Args:
        predictions: List of dicts with 'instance_id' and 'patch' keys
        ...

    Returns:
        Dict mapping instance_id -> bool (resolved or not)
    """
    from swebench.harness.run_evaluation import main as run_evaluation
    from swebench.harness.constants import (
        KEY_INSTANCE_ID,
        KEY_MODEL,
        KEY_PREDICTION,
    )
    import tempfile

    # Deduplicate by instance_id (take the first prediction per instance)
    seen = set()
    unique_predictions = []
    for p in predictions:
        if p["instance_id"] not in seen:
            seen.add(p["instance_id"])
            unique_predictions.append(p)

    model_name = "r2v-teacher"
    instance_ids = [p["instance_id"] for p in unique_predictions]

    formatted = [
        {
            KEY_INSTANCE_ID: p["instance_id"],
            KEY_PREDICTION: p["patch"],
            KEY_MODEL: model_name,
        }
        for p in unique_predictions
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="r2v_batch_"
    ) as f:
        json.dump(formatted, f)
        pred_file = f.name

    print(f"\nRunning swebench harness on {len(unique_predictions)} instances...")
    print(f"  Run ID: {run_id}")
    print(f"  Workers: {max_workers}")
    print(f"  Timeout: {timeout}s per instance")
    print(f"  Predictions file: {pred_file}")

    try:
        summary_report_path = run_evaluation(
            dataset_name=dataset_name,
            split=split,
            instance_ids=instance_ids,
            predictions_path=pred_file,
            max_workers=max_workers,
            force_rebuild=False,
            cache_level="env",
            clean=True,
            open_file_limit=4096,
            run_id=run_id,
            timeout=timeout,
            namespace="swebench",
            rewrite_reports=False,
            modal=False,
        )

        resolved_ids = set()
        if summary_report_path and Path(summary_report_path).exists():
            with open(summary_report_path) as rf:
                report = json.load(rf)
            resolved_ids = set(report.get("resolved_ids", []))

        return {iid: iid in resolved_ids for iid in instance_ids}
    finally:
        import os
        try:
            os.unlink(pred_file)
        except OSError:
            pass


def update_trajectories(
    traj_path: str,
    output_path: str,
    results: dict[str, bool],
    episodes: list[dict],
) -> None:
    """Write an updated JSONL with Docker-evaluated success labels."""
    # Build a lookup from episode_id to docker result
    ep_results = {}
    for ep in episodes:
        instance_id = ep["instance_id"]
        if instance_id in results:
            ep_results[ep["episode_id"]] = results[instance_id]

    updated = 0
    total = 0
    with open(traj_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line + "\n")
                continue

            total += 1
            episode_id = ep.get("episode_id", "")

            if episode_id in ep_results:
                docker_resolved = ep_results[episode_id]
                ep["success"] = docker_resolved
                ep["partial_score"] = 1.0 if docker_resolved else 0.0
                ep["docker_evaluated"] = True
                updated += 1

            fout.write(json.dumps(ep) + "\n")

    print(f"\nUpdated {updated}/{total} episodes with Docker evaluation results")
    print(f"Output: {output_path}")


def main():
    args = parse_args()
    traj_path = args.trajectories

    if not Path(traj_path).exists():
        print(f"ERROR: Trajectory file not found: {traj_path}")
        sys.exit(1)

    # Step 1: Extract patches
    print(f"Reading trajectories from: {traj_path}")
    episodes = extract_patches_from_trajectories(traj_path)
    print(f"Found {len(episodes)} SWE-bench episodes with patches")

    if not episodes:
        print("No episodes to evaluate.")
        sys.exit(0)

    # Summarize
    unique_instances = set(ep["instance_id"] for ep in episodes)
    print(f"Unique instances: {len(unique_instances)}")
    heuristic_successes = sum(1 for ep in episodes if ep["original_success"])
    print(f"Heuristic successes: {heuristic_successes}/{len(episodes)}")

    if args.dry_run:
        print("\n[DRY RUN] Would evaluate these instances:")
        for iid in sorted(unique_instances):
            print(f"  - {iid}")
        sys.exit(0)

    # Step 2: Run Docker evaluation
    start_time = time.time()
    results = run_batch_evaluation(
        predictions=episodes,
        dataset_name=args.dataset_name,
        split=args.split,
        run_id=args.run_id,
        max_workers=args.max_workers,
        timeout=args.timeout,
    )
    elapsed = time.time() - start_time

    # Step 3: Print summary
    resolved = sum(1 for v in results.values() if v)
    print(f"\n{'=' * 60}")
    print(f"Docker Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Total instances:  {len(results)}")
    print(f"  Resolved:         {resolved}")
    print(f"  Unresolved:       {len(results) - resolved}")
    print(f"  Success rate:     {resolved / len(results) * 100:.1f}%")
    print(f"  Wall time:        {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print()

    # Step 4: Update trajectories
    output_path = args.output
    if output_path is None:
        p = Path(traj_path)
        output_path = str(p.parent / f"{p.stem}_evaluated{p.suffix}")

    update_trajectories(traj_path, output_path, results, episodes)


if __name__ == "__main__":
    main()
