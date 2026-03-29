#!/usr/bin/env python3
"""
Save deterministic train/val/test splits for preference (DPO) data.

Unlike BC splits (40% of tasks, success-only, max 2 perturbations per clean),
preference splits use ALL tasks and ALL perturbations, while ensuring that
BC test/val task_ids never leak into preference train.

Task-ID split assignment is consistent with BC:
  - BC train task_ids  → preference train  (preserved)
  - BC val task_ids    → preference val    (preserved)
  - BC test task_ids   → preference test   (preserved)
  - Extra task_ids     → distributed 70/15/15 (success/failure balanced)

Usage:
    python scripts/save_preference_splits.py \\
        --trajectories data/trajectories/humaneval_noisy/trajectories.jsonl \\
        --preference-data data/candidates/humaneval_noisy_heuristic.jsonl \\
        --output-dir data/trajectories/humaneval_noisy \\
        --bc-manifest data/trajectories/humaneval_noisy/bc_split_manifest.json \\
        --seed 42

Produces:
    pref_train.jsonl           — preference pairs for training
    pref_val.jsonl             — preference pairs for validation
    pref_test.jsonl            — preference pairs for testing
    pref_split_manifest.json   — split metadata and statistics
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.trajectory import TrajectoryStore
from r2v.utils.logging import setup_logging

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split preference pairs into static train/val/test JSONL files",
    )
    parser.add_argument("--trajectories", type=str, required=True,
                        help="Path to the full noisy trajectories JSONL "
                             "(used to map episode_id → task_id)")
    parser.add_argument("--preference-data", type=str, required=True,
                        help="Path to the preference pairs JSONL (candidates)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write split files "
                             "(defaults to same dir as --trajectories)")
    parser.add_argument("--bc-manifest", type=str, default=None,
                        help="Path to bc_split_manifest.json. If omitted, "
                             "auto-detects from --output-dir.")
    parser.add_argument("--prefix", type=str, default="pref",
                        help="Filename prefix (default: 'pref' → pref_train.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _task_id_from_episode_id(episode_id: str) -> str:
    """Fallback: extract task_id by stripping _seed/_perturbed_seed suffixes."""
    cleaned = re.sub(r"_perturbed_seed\d+$", "", episode_id)
    cleaned = re.sub(r"_seed\d+$", "", cleaned)
    return cleaned


def _ratio_assign(task_ids: list[str], ratios: dict[str, float]) -> dict[str, list[str]]:
    n = len(task_ids)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    return {
        "train": task_ids[:n_train],
        "val": task_ids[n_train : n_train + n_val],
        "test": task_ids[n_train + n_val :],
    }


def _prefixed(name: str, prefix: str | None) -> str:
    return f"{prefix}_{name}" if prefix else name


def _pair_stats(
    pairs: list[dict],
    eid_to_task: dict[str, str],
) -> dict:
    """Compute detailed statistics for a set of preference pairs."""
    episode_ids: set[str] = set()
    task_ids: set[str] = set()
    score_gaps: list[float] = []
    step_indices: list[int] = []
    pairs_per_episode: Counter = Counter()
    task_episode_sets: dict[str, set[str]] = defaultdict(set)

    for p in pairs:
        eid = p.get("episode_id", "")
        episode_ids.add(eid)
        pairs_per_episode[eid] += 1

        tid = eid_to_task.get(eid) or _task_id_from_episode_id(eid)
        task_ids.add(tid)
        task_episode_sets[tid].add(eid)

        cs = p.get("chosen_score")
        rs = p.get("rejected_score")
        if cs is not None and rs is not None:
            score_gaps.append(float(cs) - float(rs))

        if "step_idx" in p:
            step_indices.append(int(p["step_idx"]))

    episodes_per_task = [len(eids) for eids in task_episode_sets.values()]
    pairs_per_task = Counter()
    for p in pairs:
        eid = p.get("episode_id", "")
        tid = eid_to_task.get(eid) or _task_id_from_episode_id(eid)
        pairs_per_task[tid] += 1
    ppt_values = list(pairs_per_task.values()) or [0]

    clean_eids = {eid for eid in episode_ids if "_perturbed_seed" not in eid}
    perturbed_eids = episode_ids - clean_eids

    stats: dict = {
        "num_pairs": len(pairs),
        "num_unique_episodes": len(episode_ids),
        "num_unique_tasks": len(task_ids),
        "task_ids": sorted(task_ids),
        "clean_episodes": len(clean_eids),
        "perturbed_episodes": len(perturbed_eids),
        "pairs_per_task": {
            "min": min(ppt_values),
            "max": max(ppt_values),
            "mean": round(statistics.mean(ppt_values), 2),
            "median": round(statistics.median(ppt_values), 2),
        },
        "episodes_per_task": {
            "min": min(episodes_per_task) if episodes_per_task else 0,
            "max": max(episodes_per_task) if episodes_per_task else 0,
            "mean": round(statistics.mean(episodes_per_task), 2) if episodes_per_task else 0,
        },
    }
    if score_gaps:
        stats["score_gap"] = {
            "min": round(min(score_gaps), 4),
            "max": round(max(score_gaps), 4),
            "mean": round(statistics.mean(score_gaps), 4),
            "median": round(statistics.median(score_gaps), 4),
        }
    if step_indices:
        stats["step_idx"] = {
            "min": min(step_indices),
            "max": max(step_indices),
            "mean": round(statistics.mean(step_indices), 2),
        }
    return stats


def main():
    args = parse_args()
    logger = setup_logging(level="INFO")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.trajectories).parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load trajectories → episode_id-to-task_id mapping ──
    logger.info("Loading trajectories from %s", args.trajectories)
    store = TrajectoryStore(args.trajectories)
    episodes = store.load_episodes()
    logger.info("Loaded %d episodes from trajectories", len(episodes))

    eid_to_task: dict[str, str] = {}
    task_to_eids: dict[str, set[str]] = defaultdict(set)
    task_success: dict[str, bool] = {}

    for ep in episodes:
        tid = ep.metadata.task_id
        eid_to_task[ep.episode_id] = tid
        task_to_eids[tid].add(ep.episode_id)
        if tid not in task_success:
            task_success[tid] = ep.success

    all_task_ids = sorted(task_to_eids.keys())
    n_success = sum(1 for s in task_success.values() if s)
    n_failure = len(task_success) - n_success
    logger.info(
        "Trajectories: %d unique tasks (%d success, %d failure), %d episodes",
        len(all_task_ids), n_success, n_failure, len(episodes),
    )

    # ── 2. Load BC manifest for consistent task-ID assignment ──
    bc_manifest_path = (
        Path(args.bc_manifest)
        if args.bc_manifest
        else output_dir / "bc_split_manifest.json"
    )
    bc_task_assignment: dict[str, str] = {}

    if bc_manifest_path.exists():
        with open(bc_manifest_path) as f:
            bc_manifest = json.load(f)
        for split_name in ("train", "val", "test"):
            for tid in bc_manifest["splits"][split_name].get("task_ids", []):
                bc_task_assignment[tid] = split_name
        bc_counts = Counter(bc_task_assignment.values())
        logger.info(
            "BC manifest loaded (%s): %d task_ids — train=%d, val=%d, test=%d",
            bc_manifest_path.name,
            len(bc_task_assignment),
            bc_counts["train"], bc_counts["val"], bc_counts["test"],
        )
    else:
        logger.warning(
            "BC manifest not found at %s — splitting all tasks fresh "
            "(no BC consistency guarantee)", bc_manifest_path,
        )

    # ── 3. Assign task_ids to splits ──
    pref_assignment: dict[str, str] = {}

    # Honour BC assignments for task_ids that exist in current data
    bc_honoured = 0
    for tid, split_name in bc_task_assignment.items():
        if tid in task_to_eids:
            pref_assignment[tid] = split_name
            bc_honoured += 1

    # Distribute extra task_ids 70/15/15 with success/failure balance
    extra_task_ids = sorted(set(all_task_ids) - set(pref_assignment.keys()))

    if extra_task_ids:
        rng = random.Random(args.seed)
        success_extra = [tid for tid in extra_task_ids if task_success.get(tid, False)]
        failure_extra = [tid for tid in extra_task_ids if not task_success.get(tid, False)]
        rng.shuffle(success_extra)
        rng.shuffle(failure_extra)

        for group in (success_extra, failure_extra):
            assignment = _ratio_assign(group, SPLIT_RATIOS)
            for split_name, tids in assignment.items():
                for tid in tids:
                    pref_assignment[tid] = split_name

        logger.info(
            "Extra task_ids (not in BC): %d total (%d success, %d failure) → 70/15/15",
            len(extra_task_ids), len(success_extra), len(failure_extra),
        )

    # ── Verify no task_id overlap across splits ──
    split_to_tids: dict[str, set[str]] = defaultdict(set)
    for tid, sname in pref_assignment.items():
        split_to_tids[sname].add(tid)

    train_tids = split_to_tids["train"]
    val_tids = split_to_tids["val"]
    test_tids = split_to_tids["test"]

    train_val_overlap = train_tids & val_tids
    train_test_overlap = train_tids & test_tids
    val_test_overlap = val_tids & test_tids

    if train_val_overlap or train_test_overlap or val_test_overlap:
        logger.error("FATAL: Task-ID overlap detected across splits!")
        if train_val_overlap:
            logger.error("  train ∩ val:  %s", sorted(train_val_overlap))
        if train_test_overlap:
            logger.error("  train ∩ test: %s", sorted(train_test_overlap))
        if val_test_overlap:
            logger.error("  val ∩ test:   %s", sorted(val_test_overlap))
        sys.exit(1)

    assert len(train_tids) + len(val_tids) + len(test_tids) == len(pref_assignment), \
        "Task count mismatch: splits don't sum to total assignments"

    logger.info("✓ Task-ID overlap check passed (train ∩ val ∩ test = ∅)")

    # Build episode_id → split mapping (all episodes for each task)
    eid_to_split: dict[str, str] = {}
    for tid, split_name in pref_assignment.items():
        for eid in task_to_eids[tid]:
            eid_to_split[eid] = split_name

    pref_counts = Counter(pref_assignment.values())
    logger.info(
        "Preference task assignment: %d total — train=%d, val=%d, test=%d "
        "(%d honoured from BC, %d extra)",
        len(pref_assignment),
        pref_counts["train"], pref_counts["val"], pref_counts["test"],
        bc_honoured, len(extra_task_ids),
    )

    # ── 4. Load and split preference pairs ──
    logger.info("Loading preference pairs from %s", args.preference_data)
    split_pairs: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    skipped_no_eid = 0
    skipped_unknown = 0
    total_pairs = 0

    with open(args.preference_data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total_pairs += 1

            eid = obj.get("episode_id")
            if not eid:
                skipped_no_eid += 1
                continue

            split_name = eid_to_split.get(eid)
            if split_name is None:
                tid = _task_id_from_episode_id(eid)
                split_name = pref_assignment.get(tid)
            if split_name is None:
                skipped_unknown += 1
                continue

            split_pairs[split_name].append(obj)

    logger.info(
        "Read %d preference pairs → train=%d, val=%d, test=%d",
        total_pairs,
        len(split_pairs["train"]),
        len(split_pairs["val"]),
        len(split_pairs["test"]),
    )
    if skipped_no_eid:
        logger.warning("  Skipped %d pairs (no episode_id field)", skipped_no_eid)
    if skipped_unknown:
        logger.warning(
            "  Skipped %d pairs (episode_id not in any known task)", skipped_unknown,
        )

    # ── 5. Save split JSONL files ──
    for split_name, pairs in split_pairs.items():
        out_path = output_dir / f"{_prefixed(split_name, args.prefix)}.jsonl"
        if out_path.exists():
            out_path.unlink()
        with open(out_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("  Wrote %s (%d pairs)", out_path.name, len(pairs))

    # ── 6. Build manifest with detailed stats ──
    all_pairs = split_pairs["train"] + split_pairs["val"] + split_pairs["test"]
    global_stats = _pair_stats(all_pairs, eid_to_task)
    global_task_ids = global_stats.pop("task_ids")

    manifest: dict = {
        "split_config": {
            "ratios": SPLIT_RATIOS,
            "seed": args.seed,
            "split_by": "task_id",
            "prefix": args.prefix,
            "data_fraction": 1.0,
            "max_perturbations_per_task": "all (no limit)",
            "success_only": False,
            "bc_manifest": str(bc_manifest_path) if bc_task_assignment else None,
            "bc_task_ids_honoured": bc_honoured,
            "extra_task_ids_distributed": len(extra_task_ids) if extra_task_ids else 0,
        },
        "global": global_stats,
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        stats = _pair_stats(split_pairs[split_name], eid_to_task)
        manifest["splits"][split_name] = stats

    # Cross-split overlap verification (saved in manifest for audit)
    m_train_tids = set(manifest["splits"]["train"].get("task_ids", []))
    m_val_tids = set(manifest["splits"]["val"].get("task_ids", []))
    m_test_tids = set(manifest["splits"]["test"].get("task_ids", []))
    manifest["integrity_checks"] = {
        "train_val_overlap": sorted(m_train_tids & m_val_tids),
        "train_test_overlap": sorted(m_train_tids & m_test_tids),
        "val_test_overlap": sorted(m_val_tids & m_test_tids),
        "all_disjoint": len(m_train_tids & m_val_tids) == 0
                        and len(m_train_tids & m_test_tids) == 0
                        and len(m_val_tids & m_test_tids) == 0,
        "total_tasks_check": (
            len(m_train_tids) + len(m_val_tids) + len(m_test_tids)
            == manifest["global"]["num_unique_tasks"]
        ),
    }

    # BC overlap analysis
    if bc_task_assignment:
        overlap = {}
        for split_name in ("train", "val", "test"):
            pref_tids = set(manifest["splits"][split_name].get("task_ids", []))
            bc_tids = {t for t, s in bc_task_assignment.items() if s == split_name}
            overlap[split_name] = {
                "pref_tasks": len(pref_tids),
                "bc_tasks_in_this_split": len(bc_tids),
                "bc_tasks_matched": len(pref_tids & bc_tids),
                "extra_pref_tasks": len(pref_tids - bc_tids),
            }

        bc_test_val = {t for t, s in bc_task_assignment.items() if s in ("test", "val")}
        pref_train_tids = set(manifest["splits"]["train"].get("task_ids", []))
        leaked = sorted(bc_test_val & pref_train_tids)
        overlap["leakage_check"] = {
            "bc_test_val_tasks_in_pref_train": len(leaked),
            "leaked_task_ids": leaked,
            "no_leakage": len(leaked) == 0,
        }
        manifest["bc_overlap"] = overlap

    manifest_path = output_dir / f"{_prefixed('split_manifest', args.prefix)}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # ── 7. Print comprehensive summary ──
    logger.info("")
    logger.info("═" * 60)
    logger.info("  Preference Static Split Summary (%s)", args.prefix)
    logger.info("═" * 60)
    g = manifest["global"]
    logger.info(
        "Total: %d pairs | %d unique episodes | %d unique tasks",
        g["num_pairs"], g["num_unique_episodes"], g["num_unique_tasks"],
    )
    logger.info(
        "  Episodes: %d clean, %d perturbed",
        g["clean_episodes"], g["perturbed_episodes"],
    )
    if "score_gap" in g:
        sg = g["score_gap"]
        logger.info(
            "  Score gaps: mean=%.4f, median=%.4f, min=%.4f, max=%.4f",
            sg["mean"], sg["median"], sg["min"], sg["max"],
        )
    logger.info("")

    header = f"  {'Split':<7} {'Pairs':>7} {'Episodes':>10} {'Tasks':>7} {'Clean':>7} {'Perturb':>9} {'Pairs/Task':>12}"
    logger.info(header)
    logger.info("  " + "─" * (len(header) - 2))
    for split_name in ("train", "val", "test"):
        s = manifest["splits"][split_name]
        ppt = s["pairs_per_task"]
        logger.info(
            "  %-7s %7d %10d %7d %7d %9d %5.1f–%.1f",
            split_name,
            s["num_pairs"],
            s["num_unique_episodes"],
            s["num_unique_tasks"],
            s["clean_episodes"],
            s["perturbed_episodes"],
            ppt["min"], ppt["max"],
        )

    # Integrity checks
    ic = manifest["integrity_checks"]
    logger.info("")
    logger.info("── Integrity Checks ──")
    if ic["all_disjoint"]:
        logger.info("  ✓ train/val/test task_ids are fully disjoint (no overlap)")
    else:
        logger.error("  ✗ OVERLAP DETECTED:")
        if ic["train_val_overlap"]:
            logger.error("    train ∩ val:  %s", ic["train_val_overlap"])
        if ic["train_test_overlap"]:
            logger.error("    train ∩ test: %s", ic["train_test_overlap"])
        if ic["val_test_overlap"]:
            logger.error("    val ∩ test:   %s", ic["val_test_overlap"])
    if ic["total_tasks_check"]:
        logger.info("  ✓ Split task counts sum to global total (no missing/duplicate tasks)")
    else:
        logger.error("  ✗ Split task counts don't sum to global total!")

    if "bc_overlap" in manifest:
        logger.info("")
        logger.info("── BC Consistency ──")
        for split_name in ("train", "val", "test"):
            o = manifest["bc_overlap"][split_name]
            logger.info(
                "  %-7s %d pref tasks = %d from BC + %d extra",
                split_name, o["pref_tasks"], o["bc_tasks_matched"], o["extra_pref_tasks"],
            )
        lc = manifest["bc_overlap"]["leakage_check"]
        if lc["no_leakage"]:
            logger.info("  ✓ No BC test/val task_ids leaked into preference train")
        else:
            logger.warning(
                "  ✗ LEAKAGE DETECTED: %d BC test/val task_ids in pref train: %s",
                lc["bc_test_val_tasks_in_pref_train"], lc["leaked_task_ids"],
            )

    logger.info("")
    logger.info("Files saved to %s:", output_dir)
    for split_name in ("train", "val", "test"):
        fname = f"{_prefixed(split_name, args.prefix)}.jsonl"
        logger.info("  %s  (%d pairs)", fname, len(split_pairs[split_name]))
    logger.info("  %s", manifest_path.name)
    logger.info("")

    if skipped_no_eid or skipped_unknown:
        logger.info(
            "Note: %d pairs skipped total (%d no episode_id, %d unknown task)",
            skipped_no_eid + skipped_unknown, skipped_no_eid, skipped_unknown,
        )


if __name__ == "__main__":
    main()
