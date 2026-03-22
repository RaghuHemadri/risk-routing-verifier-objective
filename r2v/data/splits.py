"""
Deterministic task-level train/val/test splitting.

Splits are made by *task_id* so that no task appears in more than one
partition — this prevents data leakage between splits.  All episodes
for a task (clean + perturbed) go to the same split.

Success/failure tasks are split independently at 70/15/15 so each
split preserves the overall success rate.

A split manifest (JSON) records the exact composition for
reproducibility and paper appendices.
"""

from __future__ import annotations

import logging
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Literal

from r2v.data.trajectory import Episode, TrajectoryStore

logger = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _get_perturbation_type(ep: Episode) -> str:
    pt = ep.perturbation_type
    return pt.value if hasattr(pt, "value") else str(pt)


def _is_clean(ep: Episode) -> bool:
    return _get_perturbation_type(ep) == "none"


def _subsample_perturbations(
    task_eps: list[Episode],
    max_perturbations_per_task: int,
    rng: random.Random,
) -> list[Episode]:
    """Keep at most max_perturbations_per_task episodes for one task."""
    if len(task_eps) <= max_perturbations_per_task:
        return task_eps
    clean = [e for e in task_eps if _is_clean(e)]
    noisy = [e for e in task_eps if not _is_clean(e)]
    rng.shuffle(noisy)
    budget = max(0, max_perturbations_per_task - len(clean))
    return clean + noisy[:budget]


def _split_stats(episodes: list[Episode]) -> dict[str, Any]:
    """Compute per-split statistics for the manifest."""
    task_ids = {ep.metadata.task_id for ep in episodes}
    perturbation_counts = Counter(_get_perturbation_type(ep) for ep in episodes)
    n_success = sum(1 for ep in episodes if ep.success)
    n_total = len(episodes)
    return {
        "num_episodes": n_total,
        "num_tasks": len(task_ids),
        "num_success": n_success,
        "num_failure": n_total - n_success,
        "success_rate": round(n_success / n_total, 4) if n_total else 0.0,
        "perturbation_type_counts": dict(sorted(perturbation_counts.items())),
    }


def build_split_manifest(
    splits: dict[Split, list[Episode]],
    seed: int,
    ratios: dict[str, float],
    max_perturbations_per_task: int | None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest describing the split composition."""
    manifest: dict[str, Any] = {
        "split_config": {
            "ratios": ratios,
            "seed": seed,
            "max_perturbations_per_task": max_perturbations_per_task,
            "split_by": "task_id",
            "success_balanced": True,
        },
        "splits": {},
        "global": _split_stats(
            splits["train"] + splits["val"] + splits["test"]
        ),
    }
    for split_name in ("train", "val", "test"):
        eps = splits[split_name]
        stats = _split_stats(eps)
        stats["task_ids"] = sorted({ep.metadata.task_id for ep in eps})
        manifest["splits"][split_name] = stats
    return manifest


def _log_episode_group_stats(
    label: str,
    episodes: list[Episode],
    by_task: dict[str, list[Episode]] | None = None,
) -> None:
    """Log detailed statistics for a group of episodes."""
    if not episodes:
        logger.info("%s: (empty)", label)
        return

    if by_task is None:
        by_task = defaultdict(list)
        for ep in episodes:
            by_task[ep.metadata.task_id].append(ep)

    task_ids = set(by_task.keys())
    n_success_tasks = sum(
        1 for tid in task_ids if by_task[tid][0].success
    )
    n_failure_tasks = len(task_ids) - n_success_tasks
    n_success_eps = sum(1 for ep in episodes if ep.success)

    eps_per_task = [len(eps) for eps in by_task.values()]
    perturbation_counts = Counter(_get_perturbation_type(ep) for ep in episodes)
    n_clean = perturbation_counts.pop("none", 0)
    n_perturbed = sum(perturbation_counts.values())

    logger.info("── %s ──", label)
    logger.info(
        "  Episodes : %d total (%d success, %d failure, %.1f%% success rate)",
        len(episodes), n_success_eps, len(episodes) - n_success_eps,
        n_success_eps / len(episodes) * 100,
    )
    logger.info(
        "  Tasks    : %d total (%d success, %d failure)",
        len(task_ids), n_success_tasks, n_failure_tasks,
    )
    logger.info(
        "  Episodes/task : min=%d, max=%d, mean=%.1f, median=%.1f",
        min(eps_per_task), max(eps_per_task),
        statistics.mean(eps_per_task), statistics.median(eps_per_task),
    )
    logger.info(
        "  Clean vs perturbed : %d clean, %d perturbed",
        n_clean, n_perturbed,
    )
    if perturbation_counts:
        logger.info(
            "  Perturbation types : %s",
            ", ".join(f"{k}={v}" for k, v in sorted(perturbation_counts.items())),
        )


def split_episodes(
    episodes: list[Episode],
    ratios: dict[str, float] | None = None,
    max_perturbations_per_task: int | None = None,
    seed: int = 42,
) -> dict[Split, list[Episode]]:
    """Split episodes into train/val/test by task_id with balanced success/failure.

    1. Group episodes by task_id (all clean + perturbed stay together).
    2. Separate tasks into success vs failure groups.
    3. Shuffle each group deterministically, then apply 70/15/15
       independently so each split has the same success rate.

    All episodes for a task share the same success label (perturbations
    are applied post-hoc and don't change the outcome).
    """
    if ratios is None:
        ratios = SPLIT_RATIOS

    by_task: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.metadata.task_id].append(ep)

    # ── Pre-split dataset overview ──
    logger.info("")
    _log_episode_group_stats("Initial dataset (before split)", episodes, by_task)
    logger.info(
        "  Split config : ratios=%s, max_perturbations_per_task=%s, seed=%d",
        "/".join(f"{v:.0%}" for v in ratios.values()),
        max_perturbations_per_task, seed,
    )
    logger.info("")

    rng = random.Random(seed)

    success_tasks: list[str] = []
    failure_tasks: list[str] = []
    for task_id, task_eps in by_task.items():
        if task_eps[0].success:
            success_tasks.append(task_id)
        else:
            failure_tasks.append(task_id)

    success_tasks.sort()
    failure_tasks.sort()
    rng.shuffle(success_tasks)
    rng.shuffle(failure_tasks)

    def _ratio_assign(task_ids: list[str]) -> dict[Split, list[str]]:
        n = len(task_ids)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        return {
            "train": task_ids[:n_train],
            "val": task_ids[n_train:n_train + n_val],
            "test": task_ids[n_train + n_val:],
        }

    success_assignment = _ratio_assign(success_tasks)
    failure_assignment = _ratio_assign(failure_tasks)

    result: dict[Split, list[Episode]] = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        assigned = success_assignment[split] + failure_assignment[split]
        for task_id in assigned:
            task_eps = by_task[task_id]
            if max_perturbations_per_task is not None:
                task_eps = _subsample_perturbations(
                    task_eps, max_perturbations_per_task, rng,
                )
            result[split].extend(task_eps)

    # ── Per-split logging ──
    for s in ("train", "val", "test"):
        eps = result[s]
        if not eps:
            logger.warning("Split %s is EMPTY — check data and ratios", s)
            continue
        _log_episode_group_stats(f"Split: {s}", eps)

    logger.info("")

    return result


def load_and_split(
    trajectory_path: str,
    ratios: dict[str, float] | None = None,
    max_perturbations_per_task: int | None = None,
    seed: int = 42,
    max_episodes: int | None = None,
) -> dict[Split, list[Episode]]:
    """Load from JSONL, then split.

    Parameters
    ----------
    trajectory_path
        Path to the trajectories JSONL file.
    ratios
        Split ratios (default 70/15/15).
    max_perturbations_per_task
        Cap perturbation variants per task.
    seed
        RNG seed for shuffling and subsampling.
    max_episodes
        Optional cap on total episodes to load.
    """
    store = TrajectoryStore(trajectory_path)
    episodes = store.load_episodes(max_count=max_episodes)
    logger.info("Loaded %d episodes from %s", len(episodes), trajectory_path)
    return split_episodes(
        episodes,
        ratios=ratios,
        max_perturbations_per_task=max_perturbations_per_task,
        seed=seed,
    )
