"""
Deterministic task-level train/val/test splitting and perturbation subsampling.

Splits are made by *task_id* so that no task appears in more than one
partition — this prevents data leakage between splits.

Perturbation subsampling selects a fixed number of random perturbation
variants per task to limit dataset size while preserving diversity.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict
from typing import Literal

from r2v.data.trajectory import Episode, TrajectoryStore

logger = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _stable_hash(task_id: str) -> int:
    """Return a deterministic integer hash for a task_id (seed-independent)."""
    return int(hashlib.sha256(task_id.encode("utf-8")).hexdigest(), 16)


def assign_split(task_id: str, ratios: dict[str, float] | None = None) -> Split:
    """Deterministically assign a task_id to train/val/test.

    Uses a stable hash so the assignment is reproducible across runs
    without requiring all task_ids to be known up front.
    """
    if ratios is None:
        ratios = SPLIT_RATIOS
    h = _stable_hash(task_id) % 10000 / 10000.0
    train_end = ratios["train"]
    val_end = train_end + ratios["val"]
    if h < train_end:
        return "train"
    elif h < val_end:
        return "val"
    else:
        return "test"


def split_episodes(
    episodes: list[Episode],
    ratios: dict[str, float] | None = None,
    max_perturbations_per_task: int | None = None,
    seed: int = 42,
) -> dict[Split, list[Episode]]:
    """Split episodes by task_id and optionally subsample perturbations.

    Parameters
    ----------
    episodes
        All loaded episodes.
    ratios
        Mapping of split name to fraction.  Defaults to 70/15/15.
    max_perturbations_per_task
        If set, keep at most this many perturbation variants per task
        (randomly chosen but deterministically seeded).  A "clean" episode
        (perturbation_type == "none") always counts as one slot.
    seed
        RNG seed for perturbation subsampling.

    Returns
    -------
    dict with keys "train", "val", "test", each mapping to a list of Episode.
    """
    if ratios is None:
        ratios = SPLIT_RATIOS

    by_task: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.metadata.task_id].append(ep)

    result: dict[Split, list[Episode]] = {"train": [], "val": [], "test": []}

    rng = random.Random(seed)

    for task_id, task_eps in sorted(by_task.items()):
        split = assign_split(task_id, ratios)

        if max_perturbations_per_task is not None and len(task_eps) > max_perturbations_per_task:
            clean = [e for e in task_eps if _is_clean(e)]
            noisy = [e for e in task_eps if not _is_clean(e)]
            rng.shuffle(noisy)
            budget = max(0, max_perturbations_per_task - len(clean))
            task_eps = clean + noisy[:budget]

        result[split].extend(task_eps)

    for s in ("train", "val", "test"):
        task_ids = {ep.metadata.task_id for ep in result[s]}
        logger.info(
            "Split %-5s: %4d episodes across %3d tasks",
            s, len(result[s]), len(task_ids),
        )

    return result


def load_and_split(
    trajectory_path: str,
    ratios: dict[str, float] | None = None,
    max_perturbations_per_task: int | None = None,
    seed: int = 42,
    max_episodes: int | None = None,
) -> dict[Split, list[Episode]]:
    """Convenience: load from JSONL, then split.

    Parameters
    ----------
    trajectory_path
        Path to the trajectories JSONL file.
    ratios
        Split ratios (default 70/15/15).
    max_perturbations_per_task
        Cap perturbation variants per task.
    seed
        RNG seed for subsampling.
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


def _is_clean(ep: Episode) -> bool:
    pt = ep.perturbation_type
    if hasattr(pt, "value"):
        return pt.value == "none"
    return str(pt) == "none"
