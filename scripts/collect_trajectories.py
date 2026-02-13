#!/usr/bin/env python3
"""
Collect teacher trajectories from a benchmark environment.

Usage:
    python scripts/collect_trajectories.py \
        --config configs/webarena/clean.yaml \
        --output data/trajectories/webarena_teacher \
        --num-episodes 500 \
        --seeds 1 2 3

This script:
1. Loads benchmark tasks
2. Runs the teacher LLM on each task
3. Saves trajectories in JSONL format for downstream training
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.trajectory import (
    Action,
    ActionSource,
    Episode,
    EpisodeMetadata,
    Observation,
    Step,
    TrajectoryStore,
)
from r2v.utils.config import load_config
from r2v.utils.logging import JSONLLogger, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Collect teacher trajectories")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides (key=value)")
    return parser.parse_args()


def collect_with_teacher(cfg, task, seed: int, logger) -> Episode | None:
    """Run teacher LLM on a single task and collect trajectory.

    NOTE: This is the integration point with the actual benchmark env.
    Replace the placeholder with your WebArena/SWE-bench env calls.
    """
    # ── Placeholder: integrate with actual benchmark environment ──
    # For WebArena:
    #   from webarena.environment import WebArenaEnv
    #   env = WebArenaEnv(task_id=task["task_id"], seed=seed)
    #   obs = env.reset()
    #
    # For SWE-bench:
    #   from swebench.harness import SWEBenchEnv
    #   env = SWEBenchEnv(instance_id=task["instance_id"])
    #   obs = env.reset()

    logger.info(f"Collecting trajectory for task={task.get('task_id', 'unknown')} seed={seed}")

    # Placeholder episode structure
    steps = []
    max_steps = cfg.get("inference", {}).get("step_limit", 15)
    done = False
    t = 0

    while not done and t < max_steps:
        # ── Replace with actual teacher LLM call ──
        # response = teacher.generate(context + obs)
        # action_text = parse_action(response)
        # obs_text, reward, done, info = env.step(action_text)

        # Placeholder
        obs_text = f"[Placeholder observation at step {t}]"
        action_text = f"[Placeholder teacher action at step {t}]"
        done = t >= max_steps - 1
        reward = 1.0 if done else 0.0

        step = Step(
            observation=Observation(text=obs_text, timestamp=time.time()),
            action=Action(
                text=action_text,
                source=ActionSource.TEACHER,
                log_prob=None,
            ),
            reward=reward,
        )
        steps.append(step)
        t += 1

    metadata = EpisodeMetadata(
        benchmark=cfg.get("data", {}).get("benchmark", "unknown"),
        task_id=task.get("task_id", "unknown"),
        seed=seed,
        model_name=cfg.get("teacher", {}).get("model_name", "unknown"),
    )

    return Episode(
        episode_id=f"{task.get('task_id', 'unknown')}_seed{seed}",
        metadata=metadata,
        steps=steps,
        success=reward > 0.5,
        total_reward=sum(s.reward for s in steps if s.reward),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = TrajectoryStore(output_dir / "trajectories.jsonl")
    jsonl_log = JSONLLogger(output_dir / "collection_log.jsonl")

    # Log configuration
    from r2v.utils.config import config_to_dict
    jsonl_log.log_config(config_to_dict(cfg))

    # ── Load task list ──
    # Replace with actual benchmark task loading
    # For WebArena: tasks = load_webarena_tasks(cfg.data.task_file)
    # For SWE-bench: tasks = load_swebench_instances(cfg.data.split)
    tasks = [{"task_id": f"task_{i}"} for i in range(args.num_episodes)]

    total_success = 0
    total_episodes = 0

    for seed in args.seeds:
        logger.info(f"=== Collecting with seed {seed} ===")

        for task in tasks:
            episode = collect_with_teacher(cfg, task, seed, logger)
            if episode is not None:
                store.save_episode(episode)
                total_episodes += 1
                if episode.success:
                    total_success += 1

                jsonl_log.log_metric(
                    "episode_success",
                    float(episode.success),
                    step=total_episodes,
                    task_id=task.get("task_id"),
                    seed=seed,
                )

    sr = total_success / total_episodes if total_episodes > 0 else 0.0
    logger.info(f"Collection complete: {total_episodes} episodes, SR={sr:.3f}")
    jsonl_log.log("summary", {
        "total_episodes": total_episodes,
        "total_success": total_success,
        "success_rate": sr,
    })


if __name__ == "__main__":
    main()
