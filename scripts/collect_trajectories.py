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
1. Loads benchmark tasks (WebArena or SWE-bench)
2. Initializes the appropriate environment wrapper
3. Runs the teacher LLM on each task to generate actions
4. Saves trajectories in JSONL format for downstream training
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.benchmarks.base import BenchmarkEnv, BenchmarkTask, EnvStepResult
from r2v.data.registry import DataRegistry, make_run_id
from r2v.data.trajectory import (
    Action,
    ActionSource,
    Episode,
    EpisodeMetadata,
    Observation,
    PerturbationType,
    Step,
    TrajectoryStore,
)
from r2v.models.llm_client import LLMResponse, create_llm_client_from_cfg
from r2v.utils.config import load_config
from r2v.utils.logging import JSONLLogger, setup_logging


# ── Prompt templates ─────────────────────────────────────────────

WEBARENA_SYSTEM_PROMPT = """\
You are an autonomous web agent. You interact with websites by issuing actions \
in the following format:

Actions available:
  click [element_id]       - Click on an element
  type [element_id] [text] - Type text into an element
  hover [element_id]       - Hover over an element
  scroll [up|down]         - Scroll the page
  goto [url]               - Navigate to a URL
  go_back                  - Go back in browser history
  go_forward               - Go forward in browser history
  new_tab                  - Open a new tab
  tab_focus [tab_id]       - Switch to a tab
  press [key_combo]        - Press keyboard key(s)
  stop [answer]            - Stop and submit your answer

Respond with EXACTLY ONE action per turn. Think step-by-step about which \
action to take, then output the action on its own line prefixed with "Action: "."""

WEBARENA_USER_TEMPLATE = """\
Goal: {goal}

Current URL: {url}

Observation:
{observation}

Previous actions:
{action_history}

What is your next action?"""

SWEBENCH_SYSTEM_PROMPT = """\
You are an expert software engineer tasked with fixing a GitHub issue. \
You will be given a problem statement describing a bug or feature request. \
Analyze the issue carefully and provide a patch (in unified diff format) \
that resolves it.

Respond with your analysis followed by the patch in this format:
<patch>
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context
-old line
+new line
 context
</patch>

When you are confident in your patch, end with: stop [done]"""

SWEBENCH_USER_TEMPLATE = """\
{observation}

Please analyze this issue and provide a fix as a unified diff patch."""


# ── Benchmark factory ────────────────────────────────────────────


def create_benchmark_env(cfg) -> BenchmarkEnv:
    """Instantiate the correct benchmark environment from config."""
    benchmark = cfg.get("benchmark", "webarena")

    if benchmark == "webarena":
        from r2v.data.benchmarks.webarena_env import WebArenaEnv
        return WebArenaEnv(cfg)
    elif benchmark == "swebench":
        from r2v.data.benchmarks.swebench_env import SWEBenchEnv
        return SWEBenchEnv(cfg)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Supported: webarena, swebench")


# ── Action parsing ───────────────────────────────────────────────


def parse_action_from_response(response_text: str, benchmark: str) -> str:
    """Extract the action string from the teacher LLM's response.

    For WebArena: looks for "Action: <action>" pattern.
    For SWE-bench: extracts the patch or stop command.
    """
    if benchmark == "webarena":
        # Look for "Action: <action>" pattern
        match = re.search(r"Action:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: look for a known action verb at start of any line
        action_verbs = [
            "click", "type", "hover", "scroll", "goto", "go_back",
            "go_forward", "new_tab", "tab_focus", "press", "stop",
        ]
        for line in response_text.strip().split("\n"):
            line = line.strip()
            for verb in action_verbs:
                if line.lower().startswith(verb):
                    return line
        # Last resort
        return "stop [unable to parse action]"

    elif benchmark == "swebench":
        # Check for stop command
        match = re.search(r"stop\s*\[(.+?)\]", response_text, re.IGNORECASE)
        if match:
            return f"stop [{match.group(1)}]"

        # Check for patch content - return the full response for patch extraction
        if "<patch>" in response_text or "```diff" in response_text or "---" in response_text:
            return response_text

        return response_text

    return response_text


def classify_action_type(action_text: str, benchmark: str) -> str:
    """Classify the action type for metadata."""
    if benchmark == "webarena":
        verb = action_text.strip().split()[0].lower() if action_text.strip() else "unknown"
        return verb
    elif benchmark == "swebench":
        if action_text.lower().startswith("stop"):
            return "stop"
        return "patch"
    return "unknown"


# ── Core collection logic ────────────────────────────────────────


def collect_with_teacher(
    cfg,
    env: BenchmarkEnv,
    teacher_client,
    task: BenchmarkTask,
    seed: int,
    logger,
    *,
    run_id: str = "",
    teacher_model: str = "",
    teacher_provider: str = "",
) -> Episode | None:
    """Run the teacher LLM on a single task and collect a trajectory.

    This function:
    1. Resets the environment for the task
    2. In a loop: builds a prompt -> calls teacher LLM -> parses action -> steps env
    3. Evaluates the final trajectory for correctness
    4. Returns an Episode object
    """
    benchmark = cfg.get("benchmark", "webarena")
    max_steps = cfg.get("inference", {}).get("step_limit", 15)

    logger.info(f"Collecting trajectory: task={task.task_id}, seed={seed}")

    # Select prompt templates based on benchmark
    if benchmark == "webarena":
        system_prompt = WEBARENA_SYSTEM_PROMPT
        user_template = WEBARENA_USER_TEMPLATE
    else:
        system_prompt = SWEBENCH_SYSTEM_PROMPT
        user_template = SWEBENCH_USER_TEMPLATE

    try:
        # Reset environment
        initial_obs = env.reset(task, seed=seed)
    except Exception as e:
        logger.error(f"Failed to reset env for task={task.task_id}: {e}")
        return None

    # Collection state
    steps: list[Step] = []
    action_history: list[str] = []
    current_obs = initial_obs
    current_url = task.extra.get("start_url", "N/A")
    total_cost = 0.0
    done = False
    t = 0
    wall_start = time.time()

    while not done and t < max_steps:
        # Build the prompt
        if benchmark == "webarena":
            max_obs_chars = cfg.get("webarena", {}).get("observation", {}).get("max_tokens", 4096) * 4
            user_prompt = user_template.format(
                goal=task.goal,
                url=current_url,
                observation=current_obs[:max_obs_chars],
                action_history="\n".join(action_history[-5:]) if action_history else "(none)",
            )
        else:
            user_prompt = user_template.format(observation=current_obs)

        # Call the teacher LLM
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response: LLMResponse = teacher_client.chat(messages)
            total_cost += response.cost
        except Exception as e:
            logger.error(f"Teacher LLM call failed at step {t}: {e}")
            break

        # Parse the action from the response
        action_text = parse_action_from_response(response.text, benchmark)
        action_type = classify_action_type(action_text, benchmark)
        is_stop = action_type in ("stop", "finish", "submit")
        action_history.append(f"Step {t}: {action_text[:100]}")

        # Step the environment
        try:
            if is_stop and benchmark == "webarena":
                step_result = EnvStepResult(
                    observation="[STOP]", reward=0.0, done=True
                )
            else:
                step_result = env.step(action_text)
        except Exception as e:
            logger.warning(f"Env step failed at step {t}: {e}")
            step_result = EnvStepResult(
                observation=f"[ERROR: {str(e)[:200]}]",
                reward=0.0,
                done=True,
                info={"error": str(e)},
            )

        # Record the step
        step = Step(
            step_idx=t,
            observation=Observation(
                raw_text=current_obs,
                url=current_url if benchmark == "webarena" else None,
            ),
            action=Action(
                raw_text=action_text,
                action_type=action_type,
            ),
            action_source=ActionSource.TEACHER,
            reward=step_result.reward,
            perturbation_type=PerturbationType.NONE,
        )
        steps.append(step)

        # Update state
        current_obs = step_result.observation
        if step_result.url:
            current_url = step_result.url
        done = step_result.done or is_stop
        t += 1

    wall_time = time.time() - wall_start

    # Evaluate the trajectory
    try:
        score = env.evaluate()
    except Exception as e:
        logger.warning(f"Evaluation failed for task={task.task_id}: {e}")
        score = 0.0
    success = score > 0.5

    # Build episode metadata
    metadata = EpisodeMetadata(
        task_id=task.task_id,
        template_id=task.template_id,
        goal=task.goal,
        benchmark=benchmark,
        site=task.site,
        repo=task.repo,
        difficulty=task.difficulty,
        run_id=run_id,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
    )

    episode = Episode(
        episode_id=f"{task.task_id}_seed{seed}",
        metadata=metadata,
        steps=steps,
        success=success,
        partial_score=score,
        total_cost=total_cost,
        wall_time_seconds=wall_time,
        perturbation_type=PerturbationType.NONE,
    )

    logger.info(
        f"  Task={task.task_id} seed={seed}: "
        f"steps={len(steps)}, success={success}, score={score:.2f}, "
        f"cost=${total_cost:.4f}, time={wall_time:.1f}s"
    )
    return episode


# ── CLI & main ────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect teacher trajectories from benchmark environments"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Config YAML path (e.g., configs/webarena/clean.yaml)"
    )
    parser.add_argument(
        "--output", type=str, default="data/runs",
        help=(
            "Base output directory. A versioned subdirectory "
            "<benchmark>_<provider>_<model>_<timestamp> is created inside it. "
            "Default: data/runs"
        ),
    )
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="Config overrides (key=value)"
    )
    parser.add_argument(
        "--run-tags", nargs="*", default=[],
        help="Optional tags for this run, stored in registry (e.g. noisy ablation)",
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Free-text notes stored in run_manifest.json",
    )
    parser.add_argument(
        "--run-id", type=str, default="",
        help=(
            "Override the auto-generated run ID. "
            "Useful for resuming a partially-completed run."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")

    benchmark = cfg.get("benchmark", "webarena")

    # ── Initialize teacher LLM client ────────────────────────────
    logger.info("Initializing teacher LLM client...")
    teacher_client = create_llm_client_from_cfg(cfg)
    teacher_model: str = teacher_client.config.model_name
    teacher_provider: str = teacher_client.config.provider
    logger.info(f"Teacher: provider={teacher_provider}, model={teacher_model}")

    # ── Determine run ID and output directory ────────────────────
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = make_run_id(benchmark, teacher_provider, teacher_model)

    output_dir = Path(args.output) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"  Run ID : {run_id}")
    logger.info(f"  Output : {output_dir}")
    logger.info(f"{'='*60}")

    # ── Register the run ─────────────────────────────────────────
    registry = DataRegistry()
    from r2v.utils.config import config_to_dict
    cfg_dict = config_to_dict(cfg)
    registry.begin_run(
        run_id,
        benchmark=benchmark,
        provider=teacher_provider,
        model_name=teacher_model,
        output_dir=output_dir,
        cfg=cfg_dict,
        tags=args.run_tags,
        notes=args.notes,
    )

    store = TrajectoryStore(output_dir / "trajectories.jsonl")
    jsonl_log = JSONLLogger(output_dir / "collection_log.jsonl")
    jsonl_log.log_config(cfg_dict)

    # ── Initialize benchmark environment ─────────────────────────
    logger.info(f"Setting up {benchmark} environment...")
    env = create_benchmark_env(cfg)

    # Load tasks
    tasks = env.load_tasks(cfg)
    if not tasks:
        registry.fail_run(run_id, reason="No tasks loaded from benchmark config")
        logger.error("No tasks loaded. Check your benchmark configuration.")
        sys.exit(1)

    # Limit to requested number of episodes
    if args.num_episodes < len(tasks):
        random.seed(cfg.get("project", {}).get("seed", 42))
        tasks = random.sample(tasks, args.num_episodes)

    logger.info(f"Loaded {len(tasks)} tasks, collecting with seeds={args.seeds}")

    total_success = 0
    total_episodes = 0
    total_cost = 0.0
    wall_start = time.time()
    collection_status = "running"

    try:
        for seed in args.seeds:
            logger.info(f"=== Collecting with seed {seed} ===")

            for i, task in enumerate(tasks):
                logger.info(f"  [{i + 1}/{len(tasks)}] Task: {task.task_id}")

                episode = collect_with_teacher(
                    cfg, env, teacher_client, task, seed, logger,
                    run_id=run_id,
                    teacher_model=teacher_model,
                    teacher_provider=teacher_provider,
                )

                if episode is not None:
                    store.save_episode(episode)
                    total_episodes += 1
                    total_cost += episode.total_cost
                    if episode.success:
                        total_success += 1

                    jsonl_log.log_metric(
                        "episode_result",
                        float(episode.success),
                        step=total_episodes,
                        task_id=task.task_id,
                        seed=seed,
                        num_steps=episode.num_steps,
                        score=episode.partial_score,
                        cost=episode.total_cost,
                        wall_time=episode.wall_time_seconds,
                    )
        collection_status = "done"
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user.")
        collection_status = "partial"
    except Exception as exc:
        logger.error(f"Collection failed with error: {exc}")
        collection_status = "failed"
        registry.fail_run(run_id, reason=str(exc))
        raise
    finally:
        env.close()

    wall_time = time.time() - wall_start
    sr = total_success / total_episodes if total_episodes > 0 else 0.0
    logger.info(f"=== Collection complete ===")
    logger.info(f"  Run ID  : {run_id}")
    logger.info(f"  Episodes: {total_episodes}")
    logger.info(f"  Success : {sr:.3f} ({total_success}/{total_episodes})")
    logger.info(f"  Cost    : ${total_cost:.4f}")
    logger.info(f"  Output  : {output_dir}")

    jsonl_log.log("summary", {
        "run_id": run_id,
        "total_episodes": total_episodes,
        "total_success": total_success,
        "success_rate": sr,
        "total_cost_usd": round(total_cost, 6),
        "wall_time_seconds": round(wall_time, 1),
        "benchmark": benchmark,
        "teacher_provider": teacher_provider,
        "teacher_model": teacher_model,
    })

    # ── Finalise run in registry ──────────────────────────────────
    registry.finish_run(
        run_id,
        n_episodes=total_episodes,
        n_success=total_success,
        total_cost=total_cost,
        wall_time=wall_time,
        status=collection_status,
    )
    logger.info(f"Run registered. View all runs:\n{registry.summary()}")


if __name__ == "__main__":
    main()
