#!/usr/bin/env python3
"""
Collect teacher trajectories from a benchmark environment.

Usage:
    python scripts/collect_trajectories.py \
        --config configs/humaneval/clean.yaml \
        --output data/trajectories/humaneval_teacher \
        --num-episodes 500 \
        --seeds 1 2 3

This script:
1. Loads benchmark tasks from the configured benchmark
2. Initializes the appropriate environment wrapper
3. Runs the teacher LLM on each task to generate actions
4. Saves trajectories in JSONL format for downstream training
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

_MAX_PROMPT_CHARS = 16_000  # Max observation chars in prompt

# ── TextWorld prompt templates ───────────────────────────────────

TEXTWORLD_SYSTEM_PROMPT = """\
You are a text-game agent in a TextWorld environment.

Actions available:
    look
    inventory
    examine <object>
    go <direction>
    open <object>
    close <object>
    take <object>
    drop <object>
    put <object> on/in <object>
    insert <object> into <object>
    unlock <object> with <object>
    eat <object>

Respond with EXACTLY ONE action per turn. Think step-by-step, then output \
the action on its own line prefixed with "Action: "."""

TEXTWORLD_USER_TEMPLATE = """\
Goal: {goal}

Observation:
{observation}

Previous actions:
{action_history}

What is your next action?"""


# ── HumanEval+ prompt templates ──────────────────────────────────

HUMANEVAL_SYSTEM_PROMPT = """\
You are an expert Python programmer. You solve coding problems by writing \
and testing code iteratively.

Actions available:
  write_code [your_python_code]  – Write/overwrite your solution
  test                           – Test your current solution
  submit                         – Submit your solution for evaluation

Workflow:
1. Read the function signature and docstring carefully.
2. Write your implementation using write_code [...].
3. Test it using test.
4. Fix any errors and re-test.
5. Submit when confident.

Respond with EXACTLY ONE action per turn. Think step-by-step, then output \
the action on its own line prefixed with "Action: "."""

HUMANEVAL_USER_TEMPLATE = """\
{observation}

Previous actions:
{action_history}

What is your next action?"""


# ── RTL-Repair prompt templates ───────────────────────────────

RTLREPAIR_SYSTEM_PROMPT = """\
You are an expert RTL engineer fixing buggy Verilog modules.

Actions available:
    write_code [your_verilog_code]  - Write/overwrite the current candidate patch
    test                            - Run the configured simulation/test command
    submit                          - Submit your final patch

Workflow:
1. Inspect the buggy module and the repair objective.
2. Propose a minimal, correct patch using write_code [...].
3. Run test, inspect failures, and iterate.
4. Submit when tests are passing.

Respond with EXACTLY ONE action per turn. Think step-by-step, then output \
the action on its own line prefixed with "Action: "."""

RTLREPAIR_USER_TEMPLATE = """\
Goal: {goal}

Observation:
{observation}

Previous actions:
{action_history}

What is your next action?"""


# ── Benchmark factory ────────────────────────────────────────────


def create_benchmark_env(cfg) -> BenchmarkEnv:
    """Instantiate the correct benchmark environment from config."""
    benchmark = cfg.get("benchmark", "humaneval")

    if benchmark == "humaneval":
        from r2v.data.benchmarks.humaneval_env import HumanEvalPlusEnv
        return HumanEvalPlusEnv(cfg)
    elif benchmark == "textworld":
        from r2v.data.benchmarks.textworld_env import TextWorldEnv
        return TextWorldEnv(cfg)
    elif benchmark == "rtlrepair":
        from r2v.data.benchmarks.rtlrepair_env import RTLRepairEnv
        return RTLRepairEnv(cfg)
    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. "
            f"Supported: humaneval, textworld, rtlrepair"
        )


# ── Action parsing ───────────────────────────────────────────────


def parse_action_from_response(response_text: str, benchmark: str) -> str:
    """Extract the action string from the teacher LLM's response.

    For HumanEval/TextWorld/RTLRepair: looks for "Action: <action>" pattern.
    """
    if benchmark in ("humaneval", "textworld", "rtlrepair"):
        # ── HumanEval: needs multiline-aware extraction ──────────
        # `write_code [<code>]` spans many lines; the generic single-line
        # regex below would truncate it to just `write_code [`.
        if benchmark == "humaneval":
            # 1) Full markdown code block with or without Action: prefix
            cb = re.search(r"```(?:python)?\n(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
            if cb:
                return f"write_code [{cb.group(1).strip()}]"

            # 2) write_code [...] spanning multiple lines — find opening bracket,
            #    then scan forward to the last ']' in the response.
            wc = re.search(r"write_code\s*\[", response_text, re.IGNORECASE)
            if wc:
                body = response_text[wc.end():]
                last_bracket = body.rfind("]")
                if last_bracket != -1:
                    code = body[:last_bracket].strip()
                    if code:
                        return f"write_code [{code}]"
                # Opening bracket found but no closing — take everything after it
                code = body.strip()
                if code:
                    return f"write_code [{code}]"

            # 3) test / submit are single-word, safe to match on one line
            for keyword in ("submit", "test"):
                if re.search(rf"\bAction:\s*{keyword}\b", response_text, re.IGNORECASE):
                    return keyword
                if re.search(rf"^{keyword}\b", response_text.strip(), re.IGNORECASE | re.MULTILINE):
                    return keyword

            # 4) Raw Python code anywhere in the response
            raw_py = re.search(r"^def \w+", response_text, re.MULTILINE)
            if raw_py:
                return f"write_code [{response_text[raw_py.start():].strip()}]"

            return response_text.strip().split("\n")[-1]

        if benchmark == "rtlrepair":
            cb = re.search(
                r"```(?:verilog|systemverilog)?\s*(.*?)```",
                response_text,
                re.DOTALL | re.IGNORECASE,
            )
            if cb:
                code = cb.group(1).strip()
                if code:
                    return f"write_code [{code}]"

            wc = re.search(r"write_code\s*\[", response_text, re.IGNORECASE)
            if wc:
                body = response_text[wc.end():]
                last_bracket = body.rfind("]")
                if last_bracket != -1:
                    code = body[:last_bracket].strip()
                    if code:
                        return f"write_code [{code}]"
                code = body.strip()
                if code:
                    return f"write_code [{code}]"

            # Fallback: capture a full module body even when not fenced.
            module_match = re.search(
                r"(module\s+\w+[\s\S]*?endmodule)",
                response_text,
                re.IGNORECASE,
            )
            if module_match:
                code = module_match.group(1).strip()
                if code:
                    return f"write_code [{code}]"

            for keyword in ("submit", "test"):
                if re.search(rf"\bAction:\s*{keyword}\b", response_text, re.IGNORECASE):
                    return keyword
                if re.search(rf"^{keyword}\b", response_text.strip(), re.IGNORECASE | re.MULTILINE):
                    return keyword

            raw_rtl = re.search(r"^\s*module\s+\w+", response_text, re.IGNORECASE | re.MULTILINE)
            if raw_rtl:
                return f"write_code [{response_text[raw_rtl.start():].strip()}]"

            return response_text.strip().split("\n")[-1]

        # ── Single-line "Action: <action>" pattern ──
        match = re.search(r"Action:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        if benchmark == "textworld":
            tw_verbs = [
                "look", "inventory", "examine", "go", "open", "close", "take",
                "drop", "put", "insert", "unlock", "eat",
            ]
            for line in response_text.strip().split("\n"):
                line = line.strip()
                for verb in tw_verbs:
                    if line.lower().startswith(verb):
                        return line

        return response_text.strip().split("\n")[-1]  # Last line as fallback

    return response_text


def classify_action_type(action_text: str, benchmark: str) -> str:
    """Classify the action type for metadata."""
    if benchmark == "humaneval":
        lower = action_text.strip().lower()
        if lower == "submit":
            return "submit"
        if lower.startswith("write_code") or lower.startswith("def "):
            return "write_code"
        if lower.startswith("test"):
            return "test"
        return "unknown"
    elif benchmark == "textworld":
        lower = action_text.strip().lower()
        for verb in [
            "look", "inventory", "examine", "go", "open", "close", "take",
            "drop", "put", "insert", "unlock", "eat",
        ]:
            if lower.startswith(verb):
                return verb
        if lower in ("submit", "finish", "stop"):
            return "finish"
        return "unknown"
    elif benchmark == "rtlrepair":
        lower = action_text.strip().lower()
        if lower == "submit":
            return "submit"
        if lower.startswith("test"):
            return "test"
        if lower.startswith("write_code") or lower.startswith("module ") or "endmodule" in lower:
            return "write_code"
        return "unknown"
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

    For HumanEval/TextWorld: single-turn per step with context re-build.
    """
    benchmark = cfg.get("benchmark", "humaneval")
    # data.max_episode_steps is set per benchmark; fall back to
    # inference.step_limit only if not provided.
    max_steps = (
        cfg.get("data", {}).get("max_episode_steps")
        or cfg.get("inference", {}).get("step_limit", 15)
    )

    logger.info(f"Collecting trajectory: task={task.task_id}, seed={seed}, max_steps={max_steps}")

    # Select prompt templates based on benchmark
    if benchmark == "humaneval":
        system_prompt = HUMANEVAL_SYSTEM_PROMPT
        user_template = HUMANEVAL_USER_TEMPLATE
    elif benchmark == "textworld":
        system_prompt = TEXTWORLD_SYSTEM_PROMPT
        user_template = TEXTWORLD_USER_TEMPLATE
    elif benchmark == "rtlrepair":
        system_prompt = RTLREPAIR_SYSTEM_PROMPT
        user_template = RTLREPAIR_USER_TEMPLATE
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

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
    total_cost = 0.0
    done = False
    t = 0
    wall_start = time.time()

    while not done and t < max_steps:
        # ── Build prompt / call LLM ──
        # Single-turn per step with full context re-build
        fmt_kwargs = {
            "observation": current_obs[:_MAX_PROMPT_CHARS],
            "action_history": "\n".join(action_history[-5:]) if action_history else "(none)",
        }
        if benchmark == "textworld":
            fmt_kwargs["goal"] = task.goal
        if benchmark == "rtlrepair":
            fmt_kwargs["goal"] = task.goal
        user_prompt = user_template.format(**fmt_kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            step_start = time.time()
            # Hard per-step timeout: kills the thread if the API hangs.
            _LLM_STEP_TIMEOUT = 180  # seconds
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
            with ThreadPoolExecutor(max_workers=1) as _pool:
                _fut = _pool.submit(teacher_client.chat, messages)
                try:
                    response: LLMResponse = _fut.result(timeout=_LLM_STEP_TIMEOUT)
                except FutureTimeout:
                    logger.error(
                        f"Teacher LLM call timed out at step {t} "
                        f"after {_LLM_STEP_TIMEOUT}s — aborting episode"
                    )
                    break
            total_cost += response.cost
            step_dur = time.time() - step_start
            logger.info(
                f"    step {t}: tokens_in={response.input_tokens} "
                f"tokens_out={response.output_tokens} "
                f"cost=${response.cost:.5f} time={step_dur:.1f}s"
            )
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
                url=None,
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
        help="Config YAML path (e.g., configs/humaneval/clean.yaml)"
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
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help=(
            "Number of parallel worker processes for trajectory collection. "
            "Each worker gets its own environment instance and LLM client. "
            "Recommended: 1-4 depending on API and compute budget. "
            "Default: 1 (sequential)."
        ),
    )
    parser.add_argument(
        "--resume-from", type=str, default="",
        help=(
            "Path to an existing trajectories JSONL file. Episodes whose "
            "episode_id already appears there will be skipped. Useful for "
            "resuming a partially-completed run without re-doing finished work."
        ),
    )
    return parser.parse_args()


# ── Progress helpers ──────────────────────────────────────────────


def _write_worker_progress(
    progress_dir: Path,
    worker_id: int,
    completed: int,
    total: int,
    successes: int,
    cost: float,
    last_task: str,
) -> None:
    """Atomically write a worker's progress to a JSON file."""
    progress_file = progress_dir / f"worker_{worker_id}.json"
    data = {
        "worker_id": worker_id,
        "completed": completed,
        "total": total,
        "successes": successes,
        "cost": round(cost, 6),
        "last_task": last_task,
        "updated": time.strftime("%H:%M:%S"),
    }
    tmp = progress_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(progress_file)


def _read_all_progress(progress_dir: Path) -> list[dict]:
    """Read progress from all worker JSON files."""
    results = []
    for p in sorted(progress_dir.glob("worker_*.json")):
        try:
            results.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return results


def _progress_monitor(
    progress_dir: Path,
    total_episodes: int,
    logger,
    interval: float = 30.0,
    stop_event: threading.Event | None = None,
) -> None:
    """Background thread that logs aggregate progress every *interval* seconds."""
    while not (stop_event and stop_event.is_set()):
        time.sleep(interval)
        workers = _read_all_progress(progress_dir)
        if not workers:
            continue
        done = sum(w["completed"] for w in workers)
        successes = sum(w["successes"] for w in workers)
        cost = sum(w["cost"] for w in workers)
        pct = done / total_episodes * 100 if total_episodes else 0
        sr = successes / done if done else 0
        logger.info(
            f"[PROGRESS] {done}/{total_episodes} episodes ({pct:.1f}%) | "
            f"success_rate={sr:.3f} | cost=${cost:.4f}"
        )
        for w in workers:
            logger.info(
                f"  worker {w['worker_id']}: "
                f"{w['completed']}/{w['total']} done, "
                f"last_task={w['last_task']}, updated={w['updated']}"
            )


# ── Worker function for multiprocessing ──────────────────────────


def _worker_collect(
    worker_id: int,
    task_seed_pairs: list[tuple],
    config_path: str,
    overrides: list[str],
    run_id: str,
    output_dir: str,
) -> list[dict]:
    """Worker process: creates its own env + LLM client and collects episodes.

    Each worker is fully independent — its own environment instance, LLM
    client, and logger.  Episodes are saved incrementally to a per-worker
    JSONL file and progress is written to a JSON file for monitoring.
    Results are also returned for final aggregation by the main process.
    """
    # Re-import inside worker (new process)
    from r2v.data.benchmarks.base import BenchmarkTask
    from r2v.data.trajectory import TrajectoryStore
    from r2v.models.llm_client import create_llm_client_from_cfg
    from r2v.utils.config import load_config
    from r2v.utils.logging import setup_logging

    output_path = Path(output_dir)
    log_dir = output_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = output_path / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path, overrides)
    logger = setup_logging(
        level="INFO",
        name=f"r2v.w{worker_id}",
        log_file=str(log_dir / f"worker_{worker_id}.log"),
    )
    benchmark = cfg.get("benchmark", "humaneval")

    # Each worker gets its own LLM client and environment
    teacher_client = create_llm_client_from_cfg(cfg)
    teacher_model = teacher_client.config.model_name
    teacher_provider = teacher_client.config.provider
    env = create_benchmark_env(cfg)

    # Per-worker trajectory store for incremental saving
    worker_store = TrajectoryStore(
        output_path / f"worker_{worker_id}_trajectories.jsonl"
    )

    # Pre-load tasks so we can look them up by id
    all_tasks = env.load_tasks(cfg)
    task_map = {t.task_id: t for t in all_tasks}

    total = len(task_seed_pairs)
    completed = 0
    successes = 0
    total_cost = 0.0

    logger.info(f"Worker {worker_id} starting: {total} episodes to collect")
    _write_worker_progress(progress_dir, worker_id, 0, total, 0, 0.0, "(starting)")

    results: list[dict] = []
    for task_id, seed in task_seed_pairs:
        task = task_map.get(task_id)
        if task is None:
            logger.warning(f"Task {task_id} not found — skipping")
            completed += 1
            continue
        logger.info(f"[{completed + 1}/{total}] Collecting task={task_id}, seed={seed}")
        episode = collect_with_teacher(
            cfg, env, teacher_client, task, seed, logger,
            run_id=run_id,
            teacher_model=teacher_model,
            teacher_provider=teacher_provider,
        )
        if episode is not None:
            # Save incrementally to per-worker file
            worker_store.save_episode(episode)
            results.append({
                "episode": episode,
                "task_id": task_id,
                "seed": seed,
            })
            total_cost += episode.total_cost
            if episode.success:
                successes += 1

        completed += 1
        _write_worker_progress(
            progress_dir, worker_id, completed, total,
            successes, total_cost, str(task_id),
        )

    env.close()
    logger.info(
        f"Worker {worker_id} finished: {len(results)} episodes, "
        f"{successes} successes, cost=${total_cost:.4f}"
    )
    return results


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    benchmark = cfg.get("benchmark", "humaneval")
    num_workers = max(1, args.num_workers)

    # ── Initialize teacher LLM client (main process, for metadata) ───
    # (logger needs output_dir, but we need model info for run_id, so
    #  create a temporary logger first)
    tmp_logger = setup_logging(level="INFO")
    tmp_logger.info("Initializing teacher LLM client...")
    teacher_client = create_llm_client_from_cfg(cfg)
    teacher_model: str = teacher_client.config.model_name
    teacher_provider: str = teacher_client.config.provider
    tmp_logger.info(f"Teacher: provider={teacher_provider}, model={teacher_model}")

    # ── Determine run ID and output directory ────────────────────
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = make_run_id(benchmark, teacher_provider, teacher_model)

    output_dir = Path(args.output) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Set up file + console logging for main process ───────────
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        level="INFO",
        log_file=str(log_dir / "main.log"),
    )

    logger.info(f"{'='*60}")
    logger.info(f"  Run ID  : {run_id}")
    logger.info(f"  Output  : {output_dir}")
    logger.info(f"  Log dir : {log_dir}")
    logger.info(f"  Workers : {num_workers}")
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

    # ── Initialize benchmark environment (main process) ──────────
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

    # Build full list of (task_id, seed) pairs
    task_seed_pairs = [
        (task.task_id, seed)
        for seed in args.seeds
        for task in tasks
    ]

    # ── Resume: skip already-collected episodes ───────────────────
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            logger.error(f"--resume-from file not found: {resume_path}")
            sys.exit(1)
        already_done: set[str] = set()
        with open(resume_path) as _rf:
            for _line in _rf:
                _line = _line.strip()
                if _line:
                    _ep = json.loads(_line)
                    _eid = _ep.get("episode_id", "")
                    if _eid:
                        already_done.add(_eid)
        before = len(task_seed_pairs)
        task_seed_pairs = [
            (tid, seed) for tid, seed in task_seed_pairs
            if f"{tid}_seed{seed}" not in already_done
        ]
        logger.info(
            f"--resume-from: skipping {before - len(task_seed_pairs)} already-collected "
            f"episodes ({len(task_seed_pairs)} remaining)"
        )

    logger.info(f"Total episodes to collect: {len(task_seed_pairs)}")

    total_success = 0
    total_episodes = 0
    total_cost = 0.0
    wall_start = time.time()
    collection_status = "running"

    # ── Sequential collection (num_workers == 1) ─────────────────
    if num_workers == 1:
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

    # ── Parallel collection (num_workers > 1) ────────────────────
    else:
        env.close()  # main-process env not needed; workers create their own

        # Create progress directory
        progress_dir = output_dir / "progress"
        progress_dir.mkdir(parents=True, exist_ok=True)

        # Partition task_seed_pairs across workers (round-robin)
        worker_chunks: list[list[tuple]] = [[] for _ in range(num_workers)]
        for idx, pair in enumerate(task_seed_pairs):
            worker_chunks[idx % num_workers].append(pair)

        logger.info(
            f"Distributing {len(task_seed_pairs)} episodes across "
            f"{num_workers} workers "
            f"(~{len(task_seed_pairs) // num_workers} each)"
        )
        logger.info(f"Worker logs: {log_dir}/worker_*.log")
        logger.info(f"Worker progress: {progress_dir}/worker_*.json")

        # Start background progress monitor thread
        stop_monitor = threading.Event()
        monitor = threading.Thread(
            target=_progress_monitor,
            args=(progress_dir, len(task_seed_pairs), logger),
            kwargs={"interval": 30.0, "stop_event": stop_monitor},
            daemon=True,
        )
        monitor.start()

        try:
            # Use 'spawn' context to avoid fork-safety issues with CUDA / Playwright
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_workers, mp_context=ctx
            ) as executor:
                futures = {
                    executor.submit(
                        _worker_collect,
                        wid,
                        chunk,
                        args.config,
                        args.overrides,
                        run_id,
                        str(output_dir),
                    ): wid
                    for wid, chunk in enumerate(worker_chunks)
                    if chunk  # skip empty chunks
                }

                for future in as_completed(futures):
                    wid = futures[future]
                    try:
                        results = future.result()
                    except Exception as exc:
                        logger.error(f"Worker {wid} failed: {exc}")
                        continue

                    for res in results:
                        episode = res["episode"]
                        store.save_episode(episode)
                        total_episodes += 1
                        total_cost += episode.total_cost
                        if episode.success:
                            total_success += 1

                        jsonl_log.log_metric(
                            "episode_result",
                            float(episode.success),
                            step=total_episodes,
                            task_id=res["task_id"],
                            seed=res["seed"],
                            num_steps=episode.num_steps,
                            score=episode.partial_score,
                            cost=episode.total_cost,
                            wall_time=episode.wall_time_seconds,
                        )
                    logger.info(
                        f"Worker {wid} finished: {len(results)} episodes collected"
                    )
            collection_status = "done"
        except KeyboardInterrupt:
            logger.info("Collection interrupted by user.")
            collection_status = "partial"
        except Exception as exc:
            logger.error(f"Parallel collection failed: {exc}")
            collection_status = "failed"
            registry.fail_run(run_id, reason=str(exc))
            raise
        finally:
            stop_monitor.set()
            monitor.join(timeout=5)

    wall_time = time.time() - wall_start
    sr = total_success / total_episodes if total_episodes > 0 else 0.0
    logger.info(f"=== Collection complete ===")
    logger.info(f"  Run ID  : {run_id}")
    logger.info(f"  Episodes: {total_episodes}")
    logger.info(f"  Success : {sr:.3f} ({total_success}/{total_episodes})")
    logger.info(f"  Cost    : ${total_cost:.4f}")
    logger.info(f"  Workers : {num_workers}")
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
        "num_workers": num_workers,
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
