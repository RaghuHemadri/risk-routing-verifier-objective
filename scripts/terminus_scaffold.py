"""
Terminus scaffold for Terminal-Bench trajectory collection.

Uses the official terminal-bench leaderboard agent prompt + CommandBatchResponse
schema, but executes commands through env.step() for correct tmux handling.

To wire into collect_trajectories.py, replace collect_with_teacher calls for
terminalbench with collect_terminus_trajectory, e.g.:

    _collector = (
        collect_terminus_trajectory
        if cfg.get("benchmark") == "terminalbench"
        else collect_with_teacher
    )
"""

from __future__ import annotations

import json as _json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from r2v.data.trajectory import (
    Action,
    ActionSource,
    Episode,
    EpisodeMetadata,
    Observation,
    PerturbationType,
    Step,
)
from r2v.models.llm_client import LLMResponse


class _Command(BaseModel):
    keystrokes: str
    is_blocking: bool
    timeout_sec: float
    model_config = ConfigDict(extra="forbid")


class _CommandBatchResponse(BaseModel):
    state_analysis: str
    explanation: str
    commands: list[_Command]
    is_task_complete: bool
    model_config = ConfigDict(extra="forbid")


_RESPONSE_SCHEMA = _json.dumps(_CommandBatchResponse.model_json_schema(), indent=2)

_PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / ".venv312/lib/python3.12/site-packages/terminal_bench/agents/prompt-templates/terminus.txt"
)

_PROMPT_TEMPLATE = (
    _PROMPT_TEMPLATE_PATH.read_text()
    if _PROMPT_TEMPLATE_PATH.exists()
    else (
        "You are an AI assistant tasked with solving command-line tasks in a Linux environment.\n\n"
        "Instruction:\n{instruction}\n\n"
        "Your response must be a JSON object matching this schema:\n{response_schema}\n\n"
        "Don't include markdown formatting.\n\n"
        "The current terminal state is:\n{terminal_state}"
    )
)


def collect_terminus_trajectory(
    cfg,
    env,
    teacher_client,
    task,
    seed: int,
    logger,
    *,
    run_id: str = "",
    teacher_model: str = "",
    teacher_provider: str = "",
) -> "Episode | None":
    """Collect a trajectory using the Terminus scaffold.

    Uses Terminus prompt + CommandBatchResponse JSON parsing, but executes
    each command via env.step("run [cmd]") so tmux output capture works correctly.
    """
    max_steps = (
        cfg.get("data", {}).get("max_episode_steps")
        or cfg.get("inference", {}).get("step_limit", 50)
    )

    logger.info(
        f"Collecting trajectory (Terminus): task={task.task_id}, seed={seed}, max_steps={max_steps}"
    )

    try:
        current_obs = env.reset(task, seed=seed)
    except Exception as e:
        logger.error(f"Failed to reset env for task={task.task_id}: {e}")
        return None

    # Extract terminal state from the formatted observation
    def _terminal_state(obs_text: str) -> str:
        if "Terminal output:" in obs_text:
            return obs_text.split("Terminal output:")[1].split("Available actions:")[0].strip()
        return obs_text

    steps = []
    total_cost = 0.0
    wall_start = time.time()
    step_idx = 0
    done = False

    while not done and step_idx < max_steps:
        terminal_state = _terminal_state(current_obs)

        prompt = _PROMPT_TEMPLATE.format(
            instruction=task.goal,
            response_schema=_RESPONSE_SCHEMA,
            terminal_state=terminal_state,
            history="",
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            step_start = time.time()
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(teacher_client.chat, messages)
                try:
                    response: LLMResponse = fut.result(timeout=180)
                except FutureTimeout:
                    logger.error(f"Teacher LLM timed out at step {step_idx}")
                    break
            total_cost += response.cost
            logger.info(
                f"    step {step_idx}: tokens_in={response.input_tokens} "
                f"tokens_out={response.output_tokens} "
                f"cost=${response.cost:.5f} time={time.time()-step_start:.1f}s"
            )
        except Exception as e:
            logger.error(f"Teacher LLM call failed at step {step_idx}: {e}")
            break

        # Parse CommandBatchResponse JSON
        raw_text = response.text.strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            batch = _CommandBatchResponse.model_validate_json(raw_text)
        except (ValidationError, Exception) as e:
            logger.warning(f"Failed to parse CommandBatchResponse at step {step_idx}: {e}")
            # Fall back: try to extract any shell commands and run them
            cmds = re.findall(r'"keystrokes":\s*"([^"]+)"', raw_text)
            if not cmds:
                step_idx += 1
                continue
            # Fake a batch from extracted keystrokes
            class _FakeBatch:
                commands = [type('C', (), {'keystrokes': k, 'is_blocking': True, 'timeout_sec': 30})() for k in cmds]
                is_task_complete = False
                state_analysis = ""
            batch = _FakeBatch()

        obs_before = current_obs
        last_step_result = None

        # Execute each command via env.step() — handles tmux correctly
        for cmd in batch.commands:
            ks = cmd.keystrokes.rstrip("\n")
            if not ks:
                continue

            # Map special keystrokes
            if ks in ("C-c", "^C"):
                action_text = "run [true]"  # send ctrl-c via a no-op; best effort
            else:
                action_text = f"run [{ks}]"

            try:
                from r2v.data.benchmarks.base import EnvStepResult
                last_step_result = env.step(action_text)
                current_obs = last_step_result.observation
                if last_step_result.done:
                    done = True
                    break
            except Exception as exc:
                logger.warning(f"env.step failed: {exc}")

        # Record as one step with combined action summary
        cmds_text = " && ".join(
            c.keystrokes.rstrip("\n") for c in batch.commands if c.keystrokes.strip()
        )[:200]
        action_text = f"run [{cmds_text}]"

        steps.append(Step(
            step_idx=step_idx,
            observation=Observation(raw_text=obs_before),
            action=Action(raw_text=action_text, action_type="run"),
            action_source=ActionSource.TEACHER,
            reward=last_step_result.reward if last_step_result else 0.0,
            perturbation_type=PerturbationType.NONE,
            llm_tokens_in=response.input_tokens,
            llm_tokens_out=response.output_tokens,
            llm_cost=response.cost,
        ))
        step_idx += 1

        if batch.is_task_complete:
            # Run evaluation via submit
            submit_result = env.step("submit")
            current_obs = submit_result.observation
            done = True

    wall_time = time.time() - wall_start

    try:
        score = env.evaluate()
    except Exception as e:
        logger.warning(f"Evaluation failed for task={task.task_id}: {e}")
        score = 0.0
    success = score > 0.5

    metadata = EpisodeMetadata(
        task_id=task.task_id,
        template_id=getattr(task, 'template_id', None),
        goal=task.goal,
        benchmark="terminalbench",
        site=getattr(task, 'site', None),
        repo=getattr(task, 'repo', None),
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
