"""
TextWorld benchmark environment wrapper.

Supports two modes:
1) Real TextWorld mode: load game files from a manifest and run with textworld.
2) Mock mode: lightweight built-in text quests for smoke tests and API checks.

Real mode requires:
    pip install textworld
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .base import BenchmarkEnv, BenchmarkTask, EnvStepResult

logger = logging.getLogger(__name__)


def _import_textworld():
    try:
        import textworld  # type: ignore[import-not-found]
        return textworld
    except ImportError as exc:
        raise ImportError(
            "TextWorld is not installed. Run: pip install textworld"
        ) from exc


class TextWorldEnv(BenchmarkEnv):
    """TextWorld environment with real and mock execution modes."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        tw_cfg = cfg.get("textworld", {})
        ds_cfg = tw_cfg.get("dataset", {})

        self.mock: bool = bool(tw_cfg.get("mock", False))
        self.max_instances: Optional[int] = ds_cfg.get("max_instances", None)
        self.dataset_path: str = ds_cfg.get("path", "")

        self._current_task: Optional[BenchmarkTask] = None
        self._current_success: bool = False

        # Real-mode runtime state
        self._tw = None
        self._env = None

        # Mock-mode runtime state
        self._mock_idx: int = 0
        self._mock_step: int = 0
        self._mock_tasks: list[dict[str, Any]] = [
            {
                "task_id": "tw_mock_0001",
                "goal": "Find and take the key from the table.",
                "initial_observation": "You are in a small room. A table is here. On the table you see a key.",
                "solution": ["look", "take key"],
            },
            {
                "task_id": "tw_mock_0002",
                "goal": "Open the chest after picking up the key.",
                "initial_observation": "You are in a study. You see a key on a desk and a locked chest.",
                "solution": ["take key", "unlock chest with key", "open chest"],
            },
            {
                "task_id": "tw_mock_0003",
                "goal": "Move north and pick up the apple.",
                "initial_observation": "You are in the kitchen. An exit leads north.",
                "solution": ["go north", "take apple"],
            },
        ]

    def _load_manifest(self) -> list[dict[str, Any]]:
        if not self.dataset_path:
            raise ValueError(
                "textworld.dataset.path is required in real mode. "
                "Point it to a JSON/JSONL manifest with game_file entries."
            )

        path = Path(self.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"TextWorld manifest not found: {path}")

        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            return data["tasks"]

        raise ValueError(
            "Unsupported TextWorld manifest format. Use JSON list, "
            "JSON object with tasks, or JSONL."
        )

    def load_tasks(self, cfg: Any) -> list[BenchmarkTask]:
        if self.mock:
            rows = []
            target = self.max_instances if self.max_instances is not None else len(self._mock_tasks)
            if target <= 0:
                target = len(self._mock_tasks)

            # In mock mode, expand built-in templates to the requested size.
            for idx in range(target):
                base = dict(self._mock_tasks[idx % len(self._mock_tasks)])
                base["task_id"] = f"tw_mock_{idx + 1:04d}"
                rows.append(base)
        else:
            rows = self._load_manifest()

        tasks: list[BenchmarkTask] = []
        for idx, row in enumerate(rows):
            if self.max_instances is not None and len(tasks) >= self.max_instances:
                break

            task_id = str(row.get("task_id", f"textworld_{idx:04d}"))
            goal = str(row.get("goal", row.get("objective", "Complete the TextWorld quest.")))
            game_file = row.get("game_file")

            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    goal=goal,
                    benchmark="textworld",
                    difficulty=row.get("difficulty", "standard"),
                    extra={
                        "game_file": game_file,
                        "solution": row.get("solution"),
                        "initial_observation": row.get("initial_observation"),
                    },
                )
            )

        logger.info("Loaded %d TextWorld tasks (mock=%s)", len(tasks), self.mock)
        return tasks

    def reset(self, task: BenchmarkTask, seed: int = 0) -> str:
        self._current_task = task
        self._current_success = False

        if self.mock:
            self._mock_step = 0
            obs = task.extra.get("initial_observation") or "You are in a room."
            return (
                f"Goal: {task.goal}\n\n"
                f"Observation:\n{obs}\n\n"
                "Available actions: look, inventory, examine, go, open, close, "
                "take, drop, put, insert, unlock, eat"
            )

        game_file = task.extra.get("game_file")
        if not game_file:
            raise ValueError(
                "TextWorld task missing game_file. "
                "Provide game_file in the manifest."
            )

        self._tw = _import_textworld()
        self._env = self._tw.start(game_file)
        state = self._env.reset()
        feedback = getattr(state, "feedback", str(state))
        return f"Goal: {task.goal}\n\nObservation:\n{feedback}"

    def _mock_step_env(self, action_text: str) -> EnvStepResult:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")

        solution = list(self._current_task.extra.get("solution") or [])
        action = action_text.strip().lower()

        if action in {"look", "inventory"}:
            return EnvStepResult(
                observation="You examine your surroundings carefully.",
                reward=0.0,
                done=False,
            )

        expected = solution[self._mock_step] if self._mock_step < len(solution) else None
        if expected and action == expected:
            self._mock_step += 1
            done = self._mock_step >= len(solution)
            self._current_success = done
            return EnvStepResult(
                observation="Action succeeded." if not done else "Quest completed successfully.",
                reward=1.0 if done else 0.2,
                done=done,
                success=self._current_success if done else None,
            )

        return EnvStepResult(
            observation="That action did not help. Try a different command.",
            reward=0.0,
            done=False,
        )

    def step(self, action_text: str) -> EnvStepResult:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")

        if self.mock:
            return self._mock_step_env(action_text)

        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action_text = action_text.strip()
        if not action_text:
            return EnvStepResult(
                observation="[No action provided]",
                reward=0.0,
                done=False,
            )

        try:
            state, score, done = self._env.step(action_text)
            feedback = getattr(state, "feedback", str(state))
            if done:
                self._current_success = bool(score and score > 0)
            return EnvStepResult(
                observation=feedback,
                reward=float(score),
                done=bool(done),
                success=self._current_success if done else None,
            )
        except Exception as exc:
            return EnvStepResult(
                observation=f"[ERROR] {str(exc)[:500]}",
                reward=0.0,
                done=False,
            )

    def evaluate(self) -> float:
        return 1.0 if self._current_success else 0.0

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
