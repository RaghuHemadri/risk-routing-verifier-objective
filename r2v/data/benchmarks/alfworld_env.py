"""
ALFWorld benchmark environment for embodied text-based agent evaluation.

ALFWorld provides interactive TextWorld environments where an agent must
complete household tasks (find & pick up objects, clean, heat, cool, examine,
put objects in locations) from text observations.  Each task is a multi-step
sequential decision problem with clear success/failure criteria.

Runs entirely in Python — **no Docker or browser required**.

Paper:   https://arxiv.org/abs/2010.03768
Package: pip install alfworld[full]

Requires:
    pip install alfworld
    export ALFWORLD_DATA=<path-to-alfworld-data>  (auto-downloaded on first use)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from .base import BenchmarkEnv, BenchmarkTask, EnvStepResult

logger = logging.getLogger(__name__)

# Task types in ALFWorld
TASK_TYPES = [
    "pick_and_place",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "look_at_obj_in_light",
    "pick_two_obj",
]


def _import_alfworld():
    """Import alfworld with helpful error message."""
    try:
        import alfworld
        import alfworld.agents.environment as environment
        return alfworld, environment
    except ImportError:
        raise ImportError(
            "ALFWorld is not installed. Run:\n"
            "  pip install alfworld[full]\n"
            "  export ALFWORLD_DATA=$HOME/alfworld_data\n"
            "  alfworld-download"
        )


class ALFWorldEnv(BenchmarkEnv):
    """ALFWorld text-based household task environment.

    Actions are natural language commands:
        go to {recep}              – navigate to a receptacle
        take {obj} from {recep}    – pick up an object
        put {obj} in/on {recep}    – place object in receptacle
        open {recep}               – open a receptacle
        close {recep}              – close a receptacle
        toggle {obj/recep}         – toggle on/off
        clean {obj} with {recep}   – clean an object
        heat {obj} with {recep}    – heat an object
        cool {obj} with {recep}    – cool an object
        use {recep}                – use a device
        examine {obj/recep}        – look at something closely
        inventory                  – check what you're carrying
        look                       – look around
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        alf_cfg = cfg.get("alfworld", {})

        self.max_instances: Optional[int] = alf_cfg.get("max_instances", None)
        self.task_types: Optional[list[str]] = alf_cfg.get("task_types", None)
        self.split: str = alf_cfg.get("split", "eval_out_of_distribution")

        self._env = None
        self._current_task: Optional[BenchmarkTask] = None
        self._done: bool = False
        self._success: bool = False
        self._task_list: Optional[list[dict]] = None
        self._current_game_idx: int = 0

    def _ensure_env(self):
        """Initialize the ALFWorld environment on first use."""
        if self._env is not None:
            return

        alfworld, environment = _import_alfworld()

        # ALFWorld config
        config_path = os.environ.get(
            "ALFWORLD_CONFIG",
            os.path.join(
                os.path.dirname(alfworld.__file__),
                "agents",
                "config",
                "base_config.yaml",
            ),
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"ALFWorld config not found at {config_path}. "
                "Set ALFWORLD_CONFIG env var or run: alfworld-download"
            )

        import yaml
        with open(config_path, "r") as f:
            alf_config = yaml.safe_load(f)

        # Override split
        split_map = {
            "train": "train",
            "eval_in_distribution": "eval_in_distribution",
            "eval_out_of_distribution": "eval_out_of_distribution",
        }
        alf_split = split_map.get(self.split, "eval_out_of_distribution")

        # Set up environment
        alf_config["dataset"]["eval_ood_data_path"] = os.environ.get(
            "ALFWORLD_DATA",
            alf_config.get("dataset", {}).get("eval_ood_data_path", ""),
        )
        alf_config["logic"]["domain"] = os.path.join(
            os.path.dirname(alfworld.__file__), "logic", "alfred.pddl"
        )
        alf_config["env"]["type"] = "AlfredTWEnv"

        self._alf_config = alf_config
        self._env = getattr(environment, "AlfredTWEnv")(alf_config, train_eval=alf_split)
        self._env = self._env.init_env(batch_size=1)

    def load_tasks(self, cfg: Any) -> list[BenchmarkTask]:
        """Load ALFWorld game tasks.

        Since ALFWorld uses sequential game files, we enumerate them
        and create task descriptors.
        """
        self._ensure_env()

        # Get task metadata from the environment's game files
        game_files = self._env.gamefiles if hasattr(self._env, "gamefiles") else []

        tasks = []
        for idx, gf in enumerate(game_files):
            if self.max_instances is not None and idx >= self.max_instances:
                break

            # Parse task type from game file path
            task_type = "unknown"
            for tt in TASK_TYPES:
                if tt in str(gf):
                    task_type = tt
                    break

            if self.task_types and task_type not in self.task_types:
                continue

            task_id = f"alfworld_{idx:04d}"
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    goal="",  # Set during reset from env observation
                    benchmark="alfworld",
                    difficulty=task_type,
                    extra={"game_idx": idx, "game_file": str(gf)},
                )
            )

        logger.info(f"Loaded {len(tasks)} ALFWorld tasks")
        return tasks

    def reset(self, task: BenchmarkTask, seed: int = 0) -> str:
        self._ensure_env()
        self._current_task = task
        self._done = False
        self._success = False

        game_idx = task.extra.get("game_idx", 0)

        # Skip to the correct game
        obs, info = self._env.reset()
        while self._current_game_idx < game_idx:
            obs, info = self._env.reset()
            self._current_game_idx += 1

        # Extract initial observation
        initial_obs = obs[0] if isinstance(obs, (list, tuple)) else str(obs)

        # Update the task goal from the environment
        if hasattr(info, "get"):
            task.goal = info.get("objective", initial_obs.split("\n")[0])
        elif isinstance(info, (list, tuple)) and len(info) > 0:
            task.goal = initial_obs.split("\n")[0]

        obs_text = (
            f"Task: {task.goal}\n\n"
            f"Observation:\n{initial_obs}\n\n"
            "Available actions: go to, take, put, open, close, toggle, "
            "clean, heat, cool, use, examine, inventory, look"
        )
        return obs_text

    def step(self, action_text: str) -> EnvStepResult:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")

        if self._done:
            return EnvStepResult(
                observation="[Episode already finished]",
                reward=0.0,
                done=True,
                success=self._success,
            )

        action_text = action_text.strip()
        if not action_text:
            return EnvStepResult(
                observation="[No action provided. Try: look, inventory, "
                "go to <location>, take <object> from <location>, etc.]",
                reward=0.0,
                done=False,
            )

        try:
            obs, scores, dones, infos = self._env.step([action_text])
        except Exception as e:
            logger.warning(f"ALFWorld step error: {e}")
            return EnvStepResult(
                observation=f"[ERROR] {str(e)[:500]}",
                reward=0.0,
                done=False,
            )

        obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
        score = scores[0] if isinstance(scores, (list, tuple)) else float(scores)
        done = dones[0] if isinstance(dones, (list, tuple)) else bool(dones)

        self._done = done
        self._success = score > 0

        return EnvStepResult(
            observation=obs_text,
            reward=score,
            done=done,
            success=self._success if done else None,
        )

    def evaluate(self) -> float:
        return 1.0 if self._success else 0.0

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
