"""
Abstract base class for benchmark environments.

All benchmark wrappers implement this interface so that the collection
script (collect_trajectories.py) can work uniformly across WebArena,
SWE-bench, or any future benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BenchmarkTask:
    """A single task from a benchmark."""

    task_id: str
    goal: str  # Natural-language intent / problem statement
    benchmark: str  # "webarena" or "swebench"
    template_id: Optional[str] = None
    site: Optional[str] = None  # WebArena site category
    repo: Optional[str] = None  # SWE-bench repo
    difficulty: Optional[str] = None
    config_file: Optional[str] = None  # WebArena config JSON path
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvStepResult:
    """Result of a single environment step."""

    observation: str  # Text observation (accessibility tree, logs, etc.)
    reward: float  # Step reward (usually 0 until final step)
    done: bool  # Episode terminated
    success: Optional[bool] = None  # Final success (only meaningful when done=True)
    url: Optional[str] = None  # Current URL (WebArena)
    info: dict[str, Any] = field(default_factory=dict)


class BenchmarkEnv(ABC):
    """Abstract benchmark environment.

    Follows an OpenAI Gym-like interface: reset → step → ... → close.
    """

    @abstractmethod
    def load_tasks(self, cfg: Any) -> list[BenchmarkTask]:
        """Load tasks from the benchmark dataset/config.

        Args:
            cfg: OmegaConf config section for this benchmark.

        Returns:
            List of BenchmarkTask instances.
        """
        ...

    @abstractmethod
    def reset(self, task: BenchmarkTask, seed: int = 0) -> str:
        """Reset the environment for a new task.

        Args:
            task: The task to set up.
            seed: Random seed for reproducibility.

        Returns:
            Initial text observation.
        """
        ...

    @abstractmethod
    def step(self, action_text: str) -> EnvStepResult:
        """Execute an action in the environment.

        Args:
            action_text: The action string (WebArena grammar or git patch).

        Returns:
            EnvStepResult with observation, reward, done, etc.
        """
        ...

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the current trajectory for functional correctness.

        Returns:
            Score in [0, 1]. 1.0 = fully correct.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
