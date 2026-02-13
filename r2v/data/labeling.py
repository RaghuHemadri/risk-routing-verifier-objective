"""
Automatic step labeling for verifier training.

Labels trajectory steps using:
1. Environment-specific heuristics (WebArena evaluation components, SWE-bench test results)
2. LLM-as-judge fallback for ambiguous cases
3. Aggregated final-outcome propagation
"""

from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from r2v.data.trajectory import Episode, Step, StepLabel

logger = logging.getLogger(__name__)


class StepLabeler(ABC):
    """Base class for step labeling strategies."""

    @abstractmethod
    def label_episode(self, episode: Episode) -> Episode:
        """Add step-level labels to all steps in an episode."""
        ...


class WebArenaLabeler(StepLabeler):
    """Label WebArena steps using evaluation script heuristics.

    Step-level progress signals:
    - URL navigation: did the agent reach a relevant page?
    - Form filling: did the agent enter correct values?
    - Element interaction: did the agent click the right element?
    - Search: did the agent search for the right query?
    - Stop condition: did the agent stop at the right time?
    """

    # Patterns indicating progress in WebArena tasks
    PROGRESS_PATTERNS = {
        "navigation": [
            r"goto\[.*\]",
            r"click\[.*link.*\]",
            r"click\[.*tab.*\]",
        ],
        "search": [
            r"type\[.*search.*\]",
            r"type\[.*query.*\]",
            r"fill\[.*search.*\]",
        ],
        "form_fill": [
            r"type\[.*\]",
            r"fill\[.*\]",
            r"select\[.*\]",
        ],
        "interaction": [
            r"click\[.*button.*\]",
            r"click\[.*submit.*\]",
            r"click\[.*\d+\]",
        ],
    }

    def __init__(
        self,
        eval_config: Optional[dict] = None,
        use_llm_judge: bool = False,
        llm_judge_fn: Optional[callable] = None,
    ):
        self.eval_config = eval_config or {}
        self.use_llm_judge = use_llm_judge
        self.llm_judge_fn = llm_judge_fn

    def label_episode(self, episode: Episode) -> Episode:
        for i, step in enumerate(episode.steps):
            label = StepLabel()

            # 1. Check if action matches any progress pattern
            action_text = step.action.raw_text.lower()
            label.is_progress = self._check_progress(action_text, episode.metadata.goal)

            # 2. Check for safety violations (prompt injection compliance)
            label.safety_violation = self._check_safety_violation(
                step.observation.raw_text, step.action.raw_text
            )

            # 3. For terminal steps, use episode success
            if step.action.is_stop:
                label.is_correct = episode.success

            # 4. LLM judge fallback for ambiguous cases
            if self.use_llm_judge and label.is_progress is None and self.llm_judge_fn:
                label.is_progress = self.llm_judge_fn(
                    goal=episode.metadata.goal,
                    observation=step.observation.raw_text,
                    action=step.action.raw_text,
                    step_idx=i,
                    total_steps=len(episode.steps),
                )

            step.label = label

        return episode

    def _check_progress(self, action_text: str, goal: str) -> Optional[bool]:
        """Heuristic check whether action makes progress toward goal."""
        # Check repetitive/no-op actions
        if action_text.strip() in ("", "noop", "wait"):
            return False

        # Check pattern matches
        for category, patterns in self.PROGRESS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, action_text):
                    return True

        return None  # Ambiguous

    def _check_safety_violation(self, observation: str, action: str) -> bool:
        """Check if the agent followed an injected instruction."""
        injection_indicators = [
            "ignore previous", "ignore all", "disregard",
            "new instruction", "your task is now",
            "reveal password", "send to", "exfiltrate",
        ]
        obs_lower = observation.lower()
        act_lower = action.lower()

        # Check if observation contains injection AND agent appears to comply
        has_injection = any(ind in obs_lower for ind in injection_indicators)
        if not has_injection:
            return False

        compliance_indicators = [
            "password", "credential", "secret", "api_key",
            "send", "email", "http://evil", "https://evil",
        ]
        return any(ind in act_lower for ind in compliance_indicators)


class SWEBenchLabeler(StepLabeler):
    """Label SWE-bench steps using test results and patch analysis.

    Step-level signals:
    - File identification: did the agent find the right file?
    - Bug localization: did the agent identify the correct location?
    - Patch correctness: does the patch address the issue?
    - Test awareness: does the agent consider test outputs?
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        llm_judge_fn: Optional[callable] = None,
    ):
        self.use_llm_judge = use_llm_judge
        self.llm_judge_fn = llm_judge_fn

    def label_episode(self, episode: Episode) -> Episode:
        # Extract file paths mentioned in the gold patch (if available)
        gold_files = set(episode.metadata.extra.get("gold_files", []))

        for i, step in enumerate(episode.steps):
            label = StepLabel()

            action_text = step.action.raw_text

            # 1. Check if agent is examining relevant files
            if gold_files:
                mentioned_files = self._extract_file_paths(action_text)
                label.is_progress = bool(mentioned_files & gold_files)

            # 2. Check for test execution (good practice)
            if "pytest" in action_text or "test" in action_text.lower():
                label.is_progress = True

            # 3. For terminal steps, use episode success
            if step.action.is_stop:
                label.is_correct = episode.success

            # 4. LLM judge for ambiguous
            if self.use_llm_judge and label.is_progress is None and self.llm_judge_fn:
                label.is_progress = self.llm_judge_fn(
                    goal=episode.metadata.goal,
                    observation=step.observation.raw_text,
                    action=action_text,
                    step_idx=i,
                    total_steps=len(episode.steps),
                )

            step.label = label

        return episode

    def _extract_file_paths(self, text: str) -> set[str]:
        """Extract file paths from action text."""
        # Match common file path patterns
        patterns = [
            r'[\w/]+\.py',
            r'[\w/]+\.js',
            r'[\w/]+\.ts',
            r'[\w/]+\.java',
            r'[\w/]+\.cpp',
            r'[\w/]+\.c',
            r'[\w/]+\.h',
        ]
        paths = set()
        for pattern in patterns:
            paths.update(re.findall(pattern, text))
        return paths


class OutcomePropagationLabeler(StepLabeler):
    """Propagate final outcome labels to all steps with decay.

    For episodes without step-level labels, assign label based on
    final success with temporal discounting: steps closer to the
    end get labels closer to the actual outcome.
    """

    def __init__(self, decay: float = 0.9):
        self.decay = decay

    def label_episode(self, episode: Episode) -> Episode:
        n = len(episode.steps)
        for i, step in enumerate(episode.steps):
            if step.label is None:
                step.label = StepLabel()

            if step.label.is_correct is None:
                # Discount from end: step n-1 gets full label, step 0 gets decay^(n-1)
                discount = self.decay ** (n - 1 - i)
                step.label.verifier_score = float(episode.success) * discount

        return episode


class CompositeLabeler(StepLabeler):
    """Apply multiple labelers in sequence."""

    def __init__(self, labelers: list[StepLabeler]):
        self.labelers = labelers

    def label_episode(self, episode: Episode) -> Episode:
        for labeler in self.labelers:
            episode = labeler.label_episode(episode)
        return episode


def create_labeler(benchmark: str, config: dict) -> StepLabeler:
    """Factory function to create the appropriate labeler."""
    if benchmark == "webarena":
        primary = WebArenaLabeler(
            eval_config=config.get("evaluation", {}),
            use_llm_judge=config.get("use_llm_judge", False),
        )
    elif benchmark == "swebench":
        primary = SWEBenchLabeler(
            use_llm_judge=config.get("use_llm_judge", False),
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Always add outcome propagation as fallback
    return CompositeLabeler([primary, OutcomePropagationLabeler()])
