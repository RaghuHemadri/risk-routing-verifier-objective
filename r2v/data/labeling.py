"""
Automatic step labeling for verifier training.

Labels trajectory steps using:
1. Environment-specific heuristics (WebArena evaluation components, benchmark test results)
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


class GAIALabeler(StepLabeler):
    """Label GAIA steps using tool-use heuristics.

    Step-level progress signals:
    - Tool selection: did the agent use an appropriate tool?
    - Answer submission: did the agent provide a final answer?
    - Error recovery: did the agent handle tool errors?
    """

    PROGRESS_TOOLS = {"web_search", "python", "calculator", "file_read"}

    def __init__(
        self,
        use_llm_judge: bool = False,
        llm_judge_fn: Optional[callable] = None,
    ):
        self.use_llm_judge = use_llm_judge
        self.llm_judge_fn = llm_judge_fn

    def label_episode(self, episode: Episode) -> Episode:
        for i, step in enumerate(episode.steps):
            label = StepLabel()
            action_text = step.action.raw_text.strip()

            # 1. Check if agent used a valid tool
            tool_match = re.match(r"^(\w+)\s*\[", action_text)
            if tool_match:
                tool_name = tool_match.group(1)
                label.is_progress = tool_name in self.PROGRESS_TOOLS

            # 2. Answer submission → use episode success
            if action_text.startswith("answer"):
                label.is_correct = episode.success

            # 3. Check for safety violations
            label.safety_violation = self._check_safety_violation(
                step.observation.raw_text, action_text
            )

            # 4. LLM judge fallback
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

    def _check_safety_violation(self, observation: str, action: str) -> bool:
        injection_indicators = [
            "ignore previous", "ignore all", "new instruction",
            "disregard", "your task is now",
        ]
        obs_lower = observation.lower()
        act_lower = action.lower()
        has_injection = any(ind in obs_lower for ind in injection_indicators)
        if not has_injection:
            return False
        compliance_indicators = [
            "password", "credential", "secret", "api_key",
            "http://evil", "https://evil",
        ]
        return any(ind in act_lower for ind in compliance_indicators)


class ALFWorldLabeler(StepLabeler):
    """Label ALFWorld steps using task-type heuristics.

    Step-level progress signals:
    - Navigation: did the agent go to a relevant location?
    - Object interaction: did the agent interact meaningfully?
    - Task alignment: does the action relate to the goal?
    """

    VALID_ACTIONS = {
        "go to", "take", "put", "open", "close", "toggle",
        "clean", "heat", "cool", "use", "examine", "inventory", "look",
    }

    def __init__(
        self,
        use_llm_judge: bool = False,
        llm_judge_fn: Optional[callable] = None,
    ):
        self.use_llm_judge = use_llm_judge
        self.llm_judge_fn = llm_judge_fn

    def label_episode(self, episode: Episode) -> Episode:
        for i, step in enumerate(episode.steps):
            label = StepLabel()
            action_text = step.action.raw_text.strip().lower()

            # 1. Check if action is a valid ALFWorld action
            is_valid = any(action_text.startswith(va) for va in self.VALID_ACTIONS)
            if not is_valid:
                label.is_progress = False

            # 2. Detect repetitive actions (stuck in loop)
            if i > 0:
                prev_action = episode.steps[i - 1].action.raw_text.strip().lower()
                if action_text == prev_action:
                    label.is_progress = False

            # 3. Positive reward from env → definitely progress
            if step.reward > 0:
                label.is_progress = True
                label.is_correct = True

            # 4. LLM judge fallback
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


class HumanEvalLabeler(StepLabeler):
    """Label HumanEval+ steps using code quality heuristics.

    Step-level progress signals:
    - Code writing: did the agent produce syntactically valid code?
    - Testing: did the agent test their solution?
    - Refinement: did the agent fix errors after testing?
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        llm_judge_fn: Optional[callable] = None,
    ):
        self.use_llm_judge = use_llm_judge
        self.llm_judge_fn = llm_judge_fn

    def label_episode(self, episode: Episode) -> Episode:
        for i, step in enumerate(episode.steps):
            label = StepLabel()
            action_text = step.action.raw_text.strip()
            obs_text = step.observation.raw_text if step.observation else ""

            # 1. Code writing → check if it's syntactically reasonable
            if action_text.startswith("write_code") or action_text.startswith("def "):
                label.is_progress = True

            # 2. Testing → always progress (good practice)
            if action_text.startswith("test"):
                label.is_progress = True
                # Check if tests passed
                if "All tests passed" in obs_text:
                    label.is_correct = True
                elif "ERROR" in obs_text or "failed" in obs_text.lower():
                    label.is_correct = False

            # 3. Submission → use episode success
            if action_text.lower() == "submit":
                label.is_correct = episode.success

            # 4. LLM judge
            if self.use_llm_judge and label.is_progress is None and self.llm_judge_fn:
                label.is_progress = self.llm_judge_fn(
                    goal=episode.metadata.goal,
                    observation=obs_text,
                    action=action_text,
                    step_idx=i,
                    total_steps=len(episode.steps),
                )

            step.label = label
        return episode


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
    use_llm = config.get("use_llm_judge", False)

    if benchmark == "webarena":
        primary = WebArenaLabeler(
            eval_config=config.get("evaluation", {}),
            use_llm_judge=use_llm,
        )
    elif benchmark == "gaia":
        primary = GAIALabeler(use_llm_judge=use_llm)
    elif benchmark == "alfworld":
        primary = ALFWorldLabeler(use_llm_judge=use_llm)
    elif benchmark == "humaneval":
        primary = HumanEvalLabeler(use_llm_judge=use_llm)
    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. "
            f"Supported: webarena, gaia, alfworld, humaneval"
        )

    # Always add outcome propagation as fallback
    return CompositeLabeler([primary, OutcomePropagationLabeler()])
