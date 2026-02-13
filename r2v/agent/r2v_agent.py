"""
R2V-Agent: Robust Router + Verifier-guided Agent inference loop.

Implements the full inference procedure from the proposal:
1. SLM proposes K candidates
2. Verifier scores candidates
3. Router decides fallback
4. Optional self-correction loop

This is the core agent that ties together all components at inference time.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import torch
import numpy as np

from r2v.agent.budget import InferenceBudget
from r2v.data.trajectory import (
    Action, ActionSource, Episode, EpisodeMetadata,
    Observation, Step, StepLabel
)

logger = logging.getLogger(__name__)


class R2VAgent:
    """Risk-calibrated routing agent with verifier-guided inference.

    Components:
    - policy (SLM): generates candidate actions
    - verifier: scores (context, action) pairs
    - router: decides SLM vs LLM fallback
    - teacher (LLM): fallback model for high-risk steps
    """

    def __init__(
        self,
        policy,
        verifier,
        router,
        teacher=None,
        config: dict[str, Any] = None,
    ):
        self.policy = policy
        self.verifier = verifier
        self.router = router
        self.teacher = teacher
        self.config = config or {}

        # Router threshold (can be tuned post-hoc)
        self.router_threshold = self.config.get("router_threshold", 0.5)

    def act(
        self,
        goal: str,
        observation: Observation,
        history: list[Step],
        budget: InferenceBudget,
    ) -> tuple[Action, dict[str, Any]]:
        """Select an action for the current step.

        This implements the pseudocode from the proposal.

        Args:
            goal: Task goal string
            observation: Current observation
            history: List of previous steps
            budget: Remaining compute budget

        Returns:
            Tuple of (selected action, step metadata)
        """
        step_start = time.time()
        step_meta: dict[str, Any] = {"source": None, "verifier_scores": []}

        # Build context x_t
        context = self._build_context(goal, observation, history)

        # 1. SLM proposes K candidates
        candidates = self.policy.generate_candidates(
            context,
            num_candidates=budget.K,
            max_new_tokens=budget.max_new_tokens,
        )
        budget.use_slm_call(
            tokens=sum(c["num_tokens"] for c in candidates) + len(context.split())
        )

        # 2. Verifier scores candidates
        candidate_texts = [c["text"] for c in candidates]
        scores = self.verifier.score_candidates(
            context, candidate_texts, goal
        )
        step_meta["verifier_scores"] = scores
        step_meta["candidate_actions"] = candidate_texts

        # Best SLM candidate
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        a_slm = Action(raw_text=candidate_texts[best_idx])
        best_score = scores[best_idx]

        # Compute risk score ρ(x_t)
        entropy = self.policy.compute_entropy(context)
        risk = self._compute_risk(best_score, entropy)
        step_meta["entropy"] = entropy
        step_meta["risk_score"] = risk
        step_meta["best_verifier_score"] = best_score

        # 3. Router decides fallback
        should_fallback = self._should_fallback(
            best_score, entropy, risk, budget, history
        )
        step_meta["router_decision"] = "llm" if should_fallback else "slm"

        if should_fallback and budget.llm_calls_left > 0 and self.teacher is not None:
            # LLM fallback
            teacher_action = self._query_teacher(context, goal)
            budget.use_llm_call(tokens=len(teacher_action.split()) + len(context.split()))
            selected_action = Action(raw_text=teacher_action)
            step_meta["source"] = ActionSource.LLM_FALLBACK.value
        else:
            selected_action = a_slm
            step_meta["source"] = ActionSource.SLM.value

        # 4. Optional self-correction loop
        current_score = self.verifier.score(
            context, selected_action.raw_text, goal
        )
        for correction_iter in range(budget.max_self_correct_iters):
            if current_score >= budget.accept_thresh:
                break

            # Refine action
            refined = self._refine_action(
                context, selected_action, goal, budget
            )
            refined_score = self.verifier.score(
                context, refined.raw_text, goal
            )

            if refined_score > current_score:
                selected_action = refined
                current_score = refined_score
                step_meta["source"] = ActionSource.SELF_CORRECTED.value
                step_meta["num_corrections"] = correction_iter + 1

        step_meta["final_verifier_score"] = current_score
        step_meta["wall_time"] = time.time() - step_start

        # Parse action type
        selected_action.action_type = self._parse_action_type(
            selected_action.raw_text
        )

        return selected_action, step_meta

    def run_episode(
        self,
        goal: str,
        env,
        budget: InferenceBudget,
        episode_id: str = "",
        metadata: Optional[EpisodeMetadata] = None,
    ) -> Episode:
        """Run a full episode in the environment.

        Args:
            goal: Task goal
            env: Environment with reset(), step(action) → (obs, reward, done, info)
            budget: Episode compute budget
            episode_id: Unique episode identifier
            metadata: Episode metadata

        Returns:
            Completed Episode with all steps and outcomes.
        """
        episode_start = time.time()
        obs = env.reset()
        history: list[Step] = []
        total_cost = 0.0
        num_llm_fallbacks = 0
        num_self_corrections = 0

        if metadata is None:
            metadata = EpisodeMetadata(task_id=episode_id, goal=goal)

        for step_idx in range(budget.step_limit):
            if budget.is_exhausted:
                break

            observation = Observation(
                raw_text=obs if isinstance(obs, str) else obs.get("text", str(obs)),
                url=obs.get("url") if isinstance(obs, dict) else None,
            )

            action, step_meta = self.act(goal, observation, history, budget)
            budget.use_step()

            # Execute action in environment
            try:
                next_obs, reward, done, info = env.step(action.raw_text)
            except Exception as e:
                logger.warning(f"Environment step failed: {e}")
                next_obs = f"Error: {str(e)}"
                reward = 0.0
                done = True
                info = {"error": str(e)}

            # Record step
            step = Step(
                step_idx=step_idx,
                observation=observation,
                action=action,
                action_source=ActionSource(step_meta["source"]),
                reward=reward,
                label=StepLabel(
                    verifier_score=step_meta.get("final_verifier_score"),
                    safety_violation=step_meta.get("safety_violation", False),
                ),
            )
            history.append(step)

            if step_meta["source"] == ActionSource.LLM_FALLBACK.value:
                num_llm_fallbacks += 1
            if step_meta.get("num_corrections", 0) > 0:
                num_self_corrections += step_meta["num_corrections"]

            if done or action.is_stop:
                break

            obs = next_obs

        # Build episode
        episode = Episode(
            episode_id=episode_id,
            metadata=metadata,
            steps=history,
            success=self._evaluate_success(env, info if 'info' in dir() else {}),
            total_cost=budget.total_cost,
            num_llm_fallbacks=num_llm_fallbacks,
            num_self_corrections=num_self_corrections,
            wall_time_seconds=time.time() - episode_start,
        )

        return episode

    # ============================================================
    # Private helpers
    # ============================================================

    def _build_context(
        self, goal: str, observation: Observation, history: list[Step]
    ) -> str:
        """Build agent context: x_t = (G, o_≤t, a_<t)."""
        parts = [f"Goal: {goal}\n"]
        for step in history:
            parts.append(f"Observation: {step.observation.raw_text[:1000]}")
            if step.observation.url:
                parts.append(f"URL: {step.observation.url}")
            parts.append(f"Action: {step.action.raw_text}")
        parts.append(f"Observation: {observation.raw_text[:2000]}")
        if observation.url:
            parts.append(f"URL: {observation.url}")
        return "\n".join(parts)

    def _compute_risk(self, verifier_score: float, entropy: float) -> float:
        """Compute risk score ρ(x_t) = f(1 - V, H)."""
        return 0.7 * (1.0 - verifier_score) + 0.3 * min(entropy / 5.0, 1.0)

    def _should_fallback(
        self,
        verifier_score: float,
        entropy: float,
        risk: float,
        budget: InferenceBudget,
        history: list[Step],
    ) -> bool:
        """Router decision: should we fall back to LLM?"""
        if self.router is None:
            return False

        # Build feature vector for router
        step_pct = len(history) / max(budget.step_limit, 1)
        features = torch.tensor(
            [[verifier_score, entropy, step_pct, risk]],
            dtype=torch.float32,
        )

        if hasattr(self.router, 'forward'):
            device = next(self.router.parameters()).device
            with torch.no_grad():
                prob = self.router(features.to(device)).item()
            return prob > self.router_threshold
        else:
            # Simple threshold-based fallback
            return risk > self.router_threshold

    def _query_teacher(self, context: str, goal: str) -> str:
        """Query the LLM teacher for a fallback action."""
        if self.teacher is None:
            return ""

        if hasattr(self.teacher, 'generate_candidates'):
            candidates = self.teacher.generate_candidates(
                context, num_candidates=1
            )
            return candidates[0]["text"] if candidates else ""
        elif hasattr(self.teacher, 'generate'):
            return self.teacher.generate(context)
        else:
            return ""

    def _refine_action(
        self, context: str, action: Action, goal: str, budget: InferenceBudget
    ) -> Action:
        """Refine an action using SLM + verifier feedback."""
        # Generate new candidates conditioned on the rejected action
        refine_context = (
            f"{context}\n"
            f"Previous attempt: {action.raw_text}\n"
            f"The previous action may be incorrect. Generate a better action.\n"
            f"Action:"
        )

        candidates = self.policy.generate_candidates(
            refine_context, num_candidates=max(2, budget.K // 2)
        )
        budget.use_slm_call(tokens=sum(c["num_tokens"] for c in candidates))

        if not candidates:
            return action

        scores = self.verifier.score_candidates(
            context, [c["text"] for c in candidates], goal
        )
        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        return Action(raw_text=candidates[best_idx]["text"])

    def _parse_action_type(self, action_text: str) -> str:
        """Parse the action type from action text."""
        text = action_text.strip().lower()
        if text.startswith("click"):
            return "click"
        elif text.startswith("type") or text.startswith("fill"):
            return "type"
        elif text.startswith("scroll"):
            return "scroll"
        elif text.startswith("goto"):
            return "navigate"
        elif text.startswith("stop") or text.startswith("finish"):
            return "stop"
        elif text.startswith("submit"):
            return "submit"
        else:
            return "other"

    def _evaluate_success(self, env, info: dict) -> bool:
        """Evaluate episode success using environment evaluator."""
        if hasattr(env, "evaluate"):
            return env.evaluate()
        return info.get("success", False)


# ============================================================
# Baseline agents for comparison
# ============================================================

class LLMOnlyAgent:
    """Baseline: Always use the LLM teacher (no routing)."""

    def __init__(self, teacher, config: dict = None):
        self.teacher = teacher
        self.config = config or {}

    def act(self, goal, observation, history, budget):
        context = R2VAgent._build_context(None, goal, observation, history)
        action_text = self.teacher.generate_candidates(
            context, num_candidates=1
        )[0]["text"]
        budget.use_llm_call(tokens=len(action_text.split()))
        return Action(raw_text=action_text), {"source": "teacher"}


class SLMOnlyAgent:
    """Baseline: Always use SLM (no routing, no verifier)."""

    def __init__(self, policy, config: dict = None):
        self.policy = policy
        self.config = config or {}

    def act(self, goal, observation, history, budget):
        context = R2VAgent._build_context(None, goal, observation, history)
        candidates = self.policy.generate_candidates(
            context, num_candidates=1
        )
        action_text = candidates[0]["text"] if candidates else ""
        budget.use_slm_call(tokens=len(action_text.split()))
        return Action(raw_text=action_text), {"source": "slm"}


class EntropyRouterAgent:
    """Baseline: Route to LLM when SLM entropy exceeds threshold."""

    def __init__(self, policy, teacher, threshold: float = 2.0, config: dict = None):
        self.policy = policy
        self.teacher = teacher
        self.threshold = threshold
        self.config = config or {}

    def act(self, goal, observation, history, budget):
        context = R2VAgent._build_context(None, goal, observation, history)
        entropy = self.policy.compute_entropy(context)

        if entropy > self.threshold and budget.llm_calls_left > 0:
            action_text = self.teacher.generate_candidates(
                context, num_candidates=1
            )[0]["text"]
            budget.use_llm_call()
            source = "llm_fallback"
        else:
            candidates = self.policy.generate_candidates(context, num_candidates=1)
            action_text = candidates[0]["text"] if candidates else ""
            budget.use_slm_call()
            source = "slm"

        return Action(raw_text=action_text), {"source": source, "entropy": entropy}
