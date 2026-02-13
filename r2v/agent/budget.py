"""
Compute budget management for R2V-Agent inference.

Tracks and enforces per-episode resource constraints:
- K: number of SLM candidates (quality vs latency)
- llm_calls_left: max LLM fallback calls (cost cap)
- self_correct_iters: max verifier-gated refinement iterations
- accept_thresh: minimum verifier score to accept action
- step_limit: max steps per episode
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceBudget:
    """Per-episode resource budget for R2V-Agent."""

    # Candidate generation
    K: int = 4                      # Number of SLM candidates per step
    max_new_tokens: int = 256       # Max tokens per generated action

    # LLM fallback budget
    max_llm_calls: int = 10         # Max LLM fallback calls per episode
    llm_calls_used: int = 0         # Counter

    # Self-correction
    max_self_correct_iters: int = 2
    accept_thresh: float = 0.7      # Min V_φ score to accept without correction

    # Episode limits
    step_limit: int = 30
    steps_used: int = 0

    # Cost tracking
    slm_token_count: int = 0
    llm_token_count: int = 0
    cost_slm_per_token: float = 0.0
    cost_llm_per_token: float = 0.00003  # ~GPT-4 pricing

    @property
    def llm_calls_left(self) -> int:
        return max(0, self.max_llm_calls - self.llm_calls_used)

    @property
    def steps_left(self) -> int:
        return max(0, self.step_limit - self.steps_used)

    @property
    def total_cost(self) -> float:
        return (
            self.slm_token_count * self.cost_slm_per_token
            + self.llm_token_count * self.cost_llm_per_token
        )

    @property
    def is_exhausted(self) -> bool:
        return self.steps_left <= 0

    def use_llm_call(self, tokens: int = 0):
        self.llm_calls_used += 1
        self.llm_token_count += tokens

    def use_slm_call(self, tokens: int = 0):
        self.slm_token_count += tokens

    def use_step(self):
        self.steps_used += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "K": self.K,
            "max_llm_calls": self.max_llm_calls,
            "llm_calls_used": self.llm_calls_used,
            "max_self_correct_iters": self.max_self_correct_iters,
            "accept_thresh": self.accept_thresh,
            "step_limit": self.step_limit,
            "steps_used": self.steps_used,
            "slm_token_count": self.slm_token_count,
            "llm_token_count": self.llm_token_count,
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "InferenceBudget":
        return cls(
            K=config.get("num_candidates", 4),
            max_llm_calls=config.get("max_llm_calls", 10),
            max_self_correct_iters=config.get("max_self_correct_iters", 2),
            accept_thresh=config.get("accept_threshold", 0.7),
            step_limit=config.get("step_limit", 30),
        )
