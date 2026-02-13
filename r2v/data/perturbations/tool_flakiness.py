"""
Tool flakiness perturbation: simulates unreliable tool responses.

Models real-world issues:
- Stochastic HTTP failures / timeouts
- Stale cached results (returns previous query's results)
- Result ranking shuffling (search engine non-determinism)
- Partial response truncation (network interrupts)
- Result count variation (fewer results than expected)
- Latency-induced empty responses

This is the "tool noise" axis central to the paper's contribution.
"""

from __future__ import annotations

import random
import re
from copy import deepcopy
from typing import Any

from r2v.data.perturbations.base import Perturbation
from r2v.data.trajectory import Observation, PerturbationType


# --- Error message templates ---
HTTP_ERROR_TEMPLATES = [
    "Error 500: Internal Server Error. The server encountered an unexpected condition.",
    "Error 502: Bad Gateway. The server received an invalid response.",
    "Error 503: Service Unavailable. The server is temporarily overloaded.",
    "Error 504: Gateway Timeout. The server did not respond in time.",
    "ConnectionError: Connection refused by remote host.",
    "TimeoutError: Request timed out after {timeout}s.",
    "Error: Rate limit exceeded. Please retry after {delay} seconds.",
]

STALE_CACHE_MARKER = "[Note: Results may be cached from a previous query]"

EMPTY_RESPONSE_TEMPLATES = [
    "No results found.",
    "The search returned no matching results.",
    "0 results for your query.",
    "Unable to retrieve results at this time.",
]


class ToolFlakinessPerturbation(Perturbation):
    """Simulate unreliable, noisy tool responses."""

    perturbation_type = PerturbationType.TOOL_FLAKINESS

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.failure_prob = config.get("failure_prob", 0.15)
        self.timeout_prob = config.get("timeout_prob", 0.05)
        self.stale_cache_prob = config.get("stale_cache_prob", 0.10)
        self.result_shuffle_prob = config.get("result_shuffle_prob", 0.20)
        self.partial_response_prob = config.get("partial_response_prob", 0.10)
        self.result_drop_fraction = config.get("result_drop_fraction", [0.1, 0.5])

    def perturb_observation(
        self, obs: Observation, rng: random.Random
    ) -> tuple[Observation, dict[str, Any]]:
        new_obs = deepcopy(obs)
        meta: dict[str, Any] = {"perturbations_applied": []}

        text = new_obs.raw_text

        # 1. Stochastic HTTP failure — replace entire response with error
        if rng.random() < self.failure_prob:
            error_msg = rng.choice(HTTP_ERROR_TEMPLATES)
            error_msg = error_msg.format(timeout=rng.randint(10, 60), delay=rng.randint(1, 30))
            new_obs.raw_text = error_msg
            meta["perturbations_applied"].append("http_failure")
            meta["error_type"] = error_msg.split(":")[0] if ":" in error_msg else "Error"
            return new_obs, meta

        # 2. Timeout — replace with timeout message
        if rng.random() < self.timeout_prob:
            timeout_s = rng.randint(10, 120)
            new_obs.raw_text = f"TimeoutError: Request timed out after {timeout_s}s. No response received."
            meta["perturbations_applied"].append("timeout")
            return new_obs, meta

        # 3. Result ranking shuffle (for list-like content)
        if rng.random() < self.result_shuffle_prob:
            text = self._shuffle_results(text, rng)
            meta["perturbations_applied"].append("result_shuffle")

        # 4. Stale cache — prepend marker and optionally swap some content
        if rng.random() < self.stale_cache_prob:
            text = self._inject_stale_cache(text, rng)
            meta["perturbations_applied"].append("stale_cache")

        # 5. Partial response truncation
        if rng.random() < self.partial_response_prob:
            text = self._truncate_response(text, rng)
            meta["perturbations_applied"].append("partial_truncation")

        new_obs.raw_text = text
        return new_obs, meta

    def _shuffle_results(self, text: str, rng: random.Random) -> str:
        """Shuffle list-like result items in the observation.

        Detects numbered lists, bullet lists, or line-separated items
        and randomly reorders them.
        """
        lines = text.split("\n")

        # Detect list-like blocks (numbered items, bullet points, or tabular rows)
        list_patterns = [
            r"^\s*\d+[\.\)]\s",        # "1. item" or "1) item"
            r"^\s*[-*•]\s",             # "- item" or "* item"
            r"^\s*\[.*?\]\s",           # "[link] description"
        ]

        list_blocks: list[tuple[int, int]] = []
        current_start = None

        for i, line in enumerate(lines):
            is_list_item = any(re.match(p, line) for p in list_patterns)
            if is_list_item:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None and i - current_start >= 2:
                    list_blocks.append((current_start, i))
                current_start = None

        if current_start is not None and len(lines) - current_start >= 2:
            list_blocks.append((current_start, len(lines)))

        # Shuffle items within each detected block
        for start, end in list_blocks:
            items = lines[start:end]
            rng.shuffle(items)

            # Optionally drop some items
            drop_frac = rng.uniform(*self.result_drop_fraction)
            num_keep = max(1, int(len(items) * (1 - drop_frac)))
            items = items[:num_keep]

            lines[start:end] = items

        return "\n".join(lines)

    def _inject_stale_cache(self, text: str, rng: random.Random) -> str:
        """Simulate stale cache by prepending marker and degrading freshness."""
        lines = text.split("\n")

        # Add stale cache indicator
        lines.insert(0, STALE_CACHE_MARKER)

        # Optionally mutate some values to simulate outdated data
        for i in range(len(lines)):
            if rng.random() < 0.1:  # 10% chance per line
                # Replace numbers with slightly different ones (price changes, counts)
                lines[i] = re.sub(
                    r"\$(\d+\.?\d*)",
                    lambda m: f"${float(m.group(1)) * rng.uniform(0.8, 1.2):.2f}",
                    lines[i]
                )
                lines[i] = re.sub(
                    r"\b(\d{2,})\b",
                    lambda m: str(int(int(m.group(1)) * rng.uniform(0.7, 1.3))),
                    lines[i]
                )

        return "\n".join(lines)

    def _truncate_response(self, text: str, rng: random.Random) -> str:
        """Truncate response at a random point, simulating network interruption."""
        if len(text) < 50:
            return text

        # Cut between 30-80% of the way through
        cut_point = int(len(text) * rng.uniform(0.3, 0.8))
        truncated = text[:cut_point]

        # Add truncation indicator
        truncated += "\n... [Response truncated due to connection error]"
        return truncated
