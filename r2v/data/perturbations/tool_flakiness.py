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
    "TLSHandshakeError: remote endpoint closed the connection unexpectedly.",
    "DNSResolutionError: could not resolve host for upstream tool endpoint.",
    "ProtocolError: received malformed chunked transfer encoding.",
    "ResourceExhaustedError: service quota exceeded for this project.",
]

STALE_CACHE_MARKER = "[Note: Results may be cached from a previous query]"

EMPTY_RESPONSE_TEMPLATES = [
    "No results found.",
    "The search returned no matching results.",
    "0 results for your query.",
    "Unable to retrieve results at this time.",
]

HUMANEVAL_RUNTIME_ERRORS = [
    "Traceback (most recent call last):\n  File \"solution.py\", line {line}, in <module>\n    assert candidate({arg}) == {expected}\nAssertionError",
    "NameError: name '{symbol}' is not defined",
    "TypeError: {fn}() takes {nargs} positional arguments but {given} were given",
    "RuntimeError: Test harness interrupted due to flaky sandbox state",
    "ImportError: cannot import name '{symbol}' from 'solution'",
    "SyntaxError: invalid syntax at line {line}",
    "MemoryError: out of memory while executing hidden test case",
    "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x{hex_byte} in position {line}",
]

TEXTWORLD_TOOL_ERRORS = [
    "ParserError: command queue desynchronized; retry the previous action.",
    "WorldStateWarning: room graph failed to update after movement.",
    "InventoryServiceTimeout: inventory lookup timed out after {timeout}s.",
    "ActionExecutorError: object reference became stale between turns.",
    "Text buffer corruption: latest room description may be incomplete.",
    "EnvironmentWarning: last command may have executed partially.",
]

RTLREPAIR_TOOL_ERRORS = [
    "IcarusError: compile failed due to unresolved parameter width in candidate module.",
    "SimulationTimeout: oracle testbench exceeded {timeout}s waiting for expected waveform.",
    "LinterWarningEscalated: inferred latch detected; run aborted in strict mode.",
    "VCDWriteError: failed to flush dump.vcd before simulator exit.",
    "OracleMismatch: output trace length differs from golden trace by 1 cycle.",
]

UNIT_TOKENS = ["km", "m", "kg", "g", "ms", "s", "%", "USD", "EUR"]


class ToolFlakinessPerturbation(Perturbation):
    """Simulate unreliable, noisy tool responses."""

    perturbation_type = PerturbationType.TOOL_FLAKINESS

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.benchmark = str(config.get("benchmark", "")).lower()
        self.failure_prob = config.get("failure_prob", 0.15)
        self.timeout_prob = config.get("timeout_prob", 0.05)
        self.stale_cache_prob = config.get("stale_cache_prob", 0.10)
        self.result_shuffle_prob = config.get("result_shuffle_prob", 0.20)
        self.partial_response_prob = config.get("partial_response_prob", 0.10)
        self.format_corruption_prob = config.get("format_corruption_prob", 0.08)
        self.numeric_drift_prob = config.get("numeric_drift_prob", 0.08)
        self.pagination_cutoff_prob = config.get("pagination_cutoff_prob", 0.06)
        self.textworld_feedback_corruption_prob = config.get(
            "textworld_feedback_corruption_prob", 0.0
        )
        self.result_drop_fraction = config.get("result_drop_fraction", [0.1, 0.5])

    def perturb_observation(
        self, obs: Observation, rng: random.Random
    ) -> tuple[Observation, dict[str, Any]]:
        new_obs = deepcopy(obs)
        meta: dict[str, Any] = {"perturbations_applied": []}

        text = new_obs.raw_text

        # 1. Stochastic HTTP failure — replace entire response with error
        if rng.random() < self.failure_prob:
            error_msg = self._sample_failure_message(rng)
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

        # 6. Structured format corruption (JSON/markdown/table breakage)
        if rng.random() < self.format_corruption_prob:
            text = self._corrupt_format(text, rng)
            meta["perturbations_applied"].append("format_corruption")

        # 7. Numeric drift / unit mismatch
        if rng.random() < self.numeric_drift_prob:
            text = self._inject_numeric_drift(text, rng)
            meta["perturbations_applied"].append("numeric_drift")

        # 8. Pagination / result cutoff
        if rng.random() < self.pagination_cutoff_prob:
            text = self._cutoff_results(text, rng)
            meta["perturbations_applied"].append("pagination_cutoff")

        # 9. TextWorld feedback corruption (action acknowledgement ambiguity)
        if (
            self.benchmark == "textworld"
            and rng.random() < self.textworld_feedback_corruption_prob
        ):
            text = self._corrupt_textworld_feedback(text, rng)
            meta["perturbations_applied"].append("textworld_feedback_corruption")

        new_obs.raw_text = text
        return new_obs, meta

    def _sample_failure_message(self, rng: random.Random) -> str:
        if self.benchmark == "humaneval":
            msg = rng.choice(HUMANEVAL_RUNTIME_ERRORS)
            return msg.format(
                line=rng.randint(1, 80),
                arg=rng.randint(1, 10),
                expected=rng.randint(1, 10),
                symbol=rng.choice(["result", "n", "arr", "memo"]),
                fn=rng.choice(["candidate", "solve", "helper"]),
                nargs=rng.randint(1, 3),
                given=rng.randint(4, 7),
                hex_byte=f"{rng.randint(0, 255):02x}",
            )

        if self.benchmark == "textworld":
            msg = rng.choice(TEXTWORLD_TOOL_ERRORS)
            return msg.format(timeout=rng.randint(8, 45))

        if self.benchmark == "rtlrepair":
            msg = rng.choice(RTLREPAIR_TOOL_ERRORS)
            return msg.format(timeout=rng.randint(20, 120))

        msg = rng.choice(HTTP_ERROR_TEMPLATES)
        return msg.format(timeout=rng.randint(10, 60), delay=rng.randint(1, 30))

    def _corrupt_format(self, text: str, rng: random.Random) -> str:
        """Corrupt structured output formatting without deleting all content."""
        mode = rng.choice(["json_comma", "json_quote", "markdown_fence", "table_pipe"])
        out = text

        if mode == "json_comma":
            out = re.sub(r",\s*([}\]])", r"\1", out, count=2)
        elif mode == "json_quote":
            out = re.sub(r'"([A-Za-z_][\w-]*)"\s*:', r"\1:", out, count=2)
        elif mode == "markdown_fence":
            out = out.replace("```", "``", 1)
        elif mode == "table_pipe":
            lines = out.split("\n")
            for i, line in enumerate(lines):
                if "|" in line and rng.random() < 0.5:
                    lines[i] = line.replace("|", " ", 1)
            out = "\n".join(lines)

        if out == text:
            out += "\n[warning] output format may be malformed"
        return out

    def _inject_numeric_drift(self, text: str, rng: random.Random) -> str:
        """Perturb some numeric values and occasionally swap units."""
        lines = text.split("\n")
        edited = False

        for i, line in enumerate(lines):
            if rng.random() < 0.2:
                new_line, n = re.subn(
                    r"\b(\d+(?:\.\d+)?)\b",
                    lambda m: f"{float(m.group(1)) * rng.uniform(0.85, 1.15):.2f}",
                    line,
                    count=1,
                )
                if n > 0:
                    line = new_line
                    edited = True

            if rng.random() < 0.1:
                for unit in UNIT_TOKENS:
                    if unit in line:
                        swap = rng.choice([u for u in UNIT_TOKENS if u != unit])
                        line = line.replace(unit, swap, 1)
                        edited = True
                        break

            lines[i] = line

        if not edited:
            lines.append("[warning] one or more numeric fields may be stale")
        return "\n".join(lines)

    def _cutoff_results(self, text: str, rng: random.Random) -> str:
        """Simulate pagination failures by cutting list-like blocks."""
        lines = text.split("\n")
        if len(lines) < 6:
            return text

        cutoff = rng.randint(max(2, len(lines) // 4), max(3, len(lines) // 2))
        kept = lines[:cutoff]
        kept.append("[notice] additional results available on next page (failed to load)")
        return "\n".join(kept)

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

    def _corrupt_textworld_feedback(self, text: str, rng: random.Random) -> str:
        """Inject ambiguity into short text-game feedback lines.

        TextWorld trajectories are dominated by compact feedback strings like
        "Action succeeded." and "That action did not help..."; perturbing these
        directly makes uncertainty realistic for routing decisions.
        """
        variants = [
            "Action succeeded... maybe. State synchronization pending.",
            "Command accepted, but world update may be delayed.",
            "That action may have had no effect (engine uncertain).",
            "Feedback channel dropped one event; verify with look/inventory.",
            "Action result truncated; object state could not be confirmed.",
        ]

        if "Action succeeded." in text and rng.random() < 0.7:
            return text.replace("Action succeeded.", rng.choice(variants), 1)

        if "That action did not help." in text and rng.random() < 0.7:
            return text.replace(
                "That action did not help. Try a different command.",
                "That action may have had no visible effect. Try to verify state.",
                1,
            )

        return text + "\n[notice] observation channel reported inconsistent command feedback"
