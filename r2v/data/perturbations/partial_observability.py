"""
Partial observability perturbation: degrades observation completeness.

Models real-world issues:
- DOM element hiding/removal (accessibility tree gaps)
- DOM reordering (sibling element shuffling)
- Attribute stripping (remove IDs, classes, ARIA labels)
- Log truncation (head/tail/random line dropping)
- Information masking (redact key-value pairs)
- Observation field reordering

These perturbations test whether the agent can operate robustly
with incomplete information — a critical capability for deployment.
"""

from __future__ import annotations

import random
import re
from copy import deepcopy
from typing import Any

from r2v.data.perturbations.base import Perturbation
from r2v.data.trajectory import Observation, PerturbationType


class PartialObservabilityPerturbation(Perturbation):
    """Degrade observation completeness through information removal."""

    perturbation_type = PerturbationType.PARTIAL_OBSERVABILITY

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.dom_hide_prob = config.get("dom_hide_prob", 0.15)
        self.dom_reorder_prob = config.get("dom_reorder_prob", 0.10)
        self.attribute_strip_prob = config.get("attribute_strip_prob", 0.10)
        self.log_truncation_prob = config.get("log_truncation_prob", 0.20)
        self.log_line_drop_prob = config.get("log_line_drop_prob", 0.10)
        self.info_mask_prob = config.get("info_mask_prob", 0.05)

    def perturb_observation(
        self, obs: Observation, rng: random.Random
    ) -> tuple[Observation, dict[str, Any]]:
        new_obs = deepcopy(obs)
        meta: dict[str, Any] = {"perturbations_applied": []}

        text = new_obs.raw_text

        # Apply perturbations in order of severity (least → most destructive)

        # 1. DOM reordering: shuffle sibling elements
        if rng.random() < self.dom_reorder_prob:
            text = self._reorder_elements(text, rng)
            meta["perturbations_applied"].append("dom_reorder")

        # 2. Attribute stripping: remove identifying attributes
        if rng.random() < self.attribute_strip_prob:
            text, num_stripped = self._strip_attributes(text, rng)
            meta["perturbations_applied"].append("attribute_strip")
            meta["num_attributes_stripped"] = num_stripped

        # 3. DOM element hiding: remove random elements
        if rng.random() < self.dom_hide_prob:
            text, num_hidden = self._hide_elements(text, rng)
            meta["perturbations_applied"].append("dom_hide")
            meta["num_elements_hidden"] = num_hidden

        # 4. Log line dropping: randomly remove lines
        if rng.random() < self.log_line_drop_prob:
            text, num_dropped = self._drop_lines(text, rng)
            meta["perturbations_applied"].append("line_drop")
            meta["num_lines_dropped"] = num_dropped

        # 5. Log truncation: cut from start or end
        if rng.random() < self.log_truncation_prob:
            text = self._truncate_log(text, rng)
            meta["perturbations_applied"].append("log_truncation")

        # 6. Information masking: redact key values
        if rng.random() < self.info_mask_prob:
            text, num_masked = self._mask_information(text, rng)
            meta["perturbations_applied"].append("info_mask")
            meta["num_values_masked"] = num_masked

        new_obs.raw_text = text
        return new_obs, meta

    def _reorder_elements(self, text: str, rng: random.Random) -> str:
        """Shuffle groups of lines that appear to be sibling elements.

        Detects indentation-based hierarchy in accessibility trees and
        shuffles elements at the same indentation level.
        """
        lines = text.split("\n")
        if len(lines) < 3:
            return text

        # Group consecutive lines by indentation level
        groups: list[list[int]] = []
        current_group: list[int] = []
        current_indent = -1

        for i, line in enumerate(lines):
            if not line.strip():
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_indent = -1
                continue

            indent = len(line) - len(line.lstrip())
            if indent == current_indent:
                current_group.append(i)
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [i]
                current_indent = indent

        if len(current_group) >= 2:
            groups.append(current_group)

        # Shuffle each group of siblings
        for group in groups:
            if len(group) < 2:
                continue
            group_lines = [lines[i] for i in group]
            rng.shuffle(group_lines)
            for i, idx in enumerate(group):
                lines[idx] = group_lines[i]

        return "\n".join(lines)

    def _strip_attributes(
        self, text: str, rng: random.Random
    ) -> tuple[str, int]:
        """Strip identifying attributes from accessibility tree elements.

        Removes id=, class=, aria-label=, data-* attributes from elements,
        making it harder for the agent to identify specific UI components.
        """
        count = 0

        # Pattern: [id="value"], id="value", class="value", etc.
        attribute_patterns = [
            r'\s+id="[^"]*"',
            r'\s+class="[^"]*"',
            r'\s+aria-label="[^"]*"',
            r'\s+data-[\w-]+="[^"]*"',
            r'\s+name="[^"]*"',
            r'\[id=\d+\]',
        ]

        lines = text.split("\n")
        for i in range(len(lines)):
            for pattern in attribute_patterns:
                if rng.random() < 0.5:  # Don't strip everything
                    new_line, n = re.subn(pattern, "", lines[i])
                    if n > 0:
                        lines[i] = new_line
                        count += n

        return "\n".join(lines), count

    def _hide_elements(
        self, text: str, rng: random.Random
    ) -> tuple[str, int]:
        """Remove random lines/elements from the observation.

        Simulates DOM elements that fail to load, are dynamically hidden,
        or are not captured by the accessibility tree.
        """
        lines = text.split("\n")
        if len(lines) < 5:
            return text, 0

        # Calculate how many lines to remove (5-25% of content)
        num_to_remove = max(1, int(len(lines) * rng.uniform(0.05, 0.25)))

        # Don't remove the first or last few lines (usually structural)
        removable = list(range(2, max(3, len(lines) - 2)))
        if len(removable) <= num_to_remove:
            return text, 0

        to_remove = set(rng.sample(removable, min(num_to_remove, len(removable))))
        filtered = [line for i, line in enumerate(lines) if i not in to_remove]

        return "\n".join(filtered), len(to_remove)

    def _drop_lines(
        self, text: str, rng: random.Random
    ) -> tuple[str, int]:
        """Randomly drop individual lines (simulates log gaps)."""
        lines = text.split("\n")
        if len(lines) < 3:
            return text, 0

        drop_rate = rng.uniform(0.05, 0.20)
        kept = []
        dropped = 0
        for line in lines:
            if rng.random() < drop_rate and line.strip():
                dropped += 1
            else:
                kept.append(line)

        return "\n".join(kept), dropped

    def _truncate_log(self, text: str, rng: random.Random) -> str:
        """Truncate observation from start or end.

        Simulates log buffer overflow or observation window limits.
        """
        lines = text.split("\n")
        if len(lines) < 5:
            return text

        # Calculate truncation amount (20-50% of lines)
        trunc_frac = rng.uniform(0.2, 0.5)
        num_to_keep = max(3, int(len(lines) * (1 - trunc_frac)))

        # Choose truncation direction
        direction = rng.choice(["head", "tail", "middle"])

        if direction == "head":
            # Keep tail, truncate head
            result_lines = ["[... earlier content truncated ...]"] + lines[-num_to_keep:]
        elif direction == "tail":
            # Keep head, truncate tail
            result_lines = lines[:num_to_keep] + ["[... remaining content truncated ...]"]
        else:
            # Keep head and tail, remove middle
            keep_each = num_to_keep // 2
            result_lines = (
                lines[:keep_each]
                + [f"[... {len(lines) - num_to_keep} lines omitted ...]"]
                + lines[-keep_each:]
            )

        return "\n".join(result_lines)

    def _mask_information(
        self, text: str, rng: random.Random
    ) -> tuple[str, int]:
        """Mask/redact specific values in the observation.

        Replaces key information with [REDACTED] or placeholder values,
        simulating privacy filters or incomplete data extraction.
        """
        count = 0
        lines = text.split("\n")

        # Patterns to potentially mask
        mask_targets = [
            (r'(\$\d+\.?\d*)', '[PRICE]'),         # Prices
            (r'(\b\d{3}-\d{3}-\d{4}\b)', '[PHONE]'),  # Phone numbers
            (r'([\w.]+@[\w.]+\.\w+)', '[EMAIL]'),   # Emails
            (r'(\b\d{5}(-\d{4})?\b)', '[ZIP]'),     # ZIP codes
            (r'(https?://\S+)', '[URL]'),            # URLs (aggressive)
        ]

        for i in range(len(lines)):
            for pattern, replacement in mask_targets:
                if rng.random() < 0.3:  # Don't mask everything
                    new_line, n = re.subn(pattern, replacement, lines[i])
                    if n > 0:
                        lines[i] = new_line
                        count += n

        return "\n".join(lines), count
