"""
Distractor perturbation: adds plausible but irrelevant content to observations.

Models real-world noise sources:
- Semantically similar but irrelevant search results
- Red herring error messages in logs
- Decoy UI elements with similar labels
- Plausible but wrong code suggestions (SWE-bench)
- Similar file paths / function names

These perturbations test the agent's ability to focus on task-relevant
information and ignore plausible distractors — a key challenge for smaller
models with limited context windows.
"""

from __future__ import annotations

import random
import re
from copy import deepcopy
from typing import Any

from r2v.data.perturbations.base import Perturbation
from r2v.data.trajectory import Observation, PerturbationType


# --- Distractor Templates ---

SEMANTIC_DISTRACTOR_TEMPLATES = {
    "search_results": [
        '  [{id}] link "{title}" (url: {url})\n    {description}',
    ],
    "search_titles": [
        "Similar Products You Might Like",
        "Related Items - Sponsored",
        "Customers Also Viewed",
        "Frequently Bought Together",
        "Popular in Your Area",
        "Trending Now",
    ],
    "search_descriptions": [
        "Free shipping on orders over $25. Limited time offer!",
        "Rated 4.5 stars by over 10,000 customers worldwide.",
        "Best value - Compare prices across 50+ retailers.",
        "New arrival - Just launched this season.",
        "Editor's choice - Featured in our latest collection.",
        "Sale ends today - Save up to 60% off retail price.",
    ],
}

RED_HERRING_ERRORS = [
    "Warning: Deprecated function call in module __init__.py line 42",
    "INFO: Cache miss for key 'session_data_v2' - rebuilding",
    "DEBUG: Connection pool size: 5/10 active connections",
    "Warning: Locale 'en_US.UTF-8' not found, falling back to 'C'",
    "INFO: Background task 'cleanup_temp' completed in 0.003s",
    "Warning: SSL certificate verification disabled for debug mode",
    "DEBUG: Request headers: {{'Accept': 'text/html', 'User-Agent': 'Mozilla/5.0'}}",
    "INFO: Database connection established (latency: 12ms)",
    "Warning: Memory usage at 78% - consider increasing heap size",
    "DEBUG: Rendering template 'base.html' with 3 context variables",
]

DECOY_ELEMENTS = [
    '[{id}] button "Similar Action" (disabled)',
    '[{id}] link "See Also: Related Feature"',
    '[{id}] button "Advanced Options" (collapsed)',
    '[{id}] text "Sponsored content below"',
    '[{id}] link "Help: Troubleshooting Guide"',
    '[{id}] button "Alternative Method"',
    '[{id}] link "Quick Links > {variant}"',
    '[{id}] text "Did you mean: {variant}?"',
]

SWE_BENCH_DISTRACTORS = {
    "similar_files": [
        "src/utils/helpers_v2.py",
        "src/core/handler_base.py",
        "tests/test_integration_old.py",
        "lib/compat/legacy_adapter.py",
        "src/models/base_model_mixin.py",
    ],
    "wrong_fixes": [
        "# Fix: Change the comparison operator from == to is",
        "# Fix: Add missing null check before accessing attribute",
        "# Fix: Replace deprecated API call with new version",
        "# Fix: Increase timeout to handle slow responses",
        "# Fix: Add try-except block around the file operation",
    ],
    "decoy_errors": [
        "TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'",
        "AttributeError: 'dict' object has no attribute 'items_'",
        "ImportError: cannot import name 'OldClass' from 'module'",
        "ValueError: invalid literal for int() with base 10: 'abc'",
        "KeyError: 'missing_config_key'",
    ],
}


class DistractorPerturbation(Perturbation):
    """Add plausible but irrelevant content to observations."""

    perturbation_type = PerturbationType.DISTRACTOR

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.semantic_distractor_prob = config.get("semantic_distractor_prob", 0.20)
        self.red_herring_prob = config.get("red_herring_prob", 0.15)
        self.decoy_element_prob = config.get("decoy_element_prob", 0.10)
        self.plausible_wrong_prob = config.get("plausible_wrong_prob", 0.10)

    def perturb_observation(
        self, obs: Observation, rng: random.Random
    ) -> tuple[Observation, dict[str, Any]]:
        new_obs = deepcopy(obs)
        meta: dict[str, Any] = {"perturbations_applied": [], "num_distractors_added": 0}

        text = new_obs.raw_text

        # 1. Semantic distractors (extra search results, similar items)
        if rng.random() < self.semantic_distractor_prob:
            text, n_added = self._add_semantic_distractors(text, rng)
            meta["perturbations_applied"].append("semantic_distractor")
            meta["num_distractors_added"] += n_added

        # 2. Red herring error/log messages
        if rng.random() < self.red_herring_prob:
            text, n_added = self._add_red_herrings(text, rng)
            meta["perturbations_applied"].append("red_herring")
            meta["num_distractors_added"] += n_added

        # 3. Decoy UI elements
        if rng.random() < self.decoy_element_prob:
            text, n_added = self._add_decoy_elements(text, rng)
            meta["perturbations_applied"].append("decoy_element")
            meta["num_distractors_added"] += n_added

        # 4. Plausible but wrong suggestions (SWE-bench style)
        if rng.random() < self.plausible_wrong_prob:
            text, n_added = self._add_plausible_wrong(text, rng)
            meta["perturbations_applied"].append("plausible_wrong")
            meta["num_distractors_added"] += n_added

        new_obs.raw_text = text
        return new_obs, meta

    def _add_semantic_distractors(
        self, text: str, rng: random.Random
    ) -> tuple[str, int]:
        """Add semantically similar but irrelevant search results or items."""
        lines = text.split("\n")
        num_distractors = rng.randint(2, 5)

        distractor_lines = []
        for i in range(num_distractors):
            # Generate a plausible-looking search result
            fake_id = rng.randint(100, 999)
            title = rng.choice(SEMANTIC_DISTRACTOR_TEMPLATES["search_titles"])
            desc = rng.choice(SEMANTIC_DISTRACTOR_TEMPLATES["search_descriptions"])
            url = f"https://example.com/item/{rng.randint(1000, 9999)}"

            template = rng.choice(SEMANTIC_DISTRACTOR_TEMPLATES["search_results"])
            result = template.format(
                id=fake_id, title=title, url=url, description=desc
            )
            distractor_lines.append(result)

        # Insert distractors among existing content
        insert_pos = rng.randint(
            len(lines) // 4, max(len(lines) // 4 + 1, len(lines) * 3 // 4)
        )
        for i, d in enumerate(distractor_lines):
            lines.insert(insert_pos + i, d)

        return "\n".join(lines), num_distractors

    def _add_red_herrings(
        self, text: str, rng: random.Random
    ) -> tuple[str, int]:
        """Add irrelevant but plausible error/log messages."""
        lines = text.split("\n")
        num_herrings = rng.randint(2, 4)

        selected = rng.sample(RED_HERRING_ERRORS, min(num_herrings, len(RED_HERRING_ERRORS)))

        for herring in selected:
            # Insert at random positions
            pos = rng.randint(0, len(lines))
            lines.insert(pos, herring)

        return "\n".join(lines), len(selected)

    def _add_decoy_elements(
        self, text: str, rng: random.Random
    ) -> tuple[str, int]:
        """Add decoy UI elements that look interactive but are irrelevant."""
        lines = text.split("\n")
        num_decoys = rng.randint(2, 4)

        variants = ["Settings", "Profile", "Dashboard", "Reports", "Analytics"]

        for _ in range(num_decoys):
            template = rng.choice(DECOY_ELEMENTS)
            fake_id = rng.randint(200, 999)
            variant = rng.choice(variants)
            decoy = template.format(id=fake_id, variant=variant)

            pos = rng.randint(1, max(1, len(lines) - 1))
            lines.insert(pos, decoy)

        return "\n".join(lines), num_decoys

    def _add_plausible_wrong(
        self, text: str, rng: random.Random
    ) -> tuple[str, int]:
        """Add plausible but incorrect suggestions/fixes.

        Particularly relevant for SWE-bench: suggest wrong fixes that
        look reasonable but would not resolve the actual issue.
        """
        lines = text.split("\n")
        num_wrong = rng.randint(1, 3)

        added = 0
        for _ in range(num_wrong):
            category = rng.choice(["wrong_fixes", "decoy_errors", "similar_files"])
            items = SWE_BENCH_DISTRACTORS[category]
            item = rng.choice(items)

            pos = rng.randint(0, len(lines))
            lines.insert(pos, item)
            added += 1

        return "\n".join(lines), added
