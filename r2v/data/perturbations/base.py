"""
Base classes for the perturbation framework.

Design principles:
- Each perturbation is deterministic given a seed (for reproducibility)
- Perturbations operate on Observation objects (not raw strings)
- Composable via PerturbationPipeline
- Registry pattern for config-driven instantiation
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional

from r2v.data.trajectory import Observation, Step, Episode, PerturbationType


class Perturbation(ABC):
    """Base class for observation perturbations.

    Each perturbation modifies the observation content to simulate
    real-world noise in tool outputs. Perturbations are deterministic
    given a seed for reproducibility.
    """

    perturbation_type: PerturbationType = PerturbationType.NONE

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def perturb_observation(
        self, obs: Observation, rng: random.Random
    ) -> tuple[Observation, dict[str, Any]]:
        """Apply perturbation to an observation.

        Args:
            obs: The original observation.
            rng: Seeded random number generator for reproducibility.

        Returns:
            Tuple of (perturbed observation, metadata about what was changed).
        """
        ...

    def perturb_step(self, step: Step, rng: random.Random) -> Step:
        """Apply perturbation to a full step (modifies observation in-place)."""
        if not self.enabled:
            return step

        new_step = deepcopy(step)
        new_obs, meta = self.perturb_observation(new_step.observation, rng)
        new_step.observation = new_obs
        new_step.perturbation_type = self.perturbation_type
        new_step.perturbation_params = meta
        return new_step

    def perturb_episode(self, episode: Episode, seed: int) -> Episode:
        """Apply perturbation to all steps in an episode."""
        if not self.enabled:
            return episode

        rng = random.Random(seed)
        new_episode = deepcopy(episode)
        new_episode.perturbation_type = self.perturbation_type
        new_episode.perturbation_seed = seed

        new_steps = []
        for step in new_episode.steps:
            new_steps.append(self.perturb_step(step, rng))
        new_episode.steps = new_steps
        return new_episode


class PerturbationPipeline:
    """Compose multiple perturbations applied sequentially.

    For composite perturbation scenarios (e.g., tool flakiness + injection),
    each perturbation is applied in order with sub-seeds derived from the
    master seed for reproducibility.
    """

    def __init__(self, perturbations: list[Perturbation]):
        self.perturbations = [p for p in perturbations if p.enabled]

    def perturb_episode(self, episode: Episode, seed: int) -> Episode:
        """Apply all perturbations to an episode."""
        if not self.perturbations:
            return episode

        master_rng = random.Random(seed)
        result = deepcopy(episode)

        if len(self.perturbations) > 1:
            result.perturbation_type = PerturbationType.COMPOSITE
        elif len(self.perturbations) == 1:
            result.perturbation_type = self.perturbations[0].perturbation_type

        result.perturbation_seed = seed

        for perturbation in self.perturbations:
            sub_seed = master_rng.randint(0, 2**32 - 1)
            result = perturbation.perturb_episode(result, sub_seed)
            # Restore composite type
            result.perturbation_type = (
                PerturbationType.COMPOSITE
                if len(self.perturbations) > 1
                else self.perturbations[0].perturbation_type
            )
            result.perturbation_seed = seed

        return result

    def perturb_episode_factorized(
        self, episode: Episode, seed: int
    ) -> list[Episode]:
        """Apply each perturbation separately for factorized analysis.

        Returns one perturbed episode per perturbation type (useful for
        ablation studies and understanding which perturbation is hardest).
        """
        results = []
        master_rng = random.Random(seed)

        for perturbation in self.perturbations:
            sub_seed = master_rng.randint(0, 2**32 - 1)
            perturbed = perturbation.perturb_episode(episode, sub_seed)
            results.append(perturbed)

        return results


class PerturbationRegistry:
    """Registry for config-driven perturbation instantiation."""

    _registry: dict[str, type[Perturbation]] = {}

    @classmethod
    def register(cls, name: str, perturbation_cls: type[Perturbation]):
        cls._registry[name] = perturbation_cls

    @classmethod
    def create(cls, name: str, config: dict[str, Any]) -> Perturbation:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown perturbation: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](config)

    @classmethod
    def create_pipeline(cls, config: dict[str, Any]) -> PerturbationPipeline:
        """Create a pipeline from a config dict with perturbation sections."""
        perturbations = []
        for name, sub_config in config.items():
            if isinstance(sub_config, dict) and sub_config.get("enabled", False):
                try:
                    perturbations.append(cls.create(name, sub_config))
                except ValueError:
                    pass  # Skip unknown perturbation types
        return PerturbationPipeline(perturbations)


# Register all perturbation types (done in __init__.py after imports)
def register_all():
    """Register all built-in perturbation types."""
    from r2v.data.perturbations.tool_flakiness import ToolFlakinessPerturbation
    from r2v.data.perturbations.partial_observability import PartialObservabilityPerturbation
    from r2v.data.perturbations.prompt_injection import PromptInjectionPerturbation
    from r2v.data.perturbations.distractors import DistractorPerturbation

    PerturbationRegistry.register("tool_flakiness", ToolFlakinessPerturbation)
    PerturbationRegistry.register("partial_observability", PartialObservabilityPerturbation)
    PerturbationRegistry.register("prompt_injection", PromptInjectionPerturbation)
    PerturbationRegistry.register("distractors", DistractorPerturbation)


register_all()
