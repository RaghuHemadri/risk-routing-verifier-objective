#!/usr/bin/env python3
"""
Apply perturbations to collected trajectories.

Usage:
    python scripts/generate_perturbations.py \
        --config configs/webarena/noisy.yaml \
        --input data/trajectories/webarena_teacher/trajectories.jsonl \
        --output data/trajectories/webarena_noisy/trajectories.jsonl \
        --seeds 1 2 3

This script:
1. Loads clean trajectories
2. Applies configured perturbation pipeline
3. Saves perturbed trajectories (preserving originals)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r2v.data.perturbations import (
    DistractorInjection,
    PartialObservability,
    PerturbationPipeline,
    PromptInjection,
    ToolFlakiness,
)
from r2v.data.trajectory import PerturbationType, TrajectoryStore
from r2v.utils.config import load_config
from r2v.utils.logging import JSONLLogger, setup_logging


def build_pipeline(cfg) -> PerturbationPipeline:
    """Build perturbation pipeline from config."""
    perturbations = []
    pcfg = cfg.get("perturbations", {})

    if pcfg.get("tool_flakiness", {}).get("enabled", False):
        perturbations.append(ToolFlakiness(
            prob=pcfg.tool_flakiness.get("prob", 0.15),
        ))

    if pcfg.get("partial_observability", {}).get("enabled", False):
        perturbations.append(PartialObservability(
            prob=pcfg.partial_observability.get("prob", 0.20),
        ))

    if pcfg.get("prompt_injection", {}).get("enabled", False):
        perturbations.append(PromptInjection(
            prob=pcfg.prompt_injection.get("prob", 0.10),
        ))

    if pcfg.get("distractors", {}).get("enabled", False):
        perturbations.append(DistractorInjection(
            prob=pcfg.distractors.get("prob", 0.25),
        ))

    return PerturbationPipeline(perturbations)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate perturbed trajectories")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = setup_logging(level="INFO")

    # Load clean trajectories
    input_store = TrajectoryStore(args.input)
    episodes = input_store.load_all()
    logger.info(f"Loaded {len(episodes)} clean episodes")

    # Build pipeline
    pipeline = build_pipeline(cfg)
    logger.info(f"Perturbation pipeline: {[type(p).__name__ for p in pipeline.perturbations]}")

    # Output store
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_store = TrajectoryStore(output_path)
    jsonl_log = JSONLLogger(output_path.parent / "perturbation_log.jsonl")

    total_perturbed = 0
    for seed in args.seeds:
        logger.info(f"=== Applying perturbations with seed {seed} ===")

        for episode in episodes:
            # Apply perturbations to each step's observation
            perturbed_steps = []
            for step in episode.steps:
                obs_text = step.observation.text
                perturbed_text, applied = pipeline.apply(obs_text, seed=seed)

                new_obs = step.observation
                if applied:
                    from r2v.data.trajectory import Observation
                    import time
                    new_obs = Observation(
                        text=perturbed_text,
                        timestamp=time.time(),
                        perturbation_type=applied[0] if applied else None,
                    )

                from r2v.data.trajectory import Step
                new_step = Step(
                    observation=new_obs,
                    action=step.action,
                    reward=step.reward,
                    label=step.label,
                )
                perturbed_steps.append(new_step)

            from r2v.data.trajectory import Episode
            perturbed_episode = Episode(
                episode_id=f"{episode.episode_id}_perturbed_seed{seed}",
                metadata=episode.metadata,
                steps=perturbed_steps,
                success=episode.success,
                partial_score=episode.partial_score,
            )
            output_store.save_episode(perturbed_episode)
            total_perturbed += 1

    logger.info(f"Generated {total_perturbed} perturbed episodes")
    jsonl_log.log("summary", {
        "input_episodes": len(episodes),
        "seeds": args.seeds,
        "total_perturbed": total_perturbed,
    })


if __name__ == "__main__":
    main()
