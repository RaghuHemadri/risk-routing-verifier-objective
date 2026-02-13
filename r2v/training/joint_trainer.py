"""
Joint Trainer: orchestrates the full R2V training pipeline.

Implements the overall objective:
  min_θ  L_BC + λ_pref * L_pref + λ_cons * L_cons

Training stages:
1. Behavior cloning on clean teacher trajectories
2. Verifier training (or LLM-judge setup)
3. Generate candidate actions & verifier scores → preference pairs
4. DPO preference distillation on perturbed contexts
5. Consistency regularization on paired tool outputs
6. Router training with CVaR constraint

Each stage can be run independently or as a full pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from r2v.utils.results import ExperimentResult, ResultsManager

logger = logging.getLogger(__name__)


class JointTrainer:
    """Orchestrates the full R2V training pipeline."""

    STAGES = [
        "collect_trajectories",
        "generate_perturbations",
        "label_steps",
        "train_bc",
        "train_verifier",
        "generate_candidates",
        "train_preference",
        "train_consistency",
        "train_router",
    ]

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: str = "experiments",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track progress
        self.completed_stages: list[str] = []
        self.stage_results: dict[str, Any] = {}
        self.results_manager = ResultsManager(
            str(self.output_dir / "results")
        )

        # Stage checkpoint file
        self.checkpoint_file = self.output_dir / "training_checkpoint.json"
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Resume from checkpoint if available."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
            self.completed_stages = data.get("completed_stages", [])
            self.stage_results = data.get("stage_results", {})
            logger.info(
                f"Resuming from checkpoint. Completed stages: {self.completed_stages}"
            )

    def _save_checkpoint(self):
        """Save training progress checkpoint."""
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "completed_stages": self.completed_stages,
                "stage_results": {
                    k: _serialize_results(v) for k, v in self.stage_results.items()
                },
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, default=str)

    def run_stage(self, stage_name: str, **kwargs) -> dict[str, Any]:
        """Run a single training stage."""
        if stage_name not in self.STAGES:
            raise ValueError(f"Unknown stage: {stage_name}. Available: {self.STAGES}")

        if stage_name in self.completed_stages:
            logger.info(f"Stage '{stage_name}' already completed, skipping.")
            return self.stage_results.get(stage_name, {})

        logger.info(f"Starting stage: {stage_name}")
        start_time = datetime.now()

        # Dispatch to stage handler
        handler = getattr(self, f"_stage_{stage_name}", None)
        if handler is None:
            raise NotImplementedError(f"Stage handler not implemented: {stage_name}")

        result = handler(**kwargs)

        # Record completion
        elapsed = (datetime.now() - start_time).total_seconds()
        result["elapsed_seconds"] = elapsed
        self.stage_results[stage_name] = result
        self.completed_stages.append(stage_name)
        self._save_checkpoint()

        logger.info(f"Stage '{stage_name}' completed in {elapsed:.1f}s")
        return result

    def run_full_pipeline(self, **kwargs) -> dict[str, Any]:
        """Run all training stages in sequence."""
        for stage in self.STAGES:
            self.run_stage(stage, **kwargs)

        # Save final results
        result = ExperimentResult(
            experiment_id=f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self.config,
            stage_results=self.stage_results,
        )
        self.results_manager.save_result(result)

        return self.stage_results

    # ============================================================
    # Stage handlers
    # ============================================================

    def _stage_collect_trajectories(self, **kwargs) -> dict:
        """Stage 1: Collect teacher trajectories."""
        logger.info("Collecting teacher trajectories...")
        # This stage is typically run via scripts/collect_trajectories.py
        # The handler here manages the orchestration
        return {
            "status": "requires_external_execution",
            "command": "python scripts/collect_trajectories.py",
            "description": "Run teacher model to collect successful trajectories",
        }

    def _stage_generate_perturbations(self, **kwargs) -> dict:
        """Stage 2: Generate perturbed trajectory variants."""
        logger.info("Generating perturbations...")
        return {
            "status": "requires_external_execution",
            "command": "python scripts/generate_perturbations.py",
            "description": "Apply perturbation operators to collected trajectories",
        }

    def _stage_label_steps(self, **kwargs) -> dict:
        """Stage 3: Label steps for verifier training."""
        logger.info("Labeling steps...")
        return {
            "status": "requires_external_execution",
            "command": "python scripts/label_steps.py",
            "description": "Generate step-level and final-outcome labels",
        }

    def _stage_train_bc(self, **kwargs) -> dict:
        """Stage 4: Behavior cloning on teacher trajectories."""
        from r2v.models.policy import PolicyModel
        from r2v.data.datasets import BCDataset
        from r2v.training.bc_trainer import BCTrainer

        policy = PolicyModel(self.config["policy"])
        train_dataset = BCDataset(
            trajectory_path=str(self.output_dir / "data" / "teacher_trajectories.jsonl"),
            tokenizer=policy.tokenizer,
            max_seq_len=self.config["policy"]["max_seq_len"],
        )

        trainer = BCTrainer(
            policy=policy,
            train_dataset=train_dataset,
            config=self.config["training"]["bc"],
            output_dir=str(self.output_dir / "checkpoints" / "bc"),
        )

        return trainer.train()

    def _stage_train_verifier(self, **kwargs) -> dict:
        """Stage 5: Train verifier (or confirm LLM-judge setup)."""
        verifier_config = self.config["verifier"]

        if verifier_config.get("mode") == "llm_judge":
            return {
                "status": "skipped",
                "reason": "Using LLM-as-judge verifier (no training needed)",
            }

        from r2v.models.verifier import TrainedVerifier
        from r2v.data.datasets import VerifierDataset
        from r2v.training.verifier_trainer import VerifierTrainer

        verifier = TrainedVerifier(verifier_config.get("trained", {}))
        train_dataset = VerifierDataset(
            trajectory_path=str(self.output_dir / "data" / "labeled_trajectories.jsonl"),
            tokenizer=verifier.tokenizer,
        )

        trainer = VerifierTrainer(
            verifier=verifier,
            train_dataset=train_dataset,
            config=self.config["training"]["verifier"],
            output_dir=str(self.output_dir / "checkpoints" / "verifier"),
        )

        return trainer.train()

    def _stage_generate_candidates(self, **kwargs) -> dict:
        """Stage 6: Generate K candidate actions and score with verifier."""
        return {
            "status": "requires_external_execution",
            "command": "python scripts/generate_candidates.py",
            "description": (
                "For each context in perturbed trajectories, sample K candidates "
                "from SLM and score with verifier to create preference pairs"
            ),
        }

    def _stage_train_preference(self, **kwargs) -> dict:
        """Stage 7: DPO preference distillation."""
        from r2v.models.policy import PolicyModel
        from r2v.data.datasets import PreferenceDataset
        from r2v.training.preference_trainer import PreferenceTrainer

        policy = PolicyModel(self.config["policy"])
        # Load BC-trained weights
        bc_path = self.output_dir / "checkpoints" / "bc" / "final"
        if bc_path.exists():
            policy.load(str(bc_path))

        train_dataset = PreferenceDataset(
            candidates_path=str(self.output_dir / "data" / "preference_pairs.jsonl"),
            tokenizer=policy.tokenizer,
        )

        trainer = PreferenceTrainer(
            policy=policy,
            train_dataset=train_dataset,
            config=self.config["training"]["preference"],
            output_dir=str(self.output_dir / "checkpoints" / "preference"),
        )

        return trainer.train()

    def _stage_train_consistency(self, **kwargs) -> dict:
        """Stage 8: Consistency regularization (run jointly or separately)."""
        # Typically integrated into the preference training loop
        return {
            "status": "integrated",
            "description": "Consistency loss computed jointly during preference training",
        }

    def _stage_train_router(self, **kwargs) -> dict:
        """Stage 9: Router training with CVaR constraint."""
        from r2v.models.router import Router
        from r2v.data.datasets import RouterDataset
        from r2v.training.router_trainer import RouterTrainer

        router = Router(self.config["router"])

        # Load pre-computed router features
        import jsonlines
        features_path = self.output_dir / "data" / "router_features.jsonl"
        if not features_path.exists():
            return {
                "status": "requires_external_execution",
                "command": "python scripts/generate_router_features.py",
            }

        from r2v.data.datasets import RouterExample
        examples = []
        with jsonlines.open(str(features_path)) as reader:
            for obj in reader:
                examples.append(RouterExample(**obj))

        dataset = RouterDataset(examples)

        # Split into train/eval
        from torch.utils.data import random_split
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_ds, eval_ds = random_split(dataset, [train_size, eval_size])

        trainer = RouterTrainer(
            router=router,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            config=self.config["training"]["router"],
            output_dir=str(self.output_dir / "checkpoints" / "router"),
        )

        return trainer.train()


def _serialize_results(obj: Any) -> Any:
    """Make results JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize_results(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_results(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)
