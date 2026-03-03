"""
SWE-bench benchmark environment wrapper.

Wraps the SWE-bench dataset and Docker-based evaluation harness to provide
a unified interface for teacher trajectory collection.

For SWE-bench, the "environment" is a coding task:
  - The agent receives (problem_statement + codebase context) as observations.
  - The agent produces a patch (diff) as its action.
  - Evaluation runs the test suite inside Docker to check correctness.

Unlike WebArena's multi-step browser interaction, SWE-bench is typically
a single-turn generation task: given an issue, produce a patch. We model
it as a multi-step process to unify with WebArena:
  Step 0: Initial observation (problem statement + retrieval context)
  Step 1..N-1: Agent can request more context or refine its patch
  Step N: Agent submits final patch (stop action)

Requires:
    - pip install -e ".[swebench]"
    - Docker running locally
    - HuggingFace datasets access

See SETUP_INSTRUCTIONS.md for full setup guide.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .base import BenchmarkEnv, BenchmarkTask, EnvStepResult

logger = logging.getLogger(__name__)


class SWEBenchEnv(BenchmarkEnv):
    """SWE-bench benchmark environment.

    Loads task instances from the HuggingFace `princeton-nlp/SWE-bench`
    dataset. Each instance contains:
      - instance_id: unique identifier
      - repo: GitHub repo (e.g., "django/django")
      - base_commit: commit SHA to checkout
      - problem_statement: the GitHub issue text
      - patch: gold patch (for reference only)
      - test_patch: test cases that verify the fix
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        sb_cfg = cfg.get("swebench", {})
        ds_cfg = sb_cfg.get("dataset", {})

        self.dataset_name = ds_cfg.get("name", "princeton-nlp/SWE-bench")
        self.split = ds_cfg.get("split", "test")
        self.train_split = ds_cfg.get("train_split", "train")
        self.max_instances = ds_cfg.get("max_instances", None)

        docker_cfg = sb_cfg.get("docker", {})
        self.docker_timeout = docker_cfg.get("timeout", 600)
        self.docker_memory = docker_cfg.get("memory_limit", "8g")

        self._current_task: Optional[BenchmarkTask] = None
        self._current_instance: Optional[dict] = None
        self._submitted_patch: Optional[str] = None
        self._dataset_cache: Optional[list[dict]] = None

    def _load_dataset(self) -> list[dict]:
        """Load the SWE-bench HuggingFace dataset (cached)."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. Install with:\n"
                "  pip install datasets"
            )

        logger.info(f"Loading SWE-bench dataset: {self.dataset_name} (split={self.split})")
        dataset = load_dataset(self.dataset_name, split=self.split)
        instances = [dict(inst) for inst in dataset]

        if self.max_instances is not None:
            instances = instances[: self.max_instances]

        self._dataset_cache = instances
        logger.info(f"Loaded {len(instances)} SWE-bench instances")
        return instances

    def load_tasks(self, cfg: Any) -> list[BenchmarkTask]:
        """Load SWE-bench task instances as BenchmarkTask objects."""
        instances = self._load_dataset()
        tasks = []
        for inst in instances:
            tasks.append(BenchmarkTask(
                task_id=inst["instance_id"],
                goal=inst.get("problem_statement", ""),
                benchmark="swebench",
                repo=inst.get("repo", ""),
                difficulty=inst.get("difficulty", None),
                extra={
                    "base_commit": inst.get("base_commit", ""),
                    "hints_text": inst.get("hints_text", ""),
                },
            ))
        return tasks

    def reset(self, task: BenchmarkTask, seed: int = 0) -> str:
        """Reset for a new SWE-bench instance.

        Returns the problem statement + any available context as the
        initial observation.
        """
        self._current_task = task
        self._submitted_patch = None

        # Find the full instance data
        instances = self._load_dataset()
        instance = None
        for inst in instances:
            if inst["instance_id"] == task.task_id:
                instance = inst
                break

        if instance is None:
            raise ValueError(f"Instance {task.task_id} not found in dataset.")

        self._current_instance = instance

        # Build the initial observation
        obs_parts = [
            f"=== SWE-bench Task: {task.task_id} ===",
            f"Repository: {instance.get('repo', 'N/A')}",
            f"Base Commit: {instance.get('base_commit', 'N/A')}",
            "",
            "=== Problem Statement ===",
            instance.get("problem_statement", "No problem statement available."),
        ]

        hints = instance.get("hints_text", "")
        if hints:
            obs_parts.extend(["", "=== Hints ===", hints])

        return "\n".join(obs_parts)

    def step(self, action_text: str) -> EnvStepResult:
        """Process an agent action.

        For SWE-bench, actions are either:
        1. A patch (diff) to submit as the solution
        2. A "stop" command indicating the patch is finalized

        The agent's response/patch is accumulated. When the agent emits
        a stop action or submits a patch, the episode ends.
        """
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")

        action_text = action_text.strip()

        # Check for stop/submit action
        is_done = action_text.lower().startswith("stop") or action_text.lower().startswith("submit")

        # Extract patch content
        patch = self._extract_patch(action_text)
        if patch:
            self._submitted_patch = patch

        if is_done:
            return EnvStepResult(
                observation="[SUBMITTED] Patch submitted for evaluation.",
                reward=0.0,
                done=True,
            )

        # If we found a patch, also treat the episode as done
        # (SWE-bench is effectively single-turn)
        if patch:
            return EnvStepResult(
                observation="[SUBMITTED] Patch submitted for evaluation.",
                reward=0.0,
                done=True,
            )

        return EnvStepResult(
            observation="No patch detected. Please provide a unified diff patch.",
            reward=0.0,
            done=False,
        )

    def evaluate(self, use_docker: bool = False) -> float:
        """Evaluate the submitted patch.

        Args:
            use_docker: If True, attempt full Docker harness evaluation
                (slow but accurate). If False, use heuristic evaluation
                only (fast, approximate). Default False for collection
                speed; use True or batch evaluation post-collection for
                ground-truth labels.

        Returns:
            1.0 if resolved, 0.0 if not, or a heuristic score [0,1].
        """
        if self._current_instance is None or self._submitted_patch is None:
            logger.warning("No patch submitted for evaluation.")
            return 0.0

        instance_id = self._current_instance["instance_id"]

        if use_docker:
            try:
                score = self._evaluate_with_harness(instance_id)
                if score is not None:
                    return score
            except Exception as e:
                logger.debug(f"Harness evaluation unavailable: {e}")

        # Heuristic evaluation (fast, always available)
        return self._evaluate_heuristic(instance_id)

    def _evaluate_with_harness(self, instance_id: str) -> float | None:
        """Run the full SWE-bench Docker harness. Returns None if unavailable.

        Uses swebench.harness.run_evaluation which:
        1. Builds a Docker image for the instance's repo+commit
        2. Applies the predicted patch inside the container
        3. Runs the test suite to check if the issue is resolved
        4. Writes per-instance reports to logs/run_evaluation/

        Note: This is SLOW (minutes per instance). For bulk evaluation,
        prefer the batch evaluation script (scripts/evaluate.py) which
        processes all predictions at once with parallelism.
        """
        try:
            from swebench.harness.run_evaluation import main as run_evaluation
            from swebench.harness.constants import (
                KEY_INSTANCE_ID,
                KEY_MODEL,
                KEY_PREDICTION,
                RUN_EVALUATION_LOG_DIR,
                LOG_REPORT,
            )
        except ImportError:
            return None

        # Check Docker availability before attempting evaluation
        try:
            import docker
            docker.from_env().ping()
        except Exception:
            logger.debug("Docker not available, skipping harness evaluation")
            return None

        model_name = "r2v-teacher"
        run_id = f"r2v_{instance_id}"

        prediction = {
            KEY_INSTANCE_ID: instance_id,
            KEY_PREDICTION: self._submitted_patch,
            KEY_MODEL: model_name,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="r2v_pred_"
        ) as f:
            json.dump([prediction], f)
            pred_file = f.name

        try:
            # run_evaluation returns a Path to the summary report file
            summary_report_path = run_evaluation(
                dataset_name=self.dataset_name,
                split=self.split,
                instance_ids=[instance_id],
                predictions_path=pred_file,
                max_workers=1,
                force_rebuild=False,
                cache_level="env",
                clean=True,
                open_file_limit=4096,
                run_id=run_id,
                timeout=self.docker_timeout,
                namespace="swebench",
                rewrite_reports=False,
                modal=False,
            )

            # Method 1: Check the summary report returned by main()
            if summary_report_path and Path(summary_report_path).exists():
                with open(summary_report_path) as rf:
                    report = json.load(rf)
                resolved_ids = report.get("resolved_ids", [])
                if instance_id in resolved_ids:
                    return 1.0
                return 0.0

            # Method 2: Check per-instance report file directly
            per_instance_report = (
                Path(RUN_EVALUATION_LOG_DIR)
                / run_id
                / model_name.replace("/", "__")
                / instance_id
                / LOG_REPORT
            )
            if per_instance_report.exists():
                content = per_instance_report.read_text().strip()
                if content:
                    instance_report = json.loads(content)
                    if instance_report.get(instance_id, {}).get("resolved", False):
                        return 1.0
            return 0.0
        except Exception as e:
            logger.error(f"SWE-bench evaluation failed for {instance_id}: {e}")
            return 0.0
        finally:
            try:
                os.unlink(pred_file)
            except OSError:
                pass

    def _evaluate_heuristic(self, instance_id: str) -> float:
        """Heuristic evaluation comparing predicted patch to the gold patch.

        Since the Docker harness is unavailable, we compute a similarity
        score between the predicted and gold patches.  This gives a
        non-zero signal for downstream training even without Docker:

          1.0 — predicted patch matches gold exactly (after normalisation)
          0.5-0.9 — partial overlap (correct files/hunks touched)
          0.1-0.4 — some structural similarity
          0.0 — no meaningful overlap
        """
        import difflib

        gold_patch = self._current_instance.get("patch", "")
        if not gold_patch:
            logger.warning(f"No gold patch for {instance_id}, returning 0.0")
            return 0.0

        predicted = self._submitted_patch.strip()
        gold = gold_patch.strip()

        # Exact match (normalised whitespace)
        if self._normalise_patch(predicted) == self._normalise_patch(gold):
            return 1.0

        # --- File-level overlap ---
        pred_files = set(self._extract_files_from_patch(predicted))
        gold_files = set(self._extract_files_from_patch(gold))

        if not pred_files or not gold_files:
            # Fallback to raw text similarity
            ratio = difflib.SequenceMatcher(None, predicted, gold).ratio()
            return round(ratio, 3)

        file_overlap = len(pred_files & gold_files) / len(gold_files) if gold_files else 0

        # --- Line-level similarity (on overlapping files) ---
        pred_lines = [l for l in predicted.splitlines() if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
        gold_lines = [l for l in gold.splitlines() if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]

        if pred_lines and gold_lines:
            line_ratio = difflib.SequenceMatcher(None, pred_lines, gold_lines).ratio()
        else:
            line_ratio = 0.0

        # Weighted score: file overlap + line similarity
        score = 0.4 * file_overlap + 0.6 * line_ratio
        return round(min(score, 1.0), 3)

    @staticmethod
    def _normalise_patch(patch: str) -> str:
        """Normalise a patch for comparison (strip whitespace, line numbers)."""
        import re
        lines = []
        for line in patch.splitlines():
            # Skip hunk headers (line numbers vary)
            if line.startswith("@@"):
                continue
            lines.append(line.rstrip())
        return "\n".join(lines)

    @staticmethod
    def _extract_files_from_patch(patch: str) -> list[str]:
        """Extract file paths from a unified diff."""
        import re
        files = []
        for match in re.finditer(r"^(?:---|\+\+\+) [ab]/(.+)$", patch, re.MULTILINE):
            files.append(match.group(1))
        # Deduplicate while preserving order
        seen = set()
        result = []
        for f in files:
            if f not in seen:
                seen.add(f)
                result.append(f)
        return result

    def close(self) -> None:
        """Clean up (no persistent resources for SWE-bench)."""
        self._current_task = None
        self._current_instance = None
        self._submitted_patch = None

    # ── Batch evaluation ──

    @staticmethod
    def batch_evaluate(
        predictions: list[dict],
        dataset_name: str = "princeton-nlp/SWE-bench",
        split: str = "test",
        run_id: str = "r2v_batch",
        max_workers: int = 4,
        timeout: int = 600,
    ) -> dict[str, bool]:
        """Evaluate multiple predictions using the SWE-bench Docker harness.

        This is the recommended way to evaluate after data collection.
        Much more efficient than per-episode evaluation.

        Args:
            predictions: List of dicts, each with keys:
                - instance_id: SWE-bench instance ID
                - patch: The predicted patch text
            dataset_name: HuggingFace dataset name
            split: Dataset split
            run_id: Unique identifier for this evaluation run
            max_workers: Number of parallel Docker containers
            timeout: Timeout per instance in seconds

        Returns:
            Dict mapping instance_id -> bool (resolved or not)
        """
        from swebench.harness.run_evaluation import main as run_evaluation
        from swebench.harness.constants import (
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
        )

        model_name = "r2v-teacher"
        instance_ids = [p["instance_id"] for p in predictions]

        # Format predictions for swebench harness
        formatted = [
            {
                KEY_INSTANCE_ID: p["instance_id"],
                KEY_PREDICTION: p["patch"],
                KEY_MODEL: model_name,
            }
            for p in predictions
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="r2v_batch_"
        ) as f:
            json.dump(formatted, f)
            pred_file = f.name

        try:
            summary_report_path = run_evaluation(
                dataset_name=dataset_name,
                split=split,
                instance_ids=instance_ids,
                predictions_path=pred_file,
                max_workers=max_workers,
                force_rebuild=False,
                cache_level="env",
                clean=True,
                open_file_limit=4096,
                run_id=run_id,
                timeout=timeout,
                namespace="swebench",
                rewrite_reports=False,
                modal=False,
            )

            resolved_ids = set()
            if summary_report_path and Path(summary_report_path).exists():
                with open(summary_report_path) as rf:
                    report = json.load(rf)
                resolved_ids = set(report.get("resolved_ids", []))

            return {iid: iid in resolved_ids for iid in instance_ids}
        finally:
            try:
                os.unlink(pred_file)
            except OSError:
                pass

    # ── Helpers ──

    @staticmethod
    def _extract_patch(text: str) -> Optional[str]:
        """Extract a diff/patch from the agent's response.

        Looks for content in <patch>...</patch> tags, ```diff...``` blocks
        (even unclosed), or raw unified diff lines anywhere in the text.
        """
        import re

        # Try <patch> tags
        match = re.search(r"<patch>(.*?)</patch>", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try ```diff blocks (closed)
        match = re.search(r"```(?:diff|patch)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try ```diff blocks (unclosed — model output may be truncated)
        match = re.search(r"```(?:diff|patch)?\n(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Check if it looks like a raw diff
        if text.startswith("---") or text.startswith("diff "):
            return text.strip()

        # Look for unified diff markers anywhere in the text
        match = re.search(r"(---\s+a/.+?\n\+\+\+\s+b/.+?\n@@.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None
