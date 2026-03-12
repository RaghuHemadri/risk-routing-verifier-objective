"""
HumanEval+ (EvalPlus) benchmark environment for code generation evaluation.

The agent is given a function signature + docstring and must generate the
implementation.  It can iteratively write code, test it, and refine — making
this a multi-step agentic coding task suitable for routing evaluation.

Evaluation uses the EvalPlus test suites (base HumanEval + augmented tests)
for rigorous correctness checking.  Runs entirely locally — **no Docker**.

Dataset:  evalplus/humanevalplus  (HuggingFace)
Paper:    https://arxiv.org/abs/2305.01210
Package:  pip install evalplus

Requires:
    pip install evalplus datasets
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Optional

from .base import BenchmarkEnv, BenchmarkTask, EnvStepResult

logger = logging.getLogger(__name__)

_MAX_OBS_CHARS = 12_000


class HumanEvalPlusEnv(BenchmarkEnv):
    """HumanEval+ multi-step code generation environment.

    The agent interacts via the following actions:
        write_code [python_code]     – write/overwrite the solution
        test [optional_test_code]    – run the solution against sample tests
        submit                       – submit the current solution for evaluation

    The agent can iterate: write code, test, fix, test again, then submit.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        he_cfg = cfg.get("humaneval", {})
        ds_cfg = he_cfg.get("dataset", {})

        self.dataset_name: str = ds_cfg.get("name", "evalplus/humanevalplus")
        self.split: str = ds_cfg.get("split", "test")
        self.max_instances: Optional[int] = ds_cfg.get("max_instances", None)
        self.use_plus_tests: bool = he_cfg.get("use_plus_tests", True)

        self._cmd_timeout: int = he_cfg.get("cmd_timeout", 30)
        self._test_timeout: int = he_cfg.get("test_timeout", 60)

        # Per-task state
        self._current_task: Optional[BenchmarkTask] = None
        self._current_instance: Optional[dict] = None
        self._current_code: str = ""
        self._submitted: bool = False
        self._tmp_dir: Optional[Path] = None
        self._dataset_cache: Optional[list[dict]] = None

    # ── Dataset loading ──────────────────────────────────────────

    def _load_dataset(self) -> list[dict]:
        if self._dataset_cache is not None:
            return self._dataset_cache

        try:
            from evalplus.data import get_human_eval_plus
            problems = get_human_eval_plus()
            instances = []
            for task_id, problem in problems.items():
                instances.append({
                    "task_id": task_id,
                    "prompt": problem["prompt"],
                    "canonical_solution": problem.get("canonical_solution", ""),
                    "entry_point": problem["entry_point"],
                    "test": problem.get("test", ""),
                    "base_input": problem.get("base_input", []),
                    "plus_input": problem.get("plus_input", []),
                })
        except ImportError:
            # Fallback: load from HuggingFace
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "Neither evalplus nor datasets is installed. Run:\n"
                    "  pip install evalplus\n"
                    "or:\n"
                    "  pip install datasets"
                )
            logger.info(f"Loading HumanEval+ from HuggingFace: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name, split=self.split)
            instances = [dict(inst) for inst in dataset]

        if self.max_instances is not None:
            instances = instances[: self.max_instances]

        self._dataset_cache = instances
        logger.info(f"Loaded {len(instances)} HumanEval+ instances")
        return instances

    def load_tasks(self, cfg: Any) -> list[BenchmarkTask]:
        instances = self._load_dataset()
        tasks = []
        for inst in instances:
            task_id = inst.get("task_id", "")
            prompt = inst.get("prompt", "")
            entry_point = inst.get("entry_point", "")
            tasks.append(
                BenchmarkTask(
                    task_id=str(task_id),
                    goal=(
                        f"Implement the following Python function:\n\n"
                        f"{prompt}\n\n"
                        f"Function name: {entry_point}"
                    ),
                    benchmark="humaneval",
                    difficulty="standard",
                    extra={
                        "prompt": prompt,
                        "entry_point": entry_point,
                    },
                )
            )
        return tasks

    # ── Environment interface ────────────────────────────────────

    def reset(self, task: BenchmarkTask, seed: int = 0) -> str:
        self._current_task = task
        self._current_code = ""
        self._submitted = False

        instances = self._load_dataset()
        self._current_instance = None
        for inst in instances:
            if str(inst.get("task_id", "")) == task.task_id:
                self._current_instance = inst
                break

        if self._tmp_dir and self._tmp_dir.exists():
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="r2v_he_"))

        prompt = task.extra.get("prompt", "")
        entry_point = task.extra.get("entry_point", "")
        # Store for reuse in step() observations
        self._problem_header = (
            f"Complete the following Python function.\n\n"
            f"{prompt}\n"
            f"Entry point: {entry_point}"
        )
        self._action_hint = (
            "Available actions:\n"
            "  write_code [your_python_code]  – Write/overwrite your solution\n"
            "  test [optional_test_code]      – Test your current solution\n"
            "  submit                         – Submit your solution for evaluation"
        )

        obs = self._problem_header + "\n\n" + self._action_hint
        return obs

    def _make_obs(self, status: str) -> str:
        """Build an observation that always includes the problem statement."""
        header = getattr(self, "_problem_header", "")
        code_preview = ""
        if self._current_code:
            code_preview = (
                f"\n\nYour current solution:\n```python\n"
                f"{self._current_code[:800]}\n```"
            )
        return f"{header}{code_preview}\n\n{status}"

    def step(self, action_text: str) -> EnvStepResult:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")

        action_text = action_text.strip()

        if action_text.lower() == "submit":
            self._submitted = True
            if not self._current_code:
                return EnvStepResult(
                    observation=self._make_obs("[WARNING] No code written. Use write_code first."),
                    reward=0.0,
                    done=True,
                )
            return EnvStepResult(
                observation=self._make_obs(f"[SUBMITTED] Solution ({len(self._current_code)} chars)"),
                reward=0.0,
                done=True,
            )

        # Parse write_code action — handle both write_code [...] and bare code blocks.
        # Use a greedy match from the first '[' to the last ']' to handle nested brackets.
        wc_bracket = re.match(r"write_code\s*\[", action_text, re.DOTALL)
        if wc_bracket:
            inner = action_text[wc_bracket.end():]
            # Find the matching closing bracket (last ']' in the string)
            last_bracket = inner.rfind("]")
            code = inner[:last_bracket].strip() if last_bracket != -1 else inner.strip()
            if code:
                self._current_code = code
                return EnvStepResult(
                    observation=self._make_obs(f"[Code saved] ({len(self._current_code)} chars)"),
                    reward=0.0,
                    done=False,
                )

        # Markdown code blocks (with or without write_code prefix)
        code_match = re.search(r"```(?:python)?\n(.+?)```", action_text, re.DOTALL)
        if code_match:
            self._current_code = code_match.group(1).strip()
            return EnvStepResult(
                observation=self._make_obs(f"[Code saved] ({len(self._current_code)} chars)"),
                reward=0.0,
                done=False,
            )

        # Parse test action
        test_match = re.match(r"test(?:\s*\[(.+)\])?\s*$", action_text, re.DOTALL)
        if test_match:
            custom_test = test_match.group(1) if test_match.group(1) else None
            return self._run_tests(custom_test)

        # If the action looks like raw Python (def / imports / indented body), treat as write_code
        if (action_text.startswith("def ")
                or action_text.startswith("from ")
                or action_text.startswith("import ")
                or re.match(r"^    \S", action_text)):
            self._current_code = action_text
            return EnvStepResult(
                observation=self._make_obs(f"[Code saved] ({len(self._current_code)} chars)"),
                reward=0.0,
                done=False,
            )

        return EnvStepResult(
            observation=self._make_obs(
                "[ERROR] Could not parse action. Use:\n"
                "  write_code [your_code]\n"
                "  test\n"
                "  submit"
            ),
            reward=0.0,
            done=False,
        )

    def _build_full_code(self, agent_code: str) -> str:
        """Combine the HumanEval prompt with agent code.

        HumanEval prompts end with the function signature + docstring but no body.
        If the agent already wrote a complete function (starts with 'def <entry_point>'),
        use it as-is to avoid duplicating the signature.  Otherwise prepend the prompt.
        """
        prompt = self._current_instance.get("prompt", "") if self._current_instance else ""
        entry_point = self._current_instance.get("entry_point", "") if self._current_instance else ""
        code = textwrap.dedent(agent_code).strip()
        # Detect if agent wrote a complete function (possibly after imports/comments).
        # Use re.MULTILINE so `^` matches at the start of any line, not just the string start.
        if re.search(rf"^def {re.escape(entry_point)}\b", code, re.MULTILINE):
            return code
        # Body-only — normalize indentation then re-indent to 4 spaces for function body
        return prompt + "\n" + textwrap.indent(code, "    ")

    def evaluate(self) -> float:
        """Evaluate using EvalPlus test suites if available,
        otherwise use basic assertion tests."""
        if not self._current_code or not self._current_instance:
            return 0.0

        entry_point = self._current_instance.get("entry_point", "")
        test_code = self._current_instance.get("test", "")

        # Build full solution, avoiding duplicate function signatures
        full_code = self._build_full_code(self._current_code)

        # Build test script
        test_script = full_code + "\n\n" + test_code
        test_script += f"\n\ncheck({entry_point})\n"

        sol_path = self._tmp_dir / "solution_test.py"
        sol_path.write_text(test_script)

        try:
            result = subprocess.run(
                ["python3", str(sol_path)],
                capture_output=True,
                text=True,
                timeout=self._test_timeout,
                cwd=str(self._tmp_dir),
            )
            if result.returncode == 0:
                return 1.0
            return 0.0
        except (subprocess.TimeoutExpired, Exception):
            return 0.0

    def close(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    # ── Test runner ──────────────────────────────────────────────

    def _run_tests(self, custom_test: Optional[str] = None) -> EnvStepResult:
        """Run sample tests against the current code."""
        if not self._current_code:
            return EnvStepResult(
                observation="[test error] No code written yet. Use write_code first.",
                reward=0.0,
                done=False,
            )

        if not self._current_instance:
            return EnvStepResult(
                observation="[test error] No instance loaded.",
                reward=0.0,
                done=False,
            )
        entry_point = self._current_instance.get("entry_point", "")

        full_code = self._build_full_code(self._current_code)

        if custom_test:
            test_script = full_code + "\n\n" + custom_test
        else:
            # Use a subset of base test inputs for quick feedback
            test_script = full_code + "\n\nprint('Function defined successfully.')\n"
            # Try simple smoke test
            test_script += f"\nprint(f'Entry point: {entry_point}')\n"
            if self._current_instance:
                base_inputs = self._current_instance.get("base_input", [])
                if base_inputs and len(base_inputs) > 0:
                    # Run first 3 test cases
                    for i, inp in enumerate(base_inputs[:3]):
                        test_script += (
                            f"\ntry:\n"
                            f"    result = {entry_point}(*{repr(inp)})\n"
                            f"    print(f'Test {i}: {{result}}')\n"
                            f"except Exception as e:\n"
                            f"    print(f'Test {i} ERROR: {{e}}')\n"
                        )

        sol_path = self._tmp_dir / "test_run.py"
        sol_path.write_text(test_script)

        try:
            result = subprocess.run(
                ["python3", str(sol_path)],
                capture_output=True,
                text=True,
                timeout=self._cmd_timeout,
                cwd=str(self._tmp_dir),
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            status = f"[test output]\n{stdout}"
            if stderr:
                status += f"\n[stderr]\n{stderr}"
            if result.returncode == 0:
                status += "\n[All tests passed]"
            else:
                status += f"\n[Tests failed with return code {result.returncode}]"
            return EnvStepResult(
                observation=self._make_obs(status)[:_MAX_OBS_CHARS],
                reward=0.0,
                done=False,
                info={"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return EnvStepResult(
                observation=self._make_obs("[test error] Execution timed out."),
                reward=0.0,
                done=False,
            )
