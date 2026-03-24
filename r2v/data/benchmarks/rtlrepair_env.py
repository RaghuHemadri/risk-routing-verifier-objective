"""
RTL-Repair benchmark environment wrapper.

This environment mirrors the HumanEval-style multi-step workflow but targets
Verilog repair tasks inspired by the RTL-Repair benchmark suite.

Interaction model:
  write_code [verilog_rtl]  - overwrite the current candidate RTL
  test [optional_command]   - run test/eval command on current candidate
  submit                    - submit current candidate for final evaluation

Dataset/task input:
- Real mode only: JSON/JSONL manifest with task records.

Manifest fields (per task):
  task_id: str
  goal: str
  buggy_verilog: str (or buggy_file path)
  reference_verilog: str (optional, for exact-match fallback)
  test_command: str (optional)
  evaluate_command: str (optional)
  working_dir: str (optional)
  candidate_filename: str (optional, default: candidate.v)

Command placeholders available in test/evaluate commands:
  {candidate_file}, {workdir}, {task_dir}, {buggy_file}
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Optional

from .base import BenchmarkEnv, BenchmarkTask, EnvStepResult

logger = logging.getLogger(__name__)

_MAX_OBS_CHARS = 14_000


class RTLRepairEnv(BenchmarkEnv):
    """RTL-Repair style environment for Verilog bug-fixing tasks."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        rr_cfg = cfg.get("rtlrepair", {})
        ds_cfg = rr_cfg.get("dataset", {})

        self.dataset_path: str = ds_cfg.get("path", "")
        self.max_instances: Optional[int] = ds_cfg.get("max_instances", None)
        self.prefill_with_buggy: bool = bool(rr_cfg.get("prefill_with_buggy", True))

        self._cmd_timeout: int = int(rr_cfg.get("cmd_timeout", 60))
        self._test_timeout: int = int(rr_cfg.get("test_timeout", 120))

        self._current_task: Optional[BenchmarkTask] = None
        self._current_code: str = ""
        self._submitted: bool = False
        self._last_test_passed: bool = False
        self._last_test_output: str = ""
        self._tmp_dir: Optional[Path] = None

        self._dataset_cache: Optional[list[dict[str, Any]]] = None

    # -- Dataset loading -------------------------------------------------

    def _load_manifest(self) -> list[dict[str, Any]]:
        if not self.dataset_path:
            raise ValueError(
                "rtlrepair.dataset.path is required in real mode. "
                "Point it to a JSON or JSONL manifest."
            )

        path = Path(self.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"RTL-Repair manifest not found: {path}")

        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            return data["tasks"]

        raise ValueError(
            "Unsupported RTL-Repair manifest format. Use JSON list, "
            "JSON object with tasks, or JSONL."
        )

    def _load_dataset(self) -> list[dict[str, Any]]:
        if self._dataset_cache is not None:
            return self._dataset_cache

        rows = self._load_manifest()

        if self.max_instances is not None:
            rows = rows[: self.max_instances]

        self._dataset_cache = rows
        logger.info("Loaded %d RTL-Repair tasks from manifest", len(rows))
        return rows

    def load_tasks(self, cfg: Any) -> list[BenchmarkTask]:
        rows = self._load_dataset()
        tasks: list[BenchmarkTask] = []

        for idx, row in enumerate(rows):
            task_id = str(row.get("task_id", f"rtlrepair_{idx:04d}"))
            goal = str(row.get("goal", row.get("prompt", "Repair the buggy RTL module.")))

            buggy = row.get("buggy_verilog") or row.get("buggy_code")
            buggy_file = row.get("buggy_file")
            if (not buggy) and buggy_file:
                bpath = Path(str(buggy_file))
                if bpath.exists():
                    buggy = bpath.read_text(encoding="utf-8")

            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    goal=goal,
                    benchmark="rtlrepair",
                    difficulty=str(row.get("difficulty", "standard")),
                    repo=row.get("repo"),
                    extra={
                        "buggy_verilog": buggy or "",
                        "buggy_file": row.get("buggy_file"),
                        "reference_verilog": row.get("reference_verilog") or row.get("reference_code"),
                        "reference_file": row.get("reference_file"),
                        "test_command": row.get("test_command"),
                        "evaluate_command": row.get("evaluate_command"),
                        "working_dir": row.get("working_dir"),
                        "candidate_filename": row.get("candidate_filename", "candidate.v"),
                        "expected_tokens": row.get("expected_tokens") or [],
                        "success_regex": row.get("success_regex"),
                        "fail_regex": row.get("fail_regex"),
                    },
                )
            )

        return tasks

    # -- Environment interface ------------------------------------------

    def reset(self, task: BenchmarkTask, seed: int = 0) -> str:
        self._current_task = task
        self._submitted = False
        self._last_test_passed = False
        self._last_test_output = ""

        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="r2v_rtlrepair_"))

        buggy = str(task.extra.get("buggy_verilog", "") or "")
        self._current_code = buggy if self.prefill_with_buggy else ""

        return self._make_obs("[Ready] Provide a patch with write_code, then test, then submit.")

    def step(self, action_text: str) -> EnvStepResult:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")

        action_text = action_text.strip()
        lower = action_text.lower()

        if lower == "submit":
            self._submitted = True
            score = self.evaluate()
            return EnvStepResult(
                observation=self._make_obs(
                    f"[SUBMITTED] score={score:.2f}. "
                    f"last_test_passed={self._last_test_passed}"
                ),
                reward=score,
                done=True,
                success=bool(score > 0.5),
            )

        wc_match = re.match(r"write_code\s*\[", action_text, re.DOTALL | re.IGNORECASE)
        if wc_match:
            body = action_text[wc_match.end():]
            last_bracket = body.rfind("]")
            code = body[:last_bracket].strip() if last_bracket != -1 else body.strip()
            if code:
                self._current_code = code
                return EnvStepResult(
                    observation=self._make_obs(f"[Code saved] {len(code)} chars"),
                    reward=0.0,
                    done=False,
                )

        md_code = re.search(r"```(?:verilog|systemverilog)?\n(.+?)```", action_text, re.DOTALL | re.IGNORECASE)
        if md_code:
            self._current_code = md_code.group(1).strip()
            return EnvStepResult(
                observation=self._make_obs(f"[Code saved] {len(self._current_code)} chars"),
                reward=0.0,
                done=False,
            )

        test_match = re.match(r"test(?:\s*\[(.+)\])?\s*$", action_text, re.DOTALL | re.IGNORECASE)
        if test_match:
            custom_cmd = test_match.group(1).strip() if test_match.group(1) else None
            return self._run_tests(custom_cmd)

        if self._looks_like_verilog(action_text):
            self._current_code = action_text
            return EnvStepResult(
                observation=self._make_obs(f"[Code saved] {len(self._current_code)} chars"),
                reward=0.0,
                done=False,
            )

        return EnvStepResult(
            observation=self._make_obs(
                "[ERROR] Could not parse action. Use write_code [...], test, or submit."
            ),
            reward=0.0,
            done=False,
        )

    def evaluate(self) -> float:
        if self._current_task is None or not self._current_code.strip():
            return 0.0

        eval_cmd = self._current_task.extra.get("evaluate_command")
        if eval_cmd:
            ok, out, rc = self._execute_command(str(eval_cmd), timeout=self._test_timeout)
            self._last_test_passed = ok
            self._last_test_output = out
            return 1.0 if ok else 0.0

        if self._last_test_output:
            if self._last_test_passed:
                return 1.0
            return 0.0

        ref = self._load_reference_code()
        if ref:
            return 1.0 if self._normalize_code(self._current_code) == self._normalize_code(ref) else 0.0

        expected_tokens = list(self._current_task.extra.get("expected_tokens") or [])
        if expected_tokens:
            lower = self._current_code.lower()
            ok = all(tok.lower() in lower for tok in expected_tokens)
            return 1.0 if ok else 0.0

        return 0.0

    def close(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    # -- Internals -------------------------------------------------------

    def _make_obs(self, status: str) -> str:
        task = self._current_task
        goal = task.goal if task else ""
        buggy = str(task.extra.get("buggy_verilog", "") or "") if task else ""

        candidate = self._current_code.strip()
        if candidate and len(candidate) > 4000:
            candidate = candidate[:4000] + "\n... [truncated]"

        parts = [
            f"Goal:\n{goal}",
            "",
            "Buggy RTL:",
            "```verilog",
            buggy,
            "```",
            "",
            "Current candidate:",
            "```verilog",
            candidate or "(empty)",
            "```",
            "",
            "Available actions:",
            "  write_code [your_verilog_code]",
            "  test [optional_shell_command]",
            "  submit",
            "",
            status,
        ]
        if self._last_test_output:
            out = self._last_test_output
            if len(out) > 3000:
                out = out[:3000] + "\n... [test output truncated]"
            parts += ["", "Last test output:", out]

        return "\n".join(parts)[:_MAX_OBS_CHARS]

    def _run_tests(self, custom_cmd: Optional[str] = None) -> EnvStepResult:
        if not self._current_code.strip():
            return EnvStepResult(
                observation=self._make_obs("[test error] No candidate code written yet."),
                reward=0.0,
                done=False,
            )

        if self._current_task is None:
            return EnvStepResult(
                observation=self._make_obs("[test error] No active task loaded."),
                reward=0.0,
                done=False,
            )

        cmd = custom_cmd or self._current_task.extra.get("test_command")
        if not cmd:
            msg = "[test error] No test command configured for this task."
            self._last_test_output = msg
            self._last_test_passed = False
            return EnvStepResult(observation=self._make_obs(msg), reward=0.0, done=False)

        ok, out, rc = self._execute_command(str(cmd), timeout=self._cmd_timeout)
        self._last_test_passed = ok
        self._last_test_output = out

        return EnvStepResult(
            observation=self._make_obs(out),
            reward=1.0 if ok else 0.0,
            done=False,
            info={"returncode": rc},
        )

    def _execute_command(self, command_tmpl: str, timeout: int) -> tuple[bool, str, int]:
        candidate_file = self._write_candidate_file()

        task = self._current_task
        task_dir = Path(".")
        buggy_file = ""
        if task is not None:
            wd = task.extra.get("working_dir")
            if wd:
                task_dir = Path(str(wd))
            bf = task.extra.get("buggy_file")
            if bf:
                buggy_file = str(Path(str(bf)).resolve())

        cmd = command_tmpl.format(
            candidate_file=str(candidate_file),
            workdir=str(self._tmp_dir),
            task_dir=str(task_dir.resolve()),
            buggy_file=buggy_file,
        )

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(task_dir.resolve()),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"[timeout] command exceeded {timeout}s", 124
        except Exception as exc:
            return False, f"[command error] {exc}", 1

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        merged = "[stdout]\n" + stdout
        if stderr:
            merged += "\n[stderr]\n" + stderr

        success_regex = task.extra.get("success_regex") if task else None
        fail_regex = task.extra.get("fail_regex") if task else None

        ok = result.returncode == 0
        if success_regex and re.search(str(success_regex), merged, re.IGNORECASE):
            ok = True
        if fail_regex and re.search(str(fail_regex), merged, re.IGNORECASE):
            ok = False

        merged += f"\n[returncode] {result.returncode}"
        return ok, merged[:_MAX_OBS_CHARS], int(result.returncode)

    def _write_candidate_file(self) -> Path:
        if self._tmp_dir is None:
            raise RuntimeError("Temporary directory is not initialized. Call reset() first.")

        fname = "candidate.v"
        if self._current_task is not None:
            fname = str(self._current_task.extra.get("candidate_filename", "candidate.v"))
        candidate_file = self._tmp_dir / fname
        candidate_file.parent.mkdir(parents=True, exist_ok=True)
        candidate_file.write_text(self._current_code, encoding="utf-8")
        return candidate_file

    def _load_reference_code(self) -> str:
        if self._current_task is None:
            return ""

        ref = self._current_task.extra.get("reference_verilog")
        if ref:
            return str(ref)

        ref_file = self._current_task.extra.get("reference_file")
        if ref_file:
            path = Path(str(ref_file))
            if path.exists():
                return path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _normalize_code(code: str) -> str:
        code = re.sub(r"//.*", "", code)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        code = re.sub(r"\s+", "", code)
        return code.strip()

    @staticmethod
    def _looks_like_verilog(text: str) -> bool:
        lower = text.strip().lower()
        return (
            lower.startswith("module ")
            or "endmodule" in lower
            or "always @" in lower
            or "assign " in lower
        )
