"""
GAIA benchmark environment for general AI assistant evaluation.

GAIA (General AI Assistants) proposes real-world questions that require
fundamental abilities: reasoning, multi-modality handling, web browsing,
and tool-use proficiency.  Questions are conceptually simple for humans
yet challenging for advanced AIs.

The agent interacts via text tool-calls (web_search, python, calculator,
file_read) executed locally — **no Docker required**.

Dataset:  gaia-benchmark/GAIA  (HuggingFace)
Paper:    https://arxiv.org/abs/2311.12983

Requires:
    pip install datasets
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Optional

from .base import BenchmarkEnv, BenchmarkTask, EnvStepResult

logger = logging.getLogger(__name__)

_MAX_OBS_CHARS = 12_000


class GAIAEnv(BenchmarkEnv):
    """GAIA benchmark environment with local tool execution.

    Supported tool-call actions (agent must format actions as shown):
        web_search [query]      – simulated / API web search
        python [code]           – execute Python snippet in subprocess
        calculator [expression] – evaluate a math expression
        file_read [path]        – read a file from the task's attached files
        answer [final_answer]   – submit the final answer (terminates episode)
    """

    # ── Tool-call grammar ────────────────────────────────────────
    _ACTION_RE = re.compile(
        r"^(web_search|python|calculator|file_read|answer)"
        r"\s*\[(.+)\]\s*$",
        re.DOTALL,
    )

    def __init__(self, cfg: Any):
        self.cfg = cfg
        gaia_cfg = cfg.get("gaia", {})
        ds_cfg = gaia_cfg.get("dataset", {})

        self.dataset_name: str = ds_cfg.get("name", "gaia-benchmark/GAIA")
        self.split: str = ds_cfg.get("split", "validation")
        self.level_filter: Optional[list[int]] = ds_cfg.get("levels", None)
        self.max_instances: Optional[int] = ds_cfg.get("max_instances", None)

        self._cmd_timeout: int = gaia_cfg.get("cmd_timeout", 30)

        # Per-task state
        self._current_task: Optional[BenchmarkTask] = None
        self._current_instance: Optional[dict] = None
        self._submitted_answer: Optional[str] = None
        self._dataset_cache: Optional[list[dict]] = None
        self._tmp_dir: Optional[Path] = None

    # ── Dataset loading ──────────────────────────────────────────

    def _load_dataset(self) -> list[dict]:
        if self._dataset_cache is not None:
            return self._dataset_cache

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed.  Run:\n"
                "  pip install datasets"
            )

        logger.info(
            f"Loading GAIA dataset: {self.dataset_name} (split={self.split})"
        )
        dataset = load_dataset(self.dataset_name, "2023_all", split=self.split)
        instances = [dict(inst) for inst in dataset]

        if self.level_filter:
            instances = [
                inst for inst in instances
                if inst.get("Level", inst.get("level")) in self.level_filter
            ]

        if self.max_instances is not None:
            instances = instances[: self.max_instances]

        self._dataset_cache = instances
        logger.info(f"Loaded {len(instances)} GAIA instances")
        return instances

    def load_tasks(self, cfg: Any) -> list[BenchmarkTask]:
        instances = self._load_dataset()
        tasks = []
        for inst in instances:
            task_id = inst.get("task_id", inst.get("id", ""))
            level = inst.get("Level", inst.get("level", ""))
            tasks.append(
                BenchmarkTask(
                    task_id=str(task_id),
                    goal=inst.get("Question", inst.get("question", "")),
                    benchmark="gaia",
                    difficulty=str(level),
                    extra={
                        "expected_answer": inst.get(
                            "Final answer",
                            inst.get("final_answer", ""),
                        ),
                        "annotator_metadata": inst.get("Annotator Metadata", {}),
                    },
                )
            )
        return tasks

    # ── Environment interface ────────────────────────────────────

    def reset(self, task: BenchmarkTask, seed: int = 0) -> str:
        self._current_task = task
        self._submitted_answer = None

        instances = self._load_dataset()
        self._current_instance = None
        for inst in instances:
            tid = str(inst.get("task_id", inst.get("id", "")))
            if tid == task.task_id:
                self._current_instance = inst
                break

        # Set up temp directory for file-based tasks
        if self._tmp_dir and self._tmp_dir.exists():
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="r2v_gaia_"))

        # Write any attached files
        if self._current_instance:
            file_name = self._current_instance.get("file_name", "")
            file_bytes = self._current_instance.get("file_path", "")
            if file_name and file_bytes:
                try:
                    dst = self._tmp_dir / file_name
                    if isinstance(file_bytes, bytes):
                        dst.write_bytes(file_bytes)
                    elif isinstance(file_bytes, str) and os.path.isfile(file_bytes):
                        import shutil
                        shutil.copy2(file_bytes, dst)
                except Exception as exc:
                    logger.debug(f"Could not write attached file: {exc}")

        obs_parts = [
            f"Question: {task.goal}",
            "",
            "Available tools:",
            "  web_search [query]      – Search the web",
            "  python [code]           – Execute Python code",
            "  calculator [expression] – Evaluate a math expression",
            "  file_read [filename]    – Read an attached file",
            "  answer [final_answer]   – Submit your final answer",
        ]
        attached = self._current_instance.get("file_name", "") if self._current_instance else ""
        if attached:
            obs_parts.append(f"\nAttached file: {attached}")

        return "\n".join(obs_parts)

    def step(self, action_text: str) -> EnvStepResult:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")

        action_text = action_text.strip()
        match = self._ACTION_RE.match(action_text)

        if not match:
            return EnvStepResult(
                observation=(
                    "[ERROR] Could not parse action. Use format: "
                    "tool_name [argument]\n"
                    "Available tools: web_search, python, calculator, "
                    "file_read, answer"
                ),
                reward=0.0,
                done=False,
            )

        tool, arg = match.group(1), match.group(2)

        if tool == "answer":
            self._submitted_answer = arg.strip()
            return EnvStepResult(
                observation=f"[SUBMITTED] Answer: {self._submitted_answer}",
                reward=0.0,
                done=True,
            )

        if tool == "web_search":
            return self._tool_web_search(arg)
        if tool == "python":
            return self._tool_python(arg)
        if tool == "calculator":
            return self._tool_calculator(arg)
        if tool == "file_read":
            return self._tool_file_read(arg)

        return EnvStepResult(
            observation=f"[ERROR] Unknown tool: {tool}",
            reward=0.0,
            done=False,
        )

    def evaluate(self) -> float:
        if not self._submitted_answer or not self._current_instance:
            return 0.0

        expected = str(
            self._current_instance.get(
                "Final answer",
                self._current_instance.get("final_answer", ""),
            )
        ).strip()
        predicted = self._submitted_answer.strip()

        if not expected:
            return 0.0

        # Exact match (case-insensitive, strip whitespace)
        if predicted.lower() == expected.lower():
            return 1.0

        # Normalised containment check for short expected answers
        if len(expected) < 60 and expected.lower() in predicted.lower():
            return 1.0

        # Numeric tolerance
        try:
            exp_num = float(re.sub(r"[,$%]", "", expected))
            pred_num = float(re.sub(r"[,$%]", "", predicted))
            if abs(exp_num - pred_num) < 1e-4 * max(abs(exp_num), 1):
                return 1.0
        except (ValueError, ZeroDivisionError):
            pass

        return 0.0

    def close(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    # ── Tool implementations ─────────────────────────────────────

    def _tool_web_search(self, query: str) -> EnvStepResult:
        """Simulated web search.  Returns a stub unless a search API key
        is configured (SERPER_API_KEY or SERPAPI_KEY)."""
        serper_key = os.environ.get("SERPER_API_KEY", "")
        if serper_key:
            return self._web_search_serper(query, serper_key)

        serpapi_key = os.environ.get("SERPAPI_KEY", "")
        if serpapi_key:
            return self._web_search_serpapi(query, serpapi_key)

        # Fallback: informative stub so agent can still attempt reasoning
        return EnvStepResult(
            observation=(
                "[web_search result]\n"
                f"Query: {query}\n\n"
                "No search API key is configured.  Set SERPER_API_KEY or "
                "SERPAPI_KEY to enable live web search.\n"
                "Please try to answer using your existing knowledge or "
                "other available tools."
            ),
            reward=0.0,
            done=False,
        )

    def _web_search_serper(self, query: str, api_key: str) -> EnvStepResult:
        """Call Serper.dev Google Search API."""
        import urllib.request
        import urllib.error

        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 5}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            return EnvStepResult(
                observation=f"[web_search error] {exc}",
                reward=0.0,
                done=False,
            )

        lines = [f"[web_search result] Query: {query}\n"]
        for item in data.get("organic", [])[:5]:
            lines.append(f"- {item.get('title', '')}")
            lines.append(f"  {item.get('snippet', '')}")
            lines.append(f"  URL: {item.get('link', '')}\n")
        obs = "\n".join(lines)[:_MAX_OBS_CHARS]
        return EnvStepResult(observation=obs, reward=0.0, done=False)

    def _web_search_serpapi(self, query: str, api_key: str) -> EnvStepResult:
        """Call SerpAPI Google Search."""
        import urllib.request
        import urllib.parse
        import urllib.error

        params = urllib.parse.urlencode({
            "q": query,
            "api_key": api_key,
            "num": 5,
        })
        url = f"https://serpapi.com/search.json?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            return EnvStepResult(
                observation=f"[web_search error] {exc}",
                reward=0.0,
                done=False,
            )

        lines = [f"[web_search result] Query: {query}\n"]
        for item in data.get("organic_results", [])[:5]:
            lines.append(f"- {item.get('title', '')}")
            lines.append(f"  {item.get('snippet', '')}")
            lines.append(f"  URL: {item.get('link', '')}\n")
        obs = "\n".join(lines)[:_MAX_OBS_CHARS]
        return EnvStepResult(observation=obs, reward=0.0, done=False)

    def _tool_python(self, code: str) -> EnvStepResult:
        """Execute Python code in an isolated subprocess."""
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=self._cmd_timeout,
                cwd=str(self._tmp_dir) if self._tmp_dir else None,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            obs = f"[python output]\n{stdout}"
            if stderr:
                obs += f"\n[stderr]\n{stderr}"
            return EnvStepResult(
                observation=obs[:_MAX_OBS_CHARS],
                reward=0.0,
                done=False,
                info={"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return EnvStepResult(
                observation="[python error] Execution timed out.",
                reward=0.0,
                done=False,
            )

    def _tool_calculator(self, expression: str) -> EnvStepResult:
        """Evaluate a math expression safely."""
        # Allow only safe characters and math functions
        sanitized = re.sub(r"[^0-9+\-*/().,%^ eE]", "", expression)

        # Replace common math notation
        sanitized = sanitized.replace("^", "**")

        try:
            # Use Python's compile + eval with restricted builtins
            allowed = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "pow": pow,
                "int": int,
                "float": float,
                "math": math,
            }
            result = eval(compile(expression, "<calc>", "eval"), allowed)  # noqa: S307
            return EnvStepResult(
                observation=f"[calculator result] {result}",
                reward=0.0,
                done=False,
            )
        except Exception as exc:
            return EnvStepResult(
                observation=f"[calculator error] {exc}",
                reward=0.0,
                done=False,
            )

    def _tool_file_read(self, filename: str) -> EnvStepResult:
        """Read a file from the task's attached files directory."""
        if not self._tmp_dir:
            return EnvStepResult(
                observation="[file_read error] No files attached to this task.",
                reward=0.0,
                done=False,
            )

        # Prevent path traversal
        safe_name = Path(filename).name
        filepath = self._tmp_dir / safe_name

        if not filepath.exists():
            available = [f.name for f in self._tmp_dir.iterdir() if f.is_file()]
            return EnvStepResult(
                observation=(
                    f"[file_read error] File '{safe_name}' not found.\n"
                    f"Available files: {available}"
                ),
                reward=0.0,
                done=False,
            )

        try:
            content = filepath.read_text(errors="replace")
            return EnvStepResult(
                observation=f"[file content: {safe_name}]\n{content[:_MAX_OBS_CHARS]}",
                reward=0.0,
                done=False,
            )
        except Exception as exc:
            return EnvStepResult(
                observation=f"[file_read error] {exc}",
                reward=0.0,
                done=False,
            )
