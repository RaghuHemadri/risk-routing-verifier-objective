"""
Terminal-Bench benchmark environment wrapper.

Wraps the terminal-bench framework (https://github.com/harbor-framework/terminal-bench)
to collect trajectories from terminal-based tasks.

Each task runs in an isolated Docker container; the agent sends shell commands
that execute inside the container and receives terminal output.

Interaction model:
  run [shell_command]  – execute a shell command in the container terminal
  submit               – finalize and run the evaluation test script

Dataset:
  Tasks are discovered from a local terminal-bench dataset directory
  (with the expected structure: tasks/<task_id>/task.yaml + docker-compose.yaml
  + run-tests.sh), or from a JSONL manifest for flexibility.

Config keys (under ``terminalbench``):
  dataset.path          – path to dataset root (directory of task folders) OR a
                          JSONL manifest with {task_id, goal, task_dir} per line
  dataset.max_instances – cap on number of tasks to load (default: null)
  cmd_timeout           – per-command timeout in seconds (default: 30)
  test_timeout          – test-script timeout in seconds (default: 60)
  no_rebuild            – skip Docker image rebuild if already built (default: false)
  cleanup               – remove containers after each episode (default: true)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from .base import BenchmarkEnv, BenchmarkTask, EnvStepResult

logger = logging.getLogger(__name__)

_MAX_OBS_CHARS = 14_000
_SHELL_SETTLE_SEC = 1.0   # seconds to wait after container start before first capture


class TerminalBenchEnv(BenchmarkEnv):
    """Terminal-Bench environment: runs tasks inside Docker containers via tmux."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        tb_cfg = cfg.get("terminalbench", {})
        ds_cfg = tb_cfg.get("dataset", {})

        self.dataset_path: str = ds_cfg.get("path", "")
        self.max_instances: Optional[int] = ds_cfg.get("max_instances", None)
        self.cmd_timeout: int = int(tb_cfg.get("cmd_timeout", 30))
        self.test_timeout: int = int(tb_cfg.get("test_timeout", 60))
        self.no_rebuild: bool = bool(tb_cfg.get("no_rebuild", False))
        self.cleanup: bool = bool(tb_cfg.get("cleanup", True))

        # Runtime state (reset per episode)
        self._current_task: Optional[BenchmarkTask] = None
        self._terminal: Any = None   # terminal_bench.terminal.terminal.Terminal
        self._session: Any = None    # terminal_bench.terminal.tmux_session.TmuxSession
        self._submitted: bool = False
        self._eval_score: float = 0.0
        self._logs_tmpdir: Optional[Any] = None  # tempfile.TemporaryDirectory

        self._dataset_cache: Optional[list[dict[str, Any]]] = None

    # ── Dataset loading ────────────────────────────────────────────────────

    def _load_dataset(self) -> list[dict[str, Any]]:
        """Return list of raw task dicts with keys: task_id, goal, task_dir."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        if not self.dataset_path:
            raise ValueError(
                "terminalbench.dataset.path is required. "
                "Set it to a terminal-bench task directory root or a JSONL manifest."
            )

        path = Path(self.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"TerminalBench dataset path not found: {path}")

        rows: list[dict[str, Any]]

        # ── JSONL manifest ────────────────────────────────────────────────
        if path.is_file() and path.suffix.lower() in (".jsonl", ".json"):
            rows = self._load_jsonl(path)

        # ── terminal-bench directory tree (tasks/<id>/task.yaml) ──────────
        elif path.is_dir():
            rows = self._load_from_task_tree(path)

        else:
            raise ValueError(f"Unsupported dataset path: {path} (expected directory or .jsonl/.json file)")

        if self.max_instances is not None:
            rows = rows[: self.max_instances]

        self._dataset_cache = rows
        logger.info("Loaded %d TerminalBench tasks", len(rows))
        return rows

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if raw_line:
                    rows.append(json.loads(raw_line))
        return rows

    @staticmethod
    def _load_from_task_tree(root: Path) -> list[dict[str, Any]]:
        """Discover tasks from a directory tree where each sub-dir is a task.

        Expected layout per task directory::

            <root>/<task_id>/
                task.yaml            # contains ``instruction`` field
                docker-compose.yaml
                run-tests.sh
                solution.sh          # optional reference
        """
        rows: list[dict[str, Any]] = []

        # Try to use terminal-bench's own Dataset + Task classes if available.
        try:
            import yaml  # type: ignore
            from terminal_bench.dataset.dataset import Dataset  # type: ignore
            from terminal_bench.handlers.trial_handler import TaskPaths  # type: ignore

            dataset = Dataset(path=root)
            for task_path in dataset:
                task_paths_obj = TaskPaths(task_path)
                cfg_path = task_paths_obj.task_config_path
                if not cfg_path.exists():
                    continue
                raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                instruction = raw.get("instruction") or raw.get("goal") or task_path.name
                rows.append(
                    {
                        "task_id": task_path.name,
                        "goal": instruction,
                        "task_dir": str(task_path),
                        "difficulty": raw.get("difficulty", "unknown"),
                        "category": raw.get("category"),
                        "max_agent_timeout_sec": raw.get("max_agent_timeout_sec"),
                        "max_test_timeout_sec": raw.get("max_test_timeout_sec"),
                    }
                )
            return rows

        except ImportError:
            logger.debug(
                "terminal_bench not importable; falling back to manual task-tree scan. "
                "Install with: pip install terminal-bench"
            )

        # Fallback: manual scan
        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None  # type: ignore

        for task_dir in sorted(root.iterdir()):
            if not task_dir.is_dir():
                continue
            config_file = task_dir / "task.yaml"
            if not config_file.exists():
                continue
            instruction = task_dir.name
            difficulty = "unknown"
            if yaml is not None:
                try:
                    raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                    instruction = raw.get("instruction") or raw.get("goal") or task_dir.name
                    difficulty = raw.get("difficulty", "unknown")
                except Exception:
                    pass
            rows.append(
                {
                    "task_id": task_dir.name,
                    "goal": instruction,
                    "task_dir": str(task_dir),
                    "difficulty": difficulty,
                }
            )

        return rows

    def load_tasks(self, cfg: Any) -> list[BenchmarkTask]:
        rows = self._load_dataset()
        tasks: list[BenchmarkTask] = []
        for row in rows:
            task_id = str(row.get("task_id", "unknown"))
            goal = str(row.get("goal") or row.get("instruction") or "Complete the terminal task.")
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    goal=goal,
                    benchmark="terminalbench",
                    difficulty=str(row.get("difficulty", "unknown")),
                    extra={
                        "task_dir": row.get("task_dir", ""),
                        "category": row.get("category"),
                        "max_agent_timeout_sec": row.get("max_agent_timeout_sec"),
                        "max_test_timeout_sec": row.get("max_test_timeout_sec"),
                    },
                )
            )
        _difficulty_order = {"easy": 0, "medium": 1, "hard": 2, "unknown": 3}
        tasks.sort(key=lambda t: _difficulty_order.get(t.difficulty, 3))
        return tasks

    # ── Environment interface ──────────────────────────────────────────────

    def reset(self, task: BenchmarkTask, seed: int = 0) -> str:
        """Spin up a fresh Docker container for this task and open a tmux session."""
        # Tear down any running container from a previous episode
        self._teardown()

        self._current_task = task
        self._submitted = False
        self._eval_score = 0.0

        task_dir = Path(task.extra.get("task_dir", ""))
        if not task_dir.exists():
            return (
                f"[ERROR] Task directory not found: {task_dir}. "
                "Ensure the terminal-bench dataset is installed."
            )

        docker_compose_path = task_dir / "docker-compose.yaml"
        if not docker_compose_path.exists():
            docker_compose_path = task_dir / "docker-compose.yml"
        if not docker_compose_path.exists():
            return f"[ERROR] docker-compose file not found in {task_dir}."

        # Derive container / image names (terminal-bench convention)
        safe_id = task.task_id.replace(".", "-")
        container_name = f"{safe_id}-seed{seed}"
        image_name = f"tb__{safe_id}__client"

        try:
            from terminal_bench.terminal.terminal import Terminal  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "terminal-bench is required for TerminalBenchEnv. "
                "Install it with: pip install terminal-bench"
            ) from exc

        # terminal-bench requires T_BENCH_TASK_LOGS_PATH to be set for the
        # docker-compose volume mount; create a temporary directory for logs.
        self._logs_tmpdir = tempfile.TemporaryDirectory(prefix=f"r2v_tb_{safe_id}_")
        logs_path = Path(self._logs_tmpdir.name)

        self._terminal = Terminal(
            client_container_name=container_name,
            client_image_name=image_name,
            docker_compose_path=docker_compose_path,  # Terminal expects a Path object
            sessions_logs_path=logs_path,
            no_rebuild=self.no_rebuild,
            cleanup=self.cleanup,
        )

        try:
            self._terminal.start()
        except Exception as exc:
            logger.error("Failed to start terminal for task %s: %s", task.task_id, exc)
            return f"[ERROR] Could not start Docker container: {exc}"

        try:
            self._session = self._terminal.create_session(
                "agent", is_active_stream=False, as_configured_user=True
            )
        except Exception as exc:
            logger.error("Failed to create tmux session for task %s: %s", task.task_id, exc)
            return f"[ERROR] Could not create tmux session: {exc}"

        # Give the shell a moment to settle before capturing initial state
        time.sleep(_SHELL_SETTLE_SEC)
        initial_output = ""
        try:
            initial_output = self._session.capture_pane(full_history=False) or ""
        except Exception:
            pass

        return self._make_obs(initial_output or "[Shell ready]")

    def step(self, action_text: str) -> EnvStepResult:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")
        if self._terminal is None or self._session is None:
            return EnvStepResult(
                observation="[ERROR] No active container. Call reset() first.",
                reward=0.0,
                done=True,
            )

        action_text = action_text.strip()
        lower = action_text.lower()

        # ── submit ────────────────────────────────────────────────────────
        if lower == "submit":
            self._submitted = True
            score = self.evaluate()
            self._eval_score = score
            return EnvStepResult(
                observation=self._make_obs(
                    f"[SUBMITTED] Evaluation score: {score:.2f}. "
                    f"Task {'passed' if score > 0.5 else 'failed'}."
                ),
                reward=score,
                done=True,
                success=bool(score > 0.5),
            )

        # ── run [command] ─────────────────────────────────────────────────
        cmd = self._parse_run_command(action_text)
        if cmd is None:
            return EnvStepResult(
                observation=self._make_obs(
                    "[ERROR] Unrecognised action format. Use:\n"
                    "  run [your_shell_command]\n"
                    "  submit"
                ),
                reward=0.0,
                done=False,
            )

        timed_out, output = self._send_command(cmd)
        reward = 0.0
        obs = self._make_obs(output)
        if timed_out:
            obs = self._make_obs(output + f"\n[WARNING] Command timed out after {self.cmd_timeout}s.")

        return EnvStepResult(
            observation=obs,
            reward=reward,
            done=False,
            info={"timed_out": timed_out},
        )

    def evaluate(self) -> float:
        """Run the task's test script in the container; return 1.0 on pass, 0.0 on fail."""
        if self._current_task is None or self._terminal is None:
            return 0.0

        task_dir = Path(self._current_task.extra.get("task_dir", ""))
        run_tests_path = task_dir / "run-tests.sh"
        if not run_tests_path.exists():
            logger.warning(
                "No run-tests.sh found for task %s at %s; marking as failed.",
                self._current_task.task_id, run_tests_path,
            )
            return 0.0

        # Copy run-tests.sh + tests/ directory into /tests in the container,
        # mirroring what terminal-bench's harness does in _setup_test_env().
        remote_test_dir = "/tests"
        try:
            paths_to_copy: list[Path] = [run_tests_path]
            tests_dir = task_dir / "tests"
            if tests_dir.exists():
                paths_to_copy.append(tests_dir)
            self._terminal.copy_to_container(
                paths=paths_to_copy,
                container_dir=remote_test_dir,
            )
        except Exception as exc:
            logger.warning(
                "Failed to copy test files to container for task %s: %s",
                self._current_task.task_id, exc,
            )
            return 0.0

        # Open a dedicated test session so we don't pollute the agent session.
        # The session may already exist if evaluate() was called via submit then again
        # by the outer collect loop — reuse it in that case.
        try:
            if "test" in self._terminal._sessions:
                test_session = self._terminal.get_session("test")
            else:
                test_session = self._terminal.create_session(
                    "test", is_active_stream=False, as_configured_user=True
                )
        except Exception as exc:
            logger.warning("Could not create test session: %s", exc)
            test_session = self._session  # fall back to agent session

        timeout = self.test_timeout
        if self._current_task.extra.get("max_test_timeout_sec"):
            timeout = int(self._current_task.extra["max_test_timeout_sec"])

        # Run and capture exit code via a sentinel marker
        run_cmd = f"bash {remote_test_dir}/run-tests.sh; echo 'TB_EXIT_CODE:'$?"
        timed_out, output = self._send_command(run_cmd, session=test_session, timeout=timeout)

        if timed_out:
            logger.warning("Test script timed out for task %s.", self._current_task.task_id)
            return 0.0

        match = re.search(r"TB_EXIT_CODE:(\d+)", output)
        if match:
            exit_code = int(match.group(1))
            return 1.0 if exit_code == 0 else 0.0

        # If we didn't get the sentinel (e.g. shell exited), try to infer from content
        lower = output.lower()
        if any(kw in lower for kw in ("all tests passed", "success", "ok")):
            return 1.0
        if any(kw in lower for kw in ("failed", "error", "assertion")):
            return 0.0

        logger.warning(
            "Could not determine test outcome for task %s from output: %.200s",
            self._current_task.task_id, output,
        )
        return 0.0

    def close(self) -> None:
        self._teardown()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _teardown(self) -> None:
        if self._terminal is not None:
            try:
                self._terminal.stop()
            except Exception as exc:
                logger.debug("Error stopping terminal: %s", exc)
            self._terminal = None
            self._session = None
        if self._logs_tmpdir is not None:
            try:
                self._logs_tmpdir.cleanup()
            except Exception:
                pass
            self._logs_tmpdir = None

    def _send_command(
        self,
        command: str,
        session: Any = None,
        timeout: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Send *command* to a tmux session and return (timed_out, output)."""
        sess = session if session is not None else self._session
        if timeout is None:
            timeout = self.cmd_timeout

        try:
            sess.send_keys([command, "Enter"], block=True, max_timeout_sec=float(timeout))
            output = sess.get_incremental_output() or sess.capture_pane(full_history=False) or ""
            return False, output
        except TimeoutError:
            raw = ""
            try:
                raw = sess.capture_pane(full_history=False) or ""
            except Exception:
                pass
            return True, raw
        except Exception as exc:
            return False, f"[command error] {exc}"

    @staticmethod
    def _parse_run_command(action_text: str) -> Optional[str]:
        """Extract the shell command from an action string.

        Accepts any of:
          run [<command>]
          ```bash\\n<command>\\n```
          Action: run <command>
          run <command>           (bare, without brackets)
        """
        # 1) run [...] with brackets (possibly multiline)
        bm = re.match(r"run\s*\[(.+)\]\s*$", action_text, re.DOTALL | re.IGNORECASE)
        if bm:
            return bm.group(1).strip()

        # 2) markdown bash / shell block
        md = re.search(r"```(?:bash|sh|shell)?\s*\n(.+?)```", action_text, re.DOTALL | re.IGNORECASE)
        if md:
            return md.group(1).strip()

        # 3) "Action: run <command>" (single-line, no brackets)
        am = re.search(r"Action:\s*run\s+(.+?)(?:\n|$)", action_text, re.IGNORECASE)
        if am:
            return am.group(1).strip()

        # 4) bare "run <something>" on its own line (but NOT just "run" alone)
        bm2 = re.match(r"run\s+(.+)", action_text.strip(), re.IGNORECASE | re.DOTALL)
        if bm2:
            return bm2.group(1).strip()

        return None

    def _make_obs(self, terminal_output: str) -> str:
        task = self._current_task
        goal = task.goal if task else ""

        output = terminal_output
        if len(output) > _MAX_OBS_CHARS - 500:
            output = "...[truncated]...\n" + output[-((_MAX_OBS_CHARS - 500)):]

        parts = [
            f"Task: {goal}",
            "",
            "Terminal output:",
            output,
            "",
            "Available actions:",
            "  run [your_shell_command]  – execute a command in the container",
            "  submit                    – run evaluation tests and finish",
        ]
        return "\n".join(parts)[:_MAX_OBS_CHARS]
