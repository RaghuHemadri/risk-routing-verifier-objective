"""
Logging utilities for R2V-Agent.

Provides:
- Structured console logging with levels
- wandb integration helpers
- JSONL file logger for structured metrics
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    name: str = "r2v",
) -> logging.Logger:
    """Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to write logs to file
        name: Logger name

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def init_wandb(
    project: str,
    name: str | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    mode: str = "online",
) -> wandb.run:
    """Initialize wandb run with standard settings.

    Args:
        project: W&B project name
        name: Run name
        config: Configuration dict
        tags: Tags for run organization
        group: Group name for related runs
        mode: "online", "offline", or "disabled"

    Returns:
        wandb run object
    """
    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags or [],
        group=group,
        mode=mode,
        reinit=True,
    )
    return run


def log_metrics(
    metrics: dict[str, Any],
    step: int | None = None,
    prefix: str = "",
    to_wandb: bool = True,
    to_console: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    """Log metrics to both wandb and console.

    Args:
        metrics: Dict of metric name → value
        step: Optional step number
        prefix: Prefix for metric names
        to_wandb: Whether to log to wandb
        to_console: Whether to log to console
        logger: Logger instance
    """
    prefixed = {}
    for k, v in metrics.items():
        key = f"{prefix}/{k}" if prefix else k
        prefixed[key] = v

    if to_wandb and wandb.run is not None:
        wandb.log(prefixed, step=step)

    if to_console and logger is not None:
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in prefixed.items()]
        msg = " | ".join(parts)
        if step is not None:
            msg = f"[step {step}] {msg}"
        logger.info(msg)


class JSONLLogger:
    """Append-only JSONL logger for structured experiment tracking.

    Each line is a JSON object with a timestamp and event type.
    Designed for machine-readable experiment records.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: dict[str, Any]) -> None:
        """Append a structured event to the JSONL file."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            **data,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_config(self, config: dict) -> None:
        """Log experiment configuration."""
        self.log("config", {"config": config})

    def log_metric(self, name: str, value: float, step: int | None = None, **extra) -> None:
        """Log a single metric."""
        data = {"metric": name, "value": value}
        if step is not None:
            data["step"] = step
        data.update(extra)
        self.log("metric", data)

    def log_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        """Log checkpoint save event."""
        self.log("checkpoint", {
            "path": path,
            "epoch": epoch,
            "metrics": metrics,
        })

    def log_evaluation(self, results: dict) -> None:
        """Log evaluation results."""
        self.log("evaluation", results)

    def read_all(self) -> list[dict]:
        """Read all events from the log."""
        events = []
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        return events

    def read_events(self, event_type: str) -> list[dict]:
        """Read events of a specific type."""
        return [e for e in self.read_all() if e.get("event") == event_type]
