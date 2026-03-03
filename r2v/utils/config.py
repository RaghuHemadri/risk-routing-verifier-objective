"""
Configuration management for R2V-Agent.

Wraps OmegaConf for YAML config loading with inheritance,
CLI overrides, and typed access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load a YAML config with optional CLI overrides.

    Supports config inheritance via:
    - ``_base_: ../base.yaml``  (simple string key)
    - ``defaults: [../base]``   (Hydra-style list)

    Args:
        config_path: Path to YAML config
        overrides: List of "key=value" CLI overrides

    Returns:
        Merged DictConfig
    """
    config_path = Path(config_path)
    cfg = OmegaConf.load(str(config_path))

    # Handle inheritance — try both _base_ and Hydra-style defaults
    base_path = None
    if "_base_" in cfg:
        base_path = config_path.parent / cfg._base_
        cfg = OmegaConf.merge(
            OmegaConf.load(str(base_path)),
            {k: v for k, v in cfg.items() if k != "_base_"},
        )
    elif "defaults" in cfg:
        # Hydra-style: defaults: [../base]
        defaults_list = OmegaConf.to_container(cfg.defaults, resolve=True)
        if isinstance(defaults_list, list):
            for entry in defaults_list:
                # entry can be a string like "../base" or a dict like {pkg: ...}
                if isinstance(entry, str):
                    candidate = config_path.parent / (entry + ".yaml")
                    if not candidate.exists():
                        candidate = config_path.parent / entry
                    if candidate.exists():
                        base_cfg = OmegaConf.load(str(candidate))
                        cfg = OmegaConf.merge(
                            base_cfg,
                            {k: v for k, v in cfg.items() if k != "defaults"},
                        )

    # Apply CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Resolve interpolations
    OmegaConf.resolve(cfg)

    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save config to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to plain Python dict."""
    return OmegaConf.to_container(cfg, resolve=True)


def get_output_dir(cfg: DictConfig) -> Path:
    """Get output directory from config, creating it if needed."""
    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_config(cfg: DictConfig, resolve: bool = True) -> None:
    """Pretty-print a config to console."""
    print(OmegaConf.to_yaml(cfg, resolve=resolve))
