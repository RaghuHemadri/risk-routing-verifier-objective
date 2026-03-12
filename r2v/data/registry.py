"""
Data versioning and run registry for R2V-Agent trajectory collection.

Every collection run is tracked with a unique run_id and registered in
data/registry.json. Downstream scripts (train_policy, train_verifier, etc.)
use DataRegistry to query and combine datasets across models, benchmarks,
and collection dates without hardcoding paths.

Run ID format:
    {benchmark}_{provider}_{model_slug}_{YYYYMMDDTHHMMSS}
    e.g. webarena_openai_gpt-4o_20260218T210012
         gaia_anthropic_claude-3-5-sonnet_20260219T083045

Directory layout (relative to project root):
    data/
      registry.json                          ← central index (written by registry)
      runs/
        webarena_openai_gpt-4o_20260218T210012/
          run_manifest.json                  ← config snapshot + final counts
          trajectories.jsonl                 ← episodes (each has run_id embedded)
          collection_log.jsonl               ← per-event logs

Usage:
    from r2v.data.registry import DataRegistry, make_run_id

    # During collection — create a run
    run_id = make_run_id("gaia", "openai", "gpt-4o")
    registry = DataRegistry()
    registry.begin_run(run_id, cfg=cfg_dict, output_dir=Path("data/runs") / run_id)

    # After collection — finalise
    registry.finish_run(run_id, n_episodes=300, n_success=210,
                        total_cost=1.23, wall_time=3600)

    # Downstream — query
    registry = DataRegistry()
    runs = registry.query(benchmark="gaia", model="gpt-4o")
    store = registry.merge_runs([r.run_id for r in runs])
    episodes = store.load_episodes()
"""

from __future__ import annotations

import json
import re
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from r2v.data.trajectory import Episode, TrajectoryStore


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _slugify(text: str) -> str:
    """Convert any string to a filesystem-safe slug."""
    return _SLUG_RE.sub("-", text).strip("-").lower()


def make_run_id(
    benchmark: str,
    provider: str,
    model_name: str,
    *,
    timestamp: Optional[datetime] = None,
    extra: str = "",
) -> str:
    """Generate a unique, human-readable run ID.

    Args:
        benchmark:  "webarena" | "gaia" | "humaneval" | "alfworld"
        provider:   "openai" | "anthropic" | "google" | "deepseek"
        model_name: e.g. "gpt-4o", "claude-3-5-sonnet-20241022"
        timestamp:  Defaults to UTC now.
        extra:      Optional tag appended to the ID (e.g. "noisy", "ablation").

    Returns:
        e.g. "webarena_openai_gpt-4o_20260218T210012"
             "gaia_anthropic_claude-3-5-sonnet_20260219T083045_noisy"
    """
    ts = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    parts = [benchmark, provider, _slugify(model_name), ts]
    if extra:
        parts.append(_slugify(extra))
    return "_".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# RunManifest — per-run sidecar
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunManifest:
    """Metadata record for a single collection run.

    Written as ``run_manifest.json`` inside the run directory and also
    stored inline in ``data/registry.json``.
    """

    # Identity
    run_id: str
    benchmark: str
    provider: str
    model_name: str

    # Filesystem
    output_dir: str                         # str for JSON serializability

    # Config snapshot (frozen at run start)
    config: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    started_at: str = ""                    # ISO-8601 UTC
    finished_at: str = ""
    status: str = "running"                 # running | done | failed | partial

    # Outcome counts (filled in at finish)
    n_episodes: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    total_cost_usd: float = 0.0
    wall_time_seconds: float = 0.0

    # Environment
    host: str = field(default_factory=socket.gethostname)
    git_commit: str = ""
    python_version: str = ""

    # Free-form notes
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # ── derived ──────────────────────────────────────────────────

    @property
    def trajectory_path(self) -> Path:
        return Path(self.output_dir) / "trajectories.jsonl"

    @property
    def log_path(self) -> Path:
        return Path(self.output_dir) / "collection_log.jsonl"

    @property
    def manifest_path(self) -> Path:
        return Path(self.output_dir) / "run_manifest.json"

    # ── I/O ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunManifest":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self) -> None:
        """Write run_manifest.json inside the run output directory."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ──────────────────────────────────────────────────────────────────────────────
# DataRegistry — central index
# ──────────────────────────────────────────────────────────────────────────────

class DataRegistry:
    """Central registry of all trajectory collection runs.

    Backed by ``data/registry.json`` in the project root (or a custom path).
    Thread-safe for reads; writes use a simple file lock via atomic replace.

    Example usage
    -------------
    # In collect_trajectories.py:
    registry = DataRegistry()
    run_id = make_run_id("gaia", "openai", "gpt-4o")
    registry.begin_run(run_id, cfg=cfg_dict,
                       output_dir=output_dir / run_id,
                       benchmark="gaia",
                       provider="openai",
                       model_name="gpt-4o")
    # ... collect ...
    registry.finish_run(run_id, n_episodes=300, n_success=210,
                        total_cost=1.23, wall_time=3600.0)

    # In train_policy.py:
    registry = DataRegistry()
    manifests = registry.query(benchmark="gaia", status="done")
    store = registry.merge_runs([m.run_id for m in manifests])
    episodes = list(store.iter_episodes())
    """

    DEFAULT_REGISTRY_PATH = Path("data/registry.json")

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = Path(registry_path or self.DEFAULT_REGISTRY_PATH)
        self._runs: dict[str, RunManifest] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        with open(self.registry_path) as f:
            data = json.load(f)
        for entry in data.get("runs", []):
            m = RunManifest.from_dict(entry)
            self._runs[m.run_id] = m

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Sort by started_at desc so newest runs appear first
        runs_sorted = sorted(
            self._runs.values(),
            key=lambda m: m.started_at,
            reverse=True,
        )
        payload = {
            "schema_version": "1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs": [m.to_dict() for m in runs_sorted],
        }
        # Atomic write via temp file
        tmp = self.registry_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self.registry_path)

    # ── lifecycle ─────────────────────────────────────────────────

    def begin_run(
        self,
        run_id: str,
        *,
        benchmark: str,
        provider: str,
        model_name: str,
        output_dir: Path,
        cfg: dict[str, Any],
        tags: Optional[list[str]] = None,
        notes: str = "",
    ) -> RunManifest:
        """Register a run as started. Call at the top of collect_trajectories."""
        import platform, subprocess
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_commit = "unknown"

        manifest = RunManifest(
            run_id=run_id,
            benchmark=benchmark,
            provider=provider,
            model_name=model_name,
            output_dir=str(output_dir.resolve()),
            config=cfg,
            started_at=datetime.now(timezone.utc).isoformat(),
            status="running",
            git_commit=git_commit,
            python_version=platform.python_version(),
            tags=tags or [],
            notes=notes,
        )
        self._runs[run_id] = manifest
        self._save()
        manifest.save()  # also write sidecar inside the run dir
        return manifest

    def finish_run(
        self,
        run_id: str,
        *,
        n_episodes: int,
        n_success: int,
        total_cost: float,
        wall_time: float,
        status: str = "done",
    ) -> RunManifest:
        """Mark a run as finished. Call at the end of collect_trajectories."""
        manifest = self._runs[run_id]
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        manifest.status = status
        manifest.n_episodes = n_episodes
        manifest.n_success = n_success
        manifest.success_rate = n_success / n_episodes if n_episodes > 0 else 0.0
        manifest.total_cost_usd = round(total_cost, 6)
        manifest.wall_time_seconds = round(wall_time, 1)
        self._save()
        manifest.save()
        return manifest

    def fail_run(self, run_id: str, reason: str = "") -> None:
        """Mark a run as failed (e.g. on unhandled exception)."""
        if run_id in self._runs:
            self._runs[run_id].status = "failed"
            self._runs[run_id].notes = reason or self._runs[run_id].notes
            self._save()
            self._runs[run_id].save()

    # ── querying ──────────────────────────────────────────────────

    def get_run(self, run_id: str) -> RunManifest:
        """Return the manifest for a specific run_id."""
        if run_id not in self._runs:
            raise KeyError(f"Run not found: {run_id!r}. "
                           f"Known runs: {list(self._runs)[:5]}…")
        return self._runs[run_id]

    def list_runs(self) -> list[RunManifest]:
        """All runs, newest first."""
        return sorted(self._runs.values(), key=lambda m: m.started_at, reverse=True)

    def query(
        self,
        *,
        benchmark: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[RunManifest]:
        """Filter runs by one or more criteria (all are optional).

        Args:
            benchmark: "webarena" | "gaia" | "humaneval" | "alfworld"
            provider:  "openai" | "anthropic" | "google" | "deepseek"
            model:     Substring match on model_name (e.g. "gpt-4o" matches
                       "gpt-4o-mini" and "gpt-4o").
            status:    "done" | "running" | "failed" | "partial"
            tag:       Exact match on any element of manifest.tags.
        """
        results = self.list_runs()
        if benchmark:
            results = [m for m in results if m.benchmark == benchmark]
        if provider:
            results = [m for m in results if m.provider == provider]
        if model:
            results = [m for m in results if model in m.model_name]
        if status:
            results = [m for m in results if m.status == status]
        if tag:
            results = [m for m in results if tag in m.tags]
        return results

    def latest_run(
        self,
        benchmark: str,
        model: str,
        status: str = "done",
    ) -> Optional[RunManifest]:
        """Return the most recent completed run for a given benchmark+model."""
        runs = self.query(benchmark=benchmark, model=model, status=status)
        return runs[0] if runs else None

    # ── data access ───────────────────────────────────────────────

    def iter_episodes(
        self,
        run_ids: list[str],
        *,
        deduplicate: bool = True,
    ) -> Iterator[Episode]:
        """Iterate episodes from the given run IDs.

        Args:
            run_ids:      List of run IDs to include.
            deduplicate:  If True, skip episodes with a duplicate
                          trajectory_hash (same action sequence).
                          Keeps the first occurrence (oldest run).
        """
        seen_hashes: set[str] = set()
        for run_id in run_ids:
            manifest = self.get_run(run_id)
            store = TrajectoryStore(manifest.trajectory_path)
            for ep in store.iter_episodes():
                if deduplicate:
                    h = ep.trajectory_hash
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                yield ep

    def merge_runs(
        self,
        run_ids: list[str],
        *,
        output_path: Optional[Path] = None,
        deduplicate: bool = True,
    ) -> TrajectoryStore:
        """Merge episodes from multiple runs into a single TrajectoryStore.

        If ``output_path`` is given, the merged dataset is written there.
        Otherwise, episodes are stored in memory via a temporary file under
        ``data/merged/``.

        Args:
            run_ids:       Runs to merge (order matters for deduplication).
            output_path:   Where to write the merged JSONL.
            deduplicate:   Remove duplicate trajectories (by action hash).
        """
        if output_path is None:
            slug = "_".join(run_ids[:3]) + ("_etc" if len(run_ids) > 3 else "")
            output_path = Path("data/merged") / f"{slug}.jsonl"

        merged_store = TrajectoryStore(output_path)
        # Wipe existing file so we don't double-append
        if output_path.exists():
            output_path.unlink()

        n = 0
        for ep in self.iter_episodes(run_ids, deduplicate=deduplicate):
            merged_store.save_episode(ep)
            n += 1
        return merged_store

    # ── display ───────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable table of all registered runs."""
        rows = self.list_runs()
        if not rows:
            return "No runs registered yet."
        lines = [
            f"{'RUN ID':<55} {'BENCH':<10} {'MODEL':<35} {'STATUS':<9} "
            f"{'EPS':>5} {'SR':>6} {'COST':>8}",
            "-" * 130,
        ]
        for m in rows:
            sr_str = f"{m.success_rate:.2%}" if m.n_episodes > 0 else "-"
            cost_str = f"${m.total_cost_usd:.3f}" if m.total_cost_usd > 0 else "-"
            lines.append(
                f"{m.run_id:<55} {m.benchmark:<10} {m.model_name:<35} "
                f"{m.status:<9} {m.n_episodes:>5} {sr_str:>6} {cost_str:>8}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"DataRegistry(path={self.registry_path}, "
                f"n_runs={len(self._runs)})")
