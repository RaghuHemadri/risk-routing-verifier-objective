"""
Structured results management for R2V-Agent.

Stores experiment results in machine-readable formats (JSON, JSONL, CSV)
that can be fed to LLM agents for automated paper writing.

Design:
- Each experiment run produces a ResultsBundle
- ResultsBundle contains config, training curves, evaluation results,
  ablation comparisons, and statistical tests
- ResultsManager handles I/O and aggregation across runs
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TrainingCurve:
    """Training metrics over time."""
    steps: list[int] = field(default_factory=list)
    losses: dict[str, list[float]] = field(default_factory=dict)
    eval_metrics: dict[str, list[float]] = field(default_factory=dict)

    def add_step(self, step: int, loss_dict: dict[str, float]) -> None:
        self.steps.append(step)
        for k, v in loss_dict.items():
            if k not in self.losses:
                self.losses[k] = []
            self.losses[k].append(v)

    def add_eval(self, metric_dict: dict[str, float]) -> None:
        for k, v in metric_dict.items():
            if k not in self.eval_metrics:
                self.eval_metrics[k] = []
            self.eval_metrics[k].append(v)


@dataclass
class EvalResult:
    """Evaluation result for one method on one benchmark."""
    method: str
    benchmark: str
    condition: str  # "clean" or "noisy"
    seed: int

    # Primary metrics
    success_rate: float
    success_rate_ci: tuple[float, float] | None = None

    # Robustness metrics
    worst_seed_sr: float | None = None
    cvar_failure: float | None = None
    bottom_10_sr: float | None = None
    robustness_gap: float | None = None

    # Efficiency metrics
    avg_cost: float | None = None
    avg_latency: float | None = None
    llm_call_rate: float | None = None

    # Safety
    safety_failure_rate: float | None = None

    # Calibration
    ece: float | None = None
    brier: float | None = None

    # Per-task details (optional)
    per_task_results: dict[str, Any] | None = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ComparisonResult:
    """Statistical comparison between two methods."""
    method_a: str
    method_b: str
    benchmark: str
    condition: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    effect_size: float | None = None
    test_name: str = "mcnemar"


@dataclass
class AblationResult:
    """Result of an ablation experiment."""
    name: str
    description: str
    base_method: str
    ablated_method: str
    benchmark: str
    condition: str
    base_sr: float
    ablated_sr: float
    delta: float
    delta_ci: tuple[float, float] | None = None
    p_value: float | None = None


@dataclass
class ResultsBundle:
    """Complete results for one experiment run.

    This is the main output format. Write one per experiment run,
    then aggregate with ResultsManager for paper tables/figures.
    """
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    config: dict[str, Any] = field(default_factory=dict)
    training_curves: dict[str, TrainingCurve] = field(default_factory=dict)
    eval_results: list[EvalResult] = field(default_factory=list)
    comparisons: list[ComparisonResult] = field(default_factory=list)
    ablations: list[AblationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save full bundle to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "ResultsBundle":
        """Load bundle from JSON."""
        with open(path) as f:
            data = json.load(f)
        bundle = cls(experiment_name=data["experiment_name"])
        bundle.timestamp = data.get("timestamp", "")
        bundle.config = data.get("config", {})
        bundle.metadata = data.get("metadata", {})

        # Reconstruct eval results
        for er in data.get("eval_results", []):
            bundle.eval_results.append(EvalResult(**{
                k: v for k, v in er.items()
                if k in EvalResult.__dataclass_fields__
            }))
        for cr in data.get("comparisons", []):
            bundle.comparisons.append(ComparisonResult(**{
                k: v for k, v in cr.items()
                if k in ComparisonResult.__dataclass_fields__
            }))
        for ab in data.get("ablations", []):
            bundle.ablations.append(AblationResult(**{
                k: v for k, v in ab.items()
                if k in AblationResult.__dataclass_fields__
            }))
        return bundle


class ResultsManager:
    """Manages experiment results across multiple runs.

    Provides utilities to:
    - Save/load individual run results
    - Generate summary tables (CSV, LaTeX)
    - Produce structured data for LLM paper writing
    """

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_bundle(self, bundle: ResultsBundle) -> Path:
        """Save a results bundle and return its path."""
        safe_name = bundle.experiment_name.replace("/", "_").replace(" ", "_")
        path = self.results_dir / f"{safe_name}_{bundle.timestamp[:10]}.json"
        bundle.save(path)
        return path

    def load_all_bundles(self) -> list[ResultsBundle]:
        """Load all results bundles from the results directory."""
        bundles = []
        for path in sorted(self.results_dir.glob("*.json")):
            try:
                bundles.append(ResultsBundle.load(path))
            except Exception:
                continue
        return bundles

    def generate_main_table_csv(self, output_path: str | Path | None = None) -> str:
        """Generate the main results table (Table 1 in paper).

        Columns: Method, Benchmark, Condition, SR, SR-CI, Worst-Seed,
                 CVaR-Fail, Cost, Latency, Safety-Fail-Rate
        """
        bundles = self.load_all_bundles()
        rows = []
        for bundle in bundles:
            for er in bundle.eval_results:
                ci_str = ""
                if er.success_rate_ci:
                    ci_str = f"[{er.success_rate_ci[0]:.3f}, {er.success_rate_ci[1]:.3f}]"
                rows.append({
                    "method": er.method,
                    "benchmark": er.benchmark,
                    "condition": er.condition,
                    "seed": er.seed,
                    "success_rate": f"{er.success_rate:.4f}",
                    "success_rate_ci": ci_str,
                    "worst_seed_sr": f"{er.worst_seed_sr:.4f}" if er.worst_seed_sr is not None else "",
                    "cvar_failure": f"{er.cvar_failure:.4f}" if er.cvar_failure is not None else "",
                    "bottom_10_sr": f"{er.bottom_10_sr:.4f}" if er.bottom_10_sr is not None else "",
                    "avg_cost": f"{er.avg_cost:.4f}" if er.avg_cost is not None else "",
                    "avg_latency": f"{er.avg_latency:.4f}" if er.avg_latency is not None else "",
                    "llm_call_rate": f"{er.llm_call_rate:.4f}" if er.llm_call_rate is not None else "",
                    "safety_failure_rate": f"{er.safety_failure_rate:.4f}" if er.safety_failure_rate is not None else "",
                    "ece": f"{er.ece:.4f}" if er.ece is not None else "",
                    "brier": f"{er.brier:.4f}" if er.brier is not None else "",
                })

        if output_path is None:
            output_path = self.results_dir / "main_table.csv"

        output_path = Path(output_path)
        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return str(output_path)

    def generate_comparison_csv(self, output_path: str | Path | None = None) -> str:
        """Generate pairwise comparison table for statistical tests."""
        bundles = self.load_all_bundles()
        rows = []
        for bundle in bundles:
            for comp in bundle.comparisons:
                rows.append(asdict(comp))

        if output_path is None:
            output_path = self.results_dir / "comparisons.csv"

        output_path = Path(output_path)
        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return str(output_path)

    def generate_ablation_csv(self, output_path: str | Path | None = None) -> str:
        """Generate ablation study table."""
        bundles = self.load_all_bundles()
        rows = []
        for bundle in bundles:
            for ab in bundle.ablations:
                rows.append(asdict(ab))

        if output_path is None:
            output_path = self.results_dir / "ablations.csv"

        output_path = Path(output_path)
        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return str(output_path)

    def generate_llm_summary(self, output_path: str | Path | None = None) -> str:
        """Generate a structured JSON summary optimized for LLM paper writing.

        Produces a single file with all results organized by:
        1. Main results table
        2. Statistical comparisons
        3. Ablation studies
        4. Key findings (auto-extracted)

        This output is designed to be copy-pasted into an LLM prompt
        for automated paper section writing.
        """
        bundles = self.load_all_bundles()

        # Aggregate eval results by method
        methods = {}
        for bundle in bundles:
            for er in bundle.eval_results:
                key = (er.method, er.benchmark, er.condition)
                if key not in methods:
                    methods[key] = []
                methods[key].append(er)

        # Build summary
        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "num_experiments": len(bundles),
            "main_results": [],
            "comparisons": [],
            "ablations": [],
            "key_findings": [],
        }

        # Main results (aggregate across seeds)
        for (method, benchmark, condition), ers in methods.items():
            srs = [er.success_rate for er in ers]
            import numpy as np
            summary["main_results"].append({
                "method": method,
                "benchmark": benchmark,
                "condition": condition,
                "num_seeds": len(ers),
                "mean_sr": float(np.mean(srs)),
                "std_sr": float(np.std(srs)),
                "min_sr": float(np.min(srs)),
                "max_sr": float(np.max(srs)),
                "avg_cost": float(np.mean([er.avg_cost for er in ers if er.avg_cost])) if any(er.avg_cost for er in ers) else None,
                "avg_safety_failure_rate": float(np.mean([er.safety_failure_rate for er in ers if er.safety_failure_rate is not None])) if any(er.safety_failure_rate is not None for er in ers) else None,
            })

        # Comparisons
        for bundle in bundles:
            for comp in bundle.comparisons:
                summary["comparisons"].append({
                    "method_a": comp.method_a,
                    "method_b": comp.method_b,
                    "metric": comp.metric,
                    "benchmark": comp.benchmark,
                    "difference": comp.difference,
                    "ci": [comp.ci_lower, comp.ci_upper],
                    "p_value": comp.p_value,
                    "significant": comp.significant,
                })

        # Ablations
        for bundle in bundles:
            for ab in bundle.ablations:
                summary["ablations"].append({
                    "name": ab.name,
                    "description": ab.description,
                    "delta": ab.delta,
                    "p_value": ab.p_value,
                })

        # Auto-extract key findings
        for res in summary["main_results"]:
            if res["condition"] == "noisy" and res["mean_sr"] > 0:
                summary["key_findings"].append(
                    f"{res['method']} achieves {res['mean_sr']:.1%} SR on {res['benchmark']} (noisy)"
                )

        if output_path is None:
            output_path = self.results_dir / "llm_summary.json"

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return str(output_path)

    def generate_latex_table(self, output_path: str | Path | None = None) -> str:
        """Generate LaTeX table for paper inclusion."""
        bundles = self.load_all_bundles()

        # Collect best results per method/benchmark/condition
        best = {}
        for bundle in bundles:
            for er in bundle.eval_results:
                key = (er.method, er.benchmark, er.condition)
                if key not in best or er.success_rate > best[key].success_rate:
                    best[key] = er

        # Generate LaTeX
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Main results across benchmarks and conditions.}",
            r"\label{tab:main-results}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lcccccccc}",
            r"\toprule",
            r"Method & Bench & Cond & SR$\uparrow$ & Worst-Seed$\uparrow$ & CVaR-Fail$\downarrow$ & Cost$\downarrow$ & Safety$\downarrow$ & ECE$\downarrow$ \\",
            r"\midrule",
        ]

        for (method, bench, cond), er in sorted(best.items()):
            sr = f"{er.success_rate:.3f}"
            ws = f"{er.worst_seed_sr:.3f}" if er.worst_seed_sr is not None else "--"
            cv = f"{er.cvar_failure:.3f}" if er.cvar_failure is not None else "--"
            co = f"{er.avg_cost:.2f}" if er.avg_cost is not None else "--"
            sf = f"{er.safety_failure_rate:.3f}" if er.safety_failure_rate is not None else "--"
            ece = f"{er.ece:.3f}" if er.ece is not None else "--"
            lines.append(
                f"{method} & {bench} & {cond} & {sr} & {ws} & {cv} & {co} & {sf} & {ece} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ])

        latex = "\n".join(lines)

        if output_path is None:
            output_path = self.results_dir / "main_table.tex"

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            f.write(latex)

        return str(output_path)
