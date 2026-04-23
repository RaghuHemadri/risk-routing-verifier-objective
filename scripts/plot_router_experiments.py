#!/usr/bin/env python3
"""Generate plots and LaTeX/CSV tables from router experiment outputs.

Reads:
- results/router_experiments/metrics_long.csv
- results/router_experiments/experiment_manifest.jsonl (optional)

Writes:
- results/router_experiments/plots/*.png
- results/router_experiments/tables/*.csv
- results/router_experiments/tables/*.tex
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and tabulate router experiments")
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/router_experiments_single",
        help="Root directory produced by run_router_experiments.sh",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Optional explicit metrics CSV path",
    )
    return parser.parse_args()


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    rows: list[dict] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for k in (
                "success_rate",
                "success_rate_ci_low",
                "success_rate_ci_high",
                "worst_seed_sr",
                "cvar_failure",
                "avg_cost",
                "llm_call_rate",
                "ece",
                "brier",
                "alpha",
                "epsilon",
            ):
                parsed[k] = _to_float(parsed.get(k))

            method = (parsed.get("method") or "").strip()
            threshold = None
            if method.startswith("r2v@"):
                try:
                    threshold = float(method.split("@", 1)[1])
                except ValueError:
                    threshold = None
            parsed["threshold"] = threshold
            rows.append(parsed)
    return rows


def ensure_dirs(results_root: Path) -> tuple[Path, Path]:
    plots_dir = results_root / "plots"
    tables_dir = results_root / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, tables_dir


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, headers: list[str], rows: list[list[str]], caption: str, label: str) -> None:
    colspec = "l" + "r" * (len(headers) - 1)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(row) + " \\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_latest(rows: list[dict]) -> list[dict]:
    """Keep latest row per (category, variant, benchmark, model, method, alpha, epsilon)."""
    latest: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("category"),
            row.get("variant"),
            row.get("benchmark"),
            row.get("model"),
            row.get("method"),
            row.get("alpha"),
            row.get("epsilon"),
        )
        latest[key] = row
    return list(latest.values())


def build_main_table(rows: list[dict], tables_dir: Path) -> None:
    main_rows = [r for r in rows if r.get("category") == "main" and r.get("variant") == "main_results"]
    if not main_rows:
        return

    out_rows: list[dict] = []
    for r in sorted(main_rows, key=lambda x: (x.get("benchmark", ""), x.get("model", ""), x.get("method", ""))):
        out_rows.append(
            {
                "benchmark": r.get("benchmark"),
                "model": r.get("model"),
                "method": r.get("method"),
                "sr": f"{r.get('success_rate', float('nan')):.4f}",
                "cvar_f": f"{r.get('cvar_failure', float('nan')):.4f}",
                "cost": f"{r.get('avg_cost', float('nan')):.4f}",
                "llm_rate": f"{r.get('llm_call_rate', float('nan')):.4f}",
            }
        )

    write_csv(
        tables_dir / "main_results.csv",
        out_rows,
        ["benchmark", "model", "method", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["method"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "main_results.tex",
        ["Benchmark", "Model", "Method", "SR", "CVaR-F", "Cost", "LLM\\%"],
        latex_rows,
        caption="Main router results across benchmarks.",
        label="tab:router-main-results",
    )


def build_feature_ablation_table(rows: list[dict], tables_dir: Path) -> None:
    feats = [r for r in rows if r.get("category") == "feature_ablation"]
    if not feats:
        return

    out_rows: list[dict] = []
    for r in sorted(feats, key=lambda x: (x.get("benchmark", ""), x.get("model", ""), x.get("variant", ""), x.get("method", ""))):
        out_rows.append(
            {
                "benchmark": r.get("benchmark"),
                "model": r.get("model"),
                "variant": r.get("variant"),
                "method": r.get("method"),
                "sr": f"{r.get('success_rate', float('nan')):.4f}",
                "cvar_f": f"{r.get('cvar_failure', float('nan')):.4f}",
                "cost": f"{r.get('avg_cost', float('nan')):.4f}",
                "llm_rate": f"{r.get('llm_call_rate', float('nan')):.4f}",
            }
        )

    write_csv(
        tables_dir / "feature_ablations.csv",
        out_rows,
        ["benchmark", "model", "variant", "method", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["variant"], r["method"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "feature_ablations.tex",
        ["Benchmark", "Model", "Variant", "Method", "SR", "CVaR-F", "Cost", "LLM\\%"],
        latex_rows,
        caption="Router feature ablation results.",
        label="tab:router-feature-ablations",
    )


def build_main_ablation_table(rows: list[dict], tables_dir: Path) -> None:
    ab_rows = [r for r in rows if r.get("category") == "main_ablation"]
    if not ab_rows:
        return

    out_rows: list[dict] = []
    for r in sorted(ab_rows, key=lambda x: (x.get("benchmark", ""), x.get("model", ""), x.get("variant", ""), x.get("method", ""))):
        out_rows.append(
            {
                "benchmark": r.get("benchmark"),
                "model": r.get("model"),
                "variant": r.get("variant"),
                "method": r.get("method"),
                "sr": f"{r.get('success_rate', float('nan')):.4f}",
                "cvar_f": f"{r.get('cvar_failure', float('nan')):.4f}",
                "cost": f"{r.get('avg_cost', float('nan')):.4f}",
                "llm_rate": f"{r.get('llm_call_rate', float('nan')):.4f}",
            }
        )

    write_csv(
        tables_dir / "main_ablations.csv",
        out_rows,
        ["benchmark", "model", "variant", "method", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["variant"], r["method"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "main_ablations.tex",
        ["Benchmark", "Model", "Variant", "Method", "SR", "CVaR-F", "Cost", "LLM\\%"],
        latex_rows,
        caption="Main ablations (no DPO, no BC/no DPO, lambda variants).",
        label="tab:router-main-ablations",
    )


def build_cvar_sweep_table(rows: list[dict], tables_dir: Path) -> None:
    sweep_rows = [r for r in rows if r.get("category") == "cvar_sweep" and r.get("method") == "r2v"]
    if not sweep_rows:
        return

    out_rows: list[dict] = []
    for r in sorted(
        sweep_rows,
        key=lambda x: (
            x.get("benchmark", ""),
            x.get("model", ""),
            x.get("alpha") if x.get("alpha") is not None else math.inf,
            x.get("epsilon") if x.get("epsilon") is not None else math.inf,
        ),
    ):
        out_rows.append(
            {
                "benchmark": r.get("benchmark"),
                "model": r.get("model"),
                "alpha": f"{r.get('alpha', float('nan')):.3f}",
                "epsilon": f"{r.get('epsilon', float('nan')):.3f}",
                "sr": f"{r.get('success_rate', float('nan')):.4f}",
                "cvar_f": f"{r.get('cvar_failure', float('nan')):.4f}",
                "cost": f"{r.get('avg_cost', float('nan')):.4f}",
                "llm_rate": f"{r.get('llm_call_rate', float('nan')):.4f}",
            }
        )

    write_csv(
        tables_dir / "cvar_sweeps.csv",
        out_rows,
        ["benchmark", "model", "alpha", "epsilon", "sr", "cvar_f", "cost", "llm_rate"],
    )

    latex_rows = [
        [r["benchmark"], r["model"], r["alpha"], r["epsilon"], r["sr"], r["cvar_f"], r["cost"], r["llm_rate"]]
        for r in out_rows
    ]
    write_latex_table(
        tables_dir / "cvar_sweeps.tex",
        ["Benchmark", "Model", "Alpha", "Epsilon", "SR", "CVaR-F", "Cost", "LLM\\%"],
        latex_rows,
        caption="CVaR alpha/epsilon sensitivity sweeps.",
        label="tab:router-cvar-sweeps",
    )


def plot_figure_tradeoff(rows: list[dict], plots_dir: Path) -> None:
    fig_rows = [r for r in rows if r.get("category") == "figure"]
    if not fig_rows:
        return

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in fig_rows:
        grouped[(str(r.get("benchmark")), str(r.get("model")))].append(r)

    for (benchmark, model), group in grouped.items():
        plt.figure(figsize=(7, 5))
        methods = ["slm_only", "llm_only", "entropy_router"]

        for m in methods:
            mm = [r for r in group if r.get("method") == m]
            if not mm:
                continue
            r0 = mm[-1]
            x = r0.get("llm_call_rate")
            y = r0.get("success_rate")
            if x is None or y is None:
                continue
            plt.scatter([x], [y], s=90, label=m)

        r2v_points = [r for r in group if str(r.get("method", "")).startswith("r2v@")]
        if r2v_points:
            r2v_points = sorted(r2v_points, key=lambda r: (r.get("threshold") if r.get("threshold") is not None else math.inf))
            xs = [r["llm_call_rate"] for r in r2v_points if r.get("llm_call_rate") is not None]
            ys = [r["success_rate"] for r in r2v_points if r.get("success_rate") is not None]
            if xs and ys:
                plt.plot(xs, ys, marker="o", linewidth=2, label="r2v threshold sweep")

        plt.xlabel("LLM call rate")
        plt.ylabel("Success rate")
        plt.title(f"Method tradeoff: {benchmark} / {model}")
        plt.grid(alpha=0.3)
        plt.legend()
        out = plots_dir / f"figure_tradeoff_{benchmark}_{model}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close()


def plot_feature_ablation_sr(rows: list[dict], plots_dir: Path) -> None:
    feat_rows = [r for r in rows if r.get("category") == "feature_ablation" and r.get("method") in {"r2v", "verifier_router"}]
    if not feat_rows:
        return

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in feat_rows:
        grouped[(str(r.get("benchmark")), str(r.get("model")))].append(r)

    for (benchmark, model), group in grouped.items():
        group = sorted(group, key=lambda r: str(r.get("variant")))
        labels = [str(r.get("variant")) for r in group]
        vals = [r.get("success_rate") for r in group]

        if not any(v is not None for v in vals):
            continue

        xs = list(range(len(labels)))
        ys = [v if v is not None else 0.0 for v in vals]

        plt.figure(figsize=(max(8, 0.8 * len(labels)), 5))
        plt.bar(xs, ys)
        plt.xticks(xs, labels, rotation=35, ha="right")
        plt.ylabel("Success rate")
        plt.title(f"Feature ablation SR: {benchmark} / {model}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = plots_dir / f"feature_ablation_sr_{benchmark}_{model}.png"
        plt.savefig(out, dpi=220)
        plt.close()


def plot_cvar_sweeps(rows: list[dict], plots_dir: Path) -> None:
    sweep_rows = [r for r in rows if r.get("category") == "cvar_sweep" and r.get("method") == "r2v"]
    if not sweep_rows:
        return

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in sweep_rows:
        grouped[(str(r.get("benchmark")), str(r.get("model")))].append(r)

    metrics = [
        ("success_rate", "SR"),
        ("cvar_failure", "CVaR-F"),
        ("avg_cost", "Cost"),
        ("llm_call_rate", "LLM%"),
    ]

    for (benchmark, model), group in grouped.items():
        eps_values = sorted({r.get("epsilon") for r in group if r.get("epsilon") is not None})
        if not eps_values:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()

        for ax, (metric_key, metric_title) in zip(axes, metrics):
            for eps in eps_values:
                subset = [r for r in group if r.get("epsilon") == eps and r.get("alpha") is not None and r.get(metric_key) is not None]
                subset = sorted(subset, key=lambda r: float(r["alpha"]))
                if not subset:
                    continue
                xs = [float(r["alpha"]) for r in subset]
                ys = [float(r[metric_key]) for r in subset]
                ax.plot(xs, ys, marker="o", label=f"epsilon={eps:g}")

            ax.set_title(metric_title)
            ax.set_xlabel("alpha")
            ax.grid(alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))

        fig.suptitle(f"CVaR sweeps: {benchmark} / {model}")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = plots_dir / f"cvar_sweeps_{benchmark}_{model}.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    metrics_path = Path(args.metrics_path) if args.metrics_path else (results_root / "metrics_long.csv")

    rows = load_rows(metrics_path)
    rows = aggregate_latest(rows)

    plots_dir, tables_dir = ensure_dirs(results_root)

    build_main_table(rows, tables_dir)
    build_feature_ablation_table(rows, tables_dir)
    build_main_ablation_table(rows, tables_dir)
    build_cvar_sweep_table(rows, tables_dir)

    plot_figure_tradeoff(rows, plots_dir)
    plot_feature_ablation_sr(rows, plots_dir)
    plot_cvar_sweeps(rows, plots_dir)

    print(f"Saved plots to: {plots_dir}")
    print(f"Saved tables to: {tables_dir}")


if __name__ == "__main__":
    main()
