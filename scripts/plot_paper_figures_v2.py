#!/usr/bin/env python3
"""Generate publication-quality figures for R2V-Agent NeurIPS 2026 paper.

Reads v2 CSV tables from results/paper_experiments_v2/tables/
and metrics_long.csv, writes PDF + PNG to results/paper_experiments_v2/plots_v2/.

Usage:
    source ~/.venv/bin/activate
    python scripts/plot_paper_figures_v2.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Global style — NeurIPS-appropriate: 10pt base, compact, no serif
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ---------------------------------------------------------------------------
# Colour / marker palette — consistent across all figures
# ---------------------------------------------------------------------------
_C = {
    "r2v":              "#2166AC",   # blue
    "slm_only":         "#999999",   # gray
    "llm_only":         "#333333",   # dark gray
    "entropy_router":   "#D6604D",   # red-orange
    "heuristic_router": "#4DAC26",   # green
    "oracle_router":    "#762A83",   # purple
}
_LABEL = {
    "r2v":              "R2V (ours)",
    "slm_only":         "SLM-only",
    "llm_only":         "LLM-only",
    "entropy_router":   "Entropy-Router",
    "heuristic_router": "Heuristic-Router",
    "oracle_router":    "Oracle-Router",
}
_MARKER = {
    "r2v":              "D",   # diamond
    "slm_only":         "s",   # square
    "llm_only":         "^",   # triangle up
    "entropy_router":   "o",   # circle
    "heuristic_router": "v",   # triangle down
    "oracle_router":    "*",   # star
}
_MODEL_LABEL = {
    "gemma":  "Gemma-7B",
    "llama":  "LLaMA-3.1-8B",
    "qwen7":  "Qwen-7B",
    "qwen14": "Qwen-14B",
}
_MODEL_COLORS = ["#2166AC", "#5AAE61", "#F4A582", "#762A83"]  # 4 distinct model colours

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
TABLES = ROOT / "results" / "paper_experiments_v2" / "tables"
METRICS = ROOT / "results" / "paper_experiments_v2" / "metrics_long.csv"
OUT = ROOT / "results" / "paper_experiments_v2" / "plots_v2"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return [r for r in reader]


def _flt(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p)
    print(f"  Saved {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 1 — Pareto frontier: SR vs LLM call rate (two benchmarks)
# ---------------------------------------------------------------------------
def plot_pareto(rows: list[dict]) -> None:
    """Two-panel scatter plot: HumanEval+ (left) and TextWorld (right)."""
    benchmarks = [("HumanEval+", "humaneval"), ("TextWorld", "textworld")]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0))

    for ax, (bench_label, bench_key) in zip(axes, benchmarks):
        bench_rows = [r for r in rows
                      if r["benchmark"].lower() == bench_key
                      and r["method"].lower() != "llm_only"]

        # Group by method key
        by_method: dict[str, list[dict]] = {}
        for r in bench_rows:
            m = r["method"].lower().replace("-", "_").replace(" ", "_")
            by_method.setdefault(m, []).append(r)

        # SLM-only, entropy, oracle as single reference points (avg over models)
        for mkey in ("slm_only", "entropy_router", "oracle_router"):
            pts = by_method.get(mkey, [])
            if not pts:
                continue
            sr_vals  = [_flt(r["sr"]) for r in pts]
            lr_vals  = [_flt(r["llm_rate"]) for r in pts]
            sr  = np.mean(sr_vals)
            lr  = np.mean(lr_vals)
            ax.scatter(lr * 100, sr * 100,
                       color=_C[mkey], marker=_MARKER[mkey],
                       s=60, zorder=5,
                       label=_LABEL[mkey])

        # Heuristic router — show spread across models
        pts = by_method.get("heuristic_router", [])
        if pts:
            sr_vals = [_flt(r["sr"]) for r in pts]
            lr_vals = [_flt(r["llm_rate"]) for r in pts]
            ax.scatter([lr * 100 for lr in lr_vals],
                       [sr * 100 for sr in sr_vals],
                       color=_C["heuristic_router"], marker=_MARKER["heuristic_router"],
                       s=40, alpha=0.7, zorder=4,
                       label=_LABEL["heuristic_router"])

        # R2V — coloured by model
        model_order = ["gemma", "llama", "qwen7", "qwen14"]
        r2v_pts = by_method.get("r2v", [])
        r2v_by_model = {r["model"].lower(): r for r in r2v_pts}
        for idx, model_key in enumerate(model_order):
            r = r2v_by_model.get(model_key)
            if r is None:
                continue
            sr  = _flt(r["sr"])
            lr  = _flt(r["llm_rate"])
            ci_lo = _flt(r.get("sr_ci", "[0,0]").strip("[]").split(",")[0])
            ci_hi = _flt(r.get("sr_ci", "[0,0]").strip("[]").split(",")[1])
            err_lo = (sr - ci_lo) * 100
            err_hi = (ci_hi - sr) * 100
            col = _MODEL_COLORS[idx]
            ax.errorbar(lr * 100, sr * 100,
                        yerr=[[err_lo], [err_hi]],
                        fmt=_MARKER["r2v"], color=col,
                        markersize=7, capsize=3, linewidth=1.2,
                        zorder=6,
                        label=f"R2V – {_MODEL_LABEL.get(model_key, model_key)}")

        ax.set_title(bench_label, fontweight="bold")
        ax.set_xlabel("LLM escalation rate (%)")
        ax.set_ylabel("Task success rate (%)")

        # Y-axis range
        if bench_key == "textworld":
            ax.set_ylim(58, 103)
        else:
            ax.set_ylim(88, 103)

    # Build single legend under both panels
    handles, labels = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    # Merge unique entries
    seen = set()
    merged_h, merged_l = [], []
    for h, lb in list(zip(handles, labels)) + list(zip(h1, l1)):
        if lb not in seen:
            seen.add(lb)
            merged_h.append(h)
            merged_l.append(lb)

    # Group: baselines first, then R2V models
    base_idx = [i for i, lb in enumerate(merged_l)
                if lb in (_LABEL["slm_only"], _LABEL["entropy_router"],
                          _LABEL["oracle_router"], _LABEL["heuristic_router"])]
    r2v_idx  = [i for i, lb in enumerate(merged_l) if lb not in
                {merged_l[j] for j in base_idx}]
    order = base_idx + r2v_idx
    merged_h = [merged_h[i] for i in order]
    merged_l = [merged_l[i] for i in order]

    fig.legend(merged_h, merged_l,
               loc="lower center",
               ncol=4,
               bbox_to_anchor=(0.5, -0.12),
               frameon=True,
               fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, "figure1_pareto")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — λ_cons sweep: SR and LLM-rate vs λ for HumanEval+ (+ TextWorld)
# ---------------------------------------------------------------------------
def plot_lambda_sweep(metrics_path: Path) -> None:
    """Two-panel: (a) SR vs λ, (b) LLM-call-rate vs λ for HumanEval+.
    TextWorld results are constant and shown as a reference annotation.
    """
    _LAMBDA_MAP = {
        "no_consistency": 0.0,
        "lam_0_05": 0.05, "consistency_lambda_0_05": 0.05,
        "lam_0_2":  0.20, "consistency_lambda_0_2":  0.20,
        "lam_0_5":  0.50, "consistency_lambda_0_5":  0.50,
        "lam_1_0":  1.00, "consistency_lambda_1_0":  1.00,
    }

    data: dict[tuple, dict] = {}  # (benchmark, model) -> {lam: (sr, llm_rate)}
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] != "lambda_ablation":
                continue
            if row["method"].lower() != "r2v":
                continue
            variant = row["variant"]
            if variant not in _LAMBDA_MAP:
                continue
            lam = _LAMBDA_MAP[variant]
            bench = row["benchmark"].lower()
            model = row["model"].lower()
            sr    = _flt(row["success_rate"])
            lr    = _flt(row["llm_call_rate"])
            key   = (bench, model)
            data.setdefault(key, {})[lam] = (sr, lr)

    lambda_vals = sorted({0.0, 0.05, 0.20, 0.50, 1.00})

    fig, (ax_sr, ax_lr) = plt.subplots(1, 2, figsize=(6.75, 3.0))

    model_styles = {
        ("humaneval", "qwen7"):  ("Qwen-7B",   _MODEL_COLORS[0], "o-"),
        ("humaneval", "qwen14"): ("Qwen-14B",  _MODEL_COLORS[3], "s--"),
        ("textworld", "qwen7"):  ("TextWorld / Qwen-7B",  _MODEL_COLORS[1], "^:"),
        ("textworld", "qwen14"): ("TextWorld / Qwen-14B", _MODEL_COLORS[2], "v:"),
    }

    for key, style_info in model_styles.items():
        label, col, style = style_info
        pts = data.get(key, {})
        if not pts:
            continue
        xs = [l for l in lambda_vals if l in pts]
        srs  = [pts[l][0] * 100 for l in xs]
        lrs  = [pts[l][1] * 100 for l in xs]
        ax_sr.plot(xs, srs, style, color=col, label=label, markersize=5)
        ax_lr.plot(xs, lrs, style, color=col, label=label, markersize=5)

    ax_sr.set_xlabel(r"$\lambda_{\mathrm{cons}}$")
    ax_sr.set_ylabel("Task success rate (%)")
    ax_sr.set_title(r"(a) SR vs. $\lambda_{\mathrm{cons}}$", fontweight="bold")
    ax_sr.set_xticks(lambda_vals)

    ax_lr.set_xlabel(r"$\lambda_{\mathrm{cons}}$")
    ax_lr.set_ylabel("LLM escalation rate (%)")
    ax_lr.set_title(r"(b) LLM rate vs. $\lambda_{\mathrm{cons}}$", fontweight="bold")
    ax_lr.set_xticks(lambda_vals)

    handles, labels = ax_sr.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, "figure2_lambda_sweep")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Feature group ablation (horizontal bar charts)
# ---------------------------------------------------------------------------
def plot_feature_ablation(rows: list[dict]) -> None:
    """Two sub-figures: HumanEval+ and TextWorld feature ablation."""
    bench_data = {}
    for r in rows:
        bench = r["benchmark"]
        feat  = r.get("features") or r.get("variant") or r.get("feature_set", "?")
        sr    = _flt(r.get("sr", "nan"))
        llm   = _flt(r.get("llm_rate", "nan"))
        cost  = _flt(r.get("cost", "nan"))
        bench_data.setdefault(bench, []).append((feat, sr, llm, cost))

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.5))

    _FEAT_ORDER = [
        "All features",
        "Verifier + entropy",
        "Verifier only",
        "Entropy only",
        "Log-prob only",
        "Step/context only",
        "w/o verifier",
        "w/o entropy",
    ]

    _FEAT_COLOR = {
        "All features":      "#2166AC",
        "Verifier + entropy":"#5AAE61",
        "Verifier only":     "#4DAC26",
        "Entropy only":      "#D6604D",
        "Log-prob only":     "#762A83",
        "Step/context only": "#F4A582",
        "w/o verifier":      "#92C5DE",
        "w/o entropy":       "#80CDC1",
    }

    bench_titles = {"HumanEval+": "(a) HumanEval+", "TextWorld": "(b) TextWorld"}

    for ax, (bench, pts) in zip(axes, sorted(bench_data.items())):
        # Sort by the predefined order
        ordered = sorted(pts, key=lambda x: _FEAT_ORDER.index(x[0])
                         if x[0] in _FEAT_ORDER else 99)

        labels = [p[0] for p in ordered]
        srs    = [p[1] * 100 for p in ordered]
        colors = [_FEAT_COLOR.get(l, "#aaaaaa") for l in labels]

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, srs, color=colors, height=0.65, edgecolor="white")

        # Reference line: SLM-only and Oracle
        slm_sr = 91.9 if "Human" in bench else 65.2
        oracle_sr = 100.0
        ax.axvline(slm_sr, color="#999999", linestyle=":", linewidth=1.2,
                   label=f"SLM-only ({slm_sr:.0f}%)")
        ax.axvline(oracle_sr, color="#333333", linestyle="--", linewidth=1.2,
                   label="LLM-only (100%)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("Task success rate (%)")
        ax.set_title(bench_titles.get(bench, bench), fontweight="bold")
        xmin = min(55, min(srs) - 2)
        ax.set_xlim(xmin, 101)
        ax.invert_yaxis()
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout(pad=1.5)
    _save(fig, "figure3_feature_ablation")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — CVaR hyperparameter sensitivity heatmap
# ---------------------------------------------------------------------------
def plot_cvar_sensitivity(rows: list[dict]) -> None:
    """Scatter: ε on x, SR on y, coloured by α."""
    alpha_vals  = sorted({_flt(r["alpha"]) for r in rows})
    cmap = matplotlib.colormaps.get_cmap("plasma").resampled(len(alpha_vals))
    alpha_to_color = {a: cmap(i) for i, a in enumerate(alpha_vals)}

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for a in alpha_vals:
        pts = [r for r in rows if abs(_flt(r["alpha"]) - a) < 1e-4]
        epsilons = [_flt(r["epsilon"]) for r in pts]
        srs      = [_flt(r["sr"]) * 100 for r in pts]
        ax.scatter(epsilons, srs,
                   color=alpha_to_color[a],
                   s=50, zorder=4,
                   label=f"α={a:.2f}")

    ax.set_xlabel("Constraint slack ε")
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("CVaR hyperparameter sensitivity\n(combined benchmarks)", fontweight="bold")
    ax.set_ylim(97.5, 100.2)
    ax.legend(fontsize=7, title="CVaR tail α", title_fontsize=7, ncol=2)

    fig.tight_layout()
    _save(fig, "figure4_cvar_sensitivity")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Closed-source adaptation: SR across feature variants & models
# ---------------------------------------------------------------------------
def plot_closed_source(rows: list[dict]) -> None:
    """Grouped bar chart: 3 feature variants × 4 models for each benchmark."""
    bench_order = ["HumanEval+", "TextWorld"]
    variants = ["All features (full)", "No entropy (trained)", "Verifier pseudo-entropy (trained)"]
    var_labels = ["Full features", "No entropy", "Pseudo-entropy"]
    var_colors = ["#2166AC", "#D6604D", "#4DAC26"]
    model_order = ["Gemma-7B", "LLaMA-3.1-8B", "Qwen-14B", "Qwen-7B"]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0), sharey=False)

    for ax, bench in zip(axes, bench_order):
        bench_rows = [r for r in rows if r["benchmark"] == bench]
        n_models = len(model_order)
        n_vars   = len(variants)
        x = np.arange(n_models)
        width = 0.28

        for vi, (var, vlabel, vcol) in enumerate(zip(variants, var_labels, var_colors)):
            srs = []
            for m in model_order:
                pts = [r for r in bench_rows
                       if r["model"] == m and r["router"] == var]
                sr  = _flt(pts[0]["sr"]) * 100 if pts else float("nan")
                srs.append(sr)
            offset = (vi - 1) * width
            bars = ax.bar(x + offset, srs, width, label=vlabel,
                          color=vcol, alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("LLaMA-3.1-8B", "LLaMA-8B") for m in model_order],
                           fontsize=7.5)
        ax.set_ylabel("Task success rate (%)")
        title_str = f"({'a' if bench == 'HumanEval+' else 'b'}) {bench}"
        ax.set_title(title_str, fontweight="bold")
        lo = 88 if "Human" in bench else 60
        ax.set_ylim(lo, 102)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "figure5_closed_source")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading data...")
    main_rows    = _read_csv(TABLES / "main_results_with_ci.csv")
    cvar_rows    = _read_csv(TABLES / "cvar_sweeps.csv")
    feat_rows    = _read_csv(TABLES / "feature_ablation_extended.csv")
    closed_rows  = _read_csv(TABLES / "feature_transform_comparison.csv")

    print("Figure 1: Pareto frontier...")
    plot_pareto(main_rows)

    print("Figure 2: λ_cons sweep...")
    plot_lambda_sweep(METRICS)

    print("Figure 3: Feature ablation...")
    plot_feature_ablation(feat_rows)

    print("Figure 4: CVaR sensitivity...")
    plot_cvar_sensitivity(cvar_rows)

    print("Figure 5: Closed-source adaptation...")
    plot_closed_source(closed_rows)

    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()
