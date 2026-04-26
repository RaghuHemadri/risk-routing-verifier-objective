#!/usr/bin/env python3
"""Reconstruct metrics_long.csv from existing bundle JSON files.

Scans a results root produced by run_paper_experiments.sh and rebuilds the
metrics_long.csv that would have been written incrementally by record_metrics().
Useful when the pipeline ran but the CSV was never written (e.g. the results root
changed between runs, or an older version of the script was used).

Usage:
    python scripts/rebuild_metrics_csv.py --results-root results/paper_experiments
    python scripts/rebuild_metrics_csv.py --results-root results/paper_experiments_v2
    python scripts/rebuild_metrics_csv.py \\
        --results-root results/paper_experiments \\
        --output results/paper_experiments_v2/metrics_long.csv
"""
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


def _latest_bundle(structured_dir: Path) -> Path | None:
    """Return the most-recently-modified eval_*.json in a structured_results dir."""
    candidates = [
        p for p in structured_dir.glob("eval_*.json") if "llm_summary" not in p.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _infer_meta(bundle_path: Path, results_root: Path) -> dict:
    """Infer (category, variant, benchmark, model, split, alpha, epsilon) from path."""
    rel = bundle_path.relative_to(results_root)
    parts = rel.parts  # e.g. ('main','textworld','qwen7','main_results','structured_results','eval_*.json')

    category = variant = benchmark = model = "unknown"
    split = "test"
    alpha = epsilon = None

    top = parts[0] if parts else ""

    if top == "main" and len(parts) >= 5:
        # main/{benchmark}/{model}/{run_type}/structured_results/...
        benchmark = parts[1]
        model = parts[2]
        run_type = parts[3]
        if run_type == "main_results":
            category, variant = "main", "main_results"
        elif run_type == "pareto_sweep":
            category, variant = "figure", "pareto_sweep"
        elif run_type.startswith("feature_transform_"):
            category = "feature_transform"
            variant = run_type[len("feature_transform_"):]
        else:
            category, variant = "main", run_type

    elif top == "feature_ablation" and len(parts) >= 5:
        # feature_ablation/{benchmark}/{model}/{ablation}/structured_results/...
        benchmark = parts[1]
        model = parts[2]
        variant = parts[3]
        category = "feature_ablation"

    elif top == "lambda_ablation" and len(parts) >= 5:
        # lambda_ablation/{benchmark}/{model}/{variant}/structured_results/...
        benchmark = parts[1]
        model = parts[2]
        variant = parts[3]
        category = "lambda_ablation"

    elif top == "cvar_sweep" and len(parts) >= 3:
        # cvar_sweep/alpha_{a}_eps_{e}/structured_results/...
        run_name = parts[1]
        category = "cvar_sweep"
        variant = run_name
        benchmark = "all"
        model = "all"
        # parse alpha / epsilon from name: alpha_0_05_eps_0_10
        m = re.match(r"alpha_([\d_]+)_eps_([\d_]+)$", run_name)
        if m:
            alpha = float(m.group(1).replace("_", "."))
            epsilon = float(m.group(2).replace("_", "."))

    return {
        "category": category,
        "variant": variant,
        "benchmark": benchmark,
        "model": model,
        "split": split,
        "alpha": alpha,
        "epsilon": epsilon,
    }


def _rows_from_bundle(bundle_path: Path, results_root: Path) -> list[dict]:
    meta = _infer_meta(bundle_path, results_root)
    with open(bundle_path, encoding="utf-8") as f:
        bundle = json.load(f)

    rows = []
    for er in bundle.get("eval_results", []):
        if er.get("worst_seed_sr") is None and er.get("cvar_failure") is None:
            continue
        # Use benchmark from eval_result when available (cvar_sweep bundles have "unknown")
        bm = er.get("benchmark") or meta["benchmark"]
        if bm in ("unknown", "", None):
            bm = meta["benchmark"]
        rows.append({
            "timestamp": datetime.utcnow().isoformat(),
            "category": meta["category"],
            "variant": meta["variant"],
            "benchmark": bm,
            "model": meta["model"],
            "split": meta["split"],
            "method": er.get("method"),
            "seed": er.get("seed"),
            "success_rate": er.get("success_rate"),
            "success_rate_ci_low": (er.get("success_rate_ci") or [None, None])[0],
            "success_rate_ci_high": (er.get("success_rate_ci") or [None, None])[1],
            "worst_seed_sr": er.get("worst_seed_sr"),
            "cvar_failure": er.get("cvar_failure"),
            "avg_cost": er.get("avg_cost"),
            "llm_call_rate": er.get("llm_call_rate"),
            "ece": er.get("ece"),
            "brier": er.get("brier"),
            "alpha": meta["alpha"],
            "epsilon": meta["epsilon"],
            "bundle_path": str(bundle_path),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild metrics_long.csv from bundle JSONs")
    parser.add_argument("--results-root", required=True,
                        help="Root results directory (e.g. results/paper_experiments)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <results-root>/metrics_long.csv)")
    parser.add_argument("--dedup", action="store_true", default=True,
                        help="Keep only the latest bundle per structured_results dir (default: on)")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_path = Path(args.output) if args.output else results_root / "metrics_long.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all structured_results directories under the root.
    structured_dirs = sorted(results_root.rglob("structured_results"))
    print(f"Found {len(structured_dirs)} structured_results dirs under {results_root}")

    all_rows: list[dict] = []
    skipped = 0
    for sd in structured_dirs:
        bundle = _latest_bundle(sd) if args.dedup else None
        if bundle is None and not args.dedup:
            # Non-dedup: process all eval_*.json
            for p in sorted(sd.glob("eval_*.json")):
                if "llm_summary" not in p.name:
                    all_rows.extend(_rows_from_bundle(p, results_root))
        elif bundle is not None:
            rows = _rows_from_bundle(bundle, results_root)
            if rows:
                all_rows.extend(rows)
            else:
                skipped += 1

    print(f"Total rows: {len(all_rows)}  (skipped {skipped} empty/filtered bundles)")

    if not all_rows:
        print("WARNING: No rows to write — check that bundle JSONs contain eval_results.")
        return

    fieldnames = list(all_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Written: {out_path}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
