#!/usr/bin/env bash
# R2V-Agent: NeurIPS paper experiment runner
#
# Phases (all enabled by default; use --skip-phase N or --only-phase N to subset):
#   1  Train main router on heuristic/main data
#   2  Main evaluation + threshold-sweep Pareto (eval-only)
#   3  Feature ablation (eval-only, re-uses Phase 1 router)
#   4  Consistency-λ ablation — train one router per variant + eval
#   5  CVaR hyperparameter sweep — train one router per α/ε + eval
#   6  Generate all paper plots and tables
#
# Usage:
#   bash scripts/run_paper_experiments.sh
#   bash scripts/run_paper_experiments.sh --skip-phase 4 5
#   bash scripts/run_paper_experiments.sh --only-phase 6
#   bash scripts/run_paper_experiments.sh --dry-run
#   bash scripts/run_paper_experiments.sh --parallel-jobs 4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH="${DATASET_PATH:-data/router_dataset/unified_router_features.parquet}"
RESULTS_ROOT="${RESULTS_ROOT:-results/paper_experiments}"
MAIN_ROUTER_DIR="${MAIN_ROUTER_DIR:-outputs/router/paper_main}"
# Feature-transform variants: trained with features permanently modified,
# not just masked at eval time.  Critical for the no-entropy and
# verifier-pseudo-entropy ablations (the latter matters for closed-source models
# where true SLM entropy is unavailable).
NO_ENTROPY_ROUTER_DIR="${NO_ENTROPY_ROUTER_DIR:-outputs/router/paper_main_no_entropy}"
VPE_ROUTER_DIR="${VPE_ROUTER_DIR:-outputs/router/paper_main_verifier_pseudo_entropy}"
LAMBDA_ROUTERS_DIR="${LAMBDA_ROUTERS_DIR:-outputs/router/paper_lambda}"
CVAR_ROUTERS_DIR="${CVAR_ROUTERS_DIR:-outputs/router/paper_cvar}"
BASE_CONFIG="${BASE_CONFIG:-configs/base.yaml}"

SEED="${SEED:-42}"
DRY_RUN=false
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"

# Threshold values for Pareto curve (Phase 2) — eval-only, no training.
THRESHOLDS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# CVaR sweep values (Phase 5).
CVAR_ALPHAS=(0.05 0.10 0.15 0.20 0.30 0.40 0.50)
CVAR_EPSILONS=(0.05 0.10 0.20 0.30 0.40)

# Phase control: populated by --skip-phase / --only-phase.
declare -A SKIP_PHASE=()

# ── Argument parsing ───────────────────────────────────────────────────────────
print_help() {
    cat <<'EOF'
Usage: bash scripts/run_paper_experiments.sh [options]

Options:
  --dataset PATH          Parquet features file  [data/router_dataset/unified_router_features.parquet]
  --results-root PATH     Output root            [results/paper_experiments]
  --main-router-dir PATH  Main router checkpoint dir
  --seed INT              Random seed            [42]
  --skip-phase N...       Skip phases (1-6), e.g. --skip-phase 4 5
  --only-phase N...       Run only listed phases
  --parallel-jobs N       Parallel training jobs [1]
  --dry-run               Print commands without executing
  -h, --help              Show this help
EOF
}

only_phases=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)        DATASET_PATH="$2";   shift 2 ;;
        --results-root)   RESULTS_ROOT="$2";   shift 2 ;;
        --main-router-dir) MAIN_ROUTER_DIR="$2"; shift 2 ;;
        --seed)           SEED="$2";           shift 2 ;;
        --skip-phase)
            shift
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
                SKIP_PHASE["$1"]=1; shift
            done
            ;;
        --only-phase)
            shift
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
                only_phases+=("$1"); shift
            done
            ;;
        --parallel-jobs)  PARALLEL_JOBS="$2";  shift 2 ;;
        --dry-run)        DRY_RUN=true;        shift ;;
        -h|--help)        print_help; exit 0 ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

# If --only-phase given, skip everything except those phases.
if [[ ${#only_phases[@]} -gt 0 ]]; then
    for p in 1 2 3 4 5 6; do
        SKIP_PHASE[$p]=1
    done
    for p in "${only_phases[@]}"; do
        unset "SKIP_PHASE[$p]"
    done
fi

phase_enabled() { [[ -z "${SKIP_PHASE[$1]+x}" ]]; }

mkdir -p "${RESULTS_ROOT}" "${RESULTS_ROOT}/logs"

MANIFEST_PATH="${RESULTS_ROOT}/experiment_manifest.jsonl"
METRICS_PATH="${RESULTS_ROOT}/metrics_long.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────
run_cmd() {
    local label="$1"; shift
    echo
    echo "──────────────────────────────────────────────────────────────────"
    echo "  ${label}"
    echo "──────────────────────────────────────────────────────────────────"
    printf '  $ %q ' "$@"; echo
    [[ "${DRY_RUN}" == true ]] && { echo "  [DRY RUN]"; return 0; }
    "$@"
}

config_for_benchmark() {
    case "$1" in
        humaneval)     echo "configs/humaneval/noisy.yaml" ;;
        terminalbench) echo "configs/textworld/noisy.yaml" ;;
        textworld)     echo "configs/textworld/noisy.yaml" ;;
        *)             echo "${BASE_CONFIG}" ;;
    esac
}

# Map (benchmark, lambda_value_string) → dataset variant name.
# lambda_value_string: "0.0" | "0.05" | "0.2" | "0.5" | "1.0"
lambda_variant() {
    local benchmark="$1" lam="$2"
    if [[ "${lam}" == "0.0" ]]; then
        echo "no_consistency"
    elif [[ "${benchmark}" == "terminalbench" ]]; then
        echo "consistency_lambda_${lam}"
    else
        echo "lam_${lam}"
    fi
}

# record_metrics <bundle_path> <category> <variant> <benchmark> <model> <split>
#                <router_path> [--alpha A] [--epsilon E]
record_metrics() {
    local bundle_path="$1" category="$2" variant="$3"
    local benchmark="$4" model="$5" split="$6" router_path="$7"
    shift 7
    local alpha="None" epsilon="None"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --alpha)   alpha="$2";   shift 2 ;;
            --epsilon) epsilon="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    [[ "${DRY_RUN}" == true ]] && return 0
    [[ -z "${bundle_path}" || ! -f "${bundle_path}" ]] && {
        echo "WARN: bundle missing for metrics: ${bundle_path}"; return 0
    }

    python - "${bundle_path}" "${MANIFEST_PATH}" "${METRICS_PATH}" \
        "${category}" "${variant}" "${benchmark}" "${model}" "${split}" \
        "${router_path}" "${alpha}" "${epsilon}" <<'PY'
import csv, json, os, sys
from datetime import datetime

(bundle_path, manifest_path, metrics_path,
 category, variant, benchmark, model, split, router_path,
 alpha_s, epsilon_s) = sys.argv[1:]

alpha   = float(alpha_s)   if alpha_s   != "None" else None
epsilon = float(epsilon_s) if epsilon_s != "None" else None

with open(bundle_path, "r", encoding="utf-8") as f:
    bundle = json.load(f)

os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "bundle_path": bundle_path,
        "category": category, "variant": variant,
        "benchmark": benchmark, "model": model,
        "split": split, "router_path": router_path,
    }) + "\n")

rows = []
for er in bundle.get("eval_results", []):
    if er.get("worst_seed_sr") is None and er.get("cvar_failure") is None:
        continue
    rows.append({
        "timestamp": datetime.utcnow().isoformat(),
        "category": category, "variant": variant,
        "benchmark": benchmark, "model": model,
        "split": split,
        "method": er.get("method"),
        "seed": er.get("seed"),
        "success_rate": er.get("success_rate"),
        "success_rate_ci_low":  (er.get("success_rate_ci") or [None, None])[0],
        "success_rate_ci_high": (er.get("success_rate_ci") or [None, None])[1],
        "worst_seed_sr": er.get("worst_seed_sr"),
        "cvar_failure":  er.get("cvar_failure"),
        "avg_cost":      er.get("avg_cost"),
        "llm_call_rate": er.get("llm_call_rate"),
        "ece":     er.get("ece"),
        "brier":   er.get("brier"),
        "alpha":   alpha,
        "epsilon": epsilon,
        "bundle_path": bundle_path,
    })

if rows:
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)
PY
}

latest_bundle() {
    ls -1t "${1}"/structured_results/*.json 2>/dev/null | head -n1 || true
}

# ── Enumerate benchmark/model combos in the dataset ───────────────────────────
mapfile -t HEURISTIC_COMBOS < <(
    python - "${DATASET_PATH}" <<'PY'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1], columns=["benchmark","model","variant","category","split"])
df = df[(df.split=="test") & (df.variant=="heuristic") & (df.category=="main")]
for _, row in df[["benchmark","model"]].drop_duplicates().sort_values(["benchmark","model"]).iterrows():
    print(f"{row.benchmark}|{row.model}")
PY
)

# Lambda ablation combos: (benchmark, model, variant_name, lambda_value)
mapfile -t LAMBDA_COMBOS < <(
    python - "${DATASET_PATH}" <<'PY'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1], columns=["benchmark","model","variant","category","split"])
df = df[(df.split=="test") & (df.category=="ablation")]
for _, row in (df[["benchmark","model","variant"]]
               .drop_duplicates()
               .sort_values(["benchmark","model","variant"])
               .iterrows()):
    v = row.variant
    # Normalise variant → lambda float string
    if v == "no_consistency":
        lam = "0.0"
    elif v.startswith("consistency_lambda_"):
        lam = v.replace("consistency_lambda_", "")
    elif v.startswith("lam_"):
        lam = v.replace("lam_", "")
    else:
        continue
    print(f"{row.benchmark}|{row.model}|{v}|{lam}")
PY
)

echo
echo "R2V-Agent paper experiment runner"
echo "  dataset:     ${DATASET_PATH}"
echo "  results:     ${RESULTS_ROOT}"
echo "  main router: ${MAIN_ROUTER_DIR}"
echo "  dry_run:     ${DRY_RUN}"
echo "  heuristic combos: ${#HEURISTIC_COMBOS[@]}"
echo "  lambda combos:    ${#LAMBDA_COMBOS[@]}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Train main router (heuristic/main category, all benchmarks)
# ══════════════════════════════════════════════════════════════════════════════
if phase_enabled 1; then
    echo
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 1: Train main router                             ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    # ── Full-feature router ──────────────────────────────────────────────────
    if [[ -f "${MAIN_ROUTER_DIR}/router_final.pt" ]]; then
        echo "INFO: Reusing existing main router: ${MAIN_ROUTER_DIR}/router_final.pt"
    else
        mkdir -p "${MAIN_ROUTER_DIR}"
        run_cmd "Train main router (all features)" \
            python scripts/train_router.py \
            --config "${BASE_CONFIG}" \
            --features "${DATASET_PATH}" \
            --output "${MAIN_ROUTER_DIR}" \
            --train-split train \
            --val-split val \
            --category-filter main \
            --variant-filter heuristic \
            --feature-transform none \
            --overrides \
                "project.seed=${SEED}" \
                "logging.wandb_mode=disabled"
    fi

    # ── No-entropy router ────────────────────────────────────────────────────
    # Trains with entropy (feature 0) permanently zeroed out.
    # Shows whether the router can compensate with verifier signals alone.
    if [[ -f "${NO_ENTROPY_ROUTER_DIR}/router_final.pt" ]]; then
        echo "INFO: Reusing existing no-entropy router: ${NO_ENTROPY_ROUTER_DIR}/router_final.pt"
    else
        mkdir -p "${NO_ENTROPY_ROUTER_DIR}"
        run_cmd "Train no-entropy router (feature-transform=no_entropy)" \
            python scripts/train_router.py \
            --config "${BASE_CONFIG}" \
            --features "${DATASET_PATH}" \
            --output "${NO_ENTROPY_ROUTER_DIR}" \
            --train-split train \
            --val-split val \
            --category-filter main \
            --variant-filter heuristic \
            --feature-transform no_entropy \
            --overrides \
                "project.seed=${SEED}" \
                "logging.wandb_mode=disabled"
    fi

    # ── Verifier-pseudo-entropy router ───────────────────────────────────────
    # Replaces SLM entropy (feature 0) with entropy computed from verifier
    # scores.  Key for closed-source model settings where true SLM entropy
    # is not observable.
    if [[ -f "${VPE_ROUTER_DIR}/router_final.pt" ]]; then
        echo "INFO: Reusing existing verifier-pseudo-entropy router: ${VPE_ROUTER_DIR}/router_final.pt"
    else
        mkdir -p "${VPE_ROUTER_DIR}"
        run_cmd "Train verifier-pseudo-entropy router (feature-transform=verifier_pseudo_entropy)" \
            python scripts/train_router.py \
            --config "${BASE_CONFIG}" \
            --features "${DATASET_PATH}" \
            --output "${VPE_ROUTER_DIR}" \
            --train-split train \
            --val-split val \
            --category-filter main \
            --variant-filter heuristic \
            --feature-transform verifier_pseudo_entropy \
            --overrides \
                "project.seed=${SEED}" \
                "logging.wandb_mode=disabled"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Main eval + threshold-sweep Pareto (eval-only, uses Phase 1 router)
#
# Produces:
#   - Table 1 (main results): SR / CVaR-F / Cost / LLM% per benchmark × model
#   - Figure 1 data (Pareto): threshold sweep SR vs LLM%
# ══════════════════════════════════════════════════════════════════════════════
if phase_enabled 2; then
    echo
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 2: Main eval + Pareto threshold sweep            ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    for combo in "${HEURISTIC_COMBOS[@]}"; do
        IFS='|' read -r benchmark model <<< "${combo}"
        config="$(config_for_benchmark "${benchmark}")"
        eval_root="${RESULTS_ROOT}/main/${benchmark}/${model}"
        mkdir -p "${eval_root}"

        # ── Single fixed threshold for the main results table ──
        main_eval_dir="${eval_root}/main_results"
        run_cmd "Main results [${benchmark}/${model}]" \
            python scripts/evaluate.py \
            --config "${config}" \
            --features "${DATASET_PATH}" \
            --router-path "${MAIN_ROUTER_DIR}/router_final.pt" \
            --output "${main_eval_dir}" \
            --split test \
            --benchmark-filter "${benchmark}" \
            --model-filter "${model}" \
            --variant-filter heuristic \
            --category-filter main \
            --methods r2v slm_only llm_only entropy_router oracle_router heuristic_router \
            --overrides "project.seed=${SEED}"

        bundle="$(latest_bundle "${main_eval_dir}")"
        record_metrics "${bundle}" "main" "main_results" \
            "${benchmark}" "${model}" "test" "${MAIN_ROUTER_DIR}/router_final.pt"

        # ── Threshold sweep for the Pareto figure (eval-only, no training) ──
        pareto_eval_dir="${eval_root}/pareto_sweep"
        run_cmd "Pareto threshold sweep [${benchmark}/${model}]" \
            python scripts/evaluate.py \
            --config "${config}" \
            --features "${DATASET_PATH}" \
            --router-path "${MAIN_ROUTER_DIR}/router_final.pt" \
            --output "${pareto_eval_dir}" \
            --split test \
            --benchmark-filter "${benchmark}" \
            --model-filter "${model}" \
            --variant-filter heuristic \
            --category-filter main \
            --methods r2v slm_only llm_only entropy_router oracle_router \
            --router-threshold-sweep "${THRESHOLDS[@]}" \
            --overrides "project.seed=${SEED}"

        bundle="$(latest_bundle "${pareto_eval_dir}")"
        record_metrics "${bundle}" "figure" "pareto_sweep" \
            "${benchmark}" "${model}" "test" "${MAIN_ROUTER_DIR}/router_final.pt"

        # ── Feature-transform router evaluations ─────────────────────────────
        # Each router was trained with a different feature transform baked in;
        # eval must use the matching --feature-transform flag so normalisation
        # stats align with what the router saw during training.

        # no_entropy router
        if [[ -f "${NO_ENTROPY_ROUTER_DIR}/router_final.pt" ]] || [[ "${DRY_RUN}" == true ]]; then
            no_ent_eval_dir="${eval_root}/feature_transform_no_entropy"
            run_cmd "Feature-transform eval: no_entropy [${benchmark}/${model}]" \
                python scripts/evaluate.py \
                --config "${config}" \
                --features "${DATASET_PATH}" \
                --router-path "${NO_ENTROPY_ROUTER_DIR}/router_final.pt" \
                --output "${no_ent_eval_dir}" \
                --split test \
                --benchmark-filter "${benchmark}" \
                --model-filter "${model}" \
                --variant-filter heuristic \
                --category-filter main \
                --feature-transform no_entropy \
                --methods r2v \
                --overrides "project.seed=${SEED}"
            bundle="$(latest_bundle "${no_ent_eval_dir}")"
            record_metrics "${bundle}" "feature_transform" "no_entropy" \
                "${benchmark}" "${model}" "test" "${NO_ENTROPY_ROUTER_DIR}/router_final.pt"
        fi

        # verifier_pseudo_entropy router
        if [[ -f "${VPE_ROUTER_DIR}/router_final.pt" ]] || [[ "${DRY_RUN}" == true ]]; then
            vpe_eval_dir="${eval_root}/feature_transform_verifier_pseudo_entropy"
            run_cmd "Feature-transform eval: verifier_pseudo_entropy [${benchmark}/${model}]" \
                python scripts/evaluate.py \
                --config "${config}" \
                --features "${DATASET_PATH}" \
                --router-path "${VPE_ROUTER_DIR}/router_final.pt" \
                --output "${vpe_eval_dir}" \
                --split test \
                --benchmark-filter "${benchmark}" \
                --model-filter "${model}" \
                --variant-filter heuristic \
                --category-filter main \
                --feature-transform verifier_pseudo_entropy \
                --methods r2v \
                --overrides "project.seed=${SEED}"
            bundle="$(latest_bundle "${vpe_eval_dir}")"
            record_metrics "${bundle}" "feature_transform" "verifier_pseudo_entropy" \
                "${benchmark}" "${model}" "test" "${VPE_ROUTER_DIR}/router_final.pt"
        fi
    done
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Feature ablation (eval-only, re-uses Phase 1 router)
#
# Produces: Table 2 (feature importance)
# Feature indices (15-dim vector):
#   0: entropy  1-5: verifier stats (spread,mean,std,best,worst)
#   6-8: action logprob stats  9: candidate consistency  10: semantic entropy
#   11: step count  12: horizon fraction  13: context length  14: goal length
# ══════════════════════════════════════════════════════════════════════════════
if phase_enabled 3; then
    echo
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: Feature ablation (eval-only)                  ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    # Limit to the primary model per benchmark to keep the table concise.
    declare -A PRIMARY_MODEL=(
        ["humaneval"]="qwen7"
        ["textworld"]="qwen7"
        ["terminalbench"]="qwen7"
    )

    declare -A FEATURE_ABLATIONS=(
        ["all_features"]=""
        ["entropy_only"]="0"
        ["verifier_only"]="1 2 3 4 5"
        ["verifier_plus_entropy"]="0 1 2 3 4 5"
        ["no_verifier"]="0 6 7 8 9 10 11 12 13 14"
        ["no_entropy"]="1 2 3 4 5 6 7 8 9 10 11 12 13 14"
        ["logprob_only"]="6 7 8"
        ["step_context_only"]="11 12 13 14"
    )

    for benchmark in humaneval textworld terminalbench; do
        model="${PRIMARY_MODEL[${benchmark}]}"
        config="$(config_for_benchmark "${benchmark}")"
        feat_root="${RESULTS_ROOT}/feature_ablation/${benchmark}/${model}"
        mkdir -p "${feat_root}"

        for ablation_name in "${!FEATURE_ABLATIONS[@]}"; do
            mask="${FEATURE_ABLATIONS[${ablation_name}]}"
            eval_dir="${feat_root}/${ablation_name}"

            if [[ -z "${mask}" ]]; then
                run_cmd "Feature ablation: ${ablation_name} [${benchmark}/${model}]" \
                    python scripts/evaluate.py \
                    --config "${config}" \
                    --features "${DATASET_PATH}" \
                    --router-path "${MAIN_ROUTER_DIR}/router_final.pt" \
                    --output "${eval_dir}" \
                    --split test \
                    --benchmark-filter "${benchmark}" \
                    --model-filter "${model}" \
                    --variant-filter heuristic \
                    --category-filter main \
                    --methods r2v \
                    --overrides "project.seed=${SEED}"
            else
                # shellcheck disable=SC2068
                run_cmd "Feature ablation: ${ablation_name} [${benchmark}/${model}]" \
                    python scripts/evaluate.py \
                    --config "${config}" \
                    --features "${DATASET_PATH}" \
                    --router-path "${MAIN_ROUTER_DIR}/router_final.pt" \
                    --output "${eval_dir}" \
                    --split test \
                    --benchmark-filter "${benchmark}" \
                    --model-filter "${model}" \
                    --variant-filter heuristic \
                    --category-filter main \
                    --methods r2v \
                    --feature-mask ${mask} \
                    --overrides "project.seed=${SEED}"
            fi

            bundle="$(latest_bundle "${eval_dir}")"
            record_metrics "${bundle}" "feature_ablation" "${ablation_name}" \
                "${benchmark}" "${model}" "test" "${MAIN_ROUTER_DIR}/router_final.pt"
        done
    done
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Consistency-λ ablation
#
# For each (benchmark, model, lambda) combo: train a dedicated router on that
# variant's data, then evaluate.  This shows the full end-to-end effect of λ_cons.
#
# Produces: Figure 2 (SR and CVaR-F vs λ) + companion Table 3.
# ══════════════════════════════════════════════════════════════════════════════
if phase_enabled 4; then
    echo
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 4: Consistency-λ ablation (train + eval)         ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    run_lambda_combo() {
        local benchmark="$1" model="$2" variant="$3" lam="$4"
        local safe_variant="${variant//./_}"
        local router_dir="${LAMBDA_ROUTERS_DIR}/${benchmark}_${model}_${safe_variant}"
        local eval_dir="${RESULTS_ROOT}/lambda_ablation/${benchmark}/${model}/${safe_variant}"
        local log_file="${RESULTS_ROOT}/logs/lambda_${benchmark}_${model}_${safe_variant}.log"
        local config; config="$(config_for_benchmark "${benchmark}")"

        mkdir -p "${router_dir}" "${eval_dir}"

        if [[ -f "${router_dir}/router_final.pt" ]]; then
            echo "INFO: Reusing lambda router: ${router_dir}/router_final.pt"
        else
            run_cmd "Train λ=${lam} router [${benchmark}/${model}/${variant}]" \
                python scripts/train_router.py \
                --config "${config}" \
                --features "${DATASET_PATH}" \
                --output "${router_dir}" \
                --train-split train \
                --val-split val \
                --benchmark-filter "${benchmark}" \
                --model-filter "${model}" \
                --variant-filter "${variant}" \
                --category-filter ablation \
                --feature-transform none \
                --overrides \
                    "project.seed=${SEED}" \
                    "logging.wandb_mode=disabled"
        fi

        run_cmd "Eval λ=${lam} [${benchmark}/${model}/${variant}]" \
            python scripts/evaluate.py \
            --config "${config}" \
            --features "${DATASET_PATH}" \
            --router-path "${router_dir}/router_final.pt" \
            --output "${eval_dir}" \
            --split test \
            --benchmark-filter "${benchmark}" \
            --model-filter "${model}" \
            --variant-filter "${variant}" \
            --category-filter ablation \
            --methods r2v slm_only llm_only entropy_router \
            --overrides "project.seed=${SEED}"

        local bundle; bundle="$(latest_bundle "${eval_dir}")"
        record_metrics "${bundle}" "lambda_ablation" "${variant}" \
            "${benchmark}" "${model}" "test" "${router_dir}/router_final.pt"
    }
    export -f run_lambda_combo run_cmd record_metrics latest_bundle config_for_benchmark lambda_variant

    for combo in "${LAMBDA_COMBOS[@]}"; do
        IFS='|' read -r benchmark model variant lam <<< "${combo}"
        if [[ "${PARALLEL_JOBS}" -gt 1 ]]; then
            run_lambda_combo "${benchmark}" "${model}" "${variant}" "${lam}" &
            while (( $(jobs -rp | wc -l) >= PARALLEL_JOBS )); do wait -n; done
        else
            run_lambda_combo "${benchmark}" "${model}" "${variant}" "${lam}"
        fi
    done
    while (( $(jobs -rp | wc -l) > 0 )); do wait -n; done
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — CVaR hyperparameter sweep (train + eval per α/ε combo)
#
# Produces: Figure 3 (SR/CVaR-F/Cost/LLM% vs cvar_alpha for each epsilon).
# ══════════════════════════════════════════════════════════════════════════════
if phase_enabled 5; then
    echo
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 5: CVaR hyperparameter sweep (train + eval)      ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    run_cvar_combo() {
        local alpha="$1" epsilon="$2"
        local safe_alpha="${alpha//./_}" safe_eps="${epsilon//./_}"
        local run_name="alpha_${safe_alpha}_eps_${safe_eps}"
        local router_dir="${CVAR_ROUTERS_DIR}/${run_name}"
        local eval_dir="${RESULTS_ROOT}/cvar_sweep/${run_name}"
        local log_file="${RESULTS_ROOT}/logs/cvar_${run_name}.log"

        mkdir -p "${router_dir}" "${eval_dir}"

        if [[ -f "${router_dir}/router_final.pt" ]]; then
            echo "INFO: Reusing CVaR router: ${router_dir}/router_final.pt"
        else
            run_cmd "Train CVaR router α=${alpha} ε=${epsilon}" \
                python scripts/train_router.py \
                --config "${BASE_CONFIG}" \
                --features "${DATASET_PATH}" \
                --output "${router_dir}" \
                --train-split train \
                --val-split val \
                --category-filter main \
                --variant-filter heuristic \
                --feature-transform none \
                --overrides \
                    "project.seed=${SEED}" \
                    "logging.wandb_mode=disabled" \
                    "training.router.cvar_alpha=${alpha}" \
                    "training.router.cvar_epsilon=${epsilon}"
        fi

        run_cmd "Eval CVaR α=${alpha} ε=${epsilon}" \
            python scripts/evaluate.py \
            --config "${BASE_CONFIG}" \
            --features "${DATASET_PATH}" \
            --router-path "${router_dir}/router_final.pt" \
            --output "${eval_dir}" \
            --split test \
            --category-filter main \
            --variant-filter heuristic \
            --methods r2v slm_only llm_only entropy_router \
            --overrides "project.seed=${SEED}"

        local bundle; bundle="$(latest_bundle "${eval_dir}")"
        record_metrics "${bundle}" "cvar_sweep" "${run_name}" \
            "all" "all" "test" "${router_dir}/router_final.pt" \
            --alpha "${alpha}" --epsilon "${epsilon}"
    }
    export -f run_cvar_combo run_cmd record_metrics latest_bundle

    for alpha in "${CVAR_ALPHAS[@]}"; do
        for epsilon in "${CVAR_EPSILONS[@]}"; do
            if [[ "${PARALLEL_JOBS}" -gt 1 ]]; then
                run_cvar_combo "${alpha}" "${epsilon}" &
                while (( $(jobs -rp | wc -l) >= PARALLEL_JOBS )); do wait -n; done
            else
                run_cvar_combo "${alpha}" "${epsilon}"
            fi
        done
    done
    while (( $(jobs -rp | wc -l) > 0 )); do wait -n; done
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Generate all paper plots and tables
# ══════════════════════════════════════════════════════════════════════════════
if phase_enabled 6; then
    echo
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 6: Generate plots and tables                     ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    if [[ ! -f "${METRICS_PATH}" ]]; then
        echo "WARN: No metrics file found at ${METRICS_PATH} — skipping plotting."
    else
        run_cmd "Generate paper plots and LaTeX/CSV tables" \
            python scripts/plot_router_experiments.py \
            --results-root "${RESULTS_ROOT}" \
            --metrics-path "${METRICS_PATH}"
    fi
fi

echo
echo "══════════════════════════════════════════════════════════════════"
echo "Paper experiment pipeline complete."
echo "  Results root:    ${RESULTS_ROOT}"
echo "  Metrics CSV:     ${METRICS_PATH}"
echo "  Plots:           ${RESULTS_ROOT}/plots/"
echo "  Tables:          ${RESULTS_ROOT}/tables/"
echo "══════════════════════════════════════════════════════════════════"
