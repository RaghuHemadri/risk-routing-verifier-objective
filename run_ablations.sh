#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Ablation Runner: DPO β, Consistency λ, DPO Margin Sweep
# ══════════════════════════════════════════════════════════════════
#
# Runs ablation sweeps for qwen7/qwen14 across benchmarks.
# Each ablation retrains DPO (stage 3) from the existing BC checkpoint,
# then generates router features (stage 4), and optionally scores on CPU.
#
# Usage:
#   bash run_ablations.sh                           # full run, all ablations
#   bash run_ablations.sh --dry-run                 # preview commands
#   bash run_ablations.sh --model qwen7             # single model
#   bash run_ablations.sh --benchmark humaneval     # single benchmark
#   bash run_ablations.sh --ablation beta           # single ablation type
#   bash run_ablations.sh --skip-dpo                # skip DPO, only router features
#   bash run_ablations.sh --skip-router-features    # skip router features, only DPO
#   bash run_ablations.sh --only-score              # CPU scoring only (no DPO/generate)
#   bash run_ablations.sh --gpus 2                  # override GPU count
#
# Ablation types (--ablation):
#   beta     DPO β sweep: 0.01, 0.05, [0.1], 0.2, 0.5
#   lambda   Consistency λ sweep: 0.0 (disabled), 0.05, [0.1], 0.5
#   margin   DPO min_score_gap sweep: 0.0, [0.1], 0.2, 0.4
#   all      Run all three (default)
#
# Prerequisites:
#   - BC checkpoints must exist for each model/benchmark
#   - Preference splits must exist (from run_pipeline_updated.sh stages 1-2)
#
# ══════════════════════════════════════════════════════════════════

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Defaults ─────────────────────────────────────────────────
FILTER_MODEL=""
FILTER_BENCHMARK=""
FILTER_ABLATION="all"
NUM_GPUS="${NUM_GPUS:-2}"
DRY_RUN=false
SKIP_DPO=false
SKIP_ROUTER_FEATURES=false
ONLY_SCORE=false

DPO_BATCH_SIZE="${DPO_BATCH_SIZE:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-32}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-32}"
ROUTER_K="${ROUTER_K:-5}"

QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"
COMMON_OVERRIDES="logging.wandb_mode=disabled"

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)                FILTER_MODEL="$2"; shift 2 ;;
        --benchmark)            FILTER_BENCHMARK="$2"; shift 2 ;;
        --ablation)             FILTER_ABLATION="$2"; shift 2 ;;
        --gpus)                 NUM_GPUS="$2"; shift 2 ;;
        --dpo-batch-size)       DPO_BATCH_SIZE="$2"; shift 2 ;;
        --dpo-grad-accum)       DPO_GRAD_ACCUM="$2"; shift 2 ;;
        --lora-r)               LORA_R="$2"; shift 2 ;;
        --lora-alpha)           LORA_ALPHA="$2"; shift 2 ;;
        --router-batch-size)    ROUTER_BATCH_SIZE="$2"; shift 2 ;;
        --skip-dpo)             SKIP_DPO=true; shift ;;
        --skip-router-features) SKIP_ROUTER_FEATURES=true; shift ;;
        --only-score)           ONLY_SCORE=true; shift ;;
        --dry-run)              DRY_RUN=true; shift ;;
        --help|-h)
            head -35 "$0" | tail -33
            exit 0
            ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

# ── Model registry ───────────────────────────────────────────
declare -A HF_ID=(
    [qwen7]="Qwen/Qwen2.5-Coder-7B-Instruct"
    [qwen14]="Qwen/Qwen2.5-Coder-14B-Instruct"
)

declare -A BC_DIR_TAG=(
    [qwen7]="qwen_coder_7b"
    [qwen14]="qwen_coder_14b"
)

MODELS=(qwen7 qwen14)
BENCHMARKS=(humaneval)

# ── Ablation definitions ─────────────────────────────────────
# Format: KEY|OVERRIDE_STRING|DESCRIPTION
# The override string is appended to the DPO training --overrides.
#
# Design: ONE-AT-A-TIME (standard NeurIPS ablation protocol).
#   - Each sweep varies ONE hyperparameter while holding the other two at defaults.
#   - Defaults: beta=0.1, lambda_cons=0.1 (enabled), min_score_gap=0.1
#   - The existing non-ablation DPO checkpoint (humaneval_noisy_dpo_{model}/final)
#     serves as the shared baseline — no need to retrain the default config.
#
# Why one-at-a-time (not grid):
#   - Isolates the contribution of each design choice (the point of ablations)
#   - Standard practice in RouteLLM, f-DPO, SCoRe, etc.
#   - Grid search is for HP tuning, not for ablation tables

BETA_ABLATIONS=(
    "dpo_beta_0.01|training.preference.beta=0.01|DPO beta=0.01 (near-unconstrained, max robust to noisy labels)"
    "dpo_beta_0.05|training.preference.beta=0.05|DPO beta=0.05 (weak KL constraint)"
    "dpo_beta_0.2|training.preference.beta=0.2|DPO beta=0.2 (stronger KL constraint)"
    "dpo_beta_0.5|training.preference.beta=0.5|DPO beta=0.5 (strong KL, may over-constrain)"
)

LAMBDA_ABLATIONS=(
    "no_consistency|training.consistency.enabled=false|Consistency disabled (lambda=0)"
    "consistency_lambda_0.05|training.consistency.lambda_cons=0.05 training.consistency.enabled=true|Consistency lambda=0.05 (light regularization)"
    "consistency_lambda_0.5|training.consistency.lambda_cons=0.5 training.consistency.enabled=true|Consistency lambda=0.5 (strong regularization)"
)

MARGIN_ABLATIONS=(
    "dpo_margin_0.0|training.preference.min_score_gap=0.0|No margin filter (keep all pairs including noisy ones)"
    "dpo_margin_0.2|training.preference.min_score_gap=0.2|Margin=0.2 (moderate filter, drops ambiguous pairs)"
    "dpo_margin_0.4|training.preference.min_score_gap=0.4|Margin=0.4 (strict filter, fewer but cleaner pairs)"
)

# ── Colour helpers ───────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()   { echo -e "${CYAN}[INFO]${NC}   $*"; }
ok()     { echo -e "${GREEN}[OK]${NC}     $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}   $*"; }
err()    { echo -e "${RED}[ERR]${NC}    $*"; }
header() { echo -e "\n${BOLD}══════════════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $*${NC}"; echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"; }

run_cmd() {
    local label="$1"; shift
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  ${label}"
    echo "────────────────────────────────────────────────────"
    echo "$ $*"
    echo ""
    if [[ ${DRY_RUN} == false ]]; then
        "$@"
    else
        echo "  [DRY RUN] Skipping execution"
    fi
}

# ── Collect ablations to run ─────────────────────────────────
declare -a ABLATIONS_TO_RUN=()

case "${FILTER_ABLATION}" in
    beta)   ABLATIONS_TO_RUN=("${BETA_ABLATIONS[@]}") ;;
    lambda) ABLATIONS_TO_RUN=("${LAMBDA_ABLATIONS[@]}") ;;
    margin) ABLATIONS_TO_RUN=("${MARGIN_ABLATIONS[@]}") ;;
    all)    ABLATIONS_TO_RUN=("${BETA_ABLATIONS[@]}" "${LAMBDA_ABLATIONS[@]}" "${MARGIN_ABLATIONS[@]}") ;;
    *)      echo "ERROR: Unknown --ablation '${FILTER_ABLATION}'. Choose: beta, lambda, margin, all"; exit 1 ;;
esac

# ── Filter models/benchmarks ────────────────────────────────
if [[ -n "${FILTER_MODEL}" ]]; then
    if [[ -z "${HF_ID[${FILTER_MODEL}]+_}" ]]; then
        echo "ERROR: Unknown model '${FILTER_MODEL}'. Choose: qwen7, qwen14"
        exit 1
    fi
    MODELS=("${FILTER_MODEL}")
fi

if [[ -n "${FILTER_BENCHMARK}" ]]; then
    BENCHMARKS=("${FILTER_BENCHMARK}")
fi

# ── Banner ───────────────────────────────────────────────────
header "Ablation Sweep Runner"
echo ""
echo "  Models:       ${MODELS[*]}"
echo "  Benchmarks:   ${BENCHMARKS[*]}"
echo "  Ablation set: ${FILTER_ABLATION} (${#ABLATIONS_TO_RUN[@]} configs)"
echo "  GPUs:         ${NUM_GPUS}"
echo "  DPO batch:    ${DPO_BATCH_SIZE} × ${DPO_GRAD_ACCUM} accum"
echo "  LoRA:         r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "  Dry run:      ${DRY_RUN}"
echo "  Skip DPO:     ${SKIP_DPO}"
echo "  Skip router:  ${SKIP_ROUTER_FEATURES}"
echo "  Score only:   ${ONLY_SCORE}"
echo ""

TOTAL_RUNS=$(( ${#MODELS[@]} * ${#BENCHMARKS[@]} * ${#ABLATIONS_TO_RUN[@]} ))
echo "  Total ablation runs: ${TOTAL_RUNS}"
echo ""

# ── Print ablation plan ──────────────────────────────────────
echo "  Ablation plan:"
echo "  ┌──────────────────────────────────┬────────────────────────────────────────────────┐"
printf "  │ %-32s │ %-46s │\n" "Key" "Description"
echo "  ├──────────────────────────────────┼────────────────────────────────────────────────┤"
for entry in "${ABLATIONS_TO_RUN[@]}"; do
    IFS='|' read -r key overrides desc <<< "${entry}"
    printf "  │ %-32s │ %-46s │\n" "${key}" "${desc}"
done
echo "  └──────────────────────────────────┴────────────────────────────────────────────────┘"
echo ""

# ── Ensure preference splits exist ───────────────────────────
ensure_pref_splits() {
    local model_short="$1"
    local bench="$2"
    local split_dir="data/trajectories/${bench}_noisy"
    local pref_prefix="pref_${model_short}"
    local pref_train="${split_dir}/${pref_prefix}_train.jsonl"
    local pref_val="${split_dir}/${pref_prefix}_val.jsonl"

    if [[ -f "${pref_train}" && -f "${pref_val}" ]]; then
        return 0
    fi
    err "Preference splits missing for ${model_short}/${bench}:"
    err "  Expected: ${pref_train}"
    err "  Expected: ${pref_val}"
    err "  Run: bash run_pipeline_updated.sh --model ${model_short} --from 2 --only 2"
    return 1
}

ensure_pref_val_half() {
    local model_short="$1"
    local bench="$2"
    local split_dir="data/trajectories/${bench}_noisy"
    local pref_prefix="pref_${model_short}"
    local val_half="${split_dir}/${pref_prefix}_val_half.jsonl"
    local val_rest="${split_dir}/${pref_prefix}_val_rest.jsonl"
    local val_data="${split_dir}/${pref_prefix}_val.jsonl"

    if [[ -f "${val_half}" && -f "${val_rest}" ]]; then
        return 0
    fi
    if [[ ! -f "${val_data}" ]]; then
        err "Preference val data missing: ${val_data}"
        return 1
    fi
    info "Creating 50/50 val split for ${model_short}/${bench}..."
    python - "${val_data}" "${val_half}" "${val_rest}" <<'PY'
import random, sys
src, out_a, out_b = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    rows = [ln for ln in f if ln.strip()]
rng = random.Random(42)
idxs = list(range(len(rows)))
rng.shuffle(idxs)
cut = len(rows) // 2
keep = set(idxs[:cut])
with open(out_a, "w", encoding="utf-8") as fa, open(out_b, "w", encoding="utf-8") as fb:
    for i, row in enumerate(rows):
        (fa if i in keep else fb).write(row)
print(f"  val_total={len(rows)} val_half={cut} val_rest={len(rows)-cut}")
PY
}

# ── Main loop ────────────────────────────────────────────────
LOGDIR="${SCRIPT_DIR}/logs/ablations"
mkdir -p "${LOGDIR}"

RUN_IDX=0
FAILED_RUNS=()
SUCCEEDED_RUNS=()
SKIPPED_RUNS=()

SUMMARY_FILE="${LOGDIR}/ablation_summary_$(date +%Y%m%d_%H%M%S).txt"

for MODEL_SHORT in "${MODELS[@]}"; do
    MODEL_HF="${HF_ID[${MODEL_SHORT}]}"
    MODEL_TAG="${BC_DIR_TAG[${MODEL_SHORT}]}"

    for BENCH in "${BENCHMARKS[@]}"; do
        CONFIG_NOISY="configs/${BENCH}/noisy.yaml"
        SPLIT_DIR="data/trajectories/${BENCH}_noisy"
        NOISY_TRAJECTORIES="${SPLIT_DIR}/trajectories.jsonl"
        BC_CHECKPOINT="outputs/policy/${BENCH}_noisy_bc_${MODEL_TAG}/final"
        PREF_PREFIX="pref_${MODEL_SHORT}"
        PREF_TRAIN_DATA="${SPLIT_DIR}/${PREF_PREFIX}_train.jsonl"
        PREF_VAL_HALF_DATA="${SPLIT_DIR}/${PREF_PREFIX}_val_half.jsonl"
        CANDIDATES_FILE="data/candidates/${BENCH}_noisy_dpo_prefs_heuristic_${MODEL_SHORT}.jsonl"

        # ── Validate prerequisites ───────────────────────────
        header "Validating prerequisites: ${MODEL_SHORT} / ${BENCH}"

        if [[ ! -f "${CONFIG_NOISY}" ]]; then
            warn "Config not found: ${CONFIG_NOISY} — skipping ${BENCH}"
            continue
        fi

        if [[ ! -f "${NOISY_TRAJECTORIES}" ]]; then
            warn "Trajectories not found: ${NOISY_TRAJECTORIES} — skipping ${BENCH}"
            continue
        fi

        if [[ ! -d "${BC_CHECKPOINT}" ]]; then
            warn "BC checkpoint not found: ${BC_CHECKPOINT} — skipping ${MODEL_SHORT}/${BENCH}"
            continue
        fi

        ok "Config:       ${CONFIG_NOISY}"
        ok "Trajectories: ${NOISY_TRAJECTORIES}"
        ok "BC checkpoint: ${BC_CHECKPOINT}"

        if [[ ${DRY_RUN} == false ]]; then
            ensure_pref_splits "${MODEL_SHORT}" "${BENCH}" || continue
            ensure_pref_val_half "${MODEL_SHORT}" "${BENCH}" || continue
        fi

        # GPU keepalive for qwen14
        GPU_KEEPALIVE="0"
        if [[ "${MODEL_SHORT}" == "qwen14" ]]; then
            GPU_KEEPALIVE="0.2"
        fi

        for entry in "${ABLATIONS_TO_RUN[@]}"; do
            IFS='|' read -r ABL_KEY ABL_OVERRIDES ABL_DESC <<< "${entry}"
            RUN_IDX=$(( RUN_IDX + 1 ))

            # ── Derived ablation-specific paths ──────────────
            DPO_OUTPUT="outputs/policy/${BENCH}_noisy_dpo_${MODEL_SHORT}_abl_${ABL_KEY}"
            DPO_CHECKPOINT="${DPO_OUTPUT}/final"
            ROUTER_FEATURES_FILE="data/router_features/${BENCH}_noisy_router_features_heuristic_${MODEL_SHORT}_abl_${ABL_KEY}.jsonl"
            LOGFILE="${LOGDIR}/${BENCH}_${MODEL_SHORT}_${ABL_KEY}.log"

            header "[${RUN_IDX}/${TOTAL_RUNS}] ${MODEL_SHORT} / ${BENCH} / ${ABL_KEY}"
            echo "  ${ABL_DESC}"
            echo "  Override:       ${ABL_OVERRIDES}"
            echo "  DPO output:     ${DPO_OUTPUT}"
            echo "  Router features:${ROUTER_FEATURES_FILE}"
            echo "  Log:            ${LOGFILE}"
            echo ""

            # ──────────────────────────────────────────────────
            # Phase 1: DPO Training (stage 3)
            # ──────────────────────────────────────────────────
            if [[ "${SKIP_DPO}" == false && "${ONLY_SCORE}" == false ]]; then

                if [[ -d "${DPO_CHECKPOINT}" ]]; then
                    info "DPO checkpoint already exists: ${DPO_CHECKPOINT} — skipping DPO"
                else
                    # Build the DPO override array: base overrides + ablation overrides
                    DPO_ARGS=(
                        scripts/train_policy.py
                        --config "${CONFIG_NOISY}"
                        --output "${DPO_OUTPUT}"
                        --stage preference
                        --resume "${BC_CHECKPOINT}"
                        --pref-train-data "${PREF_TRAIN_DATA}"
                        --pref-val-data "${PREF_VAL_HALF_DATA}"
                        --overrides
                            "${QUANT_OVERRIDE}"
                            "${COMMON_OVERRIDES}"
                            "policy.model_name=${MODEL_HF}"
                            "policy.lora.r=${LORA_R}"
                            "policy.lora.alpha=${LORA_ALPHA}"
                            "training.preference.batch_size=${DPO_BATCH_SIZE}"
                            "training.preference.gradient_accumulation_steps=${DPO_GRAD_ACCUM}"
                            "training.preference.concat_pairs=false"
                            "training.preference.gpu_keepalive_interval=${GPU_KEEPALIVE}"
                    )
                    # Append ablation-specific overrides (space-separated → individual args)
                    for ov in ${ABL_OVERRIDES}; do
                        DPO_ARGS+=("${ov}")
                    done

                    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

                    START_T=$(date +%s)
                    DPO_RC=0

                    if [[ ${DRY_RUN} == true ]]; then
                        if [[ ${NUM_GPUS} -gt 1 ]]; then
                            run_cmd "DPO [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH} (${NUM_GPUS} GPUs)" \
                                accelerate launch --num_processes="${NUM_GPUS}" --multi_gpu \
                                "${DPO_ARGS[@]}"
                        else
                            run_cmd "DPO [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH} (single GPU)" \
                                python "${DPO_ARGS[@]}"
                        fi
                    else
                        if [[ ${NUM_GPUS} -gt 1 ]]; then
                            run_cmd "DPO [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH} (${NUM_GPUS} GPUs)" \
                                accelerate launch --num_processes="${NUM_GPUS}" --multi_gpu \
                                "${DPO_ARGS[@]}" \
                                2>&1 | tee -a "${LOGFILE}" || DPO_RC=$?
                        else
                            run_cmd "DPO [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH} (single GPU)" \
                                python "${DPO_ARGS[@]}" \
                                2>&1 | tee -a "${LOGFILE}" || DPO_RC=$?
                        fi
                        ELAPSED=$(( $(date +%s) - START_T ))

                        if [[ ${DPO_RC} -ne 0 ]]; then
                            if [[ -d "${DPO_CHECKPOINT}" ]]; then
                                warn "DPO exited with code ${DPO_RC}, but checkpoint exists — continuing"
                            else
                                err "DPO FAILED (exit ${DPO_RC}) for ${ABL_KEY} — no checkpoint"
                                FAILED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY} [DPO failed]")
                                echo "FAILED|${MODEL_SHORT}|${BENCH}|${ABL_KEY}|DPO|exit=${DPO_RC}|${ELAPSED}s" >> "${SUMMARY_FILE}"
                                continue
                            fi
                        fi
                        ok "DPO done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s: ${DPO_CHECKPOINT}"
                        echo "DPO_OK|${MODEL_SHORT}|${BENCH}|${ABL_KEY}|${ELAPSED}s" >> "${SUMMARY_FILE}"
                    fi
                fi
            fi

            # ──────────────────────────────────────────────────
            # Phase 2: Router Feature Generation (stage 4, GPU)
            # ──────────────────────────────────────────────────
            if [[ "${SKIP_ROUTER_FEATURES}" == false && "${ONLY_SCORE}" == false ]]; then

                if [[ -f "${ROUTER_FEATURES_FILE}" ]]; then
                    EXISTING_LINES=$(wc -l < "${ROUTER_FEATURES_FILE}")
                    info "Router features already exist (${EXISTING_LINES} lines): ${ROUTER_FEATURES_FILE}"
                    info "Will RESUME (append missing records)"
                fi

                # The DPO checkpoint must exist (either from this run or prior)
                EFFECTIVE_DPO="${DPO_CHECKPOINT}"
                if [[ ! -d "${EFFECTIVE_DPO}" ]]; then
                    # Fall back to default (non-ablation) checkpoint for baseline comparison
                    EFFECTIVE_DPO="outputs/policy/${BENCH}_noisy_dpo_${MODEL_SHORT}/final"
                    if [[ ! -d "${EFFECTIVE_DPO}" ]]; then
                        warn "No DPO checkpoint found for ${ABL_KEY} — skipping router features"
                        SKIPPED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY} [no DPO checkpoint]")
                        continue
                    fi
                    warn "Using default DPO checkpoint: ${EFFECTIVE_DPO}"
                fi

                VERIFIER_OVERRIDE="verifier.mode=heuristic verifier.heuristic.run_code=true verifier.heuristic.benchmark=${BENCH}"

                export POLICY_PATH="${EFFECTIVE_DPO}"
                export TRAJECTORIES="${NOISY_TRAJECTORIES}"
                export CONFIG="${CONFIG_NOISY}"
                export OUTPUT="${ROUTER_FEATURES_FILE}"
                export K="${ROUTER_K}"
                export BATCH_SIZE="${ROUTER_BATCH_SIZE}"
                export NUM_GPUS="${NUM_GPUS}"
                export GENERATE_ONLY="true"
                export EXTRA_OVERRIDES="policy.model_name=${MODEL_HF}"

                START_T=$(date +%s)
                FEAT_RC=0

                run_cmd "Router features (generate-only) [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH}" \
                    bash scripts/run_router_features_humaneval.sh \
                    2>&1 | tee -a "${LOGFILE}" || FEAT_RC=$?

                if [[ ${DRY_RUN} == false ]]; then
                    ELAPSED=$(( $(date +%s) - START_T ))
                    if [[ ${FEAT_RC} -ne 0 ]]; then
                        err "Router feature generation FAILED for ${ABL_KEY}"
                        FAILED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY} [router features failed]")
                        echo "FAILED|${MODEL_SHORT}|${BENCH}|${ABL_KEY}|ROUTER_FEAT|exit=${FEAT_RC}|${ELAPSED}s" >> "${SUMMARY_FILE}"
                        continue
                    fi
                    ok "Router features done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
                    echo "FEAT_OK|${MODEL_SHORT}|${BENCH}|${ABL_KEY}|${ELAPSED}s" >> "${SUMMARY_FILE}"
                fi

                # Unset exports to avoid leaking to the next iteration
                unset POLICY_PATH TRAJECTORIES CONFIG OUTPUT K BATCH_SIZE GENERATE_ONLY EXTRA_OVERRIDES
            fi

            # ──────────────────────────────────────────────────
            # Phase 3: CPU Scoring (score-only, no GPU needed)
            # ──────────────────────────────────────────────────
            if [[ "${ONLY_SCORE}" == true ]]; then

                EFFECTIVE_DPO="${DPO_CHECKPOINT}"
                if [[ ! -d "${EFFECTIVE_DPO}" ]]; then
                    EFFECTIVE_DPO="outputs/policy/${BENCH}_noisy_dpo_${MODEL_SHORT}/final"
                fi

                export POLICY_PATH="${EFFECTIVE_DPO}"
                export TRAJECTORIES="${NOISY_TRAJECTORIES}"
                export CONFIG="${CONFIG_NOISY}"
                export OUTPUT="${ROUTER_FEATURES_FILE}"
                export K="${ROUTER_K}"
                export BATCH_SIZE="${ROUTER_BATCH_SIZE}"
                export SCORE_ONLY="true"
                export EXTRA_OVERRIDES="policy.model_name=${MODEL_HF}"

                START_T=$(date +%s)
                SCORE_RC=0

                run_cmd "CPU Scoring [${ABL_KEY}]: ${MODEL_SHORT}/${BENCH}" \
                    bash scripts/run_router_features_humaneval.sh \
                    2>&1 | tee -a "${LOGFILE}" || SCORE_RC=$?

                if [[ ${DRY_RUN} == false ]]; then
                    ELAPSED=$(( $(date +%s) - START_T ))
                    if [[ ${SCORE_RC} -ne 0 ]]; then
                        err "CPU scoring FAILED for ${ABL_KEY}"
                        FAILED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY} [scoring failed]")
                        echo "FAILED|${MODEL_SHORT}|${BENCH}|${ABL_KEY}|SCORE|exit=${SCORE_RC}|${ELAPSED}s" >> "${SUMMARY_FILE}"
                    else
                        ok "Scoring done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
                        echo "SCORE_OK|${MODEL_SHORT}|${BENCH}|${ABL_KEY}|${ELAPSED}s" >> "${SUMMARY_FILE}"
                    fi
                fi

                unset POLICY_PATH TRAJECTORIES CONFIG OUTPUT K BATCH_SIZE SCORE_ONLY EXTRA_OVERRIDES
            fi

            SUCCEEDED_RUNS+=("${MODEL_SHORT}/${BENCH}/${ABL_KEY}")

        done  # ablations
    done  # benchmarks
done  # models

# ── Summary ──────────────────────────────────────────────────
header "Ablation Sweep Complete"
echo ""
echo "  Succeeded: ${#SUCCEEDED_RUNS[@]}"
echo "  Failed:    ${#FAILED_RUNS[@]}"
echo "  Skipped:   ${#SKIPPED_RUNS[@]}"
echo ""

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo "  Failed runs:"
    for f in "${FAILED_RUNS[@]}"; do
        echo "    ✗ ${f}"
    done
    echo ""
fi

if [[ ${#SKIPPED_RUNS[@]} -gt 0 ]]; then
    echo "  Skipped runs:"
    for s in "${SKIPPED_RUNS[@]}"; do
        echo "    ⊘ ${s}"
    done
    echo ""
fi

echo "  Summary log: ${SUMMARY_FILE}"
echo ""

# ── Output directory listing ─────────────────────────────────
echo "  Ablation artifacts:"
echo ""
echo "  DPO checkpoints:"
for MODEL_SHORT in "${MODELS[@]}"; do
    for BENCH in "${BENCHMARKS[@]}"; do
        shopt -s nullglob
        for d in outputs/policy/${BENCH}_noisy_dpo_${MODEL_SHORT}_abl_*/final; do
            ABL_TAG="$(basename "$(dirname "${d}")" | sed "s/${BENCH}_noisy_dpo_${MODEL_SHORT}_abl_//")"
            echo "    ${MODEL_SHORT}/${BENCH}/${ABL_TAG}: ${d}"
        done
        shopt -u nullglob
    done
done

echo ""
echo "  Router feature files:"
for MODEL_SHORT in "${MODELS[@]}"; do
    for BENCH in "${BENCHMARKS[@]}"; do
        shopt -s nullglob
        for f in data/router_features/${BENCH}_noisy_router_features_heuristic_${MODEL_SHORT}_abl_*.jsonl; do
            ABL_TAG="$(basename "${f}" .jsonl | sed "s/${BENCH}_noisy_router_features_heuristic_${MODEL_SHORT}_abl_//")"
            LINES=$(wc -l < "${f}" 2>/dev/null || echo "?")
            echo "    ${MODEL_SHORT}/${BENCH}/${ABL_TAG}: ${f} (${LINES} records)"
        done
        shopt -u nullglob
    done
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Next steps:"
echo "    1. Score on CPU (if not done): bash run_ablations.sh --only-score"
echo "    2. Train routers per ablation:"
echo "       for f in data/router_features/*_abl_*.jsonl; do"
echo "         KEY=\$(basename \"\$f\" .jsonl | grep -oP 'abl_\\K.*')"
echo "         python scripts/train_router.py \\"
echo "           --config configs/humaneval/noisy.yaml \\"
echo "           --features \"\$f\" \\"
echo "           --output outputs/router/ablations/\${KEY}"
echo "       done"
echo "    3. Evaluate: python scripts/evaluate.py ..."
echo "══════════════════════════════════════════════════════"
