#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Collect DPO Preference Trajectories (Heuristic Verifier)
# ══════════════════════════════════════════════════════════════
#
# Generates preference pairs (chosen/rejected) for DPO training by:
#   1. Loading the BC-trained SLM policy
#   2. Sampling K candidate actions per trajectory step
#   3. Scoring candidates with the heuristic verifier
#   4. Saving (best, worst) pairs as preference data
#
# This script does NOT train — only collects preference data.
#
# Usage:
#   bash run_collect_preferences.sh                  # single GPU, defaults
#   bash run_collect_preferences.sh --gpus 4         # multi-GPU sharded
#   bash run_collect_preferences.sh --dry-run        # print commands only
#   bash run_collect_preferences.sh --K 8            # 8 candidates/step
#   bash run_collect_preferences.sh --batch-size 4   # generation batch size
#   NUM_GPUS=2 bash run_collect_preferences.sh       # env-var GPU count
#
# Prerequisites:
#   - BC policy checkpoint exists at POLICY_PATH
#   - Noisy trajectory data exists at TRAJECTORIES_PATH
#   - source exports.sh (for HF_TOKEN)
#
# ══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Configuration ─────────────────────────────────────────────
BENCHMARK="humaneval"
CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"

# Policy: BC-trained Qwen2.5-Coder-7B (PEFT adapter export)
POLICY_PATH="outputs/policy/${BENCHMARK}_noisy_bc_qwen_coder_7b/final"

# Trajectories: existing noisy data
TRAJECTORIES_PATH="data/trajectories/${BENCHMARK}_noisy/trajectories.jsonl"

# Output: preference pairs
OUTPUT_DIR="data/candidates"
OUTPUT_FILE="${OUTPUT_DIR}/${BENCHMARK}_noisy_heuristic.jsonl"

# Generation parameters
NUM_GPUS="${NUM_GPUS:-1}"
K="${K:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
VERIFIER_BATCH_SIZE="${VERIFIER_BATCH_SIZE:-8}"
GEN_MICRO_BATCH="${GEN_MICRO_BATCH:-0}"
DRY_RUN=false
NO_RESUME=false
STORE_ALL_CANDIDATES=false

QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"
COMMON_OVERRIDES="logging.wandb_mode=disabled"

LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGDIR}"

# ── Argument parsing ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)       DRY_RUN=true; shift ;;
        --gpus)          NUM_GPUS="$2"; shift 2 ;;
        --K)             K="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --policy-path)   POLICY_PATH="$2"; shift 2 ;;
        --trajectories)  TRAJECTORIES_PATH="$2"; shift 2 ;;
        --output)        OUTPUT_FILE="$2"; shift 2 ;;
        --no-resume)     NO_RESUME=true; shift ;;
        --store-all)     STORE_ALL_CANDIDATES=true; shift ;;
        --help|-h)
            head -30 "$0" | tail -27
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LOGFILE="${LOGDIR}/collect_preferences_$(date +%Y%m%d_%H%M%S).log"

# ── Pre-flight checks ────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  DPO Preference Data Collection (Heuristic Verifier)"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  Benchmark:       ${BENCHMARK}"
echo "  Config:          ${CONFIG_NOISY}"
echo "  Policy:          ${POLICY_PATH}"
echo "  Trajectories:    ${TRAJECTORIES_PATH}"
echo "  Output:          ${OUTPUT_FILE}"
echo "  GPUs:            ${NUM_GPUS}"
echo "  K (candidates):  ${K}"
echo "  Batch size:      ${BATCH_SIZE}"
echo "  Max new tokens:  ${MAX_NEW_TOKENS}"
echo "  Verifier:        heuristic (rule-based, no GPU needed)"
echo "  Dry run:         ${DRY_RUN}"
echo "  Log:             ${LOGFILE}"
echo ""

ERRORS=0

if [[ ! -f "${TRAJECTORIES_PATH}" ]]; then
    echo "  ERROR: Trajectories not found: ${TRAJECTORIES_PATH}"
    ERRORS=1
else
    TRAJ_LINES=$(wc -l < "${TRAJECTORIES_PATH}")
    echo "  ✓ Trajectories: ${TRAJECTORIES_PATH} (${TRAJ_LINES} lines)"
fi

if [[ ! -d "${POLICY_PATH}" ]]; then
    # Try best/ and final/ subdirectories
    if [[ -d "${POLICY_PATH}/final" ]]; then
        POLICY_PATH="${POLICY_PATH}/final"
        echo "  ✓ Policy (final): ${POLICY_PATH}"
    elif [[ -d "${POLICY_PATH}/best" ]]; then
        POLICY_PATH="${POLICY_PATH}/best"
        echo "  ✓ Policy (best): ${POLICY_PATH}"
    else
        echo "  ERROR: Policy checkpoint not found: ${POLICY_PATH}"
        ERRORS=1
    fi
else
    echo "  ✓ Policy: ${POLICY_PATH}"
fi

if [[ ! -f "${CONFIG_NOISY}" ]]; then
    echo "  ERROR: Config not found: ${CONFIG_NOISY}"
    ERRORS=1
else
    echo "  ✓ Config: ${CONFIG_NOISY}"
fi

if [[ ${ERRORS} -ne 0 ]]; then
    echo ""
    echo "  Fix the errors above before running."
    exit 1
fi

echo ""

# ── Build common args ─────────────────────────────────────────
COMMON_ARGS=(
    --config "${CONFIG_NOISY}"
    --policy-path "${POLICY_PATH}"
    --trajectories "${TRAJECTORIES_PATH}"
    --output "${OUTPUT_FILE}"
    --K "${K}"
    --batch-size "${BATCH_SIZE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --verifier-batch-size "${VERIFIER_BATCH_SIZE}"
    --overrides
        "${QUANT_OVERRIDE}"
        "${COMMON_OVERRIDES}"
        "verifier.mode=heuristic"
        "verifier.heuristic.run_code=true"
        "verifier.heuristic.benchmark=humaneval"
        "policy.model_name=Qwen/Qwen2.5-Coder-7B-Instruct"
)

if [[ "${GEN_MICRO_BATCH}" -gt 0 ]]; then
    COMMON_ARGS+=(--gen-micro-batch-size "${GEN_MICRO_BATCH}")
fi

if [[ "${NO_RESUME}" == true ]]; then
    COMMON_ARGS+=(--no-resume)
fi

if [[ "${STORE_ALL_CANDIDATES}" == true ]]; then
    COMMON_ARGS+=(--store-all-candidates)
fi

# ── Run ───────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

run_cmd() {
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════"
    shift
    echo "$ $*"
    echo ""
    if [[ ${DRY_RUN} == false ]]; then
        "$@"
    else
        echo "  [DRY RUN] Skipping execution"
    fi
}

START_TIME=$(date +%s)

if [[ ${NUM_GPUS} -gt 1 ]]; then
    # ── Multi-GPU: use launch_candidates.sh for sharded generation ──
    run_cmd "Collecting preferences (${NUM_GPUS} GPUs, K=${K}, heuristic verifier)" \
        bash scripts/launch_candidates.sh "${NUM_GPUS}" \
            "${COMMON_ARGS[@]}" \
        2>&1 | tee -a "${LOGFILE}"
else
    # ── Single GPU ──
    run_cmd "Collecting preferences (single GPU, K=${K}, heuristic verifier)" \
        python scripts/generate_candidates.py \
            "${COMMON_ARGS[@]}" \
        2>&1 | tee -a "${LOGFILE}"
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Preference Collection Complete"
echo "══════════════════════════════════════════════════════════"
echo ""

if [[ -f "${OUTPUT_FILE}" ]]; then
    PAIR_COUNT=$(wc -l < "${OUTPUT_FILE}")
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo "  Output:     ${OUTPUT_FILE}"
    echo "  Pairs:      ${PAIR_COUNT}"
    echo "  File size:  ${FILE_SIZE}"
else
    echo "  WARNING: Output file not found at ${OUTPUT_FILE}"
    echo "  Check log: ${LOGFILE}"
fi

echo "  Duration:   $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "  Log:        ${LOGFILE}"
echo ""
echo "  Next step — train DPO policy:"
echo "    python scripts/train_policy.py \\"
echo "        --config ${CONFIG_NOISY} \\"
echo "        --output outputs/policy/${BENCHMARK}_noisy \\"
echo "        --stage preference \\"
echo "        --preference-data ${OUTPUT_FILE} \\"
echo "        --overrides ${QUANT_OVERRIDE} ${COMMON_OVERRIDES}"
echo ""
echo "══════════════════════════════════════════════════════════"
