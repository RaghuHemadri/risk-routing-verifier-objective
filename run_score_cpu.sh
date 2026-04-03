#!/usr/bin/env bash
# --------------------------------------------------------------
# CPU-only Router Feature Scoring (Phase 2: --score-only)
# --------------------------------------------------------------
#
# Runs the verifier scoring phase on CPU using cached candidates
# produced by the GPU generation phase (--generate-only) in
# run_pipeline_updated.sh.
#
# Delegates to scripts/run_router_features_humaneval.sh with
# SCORE_ONLY=true — reuses existing sharding, merging, and
# feature analysis logic.
#
# Usage:
#   bash run_score_cpu.sh --model qwen7
#   bash run_score_cpu.sh --model llama --dry-run
#   bash run_score_cpu.sh --model qwen14 --gpus 4
#
# Models (--model):
#   qwen7   Qwen/Qwen2.5-Coder-7B-Instruct
#   qwen14  Qwen/Qwen2.5-Coder-14B-Instruct
#   llama   meta-llama/Llama-3.1-8B-Instruct
#   gemma   google/gemma-2-9b-it
#
# --------------------------------------------------------------

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Defaults ─────────────────────────────────────────────────
MODEL_SHORT=""
BENCHMARK="humaneval"
NUM_GPUS="${NUM_GPUS:-2}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-32}"
ROUTER_K="${ROUTER_K:-5}"
DRY_RUN=false
PASSTHROUGH_ARGS=()

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL_SHORT="$2"; shift 2 ;;
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        --batch-size)  ROUTER_BATCH_SIZE="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --help|-h)
            head -27 "$0" | tail -25
            exit 0
            ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_SHORT}" ]]; then
    echo "ERROR: --model is required.  Choose one of: qwen7  qwen14  llama  gemma"
    exit 1
fi

# ── Model registry (mirrors run_pipeline_updated.sh) ─────────
declare -A HF_ID=(
    [qwen7]="Qwen/Qwen2.5-Coder-7B-Instruct"
    [qwen14]="Qwen/Qwen2.5-Coder-14B-Instruct"
    [llama]="meta-llama/Llama-3.1-8B-Instruct"
    [gemma]="google/gemma-2-9b-it"
)

if [[ -z "${HF_ID[${MODEL_SHORT}]+_}" ]]; then
    echo "ERROR: Unknown model '${MODEL_SHORT}'.  Choose one of: qwen7  qwen14  llama  gemma"
    exit 1
fi

MODEL_HF="${HF_ID[${MODEL_SHORT}]}"

# ── Derived paths ────────────────────────────────────────────
CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"
NOISY_TRAJECTORIES="data/trajectories/${BENCHMARK}_noisy/trajectories.jsonl"
DPO_CHECKPOINT="outputs/policy/${BENCHMARK}_noisy_dpo_${MODEL_SHORT}/final"
ROUTER_FEATURES_FILE="data/router_features/${BENCHMARK}_noisy_heuristic_${MODEL_SHORT}.jsonl"

# ── Banner ───────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "  CPU Score-Only — ${MODEL_SHORT} (${MODEL_HF})"
echo "=========================================================="
echo ""
echo "  Benchmark:       ${BENCHMARK}"
echo "  Model:           ${MODEL_SHORT} (${MODEL_HF})"
echo "  DPO checkpoint:  ${DPO_CHECKPOINT}"
echo "  Trajectories:    ${NOISY_TRAJECTORIES}"
echo "  Output:          ${ROUTER_FEATURES_FILE}"
echo "  Num shards:      ${NUM_GPUS}"
echo "  Batch size:      ${ROUTER_BATCH_SIZE}"
echo "  Dry run:         ${DRY_RUN}"
echo ""

# ── Delegate to run_router_features_humaneval.sh ─────────────
export POLICY_PATH="${DPO_CHECKPOINT}"
export TRAJECTORIES="${NOISY_TRAJECTORIES}"
export CONFIG="${CONFIG_NOISY}"
export OUTPUT="${ROUTER_FEATURES_FILE}"
export K="${ROUTER_K}"
export BATCH_SIZE="${ROUTER_BATCH_SIZE}"
export NUM_GPUS="${NUM_GPUS}"
export SCORE_ONLY="true"
export EXTRA_OVERRIDES="policy.model_name=${MODEL_HF}"

DRY_FLAG=""
[[ ${DRY_RUN} == true ]] && DRY_FLAG="--dry-run"

exec bash scripts/run_router_features_humaneval.sh ${DRY_FLAG}
