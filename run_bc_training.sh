#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Multi-Model Behavior Cloning Training
# ══════════════════════════════════════════════════════════════
#
# Trains BC policy models sequentially across multiple backbones.
#
# Usage:
#   bash run_bc_training.sh               # run all models
#   bash run_bc_training.sh --dry-run     # print commands only
#   bash run_bc_training.sh --gpus 2      # use 2 GPUs
#
# Prerequisites:
#   - Noisy trajectories exist under data/trajectories/humaneval_noisy/
#   - Environment activated (source ~/.venv/bin/activate)
#   - source exports.sh (for HF token if needed)
#
# ══════════════════════════════════════════════════════════════

# If invoked as `sh run_bc_training.sh`, re-exec under Bash.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Configuration ─────────────────────────────────────────────
BENCHMARK="humaneval"
CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"
NUM_GPUS="${NUM_GPUS:-1}"
DRY_RUN=false
BC_EPOCHS="${BC_EPOCHS:-5}"

SPLIT_DIR="data/trajectories/${BENCHMARK}_noisy"
NOISY_TRAJECTORIES="${SPLIT_DIR}/trajectories.jsonl"
BC_TRAIN_DATA="${SPLIT_DIR}/bc_train.jsonl"
BC_VAL_DATA="${SPLIT_DIR}/bc_val.jsonl"

QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"
COMMON_OVERRIDES="logging.wandb_mode=disabled"

LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGDIR}"

# ── Argument parsing ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        --gpus)     NUM_GPUS="$2"; shift 2 ;;
        --help|-h)
            head -30 "$0" | tail -27
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ensure_bc_splits() {
    if [[ -f "${BC_TRAIN_DATA}" && -f "${BC_VAL_DATA}" ]]; then
        return
    fi

    echo "  BC splits not found. Generating static BC splits..."

    if [[ ! -f "${NOISY_TRAJECTORIES}" ]]; then
        echo "  ERROR: Missing trajectories for split generation: ${NOISY_TRAJECTORIES}"
        exit 1
    fi

    SPLIT_CMD=(
        "python" scripts/save_static_splits.py
        --trajectories "${NOISY_TRAJECTORIES}"
        --output-dir "${SPLIT_DIR}"
        --prefix bc
        --success-only
        --max-perturbations-per-task 2
        --data-fraction 0.4
        --seed 42
    )

    echo "  Split generation command:"
    echo "    ${SPLIT_CMD[*]}"

    if ${DRY_RUN}; then
        echo "  [DRY RUN] Skipping BC split generation"
        return
    fi

    "${SPLIT_CMD[@]}"

    for f in "${BC_TRAIN_DATA}" "${BC_VAL_DATA}"; do
        if [[ ! -f "${f}" ]]; then
            echo "  ERROR: Expected generated split missing: ${f}"
            exit 1
        fi
    done

    echo "  ✓ BC splits generated"
}

# ── Model definitions ────────────────────────────────────────
# TAG | HF_MODEL_ID
declare -a MODEL_TAGS=(
    "qwen_coder_7b"
    "gemma_2_9b_it"
    "llama_3_1_8b_instruct"
    "qwen_coder_14b"
    "deepseek_coder_6_7b"
)

declare -A MODEL_HF_ID=(
    [qwen_coder_7b]="Qwen/Qwen2.5-Coder-7B-Instruct"
    [gemma_2_9b_it]="google/gemma-2-9b-it"
    [llama_3_1_8b_instruct]="meta-llama/Llama-3.1-8B-Instruct"
    [qwen_coder_14b]="Qwen/Qwen2.5-Coder-14B-Instruct"
    [deepseek_coder_6_7b]="deepseek-ai/deepseek-coder-6.7b-instruct"
)

# Optional per-model LoRA settings.
declare -A MODEL_LORA_R=(
    [qwen_coder_7b]="32"
    [gemma_2_9b_it]="32"
    [llama_3_1_8b_instruct]="32"
    [qwen_coder_14b]="32"
    [deepseek_coder_6_7b]="32"
)

declare -A MODEL_LORA_ALPHA=(
    [qwen_coder_7b]="64"
    [gemma_2_9b_it]="64"
    [llama_3_1_8b_instruct]="64"
    [qwen_coder_14b]="64"
    [deepseek_coder_6_7b]="64"
)

# ── Pre-flight checks ────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  Multi-Model BC Training"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  Benchmark:   ${BENCHMARK}"
echo "  Config:      ${CONFIG_NOISY}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Dry run:     ${DRY_RUN}"
echo "  BC epochs:   ${BC_EPOCHS}"
echo ""

ERRORS=0

if [[ ! -f "${CONFIG_NOISY}" ]]; then
    echo "  ERROR: Missing config: ${CONFIG_NOISY}"
    ERRORS=1
fi

if [[ ${ERRORS} -ne 0 ]]; then
    echo ""
    echo "  Fix the errors above before running."
    exit 1
fi

ensure_bc_splits

echo ""
echo "  Models to train (from pretrained backbones):"
for tag in "${MODEL_TAGS[@]}"; do
    echo "    ${tag}"
    echo "      HF base: ${MODEL_HF_ID[$tag]}"
done
echo ""

# ── Sequential training loop ─────────────────────────────────
TOTAL=${#MODEL_TAGS[@]}
CURRENT=0
FAILED=()
SUCCEEDED=()

for tag in "${MODEL_TAGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    hf_id="${MODEL_HF_ID[$tag]}"
    lora_r="${MODEL_LORA_R[$tag]}"
    lora_alpha="${MODEL_LORA_ALPHA[$tag]}"
    output_dir="outputs/policy/${BENCHMARK}_noisy_bc_${tag}"
    logfile="${LOGDIR}/bc_${tag}.log"

    echo "════════════════════════════════════════════════════════"
    echo "  [${CURRENT}/${TOTAL}] BC: ${tag}"
    echo "    HF base:   ${hf_id}"
    echo "    Output:    ${output_dir}"
    echo "    Log:       ${logfile}"
    echo "════════════════════════════════════════════════════════"
    echo ""

    OVERRIDES=(
        "${QUANT_OVERRIDE}"
        "policy.model_name=${hf_id}"
        "training.bc.epochs=${BC_EPOCHS}"
        "policy.lora.r=${lora_r}"
        "policy.lora.alpha=${lora_alpha}"
        "${COMMON_OVERRIDES}"
    )

    if [[ ${NUM_GPUS} -gt 1 ]]; then
        CMD=(
            accelerate launch
            --num_processes="${NUM_GPUS}"
            --multi_gpu
            scripts/train_policy.py
            --config "${CONFIG_NOISY}"
            --output "${output_dir}"
            --stage bc
            --train-data "${BC_TRAIN_DATA}"
            --val-data "${BC_VAL_DATA}"
            --overrides "${OVERRIDES[@]}"
        )
    else
        CMD=(
            "${PYTHON_BIN}" scripts/train_policy.py
            --config "${CONFIG_NOISY}"
            --output "${output_dir}"
            --stage bc
            --train-data "${BC_TRAIN_DATA}"
            --val-data "${BC_VAL_DATA}"
            --overrides "${OVERRIDES[@]}"
        )
    fi

    echo "  Training command:"
    echo "    ${CMD[*]}"
    echo ""

    if ${DRY_RUN}; then
        echo "  [DRY RUN] Skipping training"
        echo ""
        SUCCEEDED+=("${tag}")
        continue
    fi

    START_TIME=$(date +%s)
    echo "  Started at: $(date)"
    echo ""

    if "${CMD[@]}" 2>&1 | tee -a "${logfile}"; then
        END_TIME=$(date +%s)
        ELAPSED=$(( END_TIME - START_TIME ))
        echo ""
        echo "  ✓ ${tag} completed in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
        SUCCEEDED+=("${tag}")
    else
        END_TIME=$(date +%s)
        ELAPSED=$(( END_TIME - START_TIME ))
        echo ""
        echo "  ✗ ${tag} FAILED after $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
        echo "    Check log: ${logfile}"
        FAILED+=("${tag}")
    fi

    echo ""
done

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  BC Training Summary"
echo "══════════════════════════════════════════════════════════"
echo ""

if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo "  Succeeded (${#SUCCEEDED[@]}/${TOTAL}):"
    for tag in "${SUCCEEDED[@]}"; do
        echo "    ✓ ${tag}"
        echo "        policy: outputs/policy/${BENCHMARK}_noisy_bc_${tag}/"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "  Failed (${#FAILED[@]}/${TOTAL}):"
    for tag in "${FAILED[@]}"; do
        echo "    ✗ ${tag}  →  logs/bc_${tag}.log"
    done
fi

echo ""
echo "══════════════════════════════════════════════════════════"
