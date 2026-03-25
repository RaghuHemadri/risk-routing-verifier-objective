#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Multi-Model Verifier Training from BC Checkpoints
# ══════════════════════════════════════════════════════════════
#
# Trains verifier models sequentially, one per BC-trained backbone.
# For each model:
#   1. Merges BC best/ checkpoint into a standalone HF model dir
#   2. Updates configs/humaneval/noisy.yaml to point at the merged model
#   3. Runs scripts/train_verifier.py (frozen backbone + trained MLP head)
#   4. Saves output to outputs/verifier/humaneval_noisy_<tag>/
#
# Usage:
#   bash run_verifier_training.sh              # run all 3 models
#   bash run_verifier_training.sh --dry-run    # print commands only
#   bash run_verifier_training.sh --gpus 2     # use 2 GPUs
#
# Prerequisites:
#   - BC checkpoints exist under outputs/policy/humaneval_noisy/
#   - Verifier splits exist under data/trajectories/humaneval_noisy/
#   - Environment activated (source /ext3/env.sh && conda activate pt311)
#   - source exports.sh (for HF_TOKEN)
#
# ══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Configuration ─────────────────────────────────────────────
BENCHMARK="humaneval"
CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"
NUM_GPUS="${NUM_GPUS:-4}"
DRY_RUN=false
VERIFIER_EPOCHS="${VERIFIER_EPOCHS:-5}"
VERIFIER_BATCH_SIZE="${VERIFIER_BATCH_SIZE:-1}"
VERIFIER_GRAD_ACCUM="${VERIFIER_GRAD_ACCUM:-8}"
VERIFIER_MAX_SEQ_LEN="${VERIFIER_MAX_SEQ_LEN:-2048}"
VERIFIER_NUM_WORKERS="${VERIFIER_NUM_WORKERS:-4}"
VERIFIER_PIN_MEMORY="${VERIFIER_PIN_MEMORY:-true}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.22}"   # (1-alpha)=0.78 upweights failure class
LDAM_C="${LDAM_C:-0.5}"             # LDAM margin: Δ_j = C / n_j^(1/4)
FROM_BASE=true
VERIFIER_FREEZE_BACKBONE="${VERIFIER_FREEZE_BACKBONE:-false}"
VERIFIER_USE_LORA="${VERIFIER_USE_LORA:-true}"
VERIFIER_LORA_R="${VERIFIER_LORA_R:-16}"
VERIFIER_LORA_ALPHA="${VERIFIER_LORA_ALPHA:-32}"
VERIFIER_LORA_DROPOUT="${VERIFIER_LORA_DROPOUT:-0.05}"

SPLIT_DIR="data/trajectories/${BENCHMARK}_noisy"
VERIFIER_TRAIN_DATA="${SPLIT_DIR}/verifier_train.jsonl"
VERIFIER_VAL_DATA="${SPLIT_DIR}/verifier_val.jsonl"
VERIFIER_TEST_DATA="${SPLIT_DIR}/verifier_test.jsonl"

QUANT_OVERRIDES=(
    "policy.quantization.load_in_4bit=false"
)
COMMON_OVERRIDES=(
    "logging.wandb_mode=disabled"
    "training.verifier.use_ldam=true"
    "training.verifier.ldam_C=${LDAM_C}"
)

BC_BASE="outputs/policy/${BENCHMARK}_noisy"
MERGED_BASE="outputs/merged"

LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGDIR}"

# ── Argument parsing ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        --gpus)     NUM_GPUS="$2"; shift 2 ;;
        --from-base) FROM_BASE=true; shift ;;
        --help|-h)
            head -24 "$0" | tail -21
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Model definitions ────────────────────────────────────────
# TAG | HF_MODEL_ID | BC_CHECKPOINT_DIR_NAME | LORA_R | LORA_ALPHA
declare -a MODEL_TAGS=(
    "qwen_coder_7b"
    "qwen_coder_14b"
    "deepseek_coder_6.7b"
)

declare -A MODEL_HF_ID=(
    [qwen_coder_7b]="Qwen/Qwen2.5-Coder-7B-Instruct"
    [qwen_coder_14b]="Qwen/Qwen2.5-Coder-14B-Instruct"
    [deepseek_coder_6.7b]="deepseek-ai/deepseek-coder-6.7b-instruct"
)

declare -A MODEL_BC_DIR=(
    [qwen_coder_7b]="bc_qwen_coder"
    [qwen_coder_14b]="bc_qwen2.5-coder-14b-instruct"
    [deepseek_coder_6.7b]="bc_deepseek-coder-6.7b-instruct"
)

declare -A MODEL_LORA_R=(
    [qwen_coder_7b]="32"
    [qwen_coder_14b]="32"
    [deepseek_coder_6.7b]="32"
)

declare -A MODEL_LORA_ALPHA=(
    [qwen_coder_7b]="64"
    [qwen_coder_14b]="64"
    [deepseek_coder_6.7b]="64"
)

# ── Pre-flight checks ────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  Multi-Model Verifier Training (from BC checkpoints)"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  Benchmark:   ${BENCHMARK}"
echo "  Config:      ${CONFIG_NOISY}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Dry run:     ${DRY_RUN}"
echo "  Epochs:      ${VERIFIER_EPOCHS}"
echo "  Batch size:  ${VERIFIER_BATCH_SIZE}"
echo "  Grad accum:  ${VERIFIER_GRAD_ACCUM}"
echo "  Eff. batch:  $(( VERIFIER_BATCH_SIZE * VERIFIER_GRAD_ACCUM ))"
echo "  Max seq len: ${VERIFIER_MAX_SEQ_LEN}"
echo "  Num workers: ${VERIFIER_NUM_WORKERS}"
echo "  Pin memory:  ${VERIFIER_PIN_MEMORY}"
echo "  Variants:    focal + bce"
echo "  Source:      $([[ ${FROM_BASE} == true ]] && echo 'pretrained base models' || echo 'merged BC checkpoints')"
echo "  Freeze BB:   ${VERIFIER_FREEZE_BACKBONE}"
echo "  Use LoRA:    ${VERIFIER_USE_LORA}"
echo ""

ERRORS=0

for f in "${VERIFIER_TRAIN_DATA}" "${VERIFIER_VAL_DATA}" "${VERIFIER_TEST_DATA}"; do
    if [[ ! -f "${f}" ]]; then
        echo "  ERROR: Missing verifier split: ${f}"
        ERRORS=1
    fi
done

if [[ ${FROM_BASE} == false ]]; then
    for tag in "${MODEL_TAGS[@]}"; do
        bc_best="${BC_BASE}/${MODEL_BC_DIR[$tag]}/best"
        if [[ ! -f "${bc_best}/model.safetensors" ]]; then
            echo "  ERROR: Missing BC checkpoint: ${bc_best}/model.safetensors"
            ERRORS=1
        else
            echo "  ✓ BC best checkpoint: ${bc_best}"
        fi
    done
fi

if [[ ${ERRORS} -ne 0 ]]; then
    echo ""
    echo "  Fix the errors above before running."
    exit 1
fi

echo ""
if [[ ${FROM_BASE} == true ]]; then
    echo "  Models to train (pretrained backbones):"
else
    echo "  Models to train (backbone from BC best/ weights):"
fi
for tag in "${MODEL_TAGS[@]}"; do
    echo "    ${tag}"
    echo "      HF base:    ${MODEL_HF_ID[$tag]}"
    if [[ ${FROM_BASE} == false ]]; then
        echo "      BC weights:  ${BC_BASE}/${MODEL_BC_DIR[$tag]}/best"
        echo "      Merged to:   ${MERGED_BASE}/${tag}"
    fi
done
echo ""

if [[ ${FROM_BASE} == false ]]; then
    # Back up the original noisy.yaml
    cp "${CONFIG_NOISY}" "${CONFIG_NOISY}.bak"
    echo "  Backed up ${CONFIG_NOISY} → ${CONFIG_NOISY}.bak"
    echo ""

    # ── Cleanup trap: restore noisy.yaml on exit ──────────────────
    cleanup() {
        if [[ -f "${CONFIG_NOISY}.bak" ]]; then
            cp "${CONFIG_NOISY}.bak" "${CONFIG_NOISY}"
            echo ""
            echo "  Restored ${CONFIG_NOISY} from backup"
        fi
    }
    trap cleanup EXIT
fi

# ── Sequential training loop ─────────────────────────────────
TOTAL=${#MODEL_TAGS[@]}
TOTAL_RUNS=$(( TOTAL * 2 ))
CURRENT=0
FAILED=()
SUCCEEDED=()

for tag in "${MODEL_TAGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    hf_id="${MODEL_HF_ID[$tag]}"
    bc_best="${BC_BASE}/${MODEL_BC_DIR[$tag]}/best"
    merged_dir="${MERGED_BASE}/${tag}"
    lora_tag=""
    if [[ "${VERIFIER_USE_LORA}" == "true" ]]; then
        lora_tag="_lora-r${VERIFIER_LORA_R}"
    fi
    base_output_dir="outputs/verifier/${BENCHMARK}_noisy_${tag}${lora_tag}"
    base_logfile="${LOGDIR}/verifier_${tag}.log"

    echo "════════════════════════════════════════════════════════"
    echo "  [${CURRENT}/${TOTAL}] Verifier: ${tag}"
    echo "    HF base:     ${hf_id}"
    if [[ ${FROM_BASE} == false ]]; then
        echo "    BC best/:    ${bc_best}"
        echo "    Merged dir:  ${merged_dir}"
    fi
    echo "    Output base: ${base_output_dir}"
    echo "    Log base:    ${base_logfile}"
    echo "════════════════════════════════════════════════════════"
    echo ""

    BACKBONE_PATH="${hf_id}"
    if [[ ${FROM_BASE} == false ]]; then
        # ── Step 1: Merge BC checkpoint into a standalone HF model dir ──
        echo "  Step 1: Merging BC checkpoint → ${merged_dir}"
        MERGE_CMD=(
            python scripts/merge_bc_for_verifier.py
            --hf-model-id "${hf_id}"
            --bc-checkpoint "${bc_best}"
            --output "${merged_dir}"
            --lora-r "${MODEL_LORA_R[$tag]}"
            --lora-alpha "${MODEL_LORA_ALPHA[$tag]}"
        )
        echo "    ${MERGE_CMD[*]}"

        if ${DRY_RUN}; then
            echo "    [DRY RUN] Skipping merge"
        else
            if ! "${MERGE_CMD[@]}" 2>&1 | tee -a "${base_logfile}"; then
                echo "  ✗ Merge FAILED for ${tag}"
                FAILED+=("${tag}:merge")
                echo ""
                continue
            fi
        fi
        echo ""

        # ── Step 2: Update noisy.yaml to point at the merged model dir ──
        # Use absolute path so train_verifier.py can find it
        MERGED_ABS="$(cd "${SCRIPT_DIR}" && realpath "${merged_dir}" 2>/dev/null || echo "${SCRIPT_DIR}/${merged_dir}")"

        python3 -c "
import sys

config_path = '${CONFIG_NOISY}'
new_backbone = '${MERGED_ABS}'

with open(config_path, 'r') as f:
    lines = f.readlines()

found = False
for i, line in enumerate(lines):
    stripped = line.lstrip()
    if stripped.startswith('backbone:') and i > 0:
        indent = len(line) - len(stripped)
        lines[i] = ' ' * indent + 'backbone: ' + new_backbone + '\n'
        found = True
        break

if not found:
    print(f'WARNING: Could not find backbone line in {config_path}', file=sys.stderr)
    sys.exit(1)

with open(config_path, 'w') as f:
    f.writelines(lines)

print(f'  Updated {config_path}:')
print(f'    verifier.trained.backbone = {new_backbone}')
"
        echo ""
        BACKBONE_PATH="${MERGED_ABS}"
    fi

    # ── Step 3: Run verifier training (two variants, sequentially) ──
    for variant in focal bce; do
        output_dir="${base_output_dir}_${variant}"
        logfile="${LOGDIR}/verifier_${tag}_${variant}.log"

        OVERRIDES=(
            "${QUANT_OVERRIDES[@]}"
            "verifier.mode=trained"
            "verifier.trained.backbone=${BACKBONE_PATH}"
            "verifier.trained.freeze_backbone=${VERIFIER_FREEZE_BACKBONE}"
            "verifier.trained.lora.enabled=${VERIFIER_USE_LORA}"
            "verifier.trained.lora.r=${VERIFIER_LORA_R}"
            "verifier.trained.lora.alpha=${VERIFIER_LORA_ALPHA}"
            "verifier.trained.lora.dropout=${VERIFIER_LORA_DROPOUT}"
            "training.verifier.epochs=${VERIFIER_EPOCHS}"
            "training.verifier.batch_size=${VERIFIER_BATCH_SIZE}"
            "training.verifier.gradient_accumulation_steps=${VERIFIER_GRAD_ACCUM}"
            "training.verifier.num_workers=${VERIFIER_NUM_WORKERS}"
            "training.verifier.pin_memory=${VERIFIER_PIN_MEMORY}"
            "policy.max_seq_len=${VERIFIER_MAX_SEQ_LEN}"
            "${COMMON_OVERRIDES[@]}"
        )

        if [[ "${variant}" == "focal" ]]; then
            OVERRIDES+=(
                "training.verifier.use_focal_loss=true"
                "training.verifier.focal_gamma=${FOCAL_GAMMA}"
                "training.verifier.focal_alpha=${FOCAL_ALPHA}"
            )
        else
            OVERRIDES+=("training.verifier.use_focal_loss=false")
        fi

        if [[ ${NUM_GPUS} -gt 1 ]]; then
            CMD=(
                accelerate launch
                --num_processes="${NUM_GPUS}"
                --multi_gpu
                scripts/train_verifier.py
                --config "${CONFIG_NOISY}"
                --output "${output_dir}"
                --train-data "${VERIFIER_TRAIN_DATA}"
                --val-data "${VERIFIER_VAL_DATA}"
                --test-data "${VERIFIER_TEST_DATA}"
                --overrides "${OVERRIDES[@]}"
            )
        else
            CMD=(
                python scripts/train_verifier.py
                --config "${CONFIG_NOISY}"
                --output "${output_dir}"
                --train-data "${VERIFIER_TRAIN_DATA}"
                --val-data "${VERIFIER_VAL_DATA}"
                --test-data "${VERIFIER_TEST_DATA}"
                --overrides "${OVERRIDES[@]}"
            )
        fi

        echo "  Step 3 (${variant}): Training verifier"
        echo "    Output: ${output_dir}"
        echo "    Log:    ${logfile}"
        echo "    ${CMD[*]}"
        echo ""

        if ${DRY_RUN}; then
            echo "  [DRY RUN] Skipping training (${variant})"
            echo ""
            SUCCEEDED+=("${tag}:${variant}")
            continue
        fi

        START_TIME=$(date +%s)
        echo "  Started at: $(date)"
        echo ""

        if "${CMD[@]}" 2>&1 | tee -a "${logfile}"; then
            END_TIME=$(date +%s)
            ELAPSED=$(( END_TIME - START_TIME ))
            echo ""
            echo "  ✓ ${tag}:${variant} completed in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
            SUCCEEDED+=("${tag}:${variant}")
        else
            END_TIME=$(date +%s)
            ELAPSED=$(( END_TIME - START_TIME ))
            echo ""
            echo "  ✗ ${tag}:${variant} FAILED after $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
            echo "    Check log: ${logfile}"
            FAILED+=("${tag}:${variant}")
        fi

        echo ""
    done
done

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Verifier Training Summary"
echo "══════════════════════════════════════════════════════════"
echo ""

if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo "  Succeeded (${#SUCCEEDED[@]}/${TOTAL_RUNS}):"
    for item in "${SUCCEEDED[@]}"; do
        tag="${item%%:*}"
        variant="${item##*:}"
        echo "    ✓ ${tag}:${variant}"
        echo "        verifier:  outputs/verifier/${BENCHMARK}_noisy_${tag}_${variant}/"
        echo "        backbone:  ${MERGED_BASE}/${tag}/"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "  Failed (${#FAILED[@]}/${TOTAL_RUNS}):"
    for item in "${FAILED[@]}"; do
        tag="${item%%:*}"
        variant="${item##*:}"
        if [[ "${variant}" == "merge" ]]; then
            echo "    ✗ ${tag}:${variant}  →  logs/verifier_${tag}.log"
        else
            echo "    ✗ ${tag}:${variant}  →  logs/verifier_${tag}_${variant}.log"
        fi
    done
fi

echo ""
echo "══════════════════════════════════════════════════════════"
