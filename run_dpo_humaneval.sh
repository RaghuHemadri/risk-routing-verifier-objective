#!/bin/bash
# ══════════════════════════════════════════════════════════════
# HumanEval: DPO preference training from an existing BC checkpoint
# ══════════════════════════════════════════════════════════════
#
# Mirrors run_pipeline.sh conventions (paths, overrides, multi-GPU).
# Run after you have:
#   - BC policy saved under BC_RESUME (LoRA adapter + tokenizer, e.g. .../final)
#   - Preference pairs JSONL (from generate_candidates / collection scripts)
#
# Usage:
#   bash run_dpo_humaneval.sh
#   PREFERENCE_DATA=data/candidates/custom.jsonl bash run_dpo_humaneval.sh
#   BC_RESUME=outputs/policy/other_run/final bash run_dpo_humaneval.sh
#   bash run_dpo_humaneval.sh --dry-run
#
# Prerequisites: same as run_pipeline.sh (see that file's header).
#
# ══════════════════════════════════════════════════════════════

set -euo pipefail

cd "$(dirname "$0")"

BENCHMARK=humaneval
CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"

NUM_GPUS=${NUM_GPUS:-2}

COMMON_OVERRIDES="logging.wandb_mode=disabled"
QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"

# Noisy trajectories (used for building episode-ID → task-ID mapping)
NOISY_TRAJECTORIES=${NOISY_TRAJECTORIES:-"data/trajectories/${BENCHMARK}_noisy/trajectories.jsonl"}

# BC checkpoint directory (PolicyModel.save → adapter + tokenizer)
BC_RESUME=${BC_RESUME:-"outputs/policy/humaneval_noisy_bc_qwen_coder_7b/final"}

# Preference pairs (same default path as run_pipeline.sh stage 5 output)
PREFERENCE_DATA=${PREFERENCE_DATA:-"data/candidates/${BENCHMARK}_noisy_heuristic.jsonl"}

# Static split directory and pre-split preference files
SPLIT_DIR="data/trajectories/${BENCHMARK}_noisy"
PREF_TRAIN_DATA="${SPLIT_DIR}/pref_train.jsonl"
PREF_VAL_DATA="${SPLIT_DIR}/pref_val.jsonl"
PREF_TEST_DATA="${SPLIT_DIR}/pref_test.jsonl"

# New output directory for this DPO run (does not overwrite BC folder by default)
POLICY_OUTPUT=${POLICY_OUTPUT:-"outputs/policy/${BENCHMARK}_noisy_dpo"}

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -f "${CONFIG_NOISY}" ]]; then
    echo "ERROR: Config not found: ${CONFIG_NOISY}"
    exit 1
fi
if [[ ${DRY_RUN} == false ]]; then
    if [[ ! -d "${BC_RESUME}" ]]; then
        echo "ERROR: BC checkpoint directory not found: ${BC_RESUME}"
        exit 1
    fi
    if [[ ! -f "${PREFERENCE_DATA}" ]]; then
        echo "ERROR: Preference data not found: ${PREFERENCE_DATA}"
        echo "  Set PREFERENCE_DATA=... to your candidates / preference JSONL."
        exit 1
    fi
fi
if [[ ! -f "${NOISY_TRAJECTORIES}" && ${DRY_RUN} == false ]]; then
    echo "WARN: Trajectories file missing: ${NOISY_TRAJECTORIES}"
    echo "  DPO episode-ID filtering will be skipped unless you set NOISY_TRAJECTORIES."
fi

run_cmd() {
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════"
    shift
    echo "$ $*"
    echo ""
    if [[ ${DRY_RUN} == false ]]; then
        "$@"
    fi
}

# ── Ensure static preference splits exist ─────────────────
ensure_pref_splits() {
    if [[ -f "${PREF_TRAIN_DATA}" && -f "${PREF_VAL_DATA}" && -f "${PREF_TEST_DATA}" ]]; then
        echo "✓ Preference splits already exist"
        echo "  train: ${PREF_TRAIN_DATA}"
        echo "  val:   ${PREF_VAL_DATA}"
        echo "  test:  ${PREF_TEST_DATA}"
        return
    fi
    echo "Generating static preference splits (all tasks, all perturbations, BC-aligned)..."
    python scripts/save_preference_splits.py \
        --trajectories "${NOISY_TRAJECTORIES}" \
        --preference-data "${PREFERENCE_DATA}" \
        --output-dir "${SPLIT_DIR}" \
        --seed 42
    echo "✓ Preference splits saved to ${SPLIT_DIR}"
}

if [[ ${DRY_RUN} == false ]]; then
    ensure_pref_splits
fi

ARGS=(
    scripts/train_policy.py
    --config "${CONFIG_NOISY}"
    --output "${POLICY_OUTPUT}"
    --stage preference
    --resume "${BC_RESUME}"
    --pref-train-data "${PREF_TRAIN_DATA}"
    # --pref-val-data "${PREF_VAL_DATA}"  # validation disabled for now
    --overrides
    "${QUANT_OVERRIDE}"
    "${COMMON_OVERRIDES}"
    training.preference.batch_size=2
    training.preference.gradient_accumulation_steps=16
    training.preference.concat_pairs=false
)

if [[ ${NUM_GPUS} -gt 1 ]]; then
    run_cmd "DPO (HumanEval, ${NUM_GPUS} GPUs, resume from BC)" \
        accelerate launch --num_processes="${NUM_GPUS}" --multi_gpu \
        "${ARGS[@]}"
else
    run_cmd "DPO (HumanEval, single GPU, resume from BC)" \
        python "${ARGS[@]}"
fi

echo ""
if [[ ${DRY_RUN} == false ]]; then
    echo "✓ DPO run complete. Final weights: ${POLICY_OUTPUT}/final"
    echo "  (config saved to ${POLICY_OUTPUT}/config.yaml)"
else
    echo "(dry-run: no training executed)"
fi
