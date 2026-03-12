#!/bin/bash
# ══════════════════════════════════════════════════════════════
# R2V-Agent: Full Pipeline — Local Execution (Multi-Benchmark)
# ══════════════════════════════════════════════════════════════
#
# Usage:
#   BENCHMARK=gaia      bash run_pipeline.sh          # GAIA benchmark (default)
#   BENCHMARK=alfworld  bash run_pipeline.sh          # ALFWorld benchmark
#   BENCHMARK=humaneval bash run_pipeline.sh          # HumanEval+ benchmark
#   bash run_pipeline.sh --benchmark gaia --from 3    # Resume from stage 3
#   bash run_pipeline.sh --only 5                     # Run only stage 5
#   bash run_pipeline.sh --dry-run                    # Print commands without executing
#
# Prerequisites:
#   - pip install -e ".[dev]"
#   - GPU with ≥24GB VRAM (A100/H200/4090 recommended)
#   - API key exported (GOOGLE_API_KEY, OPENAI_API_KEY, etc.)
#   - source exports.sh (for HuggingFace token)
#
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
# Benchmark name: gaia | alfworld | humaneval
BENCHMARK=${BENCHMARK:-gaia}

CONFIG_CLEAN="configs/${BENCHMARK}/clean.yaml"
CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"

# Number of GPUs available (set to 1 for single-GPU)
NUM_GPUS=${NUM_GPUS:-1}

# Teacher model overrides (uncomment/edit as needed)
TEACHER_OVERRIDES="teacher.provider=google teacher.model_name=gemini-3-flash-preview"

# Disable wandb by default for local runs
COMMON_OVERRIDES="logging.wandb_mode=disabled"

# Disable 4-bit quantization if bitsandbytes is problematic
QUANT_OVERRIDE="policy.quantization.load_in_4bit=false"

# Paths (auto-populated after each stage, parameterized by benchmark)
CLEAN_TRAJECTORIES=""           # Set after Stage 1
NOISY_TRAJECTORIES="data/trajectories/${BENCHMARK}_noisy/trajectories.jsonl"
POLICY_PATH="outputs/policy/${BENCHMARK}_noisy/final"
VERIFIER_PATH="outputs/verifier/${BENCHMARK}_noisy/final/verifier.pt"
CANDIDATES_PATH="data/candidates/${BENCHMARK}_noisy.jsonl"
ROUTER_FEATURES_PATH="data/router_features/${BENCHMARK}.jsonl"
ROUTER_PATH="outputs/router/${BENCHMARK}_noisy/router_final.pt"
RESULTS_DIR="results/${BENCHMARK}_noisy"

# ── Argument parsing ─────────────────────────────────────────
FROM_STAGE=0
ONLY_STAGE=0
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)  FROM_STAGE="$2"; shift 2 ;;
        --only)  ONLY_STAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

should_run() {
    local stage=$1
    if [[ ${ONLY_STAGE} -ne 0 ]]; then
        [[ ${stage} -eq ${ONLY_STAGE} ]]
    else
        [[ ${stage} -ge ${FROM_STAGE} ]]
    fi
}

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

# ── Stage 0: Smoke test ─────────────────────────────────────
if should_run 0; then
    SMOKE_CFG="configs/smoke_test_${BENCHMARK}.yaml"
    if [[ ! -f "${SMOKE_CFG}" ]]; then
        # Fall back to the benchmark's clean config with a small episode count
        SMOKE_CFG="${CONFIG_CLEAN}"
    fi
    run_cmd "Stage 0: Smoke test (1 instance, verify setup) [benchmark=${BENCHMARK}]" \
        python scripts/collect_trajectories.py \
            --config "${SMOKE_CFG}" \
            --output data/smoke_test \
            --num-episodes 1 \
            --seeds 1 \
            --overrides ${TEACHER_OVERRIDES} ${COMMON_OVERRIDES}
    echo "✓ Smoke test passed"
fi

# ── Stage 1: Collect clean teacher trajectories ─────────────
if should_run 1; then
    run_cmd "Stage 1: Collect clean teacher trajectories (500 tasks × 3 seeds = 1500 episodes)" \
        python scripts/collect_trajectories.py \
            --config ${CONFIG_CLEAN} \
            --output data/runs \
            --num-episodes 500 \
            --seeds 1 2 3 4 5 6 7 8 \
            --num-workers 4 \
            --overrides ${TEACHER_OVERRIDES} ${COMMON_OVERRIDES}

    # Find the latest run directory
    if [[ ${DRY_RUN} == false ]]; then
        CLEAN_TRAJECTORIES=$(ls -td data/runs/${BENCHMARK}_*/trajectories.jsonl 2>/dev/null | head -1)
        echo "✓ Clean trajectories: ${CLEAN_TRAJECTORIES}"
    fi
fi

# Auto-detect clean trajectories if resuming from a later stage
if [[ -z "${CLEAN_TRAJECTORIES}" ]]; then
    CLEAN_TRAJECTORIES=$(ls -td data/runs/${BENCHMARK}_*/trajectories.jsonl 2>/dev/null | head -1 || true)
fi

# ── Stage 2: Generate perturbations ─────────────────────────
if should_run 2; then
    if [[ -z "${CLEAN_TRAJECTORIES}" ]]; then
        echo "ERROR: No clean trajectories found. Run Stage 1 first."
        exit 1
    fi
    run_cmd "Stage 2: Generate perturbations (CPU-only, ~2 min)" \
        python -m scripts.generate_perturbations \
            --config ${CONFIG_NOISY} \
            --input "${CLEAN_TRAJECTORIES}" \
            --output "${NOISY_TRAJECTORIES}" \
            --seeds 1 2 3 \
            --include-clean
    echo "✓ Noisy trajectories: ${NOISY_TRAJECTORIES}"
fi

# ── Stage 3: Train BC policy ────────────────────────────────
if should_run 3; then
    run_cmd "Stage 3: Train policy (behavior cloning)" \
        python scripts/train_policy.py \
            --config ${CONFIG_NOISY} \
            --output outputs/policy/${BENCHMARK}_noisy \
            --stage bc \
            --trajectories "${NOISY_TRAJECTORIES}" \
            --overrides ${QUANT_OVERRIDE} ${COMMON_OVERRIDES}
    echo "✓ BC policy: ${POLICY_PATH}"
fi

# ── Stage 4: Train verifier ─────────────────────────────────
if should_run 4; then
    run_cmd "Stage 4: Train verifier" \
        python scripts/train_verifier.py \
            --config ${CONFIG_NOISY} \
            --output outputs/verifier/${BENCHMARK}_noisy \
            --trajectories "${NOISY_TRAJECTORIES}" \
            --overrides ${QUANT_OVERRIDE} verifier.mode=trained \
                training.verifier.epochs=3 training.verifier.batch_size=32 \
                policy.max_seq_len=2048 ${COMMON_OVERRIDES}
    echo "✓ Verifier: ${VERIFIER_PATH}"
fi

# ── Stage 5: Generate candidates for DPO ────────────────────
if should_run 5; then
    if [[ ${NUM_GPUS} -gt 1 ]]; then
        run_cmd "Stage 5: Generate DPO candidates (${NUM_GPUS} GPUs)" \
            bash scripts/launch_candidates.sh ${NUM_GPUS} \
                --config ${CONFIG_NOISY} \
                --policy-path ${POLICY_PATH} \
                --verifier-path ${VERIFIER_PATH} \
                --trajectories "${NOISY_TRAJECTORIES}" \
                --output "${CANDIDATES_PATH}" \
                --K 5 --batch-size 8 \
                --overrides ${QUANT_OVERRIDE} verifier.mode=trained ${COMMON_OVERRIDES}
    else
        run_cmd "Stage 5: Generate DPO candidates (single GPU)" \
            python scripts/generate_candidates.py \
                --config ${CONFIG_NOISY} \
                --policy-path ${POLICY_PATH} \
                --verifier-path ${VERIFIER_PATH} \
                --trajectories "${NOISY_TRAJECTORIES}" \
                --output "${CANDIDATES_PATH}" \
                --K 5 --batch-size 8 \
                --overrides ${QUANT_OVERRIDE} verifier.mode=trained ${COMMON_OVERRIDES}
    fi
    echo "✓ Candidates: ${CANDIDATES_PATH}"
fi

# ── Stage 6: Train policy (DPO) ─────────────────────────────
if should_run 6; then
    if [[ ${NUM_GPUS} -gt 1 ]]; then
        run_cmd "Stage 6: Train policy (DPO, ${NUM_GPUS} GPUs)" \
            accelerate launch --num_processes=${NUM_GPUS} --multi_gpu \
                scripts/train_policy.py \
                --config ${CONFIG_NOISY} \
                --output outputs/policy/${BENCHMARK}_noisy \
                --stage preference \
                --preference-data "${CANDIDATES_PATH}" \
                --overrides ${COMMON_OVERRIDES}
    else
        run_cmd "Stage 6: Train policy (DPO, single GPU)" \
            python scripts/train_policy.py \
                --config ${CONFIG_NOISY} \
                --output outputs/policy/${BENCHMARK}_noisy \
                --stage preference \
                --preference-data "${CANDIDATES_PATH}" \
                --overrides ${QUANT_OVERRIDE} ${COMMON_OVERRIDES}
    fi
    echo "✓ DPO policy: ${POLICY_PATH}"
fi

# ── Stage 7: Generate router features ───────────────────────
if should_run 7; then
    if [[ ${NUM_GPUS} -gt 1 ]]; then
        run_cmd "Stage 7: Generate router features (${NUM_GPUS} GPUs)" \
            bash scripts/launch_router_features.sh ${NUM_GPUS} \
                --config ${CONFIG_NOISY} \
                --policy-path ${POLICY_PATH} \
                --trajectories "${NOISY_TRAJECTORIES}" \
                --output "${ROUTER_FEATURES_PATH}" \
                --batch-size 16 --K 5 \
                --overrides ${QUANT_OVERRIDE} ${COMMON_OVERRIDES}
    else
        run_cmd "Stage 7: Generate router features (single GPU)" \
            python scripts/generate_router_features.py \
                --config ${CONFIG_NOISY} \
                --policy-path ${POLICY_PATH} \
                --trajectories "${NOISY_TRAJECTORIES}" \
                --output "${ROUTER_FEATURES_PATH}" \
                --batch-size 16 --K 5 \
                --overrides ${QUANT_OVERRIDE} ${COMMON_OVERRIDES}
    fi
    echo "✓ Router features: ${ROUTER_FEATURES_PATH}"
fi

# ── Stage 8: Train router ───────────────────────────────────
if should_run 8; then
    run_cmd "Stage 8: Train router (MLP, CPU-friendly)" \
        python scripts/train_router.py \
            --config ${CONFIG_NOISY} \
            --features "${ROUTER_FEATURES_PATH}" \
            --output outputs/router/${BENCHMARK}_noisy \
            --overrides ${COMMON_OVERRIDES}
    echo "✓ Router: ${ROUTER_PATH}"
fi

# ── Stage 9: Evaluate ───────────────────────────────────────
if should_run 9; then
    run_cmd "Stage 9: Evaluate all methods" \
        python scripts/evaluate.py \
            --config ${CONFIG_NOISY} \
            --features "${ROUTER_FEATURES_PATH}" \
            --trajectories "${NOISY_TRAJECTORIES}" \
            --router-path "${ROUTER_PATH}" \
            --output "${RESULTS_DIR}" \
            --seeds 1 2 3 \
            --methods r2v slm_only llm_only entropy_router \
            --overrides ${COMMON_OVERRIDES}
    echo "✓ Results: ${RESULTS_DIR}"
fi

# ── Stage 10: Ablations (optional) ──────────────────────────
if should_run 10; then
    run_cmd "Stage 10: Ablation studies" \
        python scripts/run_ablations.py \
            --config ${CONFIG_NOISY} \
            --features "${ROUTER_FEATURES_PATH}" \
            --trajectories "${NOISY_TRAJECTORIES}" \
            --router-path "${ROUTER_PATH}" \
            --output results/ablations/${BENCHMARK}_noisy \
            --overrides ${COMMON_OVERRIDES}
    echo "✓ Ablation results: results/ablations/${BENCHMARK}_noisy"
fi

echo ""
echo "══════════════════════════════════════════"
echo "  Pipeline complete!"
echo "══════════════════════════════════════════"
