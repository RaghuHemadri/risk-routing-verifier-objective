#!/bin/bash
# ══════════════════════════════════════════════════════════════
# R2V-Agent: Full Pipeline Execution via SLURM
# ══════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/slurm/run_all.sh webarena    # Full WebArena pipeline
#   bash scripts/slurm/run_all.sh swebench    # Full SWE-bench pipeline
#   bash scripts/slurm/run_all.sh all         # Both benchmarks
#
# Each stage submits as a SLURM job with dependency on the previous.
# Monitor with: squeue -u $USER
# ══════════════════════════════════════════════════════════════

set -euo pipefail

BENCHMARK=${1:-webarena}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs

run_pipeline() {
    local bench=$1
    echo "═══════════════════════════════════════════════"
    echo "  Starting pipeline for: ${bench}"
    echo "═══════════════════════════════════════════════"

    # Stage 1: Collect teacher trajectories
    JID1=$(sbatch --parsable ${SCRIPT_DIR}/01_collect.sh ${bench})
    echo "[Stage 1] Collect trajectories: Job ${JID1}"

    # Stage 2: Generate perturbations (depends on Stage 1)
    JID2=$(sbatch --parsable --dependency=afterok:${JID1} ${SCRIPT_DIR}/02_perturb.sh ${bench})
    echo "[Stage 2] Generate perturbations: Job ${JID2}"

    # Stage 3a: Train policy BC (depends on Stage 2)
    JID3=$(sbatch --parsable --dependency=afterok:${JID2} ${SCRIPT_DIR}/03_train_policy.sh ${bench} noisy bc)
    echo "[Stage 3a] Train policy (BC): Job ${JID3}"

    # Stage 3b: Train verifier (depends on Stage 1, parallel with 3a)
    JID3B=$(sbatch --parsable --dependency=afterok:${JID1} ${SCRIPT_DIR}/04_train_verifier.sh ${bench})
    echo "[Stage 3b] Train verifier: Job ${JID3B}"

    # Stage 4: Generate candidates for DPO (depends on 3a and 3b)
    JID4=$(sbatch --parsable --dependency=afterok:${JID3}:${JID3B} ${SCRIPT_DIR}/05_generate_candidates.sh ${bench})
    echo "[Stage 4] Generate candidates: Job ${JID4}"

    # Stage 5: Train policy DPO (depends on 4)
    JID5=$(sbatch --parsable --dependency=afterok:${JID4} ${SCRIPT_DIR}/03_train_policy.sh ${bench} noisy preference)
    echo "[Stage 5] Train policy (DPO): Job ${JID5}"

    # Stage 6: Generate router features (depends on 5)
    JID6=$(sbatch --parsable --dependency=afterok:${JID5} ${SCRIPT_DIR}/06_generate_router_features.sh ${bench})
    echo "[Stage 6] Generate router features: Job ${JID6}"

    # Stage 7: Train router (depends on 6)
    JID7=$(sbatch --parsable --dependency=afterok:${JID6} ${SCRIPT_DIR}/07_train_router.sh ${bench})
    echo "[Stage 7] Train router: Job ${JID7}"

    # Stage 8: Evaluate (depends on 7)
    JID8=$(sbatch --parsable --dependency=afterok:${JID7} ${SCRIPT_DIR}/08_evaluate.sh ${bench} noisy)
    echo "[Stage 8] Evaluate (noisy): Job ${JID8}"

    # Stage 8b: Evaluate clean (depends on 7, parallel)
    JID8B=$(sbatch --parsable --dependency=afterok:${JID7} ${SCRIPT_DIR}/08_evaluate.sh ${bench} clean)
    echo "[Stage 8b] Evaluate (clean): Job ${JID8B}"

    # Stage 9: Ablations (depends on 7)
    JID9=$(sbatch --parsable --dependency=afterok:${JID7} ${SCRIPT_DIR}/09_ablations.sh ${bench})
    echo "[Stage 9] Ablations: Job ${JID9}"

    echo ""
    echo "Pipeline submitted for ${bench}!"
    echo "Monitor: squeue -u \$USER"
    echo "Last job ID: ${JID9}"
}

if [ "${BENCHMARK}" = "all" ]; then
    run_pipeline webarena
    run_pipeline swebench
else
    run_pipeline ${BENCHMARK}
fi
