#!/bin/bash
# Launch multi-GPU router-feature generation with episode sharding.
#
# Each GPU gets its own process handling a non-overlapping slice of episodes.
# Both the policy and verifier are loaded independently on each GPU.
#
# Usage:
#   bash scripts/launch_router_features.sh <NUM_GPUS> [generate_router_features.py args...]
#
# Example:
#   bash scripts/launch_router_features.sh 4 \
#       --config configs/gaia/noisy.yaml \
#       --policy-path outputs/policy/gaia_noisy/final \
#       --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
#       --output data/router_features/gaia.jsonl \
#       --batch-size 4 --K 5
#
# After all shards finish the script auto-merges.  You can also merge manually:
#   python scripts/generate_router_features.py --merge \
#       --output data/router_features/gaia.jsonl
#
# Resume after a crash: just re-run the same command — each worker detects
# already-completed episodes in its shard file and skips them.

set -euo pipefail

NUM_GPUS="${1:?Usage: $0 <NUM_GPUS> [generate_router_features.py args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/generate_router_features.py"

echo "=== Launching ${NUM_GPUS} router-feature workers ==="
echo "Args: $@"
echo ""

PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    LOG="generate_router_features_shard${i}.log"
    echo "[GPU ${i}] Starting shard ${i}/${NUM_GPUS} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=${i} python "${GEN_SCRIPT}" \
        --shard-id ${i} \
        --num-shards ${NUM_GPUS} \
        "$@" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
    echo "[GPU ${i}] PID=${PIDS[-1]}"
done

echo ""
echo "All ${NUM_GPUS} workers launched.  PIDs: ${PIDS[*]}"
echo "Monitor progress:  tail -f generate_router_features_shard*.log"
echo ""

# Wait for all workers
FAILED=0
for ((i=0; i<NUM_GPUS; i++)); do
    if wait ${PIDS[$i]}; then
        echo "[GPU ${i}] ✓ Completed (PID ${PIDS[$i]})"
    else
        echo "[GPU ${i}] ✗ FAILED (PID ${PIDS[$i]}, exit=$?)"
        FAILED=$((FAILED+1))
    fi
done

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "ERROR: ${FAILED} shard(s) failed. Check logs before merging."
    exit 1
fi

# Auto-merge
echo ""
echo "=== All shards complete, merging... ==="

# Find the --output arg value from the original args
OUTPUT=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--output" ]] && ((i+1 < ${#ARGS[@]})); then
        OUTPUT="${ARGS[$((i+1))]}"
        break
    fi
done

if [ -n "${OUTPUT}" ]; then
    python "${GEN_SCRIPT}" --merge --output "${OUTPUT}"
    echo "=== Done! Final output: ${OUTPUT} ==="
else
    echo "Could not find --output arg; merge manually:"
    echo "  python scripts/generate_router_features.py --merge --output <path>"
fi
