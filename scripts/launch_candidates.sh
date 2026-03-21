#!/bin/bash
# Launch multi-GPU candidate generation with episode sharding.
#
# Each GPU gets its own process handling a non-overlapping slice of episodes.
# Both the policy model and verifier are loaded on each GPU independently
# (they fit in ~16GB bf16 each, well within a 24GB+ GPU).
#
# Usage:
#   bash scripts/launch_candidates.sh <NUM_GPUS> [generate_candidates.py args...]
#
# Example:
#   bash scripts/launch_candidates.sh 4 \
#       --config configs/humaneval/noisy.yaml \
#       --policy-path outputs/policy/humaneval_noisy/final \
#       --verifier-path outputs/verifier/humaneval_noisy/final/verifier.pt \
#       --trajectories data/trajectories/humaneval_noisy/trajectories.jsonl \
#       --output data/candidates/humaneval_noisy.jsonl \
#       --K 5 --batch-size 8
#
# After all shards finish, merge:
#   python scripts/generate_candidates.py --merge \
#       --output data/candidates/humaneval_noisy.jsonl

set -euo pipefail

NUM_GPUS="${1:?Usage: $0 <NUM_GPUS> [generate_candidates.py args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/generate_candidates.py"

echo "=== Launching ${NUM_GPUS} candidate-generation workers ==="
echo "Args: $@"
echo ""

PIDS=()

for ((i=0; i<NUM_GPUS; i++)); do
    echo "[GPU ${i}] Starting shard ${i}/${NUM_GPUS}..."
    CUDA_VISIBLE_DEVICES=${i} python "${GEN_SCRIPT}" \
        --shard-id ${i} \
        --num-shards ${NUM_GPUS} \
        "$@" \
        > "generate_candidates_shard${i}.log" 2>&1 &
    PIDS+=($!)
    echo "[GPU ${i}] PID=${PIDS[-1]}, log=generate_candidates_shard${i}.log"
done

echo ""
echo "All ${NUM_GPUS} workers launched. PIDs: ${PIDS[*]}"
echo "Monitor progress:  tail -f generate_candidates_shard*.log"
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
    echo "  python scripts/generate_candidates.py --merge --output <path>"
fi
