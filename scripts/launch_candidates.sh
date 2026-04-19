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
#       --config configs/gaia/noisy.yaml \
#       --policy-path outputs/policy/gaia_noisy/final \
#       --verifier-path outputs/verifier/gaia_noisy/final/verifier.pt \
#       --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
#       --output data/candidates/gaia_noisy.jsonl \
#       --K 5 --batch-size 8
#
# After all shards finish, merge:
#   python scripts/generate_candidates_heuristic.py --merge \
#       --output data/candidates/gaia_noisy.jsonl

set -euo pipefail

NUM_GPUS="${1:?Usage: $0 <NUM_GPUS> [generate_candidates_heuristic.py args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/generate_candidates_heuristic.py"

# Derive a run tag from --output so log names are unique per model/run.
RUN_TAG="candidates"
ARGS_ARRAY=("$@")
for ((j=0; j<${#ARGS_ARRAY[@]}; j++)); do
    if [[ "${ARGS_ARRAY[$j]}" == "--output" ]] && ((j+1 < ${#ARGS_ARRAY[@]})); then
        RUN_TAG="$(basename "${ARGS_ARRAY[$((j+1))]}" .jsonl)"
        break
    fi
done

echo "=== Launching ${NUM_GPUS} candidate-generation workers (${RUN_TAG}) ==="
echo "Args: $@"
echo ""

PIDS=()

# GPU_IDS env var overrides default 0..N-1 assignment (e.g. GPU_IDS=2,3 to skip busy GPUs)
if [[ -n "${GPU_IDS:-}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
    if [[ ${#GPU_LIST[@]} -ne ${NUM_GPUS} ]]; then
        echo "ERROR: GPU_IDS has ${#GPU_LIST[@]} entries but NUM_GPUS=${NUM_GPUS}"
        exit 1
    fi
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=("$i"); done
fi

for ((i=0; i<NUM_GPUS; i++)); do
    LOG="${RUN_TAG}_shard${i}.log"
    echo "[GPU ${GPU_LIST[$i]}] Starting shard ${i}/${NUM_GPUS}..."
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python "${GEN_SCRIPT}" \
        --shard-id ${i} \
        --num-shards ${NUM_GPUS} \
        "$@" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
    echo "[GPU ${GPU_LIST[$i]}] PID=${PIDS[-1]}, log=${LOG}"
done

echo ""
echo "All ${NUM_GPUS} workers launched. PIDs: ${PIDS[*]}"
echo "Monitor progress:  tail -f ${RUN_TAG}_shard*.log"
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
    echo "  python scripts/generate_candidates_heuristic.py --merge --output <path>"
fi
