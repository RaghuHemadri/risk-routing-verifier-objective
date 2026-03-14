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
#   python scripts/generate_candidates.py --merge \
#       --output data/candidates/gaia_noisy.jsonl

set -euo pipefail

NUM_GPUS="${1:?Usage: $0 <NUM_GPUS> [generate_candidates.py args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/generate_candidates.py"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "ERROR: Could not find python3/python in PATH"
        exit 1
    fi
fi

# Throughput-oriented defaults. Override in shell if needed.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

START_STAGGER_SECS="${START_STAGGER_SECS:-2}"

echo "=== Launching ${NUM_GPUS} candidate-generation workers ==="
echo "Python: ${PYTHON_BIN}"
echo "Args: $@"
echo ""

PIDS=()

cleanup_children() {
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "[launcher] Terminating child workers: ${PIDS[*]}"
        kill "${PIDS[@]}" 2>/dev/null || true
    fi
}

trap cleanup_children INT TERM

for ((i=0; i<NUM_GPUS; i++)); do
    echo "[GPU ${i}] Starting shard ${i}/${NUM_GPUS}..."
    CUDA_VISIBLE_DEVICES=${i} "${PYTHON_BIN}" "${GEN_SCRIPT}" \
        --shard-id ${i} \
        --num-shards ${NUM_GPUS} \
        "$@" \
        > "generate_candidates_shard${i}.log" 2>&1 &
    PIDS+=($!)
    echo "[GPU ${i}] PID=${PIDS[-1]}, log=generate_candidates_shard${i}.log"
    if (( i + 1 < NUM_GPUS )) && (( START_STAGGER_SECS > 0 )); then
        sleep "${START_STAGGER_SECS}"
    fi
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
    "${PYTHON_BIN}" "${GEN_SCRIPT}" --merge --output "${OUTPUT}"
    echo "=== Done! Final output: ${OUTPUT} ==="
else
    echo "Could not find --output arg; merge manually:"
    echo "  python scripts/generate_candidates.py --merge --output <path>"
fi
