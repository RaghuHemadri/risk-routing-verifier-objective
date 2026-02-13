#!/bin/bash
#SBATCH --job-name=r2v-train-verifier
#SBATCH --output=logs/train_verifier_%j.out
#SBATCH --error=logs/train_verifier_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nodes=1

# ── Train verifier (or prepare LLM-judge) ──

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

BENCHMARK=${1:-webarena}

echo "=== Training verifier for ${BENCHMARK} ==="

mkdir -p outputs/verifier/${BENCHMARK}

python scripts/train_verifier.py \
    --config configs/${BENCHMARK}/noisy.yaml \
    --output outputs/verifier/${BENCHMARK} \
    --trajectories data/trajectories/${BENCHMARK}_teacher/trajectories.jsonl

echo "=== Verifier training complete ==="
