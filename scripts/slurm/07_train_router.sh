#!/bin/bash
#SBATCH --job-name=r2v-train-router
#SBATCH --output=logs/train_router_%j.out
#SBATCH --error=logs/train_router_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodes=1

# ── Train risk-calibrated router ──
# Lightweight MLP — single GPU, fast training

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

BENCHMARK=${1:-webarena}

echo "=== Training router for ${BENCHMARK} ==="

mkdir -p outputs/router/${BENCHMARK}

python scripts/train_router.py \
    --config configs/${BENCHMARK}/noisy.yaml \
    --output outputs/router/${BENCHMARK} \
    --features data/router_features/${BENCHMARK}.jsonl

echo "=== Router training complete ==="
