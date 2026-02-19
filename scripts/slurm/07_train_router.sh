#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-train-router
#SBATCH --output=logs/train_router_%j.out
#SBATCH --error=logs/train_router_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

# ── Train risk-calibrated router ──

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
