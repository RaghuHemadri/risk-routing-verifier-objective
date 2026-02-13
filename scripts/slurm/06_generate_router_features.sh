#!/bin/bash
#SBATCH --job-name=r2v-router-feat
#SBATCH --output=logs/router_features_%j.out
#SBATCH --error=logs/router_features_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nodes=1

# ── Generate router training features ──

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

BENCHMARK=${1:-webarena}

echo "=== Generating router features for ${BENCHMARK} ==="

mkdir -p data/router_features

python scripts/generate_router_features.py \
    --config configs/${BENCHMARK}/noisy.yaml \
    --policy-path outputs/policy/${BENCHMARK}_noisy/final \
    --trajectories data/trajectories/${BENCHMARK}_noisy/trajectories.jsonl \
    --output data/router_features/${BENCHMARK}.jsonl

echo "=== Router feature generation complete ==="
