#!/bin/bash
#SBATCH --job-name=r2v-candidates
#SBATCH --output=logs/candidates_%j.out
#SBATCH --error=logs/candidates_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodes=1

# ── Generate candidate actions for DPO preference training ──

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

BENCHMARK=${1:-webarena}
K=${2:-5}

echo "=== Generating K=${K} candidates for ${BENCHMARK} ==="

mkdir -p data/candidates

python scripts/generate_candidates.py \
    --config configs/${BENCHMARK}/noisy.yaml \
    --policy-path outputs/policy/${BENCHMARK}_noisy/bc/final \
    --trajectories data/trajectories/${BENCHMARK}_teacher/trajectories.jsonl \
    --output data/candidates/${BENCHMARK}.jsonl \
    --K ${K}

echo "=== Candidate generation complete ==="
