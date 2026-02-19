#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-candidates
#SBATCH --output=logs/candidates_%j.out
#SBATCH --error=logs/candidates_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

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
