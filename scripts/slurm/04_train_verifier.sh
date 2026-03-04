#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-train-verifier
#SBATCH --output=logs/train_verifier_%j.out
#SBATCH --error=logs/train_verifier_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

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
