#!/bin/bash
#SBATCH --job-name=r2v-perturb
#SBATCH --output=logs/perturb_%j.out
#SBATCH --error=logs/perturb_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodes=1

# ── Generate perturbed trajectories ──
# CPU-only job (no GPU needed for perturbation generation)

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

BENCHMARK=${1:-webarena}
SEEDS=${2:-"1 2 3"}

echo "=== Generating perturbed trajectories for ${BENCHMARK} ==="

mkdir -p data/trajectories/${BENCHMARK}_noisy

python scripts/generate_perturbations.py \
    --config configs/${BENCHMARK}/noisy.yaml \
    --input data/trajectories/${BENCHMARK}_teacher/trajectories.jsonl \
    --output data/trajectories/${BENCHMARK}_noisy/trajectories.jsonl \
    --seeds ${SEEDS}

echo "=== Perturbation generation complete ==="
