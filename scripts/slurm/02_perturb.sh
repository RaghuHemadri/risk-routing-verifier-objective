#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-perturb
#SBATCH --output=logs/perturb_%j.out
#SBATCH --error=logs/perturb_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

# ── Generate perturbed trajectories ──

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
