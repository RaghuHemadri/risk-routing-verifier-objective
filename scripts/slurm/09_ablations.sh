#!/bin/bash
#SBATCH --job-name=r2v-ablations
#SBATCH --output=logs/ablations_%j.out
#SBATCH --error=logs/ablations_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --nodes=1

# ── Run ablation studies ──

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

BENCHMARK=${1:-webarena}
SEEDS=${2:-"1 2 3"}

echo "=== Running ablation studies for ${BENCHMARK} ==="

mkdir -p results/ablations/${BENCHMARK}

python scripts/run_ablations.py \
    --config configs/${BENCHMARK}/noisy.yaml \
    --policy-path outputs/policy/${BENCHMARK}_noisy/final \
    --router-path outputs/router/${BENCHMARK}/router_final.pt \
    --output results/ablations/${BENCHMARK} \
    --seeds ${SEEDS}

echo "=== Ablations complete ==="
