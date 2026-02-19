#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-ablations
#SBATCH --output=logs/ablations_%j.out
#SBATCH --error=logs/ablations_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

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
