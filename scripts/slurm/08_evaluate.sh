#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-evaluate
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

# ── Evaluate R2V-Agent and baselines ──

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

BENCHMARK=${1:-webarena}
CONDITION=${2:-noisy}
SEEDS=${3:-"1 2 3 4 5"}
METHODS=${4:-"r2v slm_only llm_only entropy_router"}

echo "=== Evaluating on ${BENCHMARK} (${CONDITION}) ==="
echo "Seeds: ${SEEDS}"
echo "Methods: ${METHODS}"

mkdir -p results/${BENCHMARK}_${CONDITION}

python scripts/evaluate.py \
    --config configs/${BENCHMARK}/${CONDITION}.yaml \
    --policy-path outputs/policy/${BENCHMARK}_${CONDITION}/final \
    --router-path outputs/router/${BENCHMARK}/router_final.pt \
    --output results/${BENCHMARK}_${CONDITION} \
    --seeds ${SEEDS} \
    --methods ${METHODS}

echo "=== Evaluation complete ==="
echo "Results: results/${BENCHMARK}_${CONDITION}/structured_results/"
