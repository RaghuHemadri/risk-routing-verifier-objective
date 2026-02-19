#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-collect
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

# ── Collect teacher trajectories ──

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

BENCHMARK=${1:-webarena}   # webarena or swebench
NUM_EPISODES=${2:-500}
SEEDS=${3:-"1 2 3"}

echo "=== Collecting ${BENCHMARK} trajectories ==="
echo "Episodes: ${NUM_EPISODES}, Seeds: ${SEEDS}"
echo "Job ID: ${SLURM_JOB_ID}"

mkdir -p logs data/trajectories/${BENCHMARK}_teacher

python scripts/collect_trajectories.py \
    --config configs/${BENCHMARK}/clean.yaml \
    --output data/trajectories/${BENCHMARK}_teacher \
    --num-episodes ${NUM_EPISODES} \
    --seeds ${SEEDS}

echo "=== Collection complete ==="
