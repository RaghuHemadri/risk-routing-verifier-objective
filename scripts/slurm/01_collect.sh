#!/bin/bash
#SBATCH --job-name=r2v-collect
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodes=1

# ── Collect teacher trajectories ──
# Adjust partition, mem, GPU type as needed for your cluster

set -euo pipefail

# Load modules (adjust for your cluster)
# module load cuda/12.1
# module load python/3.10

# Activate environment
source $HOME/.bashrc
conda activate r2v  # or: source venv/bin/activate

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
