#!/bin/bash
#SBATCH --job-name=r2v-train-policy
#SBATCH --output=logs/train_policy_%j.out
#SBATCH --error=logs/train_policy_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --nodes=1

# ── Train SLM policy (BC + DPO + Consistency) ──
# Requires 2x A100 80GB (or 4x A6000 48GB) for 13B model with LoRA
# Reduce to 1x A100 for 7B model

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

BENCHMARK=${1:-webarena}
CONDITION=${2:-noisy}     # clean or noisy
STAGE=${3:-all}           # bc, preference, or all

echo "=== Training policy: ${BENCHMARK} (${CONDITION}) stage=${STAGE} ==="
echo "GPUs: ${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
nvidia-smi

mkdir -p outputs/policy/${BENCHMARK}_${CONDITION}

# Use accelerate for multi-GPU
accelerate launch \
    --num_processes $(nvidia-smi -L | wc -l) \
    --mixed_precision bf16 \
    scripts/train_policy.py \
    --config configs/${BENCHMARK}/${CONDITION}.yaml \
    --output outputs/policy/${BENCHMARK}_${CONDITION} \
    --stage ${STAGE} \
    --trajectories data/trajectories/${BENCHMARK}_${CONDITION}/trajectories.jsonl

echo "=== Policy training complete ==="
