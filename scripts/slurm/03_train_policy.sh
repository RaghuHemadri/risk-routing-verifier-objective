#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-train-policy
#SBATCH --output=logs/train_policy_%j.out
#SBATCH --error=logs/train_policy_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

# ── Train SLM policy (BC + DPO + Consistency) ──
# 2x A100 80GB for 13B model with LoRA

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

export HF_TOKEN=""

BENCHMARK=${1:-swebench}
CONDITION=${2:-noisy}     # clean or noisy
STAGE=${3:-all}           # bc, preference, or all

echo "=== Training policy: ${BENCHMARK} (${CONDITION}) stage=${STAGE} ==="
NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)}
echo "GPUs: ${NUM_GPUS}"
nvidia-smi 2>/dev/null || echo "(nvidia-smi not available in container)"

mkdir -p outputs/policy/${BENCHMARK}_${CONDITION}

# Use accelerate for multi-GPU
accelerate launch \
    --num_processes ${NUM_GPUS} \
    --mixed_precision bf16 \
    scripts/train_policy.py \
    --config configs/${BENCHMARK}/${CONDITION}.yaml \
    --output outputs/policy/${BENCHMARK}_${CONDITION} \
    --stage ${STAGE} \
    --trajectories data/trajectories/${BENCHMARK}_${CONDITION}/trajectories.jsonl

echo "=== Policy training complete ==="
