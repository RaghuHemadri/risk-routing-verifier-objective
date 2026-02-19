#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-mrp
#SBATCH --output=logs/mrp_%j.out
#SBATCH --error=logs/mrp_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

# ══════════════════════════════════════════════════════════════
# Minimal Reproducible Prototype (MRP)
# ══════════════════════════════════════════════════════════════
#
# Quick validation run: 10 templates × 5 instances, 2 perturbation
# types, 1 GPU. Completes in ~4-6 hours.
#
# Usage:
#   sbatch scripts/slurm/mrp.sh
# ══════════════════════════════════════════════════════════════

set -euo pipefail

source $HOME/.bashrc
conda activate r2v

cd $SLURM_SUBMIT_DIR

echo "═══════════════════════════════════════════════"
echo "  R2V-Agent: Minimal Reproducible Prototype"
echo "═══════════════════════════════════════════════"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi -L)"

mkdir -p logs data/mrp outputs/mrp results/mrp

CONFIG=configs/mrp.yaml

# Step 1: Collect small set of trajectories
echo -e "\n[1/6] Collecting trajectories..."
python scripts/collect_trajectories.py \
    --config ${CONFIG} \
    --output data/mrp/trajectories \
    --num-episodes 50 \
    --seeds 1 2

# Step 2: Generate perturbations
echo -e "\n[2/6] Generating perturbations..."
python scripts/generate_perturbations.py \
    --config ${CONFIG} \
    --input data/mrp/trajectories/trajectories.jsonl \
    --output data/mrp/noisy/trajectories.jsonl \
    --seeds 1 2

# Step 3: Train policy (BC only for MRP)
echo -e "\n[3/6] Training policy (BC)..."
python scripts/train_policy.py \
    --config ${CONFIG} \
    --output outputs/mrp/policy \
    --stage bc \
    --trajectories data/mrp/noisy/trajectories.jsonl \
    --overrides training.num_epochs=1 training.batch_size=2

# Step 4: Generate router features
echo -e "\n[4/6] Generating router features..."
python scripts/generate_router_features.py \
    --config ${CONFIG} \
    --policy-path outputs/mrp/policy/bc/final \
    --trajectories data/mrp/noisy/trajectories.jsonl \
    --output data/mrp/router_features.jsonl

# Step 5: Train router
echo -e "\n[5/6] Training router..."
python scripts/train_router.py \
    --config ${CONFIG} \
    --output outputs/mrp/router \
    --features data/mrp/router_features.jsonl \
    --overrides router.num_epochs=10

# Step 6: Quick evaluation
echo -e "\n[6/6] Evaluating..."
python scripts/evaluate.py \
    --config ${CONFIG} \
    --policy-path outputs/mrp/policy/bc/final \
    --router-path outputs/mrp/router/router_final.pt \
    --output results/mrp \
    --seeds 1 2 \
    --methods r2v slm_only \
    --num-episodes 50

echo -e "\n═══════════════════════════════════════════════"
echo "  MRP Complete!"
echo "  Results: results/mrp/structured_results/"
echo "  End: $(date)"
echo "═══════════════════════════════════════════════"
