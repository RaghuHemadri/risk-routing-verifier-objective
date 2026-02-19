#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=r2v-test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rh3884@nyu.edu

# ══════════════════════════════════════════════════════════════
# R2V-Agent: Cluster Sanity Check
# ══════════════════════════════════════════════════════════════
# Usage: sbatch scripts/slurm/test_cluster.sh
# ══════════════════════════════════════════════════════════════

set -euo pipefail

echo "═══════════════════════════════════════════════"
echo "  R2V-Agent Cluster Test"
echo "  $(date)"
echo "═══════════════════════════════════════════════"

# ── 1. Basic environment ──
echo -e "\n[1/7] Environment"
echo "  User:     $USER"
echo "  Host:     $(hostname)"
echo "  Job ID:   ${SLURM_JOB_ID:-N/A}"
echo "  Work dir: $(pwd)"
echo "  Python:   $(which python 2>/dev/null || echo 'NOT FOUND')"

# ── 2. GPU check ──
echo -e "\n[2/7] GPUs"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "  CUDA visible: ${CUDA_VISIBLE_DEVICES:-not set}"
else
    echo "  nvidia-smi not found!"
fi

# ── 3. Python + conda ──
echo -e "\n[3/7] Python environment"
source $HOME/.bashrc
if conda activate r2v 2>/dev/null; then
    echo "  Conda env 'r2v' activated"
    python --version
    echo "  Python path: $(which python)"
else
    echo "  WARNING: conda env 'r2v' not found"
    echo "  Create it with: conda create -n r2v python=3.10 -y"
fi

# ── 7. Disk space ──
echo -e "\n[7/7] Disk space"
df -h . | tail -1 | awk '{print "  Available: " $4 " / " $2}'
echo "  Scratch:  $(df -h /scratch 2>/dev/null | tail -1 | awk '{print $4 " / " $2}' || echo 'N/A')"

echo -e "\n═══════════════════════════════════════════════"
echo "  All checks complete — $(date)"
echo "═══════════════════════════════════════════════"
