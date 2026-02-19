# Running Instructions — R2V-Agent on SLURM

This document provides step-by-step instructions for running the entire R2V-Agent
pipeline on a SLURM-managed cluster. You do **not** need a local GPU.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Quick Validation (MRP)](#3-quick-validation-mrp)
4. [Full Pipeline](#4-full-pipeline)
5. [Individual Stages](#5-individual-stages)
6. [Data Versioning](#6-data-versioning)
7. [Monitoring Jobs](#7-monitoring-jobs)
8. [Collecting Results](#8-collecting-results)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

**Cluster requirements:**
- SLURM scheduler
- GPU partition with A100 (80GB) or equivalent
- CPU partition for preprocessing
- At least 200GB disk space per benchmark

**Software:**
- Python ≥ 3.10
- CUDA ≥ 12.1
- conda or virtualenv

---

## 2. Environment Setup

```bash
# SSH into your cluster login node
ssh <your-cluster>

# Clone the repository
git clone <repo-url> ~/r2v-agent
cd ~/r2v-agent

# Option A: conda environment (recommended)
conda create -n r2v python=3.10 -y
conda activate r2v
pip install -e ".[dev]"

# Option B: virtualenv
python3.10 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Verify installation
python -c "import r2v; print(r2v.__version__)"

# Install benchmark dependencies (optional)
pip install -e ".[webarena]"   # for WebArena
pip install -e ".[swebench]"   # for SWE-bench
```

### Configure wandb (optional but recommended)

```bash
pip install wandb
wandb login  # paste your API key
# Or set: export WANDB_API_KEY=<your-key>
# To disable: export WANDB_MODE=disabled
```

### Configure API keys (if using LLM-judge verifier)

```bash
# For OpenAI teacher/verifier
export OPENAI_API_KEY=<your-key>

# For Anthropic teacher/verifier
export ANTHROPIC_API_KEY=<your-key>
```

### Adapt SLURM scripts to your cluster

Edit the following in each `scripts/slurm/*.sh`:
- `--partition=gpu` → your GPU partition name
- `--gres=gpu:a100:2` → your available GPU types
- `module load ...` → your cluster's module system
- `conda activate r2v` → your environment activation

---

## 3. Quick Validation (MRP)

The Minimal Reproducible Prototype runs the full pipeline on a tiny dataset
to verify everything works before committing to expensive runs.

```bash
cd ~/r2v-agent

# Submit MRP job (1 GPU, ~4-6 hours)
sbatch scripts/slurm/mrp.sh

# Monitor
squeue -u $USER
tail -f logs/mrp_*.out
```

**Expected output:**
- `data/mrp/` — trajectory files
- `outputs/mrp/` — trained models
- `results/mrp/structured_results/` — evaluation results

**Verify results exist:**
```bash
ls results/mrp/structured_results/
# Should see: *.json, main_table.csv, llm_summary.json
```

---

## 4. Full Pipeline

Once MRP succeeds, run the full pipeline:

```bash
# WebArena only
bash scripts/slurm/run_all.sh webarena

# SWE-bench only
bash scripts/slurm/run_all.sh swebench

# Both benchmarks
bash scripts/slurm/run_all.sh all
```

This submits ~10 SLURM jobs per benchmark with proper dependencies:

| Stage | Job | GPU | Time Est. | Depends On |
|-------|-----|-----|-----------|------------|
| 1 | Collect trajectories | 1× A100 | 12-24h | — |
| 2 | Generate perturbations | CPU | 1-4h | Stage 1 |
| 3a | Train policy (BC) | 2× A100 | 24-48h | Stage 2 |
| 3b | Train verifier | 1× A100 | 6-12h | Stage 1 |
| 4 | Generate candidates | 1× A100 | 12-24h | 3a, 3b |
| 5 | Train policy (DPO) | 2× A100 | 12-24h | Stage 4 |
| 6 | Router features | 1× A100 | 6-12h | Stage 5 |
| 7 | Train router | 1× GPU | 1-4h | Stage 6 |
| 8 | Evaluate | 1× A100 | 12-24h | Stage 7 |
| 9 | Ablations | 1× A100 | 24-48h | Stage 7 |

**Total wall time:** ~4-7 days (with job queuing)
**Total GPU hours:** ~150-250h per benchmark

---

## 5. Individual Stages

You can run stages individually for debugging or re-running:

### Collect teacher trajectories
```bash
sbatch scripts/slurm/01_collect.sh webarena 500 "1 2 3"
#                                  ^benchmark ^episodes ^seeds
```

### Generate perturbations
```bash
sbatch scripts/slurm/02_perturb.sh webarena "1 2 3"
```

### Train policy
```bash
# BC stage only
sbatch scripts/slurm/03_train_policy.sh webarena noisy bc

# DPO preference stage only (requires candidates)
sbatch scripts/slurm/03_train_policy.sh webarena noisy preference

# Both stages
sbatch scripts/slurm/03_train_policy.sh webarena noisy all
```

### Train verifier
```bash
sbatch scripts/slurm/04_train_verifier.sh webarena
```

### Generate candidates for DPO
```bash
sbatch scripts/slurm/05_generate_candidates.sh webarena 5
#                                              ^benchmark ^K
```

### Generate router features
```bash
sbatch scripts/slurm/06_generate_router_features.sh webarena
```

### Train router
```bash
sbatch scripts/slurm/07_train_router.sh webarena
```

### Evaluate
```bash
sbatch scripts/slurm/08_evaluate.sh webarena noisy "1 2 3 4 5" "r2v slm_only llm_only entropy_router"
#                                   ^bench   ^cond ^seeds      ^methods
```

### Run ablations
```bash
sbatch scripts/slurm/09_ablations.sh webarena "1 2 3"
```

---

## 6. Data Versioning

Every `collect_trajectories.py` run creates a **versioned directory** and
registers itself in `data/registry.json`.  This makes it easy to collect data
with multiple models and combine them for training without losing track of
which episodes came from where.

### Run ID format

```
{benchmark}_{provider}_{model}_{YYYYMMDDTHHMMSS}
e.g.  webarena_openai_gpt-4o_20260218T210012
      swebench_anthropic_claude-3-5-sonnet_20260219T083045
```

### Directory structure

```
data/
  registry.json                                     ← central index
  runs/
    webarena_openai_gpt-4o_20260218T210012/
      run_manifest.json                             ← config snapshot + stats
      trajectories.jsonl                            ← episodes (run_id embedded)
      collection_log.jsonl
    swebench_anthropic_claude-3-5-sonnet_20260219T083045/
      run_manifest.json
      trajectories.jsonl
      collection_log.jsonl
```

### Collect data for a specific model

`--output` is now the **base directory** (defaults to `data/runs`).  The
script automatically appends the run ID as a subdirectory.

```bash
# GPT-4o on WebArena (output → data/runs/webarena_openai_gpt-4o_<ts>/)
python scripts/collect_trajectories.py \
    --config configs/webarena/clean.yaml \
    --num-episodes 500 --seeds 1 2 3

# Claude on SWE-bench with a custom tag
python scripts/collect_trajectories.py \
    --config configs/swebench/clean.yaml \
    --num-episodes 500 --seeds 1 2 3 \
    --run-tags ablation noisy \
    --notes "using 0.3 prompt injection prob"

# Specify a custom base directory
python scripts/collect_trajectories.py \
    --config configs/webarena/clean.yaml \
    --output data/experiments/exp_01 \
    --num-episodes 100
```

The run ID is printed on start and finish:

```
============================================================
  Run ID : webarena_openai_gpt-4o_20260218T210012
  Output : data/runs/webarena_openai_gpt-4o_20260218T210012
============================================================
```

### Inspect the registry

```python
from r2v.data import DataRegistry

registry = DataRegistry()
print(registry.summary())       # tabular overview of all runs

# Filter by benchmark, model, or status
runs = registry.query(benchmark="webarena", model="gpt-4o", status="done")
for m in runs:
    print(m.run_id, m.n_episodes, f"{m.success_rate:.2%}")

# Latest completed run for a given model
latest = registry.latest_run("webarena", "gpt-4o")
print(latest.output_dir)
```

### Combine datasets from multiple models

Use `DataRegistry.merge_runs()` to create a single `TrajectoryStore` from
any subset of runs.  Duplicate trajectories (by action hash) are
automatically removed.

```python
from r2v.data import DataRegistry

registry = DataRegistry()

# All completed WebArena runs across all models
all_webarena = registry.query(benchmark="webarena", status="done")
run_ids = [m.run_id for m in all_webarena]

merged = registry.merge_runs(run_ids, output_path=Path("data/merged/webarena_all.jsonl"))
episodes = list(merged.iter_episodes())
print(f"Merged {len(episodes)} unique episodes from {len(run_ids)} runs")
```

Or iterate without writing to disk:

```python
for episode in registry.iter_episodes(run_ids, deduplicate=True):
    # episode.metadata.run_id       → which run it came from
    # episode.metadata.teacher_model → which model generated it
    # episode.metadata.teacher_provider
    ...
```

### Resume a partial run

If collection was interrupted, re-run with `--run-id` to continue under the
same ID and append to the existing JSONL:

```bash
python scripts/collect_trajectories.py \
    --config configs/webarena/clean.yaml \
    --output data/runs \
    --run-id webarena_openai_gpt-4o_20260218T210012 \
    --num-episodes 500 --seeds 1 2 3
```

### EpisodeMetadata versioning fields

Each serialised episode now contains three extra fields in `metadata`:

| Field | Example | Description |
|---|---|---|
| `run_id` | `webarena_openai_gpt-4o_20260218T210012` | Run that produced this episode |
| `teacher_model` | `gpt-4o` | Model name from config |
| `teacher_provider` | `openai` | API provider |

---

## 7. Monitoring Jobs


```bash
# View your queued/running jobs
squeue -u $USER

# View job details
scontrol show job <JOBID>

# View job output in real-time
tail -f logs/<jobname>_<JOBID>.out

# Cancel a job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER

# View completed job efficiency
sacct -j <JOBID> --format=JobID,Elapsed,MaxRSS,MaxVMSize,AllocCPUS

# View wandb dashboard (if enabled)
# https://wandb.ai/<your-username>/r2v-agent
```

### Check pipeline progress

```bash
# Quick status of all jobs
squeue -u $USER -o "%.8i %.20j %.8T %.10M %.6D %R"

# Check if outputs exist
ls -la data/trajectories/
ls -la outputs/policy/
ls -la outputs/router/
ls -la results/
```

---

## 8. Collecting Results

### Structured results for LLM paper writing

After evaluation completes, results are in `results/<benchmark>_<condition>/structured_results/`:

```bash
# View main results table
cat results/webarena_noisy/structured_results/main_table.csv

# View statistical comparisons
cat results/webarena_noisy/structured_results/comparisons.csv

# View ablation results
cat results/ablations/webarena/structured_results/ablations.csv

# The LLM summary (feed this to Claude/GPT for paper writing)
cat results/webarena_noisy/structured_results/llm_summary.json
```

### Download results to local machine

```bash
# From your local machine:
scp -r <cluster>:~/r2v-agent/results/ ./results/
scp -r <cluster>:~/r2v-agent/outputs/ ./outputs/
```

### Generate LaTeX tables

Results are auto-generated, but you can regenerate:

```python
from r2v.utils.results import ResultsManager

rm = ResultsManager("results/webarena_noisy/structured_results")
rm.generate_main_table_csv()
rm.generate_latex_table()
rm.generate_llm_summary()
```

---

## 9. Troubleshooting

### Common issues

**Out of memory (OOM):**
```bash
# Reduce batch size
--overrides training.batch_size=2 training.gradient_accumulation_steps=8

# Or use more gradient accumulation
--overrides training.gradient_accumulation_steps=16

# For 7B model with limited GPU memory
--overrides policy.load_in_4bit=true
```

**Job killed (time limit):**
```bash
# Increase time in SLURM script
#SBATCH --time=72:00:00

# Or enable checkpointing and resume
python scripts/train_policy.py --resume outputs/policy/webarena_noisy/bc/checkpoint-latest
```

**CUDA version mismatch:**
```bash
# Load correct CUDA module
module load cuda/12.1
# Verify
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**wandb issues:**
```bash
# Disable wandb for debugging
export WANDB_MODE=disabled

# Or run offline and sync later
export WANDB_MODE=offline
# After job completes:
wandb sync logs/wandb/offline-run-*
```

**Package import errors:**
```bash
# Ensure you're in the right environment
conda activate r2v
# Reinstall in development mode
pip install -e ".[dev]"
```

### Useful debug commands

```bash
# Test a single training step
python scripts/train_policy.py \
    --config configs/mrp.yaml \
    --output /tmp/test_policy \
    --stage bc \
    --trajectories data/mrp/trajectories/trajectories.jsonl \
    --overrides training.num_epochs=1 training.batch_size=1

# Interactive GPU session for debugging
srun --partition=gpu --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash
conda activate r2v
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Config Overrides

Any config value can be overridden from the command line:

```bash
python scripts/train_policy.py \
    --config configs/webarena/noisy.yaml \
    --overrides \
        training.num_epochs=5 \
        training.lr=1e-5 \
        training.batch_size=8 \
        policy.lora_r=32 \
        perturbations.prompt_injection.prob=0.2
```
