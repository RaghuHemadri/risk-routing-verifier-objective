# Running the R2V-Agent Pipeline (Local)

Instructions for running the full R2V-Agent pipeline locally on GAIA.

---

## Prerequisites

- Python ≥ 3.10, CUDA ≥ 12.1
- GPU with ≥ 24GB VRAM (A100 / H200 / RTX 4090)
- LLM API key for teacher model

```bash
# Install
pip install -e ".[dev]"

# Set API key (pick one)
export GOOGLE_API_KEY=<your-key>     # for Gemini
export OPENAI_API_KEY=<your-key>     # for GPT-4o
export ANTHROPIC_API_KEY=<your-key>  # for Claude

# HuggingFace token (needed for gated Llama model)
source exports.sh
```

---

## Quick Start

Run the entire pipeline end-to-end:

```bash
bash run_pipeline.sh
```

Resume from a specific stage (e.g., after a crash at Stage 5):

```bash
bash run_pipeline.sh --from 5
```

Run only one stage:

```bash
bash run_pipeline.sh --only 3
```

Preview all commands without executing:

```bash
bash run_pipeline.sh --dry-run
```

---

## Pipeline Stages

| Stage | Script | GPU | Description |
|-------|--------|-----|-------------|
| 0 | `collect_trajectories.py` | — | Smoke test (1 instance, verify setup) |
| 1 | `collect_trajectories.py` | — | Collect 500 clean teacher trajectories (API-only) |
| 2 | `generate_perturbations.py` | — | Generate perturbed trajectories (CPU-only, ~2 min) |
| 3 | `train_policy.py --stage bc` | 1× GPU | Behavior cloning on noisy trajectories |
| 4 | `train_verifier.py` | 1× GPU | Train step-level verifier |
| 5 | `generate_candidates.py` | 1× GPU | Generate K=5 candidates per step for DPO |
| 6 | `train_policy.py --stage preference` | 1–4× GPU | DPO preference training |
| 7 | `generate_router_features.py` | 1× GPU | Compute 13-dim routing features |
| 8 | `train_router.py` | — | Train routing MLP (CPU, ~25s) |
| 9 | `evaluate.py` | — | Evaluate all methods (CPU, ~seconds) |
| 10 | `run_ablations.py` | — | Ablation studies |

---

## Running Each Stage Manually

### Stage 0: Smoke Test

```bash
python scripts/collect_trajectories.py \
    --config configs/smoke_test.yaml \
    --output data/smoke_test \
    --num-episodes 1 \
    --seeds 1 \
    --overrides teacher.provider=google teacher.model_name=gemini-3-flash-preview
```

### Stage 1: Collect Teacher Trajectories

Collects 500 tasks × 3 seeds = 1500 episodes from GAIA using the teacher LLM. API-only, no GPU needed.

```bash
python scripts/collect_trajectories.py \
    --config configs/gaia/clean.yaml \
    --output data/runs \
    --num-episodes 500 \
    --seeds 1 2 3 \
    --num-workers 4 \
    --overrides teacher.provider=google teacher.model_name=gemini-3-flash-preview
```

Output: `data/runs/gaia_<provider>_<model>_<timestamp>/trajectories.jsonl`

### Stage 2: Generate Perturbations

Applies 4 perturbation types (tool flakiness, partial observability, prompt injection, distractors) locally. No API calls.

```bash
python -m scripts.generate_perturbations \
    --config configs/gaia/noisy.yaml \
    --input data/runs/<run_id>/trajectories.jsonl \
    --output data/trajectories/gaia_noisy/trajectories.jsonl \
    --seeds 1 2 3 \
    --include-clean
```

### Stage 3: Train Policy (Behavior Cloning)

Trains Llama-3.1-8B-Instruct with LoRA on the noisy trajectories.

```bash
python scripts/train_policy.py \
    --config configs/gaia/noisy.yaml \
    --output outputs/policy/gaia_noisy \
    --stage bc \
    --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
    --overrides policy.quantization.load_in_4bit=false logging.wandb_mode=disabled
```

Output: `outputs/policy/gaia_noisy/final` (LoRA adapters)

### Stage 4: Train Verifier

Trains step-level verifier (frozen Llama backbone + MLP heads).

```bash
python scripts/train_verifier.py \
    --config configs/gaia/noisy.yaml \
    --output outputs/verifier/gaia_noisy \
    --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
    --overrides policy.quantization.load_in_4bit=false verifier.mode=trained \
        training.verifier.epochs=3 training.verifier.batch_size=32 \
        policy.max_seq_len=2048 logging.wandb_mode=disabled
```

Output: `outputs/verifier/gaia_noisy/final/verifier.pt`

### Stage 5: Generate DPO Candidates

Samples K=5 candidate actions per step, scores with the verifier.

```bash
# Single GPU:
python scripts/generate_candidates.py \
    --config configs/gaia/noisy.yaml \
    --policy-path outputs/policy/gaia_noisy/final \
    --verifier-path outputs/verifier/gaia_noisy/final/verifier.pt \
    --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
    --output data/candidates/gaia_noisy.jsonl \
    --K 5 --batch-size 8 \
    --overrides policy.quantization.load_in_4bit=false verifier.mode=trained logging.wandb_mode=disabled

# Multi-GPU (faster):
bash scripts/launch_candidates.sh 4 \
    --config configs/gaia/noisy.yaml \
    --policy-path outputs/policy/gaia_noisy/final \
    --verifier-path outputs/verifier/gaia_noisy/final/verifier.pt \
    --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
    --output data/candidates/gaia_noisy.jsonl \
    --K 5 --batch-size 8 \
    --overrides policy.quantization.load_in_4bit=false verifier.mode=trained logging.wandb_mode=disabled
```

### Stage 6: Train Policy (DPO)

Refines the BC policy with preference pairs.

```bash
# Single GPU:
python scripts/train_policy.py \
    --config configs/gaia/noisy.yaml \
    --output outputs/policy/gaia_noisy \
    --stage preference \
    --preference-data data/candidates/gaia_noisy.jsonl \
    --overrides policy.quantization.load_in_4bit=false logging.wandb_mode=disabled

# Multi-GPU DDP:
accelerate launch --num_processes=4 --multi_gpu \
    scripts/train_policy.py \
    --config configs/gaia/noisy.yaml \
    --output outputs/policy/gaia_noisy \
    --stage preference \
    --preference-data data/candidates/gaia_noisy.jsonl \
    --overrides logging.wandb_mode=disabled
```

### Stage 7: Generate Router Features

Computes 13-dimensional feature vectors for the routing MLP.

```bash
# Single GPU:
python scripts/generate_router_features.py \
    --config configs/gaia/noisy.yaml \
    --policy-path outputs/policy/gaia_noisy/final \
    --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
    --output data/router_features/gaia.jsonl \
    --batch-size 16 --K 5

# Multi-GPU (faster):
bash scripts/launch_router_features.sh 4 \
    --config configs/gaia/noisy.yaml \
    --policy-path outputs/policy/gaia_noisy/final \
    --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
    --output data/router_features/gaia.jsonl \
    --batch-size 16 --K 5
```

### Stage 8: Train Router

Small MLP (10K params). Runs on CPU in ~25 seconds.

```bash
python scripts/train_router.py \
    --config configs/gaia/noisy.yaml \
    --features data/router_features/gaia.jsonl \
    --output outputs/router/gaia_noisy \
    --overrides logging.wandb_mode=disabled
```

Output: `outputs/router/gaia_noisy/router_final.pt`

### Stage 9: Evaluate

Offline evaluation using pre-computed features. CPU-only, runs in seconds.

```bash
python scripts/evaluate.py \
    --config configs/gaia/noisy.yaml \
    --features data/router_features/gaia.jsonl \
    --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
    --router-path outputs/router/gaia_noisy/router_final.pt \
    --output results/gaia_noisy \
    --seeds 1 2 3 \
    --methods r2v slm_only llm_only entropy_router \
    --overrides logging.wandb_mode=disabled
```

Output: `results/gaia_noisy/structured_results/`

### Stage 10: Ablation Studies (Optional)

```bash
python scripts/run_ablations.py \
    --config configs/gaia/noisy.yaml \
    --features data/router_features/gaia.jsonl \
    --trajectories data/trajectories/gaia_noisy/trajectories.jsonl \
    --router-path outputs/router/gaia_noisy/router_final.pt \
    --output results/ablations/gaia_noisy \
    --overrides logging.wandb_mode=disabled
```

---

## Multi-GPU Configuration

Set `NUM_GPUS` before running the pipeline script:

```bash
NUM_GPUS=4 bash run_pipeline.sh
```

This affects Stages 5, 6, and 7 which benefit from parallelism. All other stages use a single GPU or CPU.

---

## Directory Structure After Full Run

```
data/
  runs/<run_id>/trajectories.jsonl          # Clean teacher trajectories
  trajectories/gaia_noisy/              # Noisy (clean + perturbed)
  candidates/gaia_noisy.jsonl           # DPO preference pairs
  router_features/gaia.jsonl            # 13-dim routing features

outputs/
  policy/gaia_noisy/final/              # BC + DPO LoRA adapters
  verifier/gaia_noisy/final/            # Trained verifier
  router/gaia_noisy/                    # Routing MLP

results/
  gaia_noisy/structured_results/        # Evaluation results
  ablations/gaia_noisy/                 # Ablation results
```

---

## Troubleshooting

**Out of memory:** Reduce batch size or enable gradient accumulation:
```bash
--overrides training.batch_size=2 training.gradient_accumulation_steps=8
```

**bitsandbytes errors:** Disable 4-bit quantization (uses bf16 instead, ~16GB VRAM):
```bash
--overrides policy.quantization.load_in_4bit=false
```

**wandb auth errors:** Disable wandb:
```bash
--overrides logging.wandb_mode=disabled
```

**Resume interrupted collection:** Re-use the run ID:
```bash
python scripts/collect_trajectories.py \
    --config configs/gaia/clean.yaml \
    --run-id <run_id_from_previous_output> \
    --num-episodes 500 --seeds 1 2 3 --num-workers 4
```
