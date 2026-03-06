## Completed

- ✅ Environment setup on NYU HPC (Singularity container with `--nv` + `srun --overlap`)
- ✅ **Step 1:** Collected 900 clean SWE-bench trajectories using Gemini 3 Flash
  - Run ID: `swebench_google_gemini-3-flash-preview_20260303T004504`
  - 300 tasks × 3 seeds, 13.3% success rate, $0.60, ~2.4h
- ✅ **Step 2:** Generated 2700 perturbed trajectories (+ 900 clean copies = 3600 total)
  - 4 perturbation types (composite), 3 seeds, 79 seconds CPU-only
  - Output: `data/trajectories/swebench_noisy/trajectories.jsonl`
- ✅ **Step 3a:** Trained BC policy on NYU Greene HPC (2× H200)
  - Llama-3.1-8B-Instruct + LoRA, bf16, 480 BC examples (385 train / 95 val)
  - Loss: 3.55 → 0.11 in 3 epochs, ~17 min wall time
  - Output: `outputs/policy/swebench_noisy/final`
- ✅ **Step 3b:** Trained verifier on NYU Greene HPC (1× H200)
  - Llama-3.1-8B-Instruct (frozen backbone) + MLP heads (4.2M trainable, 0.05%)
  - 36,144 step-level examples, loss: 0.58 → 0.21, accuracy: 86.8% → 91.3%
  - 3 epochs, ~2h wall time with dynamic padding
  - Output: `outputs/verifier/swebench_noisy/final/verifier.pt`
- ✅ **Step 4:** Generated K=5 candidates per step (2× H200, 2 shards)
  - 4,015 preference pairs from 3600 trajectories
  - Output: `data/candidates/swebench_noisy.jsonl`
- ✅ **Step 6:** Generated router features (4× H200, 4 shards)
  - 4,016 feature vectors (13-dim), batched GPU inference, `--batch-size 16`
  - Output: `data/router_features/swebench.jsonl`
  - Git commit: `0353417`
- ✅ **Step 5:** DPO preference training (4× H200, Accelerate DDP)
  - 457 preference pairs (from 4,015 candidates, filtered by score gap ≥ 0.1)
  - bf16 (4-bit auto-disabled for DDP), 2 epochs, ~72s wall time
  - Output: `outputs/policy/swebench_noisy/final`
  - Git commits: `aa7a55a`, `9275759`, `ee5bc86`
- ✅ **Step 7:** Trained router on generated features (1× H200, CPU-bound MLP)
  - MLP (13→128→64→1), 10,499 params, Lagrangian CVaR objective
  - 4,016 training samples, 20 epochs, ~25s wall time
  - Acc=22.1% (88% class imbalance), λ=1→11.6, temp scaling=10.0 (upper bound)
  - Output: `outputs/router/swebench_noisy/router_final.pt`
  - Git commit: `cec1a0c`

## Next

- **Step 8:** Evaluate all methods (R2V, SLM-only, LLM-only, entropy router)
  ```bash
  python scripts/evaluate.py \
      --config configs/swebench/noisy.yaml \
      --features data/router_features/swebench.jsonl \
      --trajectories data/trajectories/swebench_noisy/trajectories.jsonl \
      --router-path outputs/router/swebench_noisy/router_final.pt \
      --output results/swebench_noisy \
      --seeds 1 2 3 \
      --methods r2v slm_only llm_only entropy_router \
      --overrides logging.wandb_mode=disabled
  ```
- **Step 9:** Ablation studies

## HPC Notes

- **GPU access in Singularity:** Must use `srun --jobid=<JID> --overlap --cpu-bind=none --pty` to attach to SLURM GPU job, then `singularity exec --nv` inside that shell. SSH-ing directly to compute nodes does not expose GPUs.
- **4-bit quantization:** Disabled (`load_in_4bit=false`) because bitsandbytes is broken in the Singularity container. bf16 works fine on H200.
- **Gradient checkpointing:** Requires `use_reentrant=False` + `enable_input_require_grads()` for LoRA compatibility.
- **DataLoader:** `num_workers=2, pin_memory=True, persistent_workers=True` for prefetching. If pickle errors in container, fall back to `num_workers=0`.
- **DPO multi-GPU:** `accelerate launch --num_processes=4 --multi_gpu` works. 4-bit quantization auto-disabled when `WORLD_SIZE > 1`.
- **wandb:** Not configured on HPC. Always pass `--overrides logging.wandb_mode=disabled` to avoid auth errors.
- **HF token:** Set via `exports.sh` (`HF_HOME`, `HF_TOKEN`) — needed for gated Llama model.