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

## In Progress

- **Step 3b:** Train verifier model
  - Same HPC setup, uses labeled trajectories from Step 2
  - Config `verifier.mode` must be set to `trained` (default is `llm_judge` which skips training)

## Next

- **Step 4:** Generate K=5 candidates per task using BC-trained policy
  - Uses `outputs/policy/swebench_noisy/final` as the policy checkpoint
- **Step 5:** Train DPO preference stage (uses candidates + verifier scores)
  - Re-run `train_policy.py --stage preference --preference-data <pairs>`
- **Step 6:** Generate router features
- **Step 7:** Train router
- **Step 8:** Evaluate all methods (R2V, SLM-only, LLM-only, entropy router)
- **Step 9:** Ablation studies

## HPC Notes

- **GPU access in Singularity:** Must use `srun --jobid=<JID> --overlap --cpu-bind=none --pty` to attach to SLURM GPU job, then `singularity exec --nv` inside that shell. SSH-ing directly to compute nodes does not expose GPUs.
- **4-bit quantization:** Disabled (`load_in_4bit=false`) because bitsandbytes is broken in the Singularity container. bf16 works fine on H200.
- **Gradient checkpointing:** Requires `use_reentrant=False` + `enable_input_require_grads()` for LoRA compatibility.
- **DataLoader:** `num_workers=0, pin_memory=False` to avoid fork-based OOM in container.
- **HF token:** Set via `exports.sh` (`HF_HOME`, `HF_TOKEN`) — needed for gated Llama model.