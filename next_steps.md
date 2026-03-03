## Completed

- ✅ Environment setup on NYU HPC (Singularity container)
- ✅ Collected 900 clean SWE-bench trajectories using Gemini 3 Flash
  - Run ID: `swebench_google_gemini-3-flash-preview_20260303T004504`
  - 300 tasks × 3 seeds, 13.3% success rate, $0.60, ~2.4h
- ✅ Generated 2700 perturbed trajectories (+ 900 clean copies = 3600 total)
  - 4 perturbation types (composite), 3 seeds, 79 seconds CPU-only
  - Output: `data/trajectories/swebench_noisy/trajectories.jsonl`

## Next

- Train the SLM policy (BC), verifier, generate candidates, train DPO, train router (Steps 3-7)
  - Requires GPU (A100) — use HPC SLURM jobs or local GPU
  - See `RUNNING_INSTRUCTIONS.md` Section 5 for individual stage commands 