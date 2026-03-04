# Experiment Tracker — R2V-Agent

This document tracks all experiment runs, their configurations, and results.
Update this file after each experiment cycle with the outcomes.

---

## How Results Are Stored

Every experiment produces structured outputs in multiple formats:

| Format | Location | Purpose |
|--------|----------|---------|
| **JSON Bundle** | `results/*/structured_results/*.json` | Complete results (config + metrics + comparisons) |
| **CSV Tables** | `results/*/structured_results/main_table.csv` | Machine-readable main results |
| **Comparisons CSV** | `results/*/structured_results/comparisons.csv` | Pairwise statistical tests |
| **Ablations CSV** | `results/*/structured_results/ablations.csv` | Ablation study deltas |
| **LLM Summary** | `results/*/structured_results/llm_summary.json` | **Feed to LLM for paper writing** |
| **LaTeX Table** | `results/*/structured_results/main_table.tex` | Paper-ready tables |
| **JSONL Logs** | `results/*/evaluation_log.jsonl` | Per-event training/eval logs |
| **wandb** | Dashboard link | Interactive plots & dashboards |

### Using Results for LLM Paper Writing

1. After evaluation, copy the LLM summary:
   ```bash
   cat results/webarena_noisy/structured_results/llm_summary.json
   ```

2. Paste into your LLM prompt:
   ```
   Here are the experiment results for my paper. Please write the Results
   section based on this data:

   <paste llm_summary.json here>

   The paper is about [R2V-Agent description]. Key claims to support:
   1. R2V outperforms baselines on noisy conditions
   2. The gap widens under more severe perturbations
   3. Risk-calibrated routing improves worst-case performance
   4. Each component (BC, DPO, consistency, router) contributes
   ```

---

## Experiment Log

### Completed: Data Collection & Perturbation Generation (Steps 1-2)

| Field | Value |
|-------|-------|
| **Date** | 2026-03-03 |
| **Run ID** | `swebench_google_gemini-3-flash-preview_20260303T004504` |
| **Teacher Model** | Google Gemini 3 Flash (`gemini-3-flash-preview`) |
| **Benchmark** | SWE-bench (test split, 300 tasks) |
| **Config** | `configs/swebench/clean.yaml` (collection), `configs/swebench/noisy.yaml` (perturbations) |
| **Host** | cyberdragon |
| **Git Commit** | `654f54c` |
| **Python** | 3.12.3 |

**Step 1 — Clean Trajectory Collection:**

| Metric | Value |
|--------|-------|
| Episodes | 900 (300 tasks × 3 seeds) |
| Successes | 120 (13.3%) |
| Cost | $0.60 |
| Wall Time | ~2.4 h (4 workers) |
| Avg Steps/Episode | 1.1 |
| Output | `data/runs/swebench_google_gemini-3-flash-preview_20260303T004504/trajectories.jsonl` (9.5 MB) |

**Step 2 — Perturbation Generation:**

| Metric | Value |
|--------|-------|
| Perturbed Episodes | 2700 (900 × 3 seeds) |
| Clean Copies | 900 |
| Total Output | 3600 episodes |
| Perturbation Types | Tool flakiness, partial observability, prompt injection, distractors (composite) |
| Wall Time | 79 seconds (CPU-only) |
| Output | `data/trajectories/swebench_noisy/trajectories.jsonl` (36 MB) |

**Smoke Test (prior):** Run ID `swebench_google_gemini-3-flash-preview_20260303T003125` — 1 task, verified pipeline works.

---

### Experiment 1: Minimal Reproducible Prototype (MRP)

| Field | Value |
|-------|-------|
| **Date** | YYYY-MM-DD |
| **Config** | `configs/mrp.yaml` |
| **SLURM Job** | `sbatch scripts/slurm/mrp.sh` |
| **GPU** | 1× A100 80GB |
| **Wall Time** | ~4-6 hours |
| **Purpose** | Validate full pipeline end-to-end |
| **Status** | ☐ Not started |

**Key metrics to check:**
- [ ] Pipeline completes without errors
- [ ] Results files generated in `results/mrp/`
- [ ] Success rates are non-trivial (>0, <1)
- [ ] Router makes varied decisions

**Notes:**
```
(Add notes here after running)
```

---

### Experiment 2: WebArena Clean

| Field | Value |
|-------|-------|
| **Date** | YYYY-MM-DD |
| **Config** | `configs/webarena/clean.yaml` |
| **SLURM Job** | `08_evaluate.sh webarena clean` |
| **Seeds** | 1, 2, 3, 4, 5 |
| **Status** | ☐ Not started |

**Expected results (Table 1, Clean column):**
- R2V SR: ~70-80%
- SLM-only SR: ~55-65%
- LLM-only SR: ~75-85%

**Actual results:**
| Method | SR | 95% CI | Cost | LLM Call Rate |
|--------|----:|--------|-----:|---------------|
| R2V | — | — | — | — |
| SLM-only | — | — | — | — |
| LLM-only | — | — | — | — |
| Entropy Router | — | — | — | — |

---

### Experiment 3: WebArena Noisy

| Field | Value |
|-------|-------|
| **Date** | YYYY-MM-DD |
| **Config** | `configs/webarena/noisy.yaml` |
| **Seeds** | 1, 2, 3, 4, 5 |
| **Status** | ☐ Not started |

**Expected results (Table 1, Noisy column):**
- R2V SR: ~60-70% (key: robust under noise)
- SLM-only SR: ~35-45% (drops significantly)
- Robustness gap (clean - noisy): R2V < baselines

**Actual results:**
| Method | SR | Worst-Seed | CVaR-Fail | Cost | Safety-Fail |
|--------|----:|----------:|----------:|-----:|------------:|
| R2V | — | — | — | — | — |
| SLM-only | — | — | — | — | — |
| LLM-only | — | — | — | — | — |
| Entropy Router | — | — | — | — | — |

---

### Experiment 4: SWE-bench Clean

| Field | Value |
|-------|-------|
| **Date** | 2026-03-03 (data collected) |
| **Config** | `configs/swebench/clean.yaml` |
| **Seeds** | 1, 2, 3 |
| **Status** | ☐ Data collected, training in progress |

**Data collection results:**
| Metric | Value |
|--------|-------|
| Episodes | 900 |
| Success Rate | 13.3% (120/900) |
| Teacher Model | Gemini 3 Flash |
| Cost | $0.60 |

**Evaluation results (pending training):**
| Method | SR | 95% CI | Cost |
|--------|----:|--------|-----:|
| R2V | — | — | — |
| SLM-only | — | — | — |

---

### Experiment 5: SWE-bench Noisy

| Field | Value |
|-------|-------|
| **Date** | 2026-03-03 (perturbations generated), 2026-03-04 (BC + verifier trained) |
| **Config** | `configs/swebench/noisy.yaml` |
| **Seeds** | 1, 2, 3 |
| **Status** | ☑ BC + verifier trained, candidate generation next |

**Perturbation data:**
| Metric | Value |
|--------|-------|
| Total Episodes | 3600 (900 clean + 2700 perturbed) |
| Perturbation Types | 4 (composite: tool flakiness, partial obs, prompt injection, distractors) |
| Generation Time | 79 seconds |

**Step 3a — BC Policy Training (completed 2026-03-04):**

| Metric | Value |
|--------|-------|
| Platform | NYU Greene HPC, 2× H200 GPU, Singularity container |
| Model | Llama-3.1-8B-Instruct + LoRA (167M trainable / 8B total, 2.05%) |
| Precision | bf16 (4-bit disabled — bitsandbytes incompatible in container) |
| BC Examples | 480 (from 120 successful episodes × 3600 total) |
| Train/Val Split | 385 / 95 |
| Epochs | 3 |
| Training Loss | 3.55 → 0.11 |
| Wall Time | ~17 minutes |
| Output | `outputs/policy/swebench_noisy/final` |
| Git Commit | `f43f796` |

**Step 3b — Verifier Training (completed 2026-03-04):**

| Metric | Value |
|--------|-------|
| Platform | NYU Greene HPC, 1× H200 GPU, Singularity container |
| Backbone | Llama-3.1-8B-Instruct (frozen) + MLP heads (4.2M trainable / 8B total, 0.05%) |
| Precision | bf16, dynamic padding (median seq len 1723 tokens) |
| Verifier Examples | 36,144 step-level examples (28,916 train / 7,228 val) |
| Epochs | 3 |
| Final Loss | 0.211 |
| Final Accuracy | 91.3% |
| Wall Time | ~1h 56min |
| Output | `outputs/verifier/swebench_noisy/final/verifier.pt` |
| Git Commit | `4ca21fb` |

**Evaluation results (pending training):**
| Method | SR | Worst-Seed | CVaR-Fail | Cost |
|--------|----:|----------:|----------:|-----:|
| R2V | — | — | — | — |
| SLM-only | — | — | — | — |

---

### Experiment 6: Ablation Studies (Table 2)

| Field | Value |
|-------|-------|
| **Date** | YYYY-MM-DD |
| **Config** | `configs/webarena/noisy.yaml` + ablation overrides |
| **SLURM Job** | `09_ablations.sh webarena` |
| **Status** | ☐ Not started |

**Ablation results:**
| Ablation | SR | Δ from Full | p-value | Significant? |
|----------|----:|------------:|--------:|:------------:|
| Full R2V (baseline) | — | — | — | — |
| No preference (BC-only) | — | — | — | — |
| No consistency | — | — | — | — |
| No verifier | — | — | — | — |
| No risk calibration | — | — | — | — |
| Static threshold | — | — | — | — |
| No self-correction | — | — | — | — |

---

### Experiment 7: Per-Perturbation Analysis (Table 3)

| Perturbation Type | SR (R2V) | SR (SLM-only) | Δ |
|-------------------|----------:|---------------:|----:|
| Clean | — | — | — |
| Tool Flakiness | — | — | — |
| Partial Observability | — | — | — |
| Prompt Injection | — | — | — |
| Distractors | — | — | — |
| All Combined | — | — | — |

---

## Metric Definitions

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **SR** | Average success rate across tasks | ↑ higher is better |
| **Worst-Seed SR** | min over seeds of average SR | ↑ higher is better |
| **CVaR-Fail** | CVaR₀.₃ of failure rate | ↓ lower is better |
| **Bottom-10% SR** | Average SR of bottom 10% tasks | ↑ higher is better |
| **Cost** | Average (weighted) tool/LLM calls per episode | ↓ lower is better |
| **LLM Call Rate** | Fraction of steps escalated to teacher LLM | ↓ lower is better |
| **Safety-Fail** | Rate of safety violation episodes | ↓ lower is better |
| **ECE** | Expected Calibration Error of router | ↓ lower is better |
| **Brier** | Brier score of router confidence | ↓ lower is better |
| **Robustness Gap** | SR(clean) - SR(noisy) | ↓ lower is better |

---

## Compute Budget Tracking

| Experiment | GPUs | GPU-Hours | Wall Time | Status |
|-----------|------|-----------|-----------|--------|
| Data collection (SWE-bench) | CPU / API | 0 | 2.4h | ☑ Done |
| Perturbation generation | CPU | 0 | 79s | ☑ Done |
| BC policy training (SWE-bench noisy) | 2× H200 | ~0.6h | ~17min | ☑ Done |
| Verifier training (SWE-bench noisy) | 1× H200 | ~2h | ~2h | ☑ Done |
| Candidate generation | 1× H200 | ~TBD | ~TBD | ☐ Next |
| MRP | 1× A100 | ~6h | ~6h | ☐ |
| WebArena full | 2× A100 | ~150h | ~4d | ☐ |
| SWE-bench full | 2× A100 | ~150h | ~4d | ☐ |
| Ablations (WA) | 1× A100 | ~100h | ~2d | ☐ |
| **Total** | — | **~400h** | — | — |

---

## Dataset Statistics & Observations

### Data Pipeline Summary

| Stage | Granularity | Count | Notes |
|-------|-------------|------:|-------|
| Raw tasks | task-level | 300 | SWE-bench test split |
| Clean episodes | episode-level | 900 | 300 tasks × 3 seeds |
| Perturbed episodes | episode-level | 2700 | 900 × 3 perturbation seeds (composite) |
| Total episodes | episode-level | 3600 | 900 clean + 2700 perturbed |
| Successful episodes | episode-level | 120 | 13.3% success rate |
| BC examples | step-level | 480 | From 120 successful episodes (~4 steps avg) |
| Verifier examples | step-level | 36,144 | From all 3600 episodes (correct/incorrect step labels) |

### Token Length Distribution (Verifier Training Data)

Analysis of 4,016 sampled verifier examples (tokenized with Llama-3.1 tokenizer):

| Statistic | Token Count |
|-----------|------------:|
| Min | 52 |
| Max | 31,027 |
| Mean | 2,355 |
| Median | 1,723 |
| Std Dev | ~2,500 |
| P90 | 4,436 |
| P95 | 5,222 |
| P99 | 8,460 |

**Token length bucket distribution:**

| Bucket | % of examples |
|--------|-------------:|
| < 512 | 7.1% |
| < 1,024 | 28.6% |
| < 2,048 | 56.0% |
| < 4,096 | 85.5% |
| < 8,192 | 97.5% |
| ≥ 8,192 | ~2.5% |

**Implication:** With `max_seq_len=2048`, ~56% of examples fit without truncation. The remaining ~44% are truncated but still retain substantial context. Using `max_seq_len=4096` would cover 85.5% but at 2× memory cost.

### Perturbation Type Distribution

All perturbations are **composite** — each perturbed episode applies all four types simultaneously:

| Perturbation | Description | Effect |
|-------------|-------------|--------|
| Tool Flakiness | Random tool call failures / timeouts | Tests retry & fallback behavior |
| Partial Observability | Missing or incomplete observation fields | Tests robustness to missing info |
| Prompt Injection | Adversarial text injected into observations | Tests instruction-following fidelity |
| Distractors | Irrelevant information added to context | Tests focus & noise filtering |

### Training Data Splits

**BC Policy (Step 3a):**

| Split | Count | % |
|-------|------:|----:|
| Train | 385 | 80.2% |
| Val | 95 | 19.8% |
| **Total** | **480** | 100% |

**Verifier (Step 3b):**

| Split | Count | % |
|-------|------:|----:|
| Train | 28,916 | 80.0% |
| Val | 7,228 | 20.0% |
| **Total** | **36,144** | 100% |

### Model Architecture Details

**BC Policy (LoRA on Llama-3.1-8B-Instruct):**

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | ~167M (2.05% of 8B) |
| Precision | bf16 (full, no quantization) |
| Gradient checkpointing | Yes (`use_reentrant=False`) |
| Optimizer | AdamW (lr=2e-5, weight_decay=0.01) |

**Trained Verifier (MLP heads on frozen backbone):**

| Parameter | Value |
|-----------|-------|
| Backbone | `meta-llama/Llama-3.1-8B-Instruct` (frozen) |
| Head architecture | MLP: hidden_dim → 256 → 128 → 1 (correctness), hidden_dim → 256 → 128 → 1 (risk) |
| Trainable params | ~4.2M (0.05% of 8B) |
| Precision | bf16, dynamic padding |
| Optimizer | AdamW (lr=1e-4) |
| Batch size | 32 (effective, with dynamic padding) |
| max_seq_len | 2048 |

---

## Key Observations & Lessons Learned

### HPC Deployment (NYU Greene)

1. **GPU visibility in Singularity containers:** SSH-ing to a compute node does NOT expose GPUs. You must attach to the SLURM job's cgroup:
   ```bash
   srun --jobid=<JID> --overlap --cpu-bind=none --pty bash
   # THEN inside that shell:
   singularity exec --nv --overlay ... <sif> bash
   ```
   Without `--overlap`, srun tries to allocate new resources and hangs. Without `--cpu-bind=none`, it may fail with CPU binding errors.

2. **bitsandbytes incompatibility:** 4-bit quantization via bitsandbytes does not work inside the Singularity container (CUDA runtime mismatch). Workaround: use full bf16 precision. H200 GPUs have sufficient VRAM (80GB) for 8B models in bf16.

3. **`num_workers > 0` in DataLoader:** Causes "cannot pickle" errors inside Singularity overlays due to filesystem constraints. Fix: always set `num_workers=0, pin_memory=False`.

4. **HuggingFace cache:** Must set `HF_HOME=/scratch/rh3884/hf_cache` to avoid quota issues on `$HOME`. Model downloads are ~16GB for Llama-3.1-8B.

### Training Observations

5. **Gradient checkpointing + LoRA:** Using `gradient_checkpointing_enable()` with default `use_reentrant=True` causes "element 0 of tensors does not require grad" error when combined with LoRA. Fix: `use_reentrant=False` + `model.enable_input_require_grads()`. (Commit `f43f796`.)

6. **Dynamic padding is critical for verifier training:** Initial verifier training padded every example to `max_seq_len=4096`, wasting >60% of compute on padding tokens. With dynamic padding (pad to max-in-batch), training time dropped from **8+ hours → ~2 hours** (4× speedup). Median sequence is only 1,723 tokens.

7. **BC training converges quickly:** 3 epochs sufficient. Loss 3.55 → 0.11, no signs of overfitting on 480 examples. Likely because LoRA has limited capacity and base model already has strong language priors.

8. **Verifier accuracy is high (91.3%):** Suggests step-level correctness/incorrectness signals are learnable from trajectory data. The frozen backbone provides strong representations; only the MLP heads need training.

9. **pad_token_id warning:** When generating with Llama models, must explicitly pass `pad_token_id=tokenizer.pad_token_id` to `model.generate()` to suppress the "Setting pad_token_id to eos_token_id" warning that floods logs.

### Data Quality Observations

10. **Low success rate (13.3%):** Only 120 of 900 episodes succeed. This means BC has limited positive examples (480 steps). DPO (Step 5) is critical to leverage the negative examples via preference pairs.

11. **Step count is low (~1.1 avg):** SWE-bench episodes tend to be short (1-2 steps). This limits the diversity of step-level data compared to WebArena (which has longer episodes).

12. **Verifier data is much larger than BC data (75×):** 36,144 vs 480 examples — verifier training can use ALL episodes (success + failure), while BC can only use successful ones. This asymmetry makes the verifier the more robust component.

### Code/API Observations

13. **Stale script APIs:** `generate_candidates.py` had completely stale constructor calls and method names that didn't match the current codebase. Expect similar issues in `generate_router_features.py`, `train_router.py`, `evaluate.py`, and `run_ablations.py` — these should be audited before running.

14. **Observation attribute:** The correct attribute for raw observation text is `Observation.raw_text`, not `Observation.text` (which doesn't exist).

15. **PolicyModel constructor:** Takes a `config: dict`, not `(model_name, lora_config)` as originally coded. Loading from checkpoint: `PolicyModel(config)` then `model.load_checkpoint(path)`.

---

## Reproducibility Checklist

- [x] Fixed random seeds (1, 2, 3, 4, 5)
- [x] Config files saved with each run
- [ ] wandb logs available
- [x] JSONL event logs preserved
- [ ] Bootstrap CIs computed (95%)
- [ ] McNemar tests for all pairwise comparisons
- [ ] Holm-Bonferroni correction for multiple comparisons
- [x] Git commit hash recorded for each experiment
