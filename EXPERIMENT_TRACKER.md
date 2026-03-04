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
| **Date** | 2026-03-03 (perturbations generated), 2026-03-04 (BC trained) |
| **Config** | `configs/swebench/noisy.yaml` |
| **Seeds** | 1, 2, 3 |
| **Status** | ☑ BC trained, verifier training next |

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
| Verifier training (SWE-bench) | 2× H200 | ~TBD | ~TBD | ☐ Next |
| MRP | 1× A100 | ~6h | ~6h | ☐ |
| WebArena full | 2× A100 | ~150h | ~4d | ☐ |
| SWE-bench full | 2× A100 | ~150h | ~4d | ☐ |
| Ablations (WA) | 1× A100 | ~100h | ~2d | ☐ |
| **Total** | — | **~400h** | — | — |

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
