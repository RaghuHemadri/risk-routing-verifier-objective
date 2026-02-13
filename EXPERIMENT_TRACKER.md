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
| **Date** | YYYY-MM-DD |
| **Config** | `configs/swebench/clean.yaml` |
| **Seeds** | 1, 2, 3, 4, 5 |
| **Status** | ☐ Not started |

**Actual results:**
| Method | SR | 95% CI | Cost |
|--------|----:|--------|-----:|
| R2V | — | — | — |
| SLM-only | — | — | — |

---

### Experiment 5: SWE-bench Noisy

| Field | Value |
|-------|-------|
| **Date** | YYYY-MM-DD |
| **Config** | `configs/swebench/noisy.yaml` |
| **Seeds** | 1, 2, 3, 4, 5 |
| **Status** | ☐ Not started |

**Actual results:**
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
| MRP | 1× A100 | ~6h | ~6h | ☐ |
| WebArena full | 2× A100 | ~150h | ~4d | ☐ |
| SWE-bench full | 2× A100 | ~150h | ~4d | ☐ |
| Ablations (WA) | 1× A100 | ~100h | ~2d | ☐ |
| **Total** | — | **~400h** | — | — |

---

## Reproducibility Checklist

- [ ] Fixed random seeds (1, 2, 3, 4, 5)
- [ ] Config files saved with each run
- [ ] wandb logs available
- [ ] JSONL event logs preserved
- [ ] Bootstrap CIs computed (95%)
- [ ] McNemar tests for all pairwise comparisons
- [ ] Holm-Bonferroni correction for multiple comparisons
- [ ] Git commit hash recorded for each experiment
