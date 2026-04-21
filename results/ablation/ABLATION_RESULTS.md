# Ablation: Consistency Loss Coefficient (λ)

**Branch:** `sri/run_ablations`  
**Date:** 2026-04-21  
**Policy:** Qwen2.5-Coder-7B, DPO `_s2` checkpoints  
**Benchmarks:** TextWorld (51,798 steps), HumanEval (43,200 steps)  
**λ sweep:** {no_cons, 0.05, 0.2, 0.5, 1.0}  
**Router training:** `cost_llm=20`, `cvar_epsilon=0.1`, 20 epochs  
**Eval:** offline, test-split only, threshold sweep 0.1–0.8

---

## TextWorld Results

All λ values produce **identical results** (to 4 decimal places).

| Method | SR | Worst-Seed SR | CVaR-Fail | LLM Rate | Brier |
|---|---|---|---|---|---|
| **R2V@0.5** (all λ) | **99.1%** | **96.9%** | 0.031 | 34.8% | ~0.29–0.33 |
| Oracle | 100% | 100% | 0.000 | 35.7% | — |
| SLM only | 64.3% | 50.0% | 0.500 | 0% | — |
| LLM only | 100% | 100% | 0.000 | 100% | — |
| Heuristic | 64.3% | 50.0% | 0.500 | 0% | — |
| Entropy router | 64.3% | 50.0% | 0.500 | 0% | — |

R2V ECE by λ (all at threshold 0.5):

| λ | Brier | ECE (approx) |
|---|---|---|
| no_cons | 0.323 | high |
| 0.05 | 0.292 | high |
| 0.2 | 0.329 | high |
| 0.5 | 0.323 | high |
| 1.0 | 0.264 | high |

**Key findings:**
- R2V achieves **+34.8 pp over SLM** and matches oracle LLM efficiency (34.8% vs 35.7%)
- λ is completely irrelevant — the routing signal on TextWorld is strong enough that any (or no) consistency regularization leads to the same routing policy
- Heuristic and entropy-only routers completely fail (same as SLM), confirming the textworld verifier has near-zero per-episode discrimination; the router exploits multi-feature combinations the heuristics miss

---

## HumanEval Results

| λ | R2V@0.5 SR | R2V@0.1 SR | LLM@0.1 | Brier | Note |
|---|---|---|---|---|---|
| no_cons | 91.9% | 100% | 100% | 0.054 | Collapsed to SLM |
| **0.05** | **92.8%** | **99.7%** | **75.7%** | 0.111 | **Best — only useful λ** |
| 0.2 | 91.9% | 100% | 100% | 0.021 | Collapsed |
| 0.5 | 92.5% | 100% | 100% | 0.111 | Near-collapsed |
| 1.0 | 91.9% | 100% | 100% | 0.040 | Collapsed |
| **SLM only** | 91.9% | — | 0% | — | Baseline |
| **Heuristic** | **98.5%** | — | 19.7% | — | Best learned-free |
| Oracle | 100% | — | 8.1% | — | Upper bound |

Statistical significance (McNemar, α=0.05):
- R2V@0.5 vs SLM: p=0.25, **not significant** for all λ
- R2V@0.1 (λ=0.05) vs SLM: p<0.001, **significant** (+7.8 pp)
- R2V@0.5 vs Heuristic: p<0.001, **heuristic significantly better** for all λ

**Key findings:**
- Only λ=0.05 produces a non-collapsed router. All other values (including no_cons) output near-zero routing probability at threshold ≥ 0.2 → identical to SLM.
- The collapse is a calibration artifact: λ≥0.2 models have very low ECE (0.021) because they confidently predict "SLM succeeds" — correct 91.9% of the time, but making no routing decisions.
- Heuristic (entropy+verifier_mean thresholds) dominates the learned router: 98.5% SR at 19.7% LLM calls vs R2V@0.1 (λ=0.05): 99.7% SR at 75.7% LLM calls.
- Oracle uses only 8.1% LLM calls for 100% SR — the true routing problem is very sparse, making learned routing difficult with current features.

---

## Cross-Benchmark Summary

| Benchmark | Best λ | R2V SR | vs SLM | vs Oracle gap | vs Heuristic |
|---|---|---|---|---|---|
| TextWorld | Any | 99.1% | **+34.8 pp** | -0.9 pp | **+34.8 pp** |
| HumanEval | 0.05 | 99.7%* | +7.8 pp* | -0.3 pp* | +1.2 pp* |

*at threshold=0.1 (75.7% LLM call rate)

At matched LLM cost (~20% calls), heuristic (98.5% SR) beats best R2V (~92.8% SR) on HumanEval.

---

## Interpretation

**TextWorld:** The routing problem is "easy" — there is a clear, strong signal in the 15-dimensional feature space that separates episodes the SLM will fail. Any λ (or no consistency loss) leads to the same optimal policy. The consistency signal may be redundant when the routing signal is already saturating.

**HumanEval:** The routing problem is "hard" — only 8.1% of episodes benefit from LLM escalation, making the decision boundary sparse and noisy. The sharp λ transition (0.05 works, 0.2 doesn't) suggests consistency regularization acts as a critical regularizer preventing collapse to the trivial SLM-always policy. Too much consistency (≥0.2) over-regularizes and collapses routing anyway. The heuristic router's superiority suggests the current MLP features do not capture the episodic structure that the hand-crafted entropy+verifier_mean rule exploits.

**Recommended λ: 0.05** for HumanEval; irrelevant for TextWorld.

---

## Artifact Locations

| Artifact | Path |
|---|---|
| Feature files (JSONL, ~14MB each) | `data/router_features/final_ablations/` |
| Router checkpoints (.pt) | `outputs/router/ablation/{benchmark}_{lambda}/router_final.pt` |
| Eval results (CSV/JSON/LaTeX) | `results/ablation/{benchmark}_{lambda}/structured_results/` |
| Eval logs | `skampere2:/lfs/skampere2/0/srivatsavad/eval_*.log` |

**Feature files to upload to GitLab** (too large for git, ~140MB total):
```
data/router_features/final_ablations/humaneval_lam_0.05.jsonl
data/router_features/final_ablations/humaneval_lam_0.2.jsonl
data/router_features/final_ablations/humaneval_lam_0.5.jsonl
data/router_features/final_ablations/humaneval_lam_1.0.jsonl
data/router_features/final_ablations/humaneval_no_consistency.jsonl
data/router_features/final_ablations/textworld_lam_0.05.jsonl
data/router_features/final_ablations/textworld_lam_0.2.jsonl
data/router_features/final_ablations/textworld_lam_0.5.jsonl
data/router_features/final_ablations/textworld_lam_1.0.jsonl
data/router_features/final_ablations/textworld_no_consistency.jsonl
```
