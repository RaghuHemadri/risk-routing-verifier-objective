# Ablation Pipeline Run Log

**Branch:** `sri/run_ablations`  
**Run date:** 2026-04-21  
**Machines:** skampere2 (H200 GPUs), hyperturing1 (RTX 8000 GPUs)

---

## Overview

Full end-to-end ablation sweep over consistency loss coefficient λ ∈ {no_cons, 0.05, 0.2, 0.5, 1.0} × {humaneval, textworld} = 10 conditions.

Pipeline: GPU generation → CPU scoring → router training → offline evaluation

---

## Phase 1: Trajectory Generation (generate-only)

Generated K=5 candidates per step using `--generate-only`. Policy checkpoints used are `_s2` DPO finetuned Qwen2.5-Coder-7B.

| Experiment | Machine | GPUs | Batch | Steps | Output |
|---|---|---|---|---|---|
| humaneval_lam_0.05 | skampere2 | H200 | 512 | 43,200 | `humaneval_lam_0.05.gen_cache.jsonl` |
| humaneval_lam_0.2 | skampere2 | H200 | 512 | 43,200 | `humaneval_lam_0.2.gen_cache.jsonl` |
| humaneval_lam_0.5 | skampere2 | H200 | 512 | 43,200 | `humaneval_lam_0.5.gen_cache.jsonl` |
| humaneval_lam_1.0 | skampere2 | H200×2 shards | 512 | 43,200 | `humaneval_lam_1.0.shard_{000,001}.gen_cache.jsonl` |
| humaneval_no_cons | skampere2 | H200 | 128 | 43,200 | `humaneval_no_consistency.gen_cache.jsonl` |
| textworld_lam_0.05 | hyperturing1 (q1/q2/q4) + skampere2 (q3) | RTX8000 / H200 | 64 | 51,798 | per-shard gen_cache files |
| textworld_lam_0.2 | skampere2 | H200 | 512 | 51,798 | `textworld_lam_0.2.gen_cache.jsonl` |
| textworld_lam_0.5 | hyperturing1 (h1+h2) | RTX8000 | 64 | 51,798 | h1/h2 gen_cache files |
| textworld_lam_1.0 | hyperturing1 (h1+h2) | RTX8000 | 64 | 51,798 | h1/h2 gen_cache files |
| textworld_no_cons | hyperturing1 (h1+h2) | RTX8000 | 64 | 51,798 | h1/h2 gen_cache files |

**Notes:**
- `textworld_lam_0.05 q3` was re-generated on skampere2 H200 (hyperturing1 q3 was corrupt: max_seq_len=512 produced 96.6% zero-token trajectories)
- `humaneval_no_cons` required `--batch-size 128` (different gen_cache header)
- Gen-cache files stored on remote machines; not committed to git (too large)

---

## Phase 2: CPU Scoring (score-only)

Scoring runs the verifier on pre-generated candidates. No GPU required.

| Experiment | Machine | Speed | Duration | Output lines |
|---|---|---|---|---|
| humaneval ×5 | skampere2 CPUs | 9–12 steps/s | ~70 min | 43,200 each |
| textworld ×5 | hyperturing1 CPUs | ~600–950 steps/s | ~2 min | 51,798 each |

Verifier theory checks at completion:
- **HumanEval:** All 5 checks passing — `ent(F>S)`, `v(S>F)`, `spread(F>S)`, `cons(S>F)`, `lp(S>F)` ✓
- **TextWorld:** Only `v(S>F)` passing; verifier spread ≈ 0.009 (near-zero discrimination)

Merged scored feature files → `data/router_features/final_ablations/` (10 × ~14MB JSONL files).

---

## Phase 3: Router Training

Small MLP (128→64 hidden dims) trained with Lagrangian CVaR objective.

**Config overrides:** `cost_llm=20`, `cvar_epsilon=0.1`, `logging.wandb_mode=disabled`  
**Epochs:** 20, **Batch size:** 64, **LR:** 1e-3

| Experiment | Train samples | Val samples | Duration | Final temp |
|---|---|---|---|---|
| textworld_no_cons | ~36,258 | ~7,768 | ~2 min | ~0.57 |
| textworld_lam_0.05 | ~36,258 | ~7,768 | ~2 min | ~0.68 |
| textworld_lam_0.2 | ~36,258 | ~7,768 | ~2 min | — |
| textworld_lam_0.5 | ~36,258 | ~7,768 | ~2 min | — |
| textworld_lam_1.0 | ~36,258 | ~7,768 | ~2 min | — |
| humaneval_lam_1.0 | ~30,240 | ~6,480 | ~2 min | 3.51 |
| humaneval_lam_0.05 | ~30,240 | ~6,480 | ~2 min | — |
| humaneval_lam_0.2 | ~30,240 | ~6,480 | ~2 min | — |
| humaneval_lam_0.5 | ~30,240 | ~6,480 | ~2 min | — |
| humaneval_no_cons | ~30,240 | ~6,480 | ~2 min | — |

All training runs completed on skampere2 CPUs. Checkpoints in `outputs/router/ablation/`.

---

## Phase 4: Offline Evaluation

Evaluated on test split only. Methods: r2v@{0.1,0.2,...,0.8}, slm_only, llm_only, entropy_router, oracle_router, heuristic_router, verifier_router.

**Config overrides:** `cost_llm=20`, `logging.wandb_mode=disabled`

| Experiment | Duration | Test episodes |
|---|---|---|
| textworld ×5 | ~68 min each | ~1,120 |
| humaneval ×5 | ~41 min each | ~666 |

Results in `results/ablation/{benchmark}_{lambda}/structured_results/`:
- `main_table.csv` — per-method aggregate + per-seed metrics
- `comparisons.csv` — McNemar pairwise significance tests
- `main_table.tex` — LaTeX table for paper
- `eval_{benchmark}_noisy_{date}.json` — full structured result bundle
- `llm_summary.json` — LLM-readable narrative summary

---

## Files in This Commit

```
.gitignore                                        (updated: allow results/ablation/, outputs/router/ablation/)
scripts/generate_router_features.py               (updated: tensor_parallel_size, max_num_seqs, truncate_prompt_tokens)
results/ablation/ABLATION_RESULTS.md              (this analysis)
results/ablation/PIPELINE_RUN_LOG.md              (this file)
results/ablation/{10 experiments}/                (eval CSVs, JSONs, LaTeX)
outputs/router/ablation/{10 experiments}/         (router_final.pt, best.pt, config.yaml, training_log.jsonl)
```

## Files NOT in This Commit (upload to GitLab separately)

Feature JSONL files — ~14MB each, ~140MB total:
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
