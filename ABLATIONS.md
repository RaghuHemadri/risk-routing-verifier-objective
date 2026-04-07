# R2V-Agent Ablation Studies

Ablation plan for NeurIPS submission. Organized by priority tier based on which claims each ablation directly validates.

---

## Core Claims Being Validated

| RQ | Claim | Key Ablations |
|---|---|---|
| RQ1 | CVaR-constrained routing improves worst-case robustness over EV baselines | CVaR α/ε sensitivity, CVaR vs worst-case vs EV |
| RQ2 | Verifier-distillation is the primary signal quality driver for routing | Feature group ablations, K-candidates sweep |
| RQ3 | Favorable cost-accuracy Pareto frontier vs SLM-only and LLM-only | Oracle router, cost ratio sweep, threshold sweep |
| RQ4 | Consistency regularization generalizes across unseen perturbation seeds | Held-out perturbation type, λ_cons sensitivity |

---

## Baselines (in `scripts/evaluate.py`)

Run with `--methods slm_only llm_only entropy_router oracle_router heuristic_router verifier_router r2v`.

| Method | Flag | Description |
|---|---|---|
| SLM-only | `slm_only` | Never escalate to LLM. Lower bound on success rate. |
| LLM-only | `llm_only` | Always escalate to LLM. Upper bound on cost. |
| Entropy router | `entropy_router` | Escalate when entropy > threshold (default 2.0). |
| Oracle router | `oracle_router` | Routes to LLM exactly when SLM fails (uses ground truth). Theoretical ceiling for any routing strategy. |
| Heuristic router | `heuristic_router` | Rule-based: escalate when entropy > 2.5 OR verifier_mean < 0.4. Training-free baseline. |
| Verifier router | `verifier_router` | Escalate when best verifier score < threshold. Tests verifier signal alone. |
| R2V (full) | `r2v` | Full CVaR-constrained learned router. |

**Inference-time feature ablation** (no retraining needed):
```bash
python scripts/evaluate.py --feature-mask 0 1 2 3 4 ...  # keep only listed feature indices
```

---

## Tier 1 — Must-Have for Acceptance

Missing any of these = high rejection risk. Each directly validates a primary paper claim.

### A. CVaR Hyperparameter Sensitivity

*Why required:* Reviewers always ask "how sensitive is your key risk parameter?" Inspired by Chow et al. ICML 2017.

**α sweep** — controls which fraction of worst seeds CVaR focuses on:

| Ablation key | Override | α value | Expected behavior |
|---|---|---|---|
| `cvar_alpha_0.1` | `training.router.cvar_alpha=0.1` | 0.1 | Most risk-averse; over-escalates, high cost |
| *(default)* | — | **0.2** | Best trade-off |
| `cvar_alpha_0.3` | `training.router.cvar_alpha=0.3` | 0.3 | Less conservative |
| `cvar_alpha_0.5` | `training.router.cvar_alpha=0.5` | 0.5 | Near average-case; weakest robustness |

**ε sweep** — controls the maximum allowable CVaR failure rate:

| Ablation key | Override | ε value | Expected behavior |
|---|---|---|---|
| `cvar_eps_0.1` | `training.router.cvar_epsilon=0.1` | 0.1 | Tight constraint; higher LLM call rate |
| `cvar_eps_0.2` | `training.router.cvar_epsilon=0.2` | 0.2 | |
| *(default)* | — | **0.3** | Best trade-off |
| `cvar_eps_0.4` | `training.router.cvar_epsilon=0.4` | 0.4 | Loose constraint; lower robustness |

**Retrain:** Router only (~25s each).

---

### B. CVaR vs Worst-Case vs Expected-Value Loss

*Why required:* Justifies CVaR as the right risk measure over simpler alternatives. RouteLLM (Ong et al. NeurIPS 2024) and FrugalGPT (Chen et al. ICML 2023) use average-case; worst-case (min-max) is another standard choice.

| Ablation key | Override | Objective |
|---|---|---|
| `no_risk_calibration` | `router.cvar_alpha=1.0` | Expected value (no risk-awareness) |
| `worst_case_loss` | `training.router.robust_objective=worst_case` | Min-max over seeds (single worst seed) |
| *(default)* | — | **CVaR α=0.2** |

**Expected result:** CVaR-Fail metric: EV > worst-case > CVaR (lower is better).

**Retrain:** Router only.

---

### C. Oracle Router Upper Bound

*Why required:* Standard in model-cascading papers (FrugalGPT, EcoAssistant Chen et al. ICLR 2024). Establishes the theoretical ceiling for routing quality.

| Method | LLM call rate | SR | Meaning |
|---|---|---|---|
| `oracle_router` | Variable (only when needed) | ~1.0 | Perfect routing with ground truth |
| `r2v` | ~X% | ~Y% | Learned router approximation |

**Gap between R2V and oracle** = headroom for future routing improvements.

**Implementation:** Already in `evaluate.py`. No retraining needed.

---

### D. Feature Group Ablations

*Why required:* Validates that each feature group contributes to routing quality. Inspired by RouterBench (Hu et al. 2024). These are the cheapest experiments (router retrain ~25s each).

Feature layout:

| Index | Feature | Group |
|---|---|---|
| 0 | entropy | Entropy |
| 1 | verifier_score_spread | Verifier |
| 2 | verifier_score_mean | Verifier |
| 3 | verifier_score_std | Verifier |
| 4 | verifier_score_best | Verifier |
| 5 | horizon_fraction | Step progress |
| 6 | step_number | Step progress |
| 7 | normalized_context_length | Context |
| 8 | perturbation_tool_flakiness | Perturbation type |
| 9 | perturbation_partial_obs | Perturbation type |
| 10 | perturbation_prompt_injection | Perturbation type |
| 11 | perturbation_distractors | Perturbation type |
| 12 | perturbation_none | Perturbation type |

| Ablation key | Features kept | Tests |
|---|---|---|
| *(default)* | All 13 | Full R2V |
| `feat_verifier_only` | 0–4 | Is step/context/perturbation type info useful? |
| `feat_no_perturbation_type` | 0–7 | Does knowing perturbation type help the router? |
| `feat_no_horizon` | 0–4, 7–12 | Is step progress signal necessary? |
| `feat_entropy_only` | 0 | Minimal-signal router (single feature) |
| `feat_no_verifier` | 0, 5–12 | Verifier-less routing; entropy+step+context only |
| `feat_best_score_only` | 0, 4–12 | Do spread/std verifier stats add value vs best alone? |

**Retrain:** Router with masked input OR use `--feature-mask` flag in evaluate.py for inference-time ablation.

---

### E. K-Candidates Sweep

*Why required:* Directly tests verifier signal quality vs compute trade-off. Reviewers will ask what K does to performance and cost.

| K | Description |
|---|---|
| 1 | No ensemble; single SLM action scored |
| 3 | Minimal ensemble |
| **5** | **Default** |
| 8 | Higher quality signal (same as DPO training) |

**Expected result:** K=1 has poor routing quality (only entropy useful); K=5 hits diminishing returns.

**Retrain:** Requires re-running `scripts/generate_router_features.py --num-candidates K` then router retrain.

---

## Tier 2 — Important (Preempt Reviewer Questions)

### F. Calibration Ablations

*Why required:* Calibration is claimed in Section 7. These verify it matters.

| Ablation key | Override | Tests |
|---|---|---|
| `no_brier_loss` | `training.router.brier_weight=0.0` | Remove Brier calibration loss from router objective |
| `no_temp_scaling` | `training.router.temperature_scaling=false` | No post-hoc temperature scaling (fixed T=1.0) |

**Metric to report:** ECE (Expected Calibration Error) and Brier score before/after each removal.

**Retrain:** Router only.

---

### G. Heuristic vs Learned Router

*Why required:* "Does the router even need to be trained, or do hand-crafted rules suffice?" Every reviewer will ask this.

| Method | Type | LLM call rate | SR | CVaR-Fail |
|---|---|---|---|---|
| `heuristic_router` | Rule-based | | | |
| `entropy_router` | Single-feature threshold | | | |
| `r2v` | Learned CVaR-MLP | | | |

**Implementation:** Already in `evaluate.py`. No retraining needed.

---

### H. Held-Out Perturbation Type (Generalization)

*Why required:* "Does your router overfit to the training perturbation types?" — natural reviewer question given the perturbation one-hot features. Inspired by AgentBench (Liu et al. ICLR 2024) OOD generalization ablations.

| Ablation key | Held-out type | Test question |
|---|---|---|
| `generalize_no_flakiness` | tool_flakiness | Does router generalize to unseen tool noise? |
| `generalize_no_injection` | prompt_injection | Adversarial generalization? |
| `generalize_no_partial_obs` | partial_observability | Observation-noise generalization? |
| `generalize_no_distractors` | distractors | Distractor generalization? |

**Interesting variant:** Combine with `feat_no_perturbation_type` — train without perturbation one-hots AND without one type. If performance is preserved, the router generalizes without type supervision.

**Retrain:** Filter `RouterDataset` to exclude held-out type, retrain router.

---

### I. Policy Training Ablations

*Why required:* DPO β sensitivity is expected for any DPO-based paper (Rafailov et al. NeurIPS 2023). λ_cons validates RQ4.

**DPO β sweep:**

| Ablation key | Override | β |
|---|---|---|
| `dpo_beta_0.05` | `training.preference.beta=0.05` | Weak preference signal |
| *(default)* | — | **0.1** |
| `dpo_beta_0.2` | `training.preference.beta=0.2` | |
| `dpo_beta_0.5` | `training.preference.beta=0.5` | Strong preference signal |

**Consistency λ sweep:**

| Ablation key | Override | λ_cons |
|---|---|---|
| `no_consistency` | `training.consistency.enabled=false` | 0.0 (existing) |
| `consistency_lambda_0.05` | `training.consistency.lambda=0.05` | 0.05 |
| *(default)* | — | **0.1** |
| `consistency_lambda_0.5` | `training.consistency.lambda=0.5` | 0.5 |

**BC warmup ablation:**

| Ablation key | Override | Tests |
|---|---|---|
| `no_bc_warmup` | `training.skip_bc=true` | DPO from Llama-3.1-8B-Instruct directly (no BC stage) |

**Retrain:** Full policy retraining (GPU hours). Coordinate with compute availability.

---

### J. No BC Warmup

*Why required:* Validates the 2-stage BC → DPO pipeline. Expected by RLHF paper reviewers.

| System | BC stage | DPO stage | Expected SR |
|---|---|---|---|
| Full R2V | Yes | Yes | Best |
| `no_bc_warmup` | No | Yes | Worse (DPO from cold init) |
| `no_preference` | Yes | No | Worse (no preference signal) |

---

## Tier 3 — Nice-to-Have (Adds Depth)

### K. Cost Ratio Sensitivity

*Tests robustness of the Pareto frontier to cost assumptions.*

| Ablation key | Override | c_LLM | c_SLM | Ratio |
|---|---|---|---|---|
| `cost_ratio_10` | `training.router.cost_llm=10.0` | 10 | 1 | 10× |
| `cost_ratio_25` | `training.router.cost_llm=25.0` | 25 | 1 | 25× |
| *(default)* | — | **50** | **1** | **50×** |
| `cost_ratio_100` | `training.router.cost_llm=100.0` | 100 | 1 | 100× |

**Expected result:** Higher ratio → router more conservative (fewer LLM calls). Plot LLM% vs SR for each ratio.

**Retrain:** Router only (~25s each).

---

### L. Router Architecture

*Standard neural architecture ablation.*

| Ablation key | Override | Hidden dims | Params |
|---|---|---|---|
| `router_shallow` | `router.hidden_dims=[128]` | [128] | ~8K |
| *(default)* | — | **[128, 64]** | **~10K** |
| `router_deep` | `router.hidden_dims=[256,128,64]` | [256, 128, 64] | ~50K |

**Expected result:** Default is near-optimal; deep has marginal gain if any, justifying the lightweight design.

**Retrain:** Router only.

---

## SLM Policy Training Ablations

These are missing from the original plan and fill the gap between "router ablations" and "end-to-end system ablations." Each targets a specific design choice in the 3-stage SLM training pipeline (BC → DPO + Consistency).

**Coverage audit of what was already handled:**

| Already covered | Gap |
|---|---|
| `no_preference` (BC-only) | LoRA rank not ablated |
| `no_consistency` | Consistency temperature T=2.0 unjustified |
| `no_bc_warmup` | JSD vs KL choice not validated |
| DPO β sweep | BC noisy:clean data ratio not ablated |
| λ_cons sweep | DPO reference policy choice not tested |

---

### M. LoRA Rank Sensitivity

*Why required:* r=64 is stated but never justified. Standard expectation for any LoRA paper (Hu et al. ICLR 2022). Reviewers will immediately ask.

| Ablation key | Override | r | α | Scale (α/r) |
|---|---|---|---|---|
| `lora_r16` | `training.bc.lora_r=16` | 16 | 32 | 2.0 |
| `lora_r32` | `training.bc.lora_r=32` | 32 | 64 | 2.0 |
| *(default)* | — | **64** | **128** | **2.0** |
| `lora_r128` | `training.bc.lora_r=128` | 128 | 256 | 2.0 |

**Expected result:** Performance plateaus at r=64; r=128 does not improve and doubles adapter memory. This justifies the chosen rank.

**Retrain:** Full policy (expensive). Can share BC checkpoint and only retrain DPO stage for a cheaper estimate.

---

### N. BC Data Noisy:Clean Ratio

*Why required:* CLAUDE.md specifies "2:1 noisy/clean" for BC training as a design choice — but it is never ablated. Reviewers will ask why 2:1 specifically.

The hypothesis: mixing noisy trajectories into BC pretraining teaches the policy to handle corrupted observations before DPO fine-tuning.

| Ablation key | Ratio (noisy:clean) | Tests |
|---|---|---|
| `bc_clean_only` | 0:1 | Is noisy BC data necessary at all? |
| `bc_ratio_1_1` | 1:1 | Equal mix |
| *(default)* | **2:1** | **Current choice** |
| `bc_ratio_3_1` | 3:1 | More noise during BC |
| `bc_noisy_only` | 1:0 | Only noisy trajectories in BC |

**Key metric:** Evaluate on the worst-seed and CVaR metrics (not just average SR), since the ratio affects robustness not just mean performance.

**Retrain:** BC stage + all downstream stages. Plan this as a multi-seed experiment to reduce variance.

---

### O. Consistency Regularization Design

*Why required:* Three design choices in consistency regularization are unjustified: (1) symmetric JSD vs one-directional KL, (2) temperature T=2.0, (3) N=5 last token positions.

**JSD vs one-directional KL:**

| Ablation key | Override | Divergence | Symmetric? |
|---|---|---|---|
| `cons_forward_kl` | `training.consistency.divergence=forward_kl` | KL(P ‖ Q) | No |
| `cons_reverse_kl` | `training.consistency.divergence=reverse_kl` | KL(Q ‖ P) | No |
| *(default)* | — | **JSD** | **Yes** |

**Expected result:** JSD ≥ one-directional KL on robustness metrics because perturbation pairs are unordered (no canonical "clean" vs "noisy" direction). This validates the symmetric choice.

**Consistency temperature T:**

| Ablation key | Override | T | Effect |
|---|---|---|---|
| `cons_temp_1` | `training.consistency.temperature=1.0` | 1.0 | No smoothing; sharp distributions |
| `cons_temp_2` | *(default)* | **2.0** | **Smoothed; encourages invariance** |
| `cons_temp_4` | `training.consistency.temperature=4.0` | 4.0 | Very smooth; loses discriminability |

**Number of token positions N:**

| Ablation key | Override | N | Tests |
|---|---|---|---|
| `cons_n1` | `training.consistency.num_positions=1` | 1 | Only final action token |
| `cons_n3` | `training.consistency.num_positions=3` | 3 | |
| *(default)* | — | **5** | **Default** |
| `cons_n10` | `training.consistency.num_positions=10` | 10 | Broader distribution matching |

**Retrain:** DPO + consistency stage only (no BC retraining needed).

---

### P. DPO Reference Policy

*Why required:* The choice to use the BC-trained policy as the DPO reference (rather than the raw Llama base) is a deliberate design choice. Using the raw base as reference is the more common setup in the DPO literature (Rafailov et al. NeurIPS 2023).

| Ablation key | Reference model | Tests |
|---|---|---|
| `dpo_ref_base_llm` | Raw `Llama-3.1-8B-Instruct` | Standard DPO setup; does BC pretraining as reference matter? |
| *(default)* | **BC-trained policy** | **Prevents drift from BC checkpoint** |

**Hypothesis:** BC-trained reference leads to smaller KL divergence during DPO, allowing more precise preference alignment without forgetting BC knowledge.

**Retrain:** DPO stage only (BC checkpoint shared).

---

### Q. DPO Preference Pair Selection

*Why required:* Choosing best vs worst verifier-scored candidate is a heuristic. Reviewers may ask whether a margin threshold (only keeping pairs with large score gaps) improves quality.

| Ablation key | Override | Selection strategy |
|---|---|---|
| *(default)* | — | Best vs worst (no margin filter) |
| `dpo_margin_0.2` | `training.preference.min_margin=0.2` | Only pairs where best_score − worst_score > 0.2 |
| `dpo_margin_0.4` | `training.preference.min_margin=0.4` | Stricter margin filter |
| `dpo_random_rejected` | `training.preference.rejected=random` | Best vs random (not worst) rejected action |

**Expected result:** Margin filtering improves DPO data quality but reduces dataset size. There is a sweet spot; very strict margins hurt coverage.

**Retrain:** DPO stage only.

---

### Suggested SLM Training Table (Table 5)

| System | SR ↑ | Worst-Seed SR ↑ | CVaR-Fail ↓ |
|---|---|---|---|
| **Full R2V (BC→DPO+Cons)** | | | |
| BC-only (no DPO) | | | |
| DPO from scratch (no BC) | | | |
| DPO with base LLM reference | | | |
| BC clean-only (no noisy data) | | | |
| BC noisy-only | | | |
| No consistency regularization | | | |
| Consistency with forward KL | | | |
| LoRA r=16 | | | |
| LoRA r=128 | | | |

---

## Suggested Paper Tables

### Table 2 — Main Ablation Table

| Method | SR ↑ | Worst-Seed SR ↑ | CVaR-Fail ↓ | LLM% ↓ | ECE ↓ |
|---|---|---|---|---|---|
| **R2V-Agent (full)** | | | | | |
| − CVaR → EV router | | | | | |
| − CVaR → worst-case loss | | | | | |
| − Verifier (random scores) | | | | | |
| − DPO (BC-only) | | | | | |
| − Consistency reg | | | | | |
| − Brier calibration loss | | | | | |
| − Temperature scaling | | | | | |
| Static entropy threshold | | | | | |
| Heuristic router | | | | | |
| Oracle router (upper bound) | | | | | |

### Table 3 — Feature Ablation Table

| Features used | SR ↑ | CVaR-Fail ↓ | LLM% |
|---|---|---|---|
| All 13 features | | | |
| Verifier + entropy (5 features) | | | |
| No perturbation one-hot (8 features) | | | |
| No horizon/step features (11 features) | | | |
| Entropy only (1 feature) | | | |
| No verifier features (9 features) | | | |
| Best verifier score only | | | |

### Table 4 — Held-Out Perturbation Generalization

| Training types | Test type | SR ↑ | CVaR-Fail ↓ |
|---|---|---|---|
| All 4 types | tool_flakiness | | |
| Excl. flakiness | tool_flakiness | | |
| Excl. injection | prompt_injection | | |
| Excl. partial_obs | partial_observability | | |
| Excl. distractors | distractors | | |

### Figure 1 — CVaR α and ε Sensitivity

Line plots: x-axis = α (or ε), y-axis = {SR, CVaR-Fail, LLM%}. Shows the chosen defaults are near-optimal.

### Figure 2 — Cost-Accuracy Pareto Frontier

x-axis = LLM call rate (%), y-axis = success rate.
- R2V threshold sweep (from `--router-threshold-sweep 0.2 0.3 0.4 0.5 0.6 0.7`)
- SLM-only (leftmost point)
- LLM-only (rightmost point)
- Oracle router (upper boundary)
- Heuristic router (single point)
- Entropy router sweep

### Figure 3 — K-Candidates vs Performance

x-axis = K, y-axis = {SR, CVaR-Fail}. Shows signal quality saturates at K≈5.

---

## Running the Ablations

### Step 1 — Baselines and Inference-Time Ablations (no retraining)

```bash
# Full evaluation with all baselines
python scripts/evaluate.py \
    --config configs/humaneval/noisy.yaml \
    --features data/router_features/humaneval.jsonl \
    --trajectories data/trajectories/humaneval_noisy/trajectories.jsonl \
    --router-path outputs/router/humaneval_noisy/router_final.pt \
    --output results/humaneval_noisy \
    --methods r2v slm_only llm_only entropy_router oracle_router heuristic_router verifier_router \
    --router-threshold-sweep 0.2 0.3 0.4 0.5 0.6 0.7

# Inference-time feature ablations (fast, no retraining)
for MASK in "0 1 2 3 4" "0 1 2 3 4 5 6 7" "0" "0 5 6 7 8 9 10 11 12"; do
    python scripts/evaluate.py \
        --config configs/humaneval/noisy.yaml \
        --features data/router_features/humaneval.jsonl \
        --router-path outputs/router/humaneval_noisy/router_final.pt \
        --output results/humaneval_noisy/feat_ablation \
        --methods r2v \
        --feature-mask $MASK
done
```

### Step 2 — Router-Only Retrains (~25s each, CPU)

```bash
# CVaR alpha sweep
for ALPHA in 0.1 0.3 0.5; do
    python scripts/train_router.py \
        --config configs/humaneval/noisy.yaml \
        --features data/router_features/humaneval.jsonl \
        --output outputs/router/ablations/cvar_alpha_${ALPHA} \
        --overrides training.router.cvar_alpha=${ALPHA}
done

# CVaR epsilon sweep
for EPS in 0.1 0.2 0.4; do
    python scripts/train_router.py \
        --config configs/humaneval/noisy.yaml \
        --features data/router_features/humaneval.jsonl \
        --output outputs/router/ablations/cvar_eps_${EPS} \
        --overrides training.router.cvar_epsilon=${EPS}
done

# Worst-case loss
python scripts/train_router.py \
    --config configs/humaneval/noisy.yaml \
    --features data/router_features/humaneval.jsonl \
    --output outputs/router/ablations/worst_case \
    --overrides training.router.robust_objective=worst_case

# No Brier loss
python scripts/train_router.py \
    --config configs/humaneval/noisy.yaml \
    --features data/router_features/humaneval.jsonl \
    --output outputs/router/ablations/no_brier \
    --overrides training.router.brier_weight=0.0

# No temperature scaling
python scripts/train_router.py \
    --config configs/humaneval/noisy.yaml \
    --features data/router_features/humaneval.jsonl \
    --output outputs/router/ablations/no_temp_scaling \
    --overrides training.router.temperature_scaling=false

# Cost ratio sweep
for COST in 10 25 100; do
    python scripts/train_router.py \
        --config configs/humaneval/noisy.yaml \
        --features data/router_features/humaneval.jsonl \
        --output outputs/router/ablations/cost_${COST} \
        --overrides training.router.cost_llm=${COST}
done
```

### Step 3 — Held-Out Perturbation Generalization (router retrain)

```bash
for HELDOUT in tool_flakiness prompt_injection partial_observability distractors; do
    python scripts/train_router.py \
        --config configs/humaneval/noisy.yaml \
        --features data/router_features/humaneval.jsonl \
        --output outputs/router/ablations/generalize_no_${HELDOUT} \
        --overrides router.held_out_perturbation=${HELDOUT}
done
```

### Step 4 — Policy Retrains (GPU hours)

```bash
# DPO beta sweep (requires full pipeline from stage 6)
for BETA in 0.05 0.2 0.5; do
    bash run_pipeline.sh --from 6 \
        --overrides training.preference.beta=${BETA}
done

# Consistency lambda sweep
for LAM in 0.05 0.5; do
    bash run_pipeline.sh --from 6 \
        --overrides training.consistency.lambda=${LAM}
done

# No BC warmup
bash run_pipeline.sh --from 3b \
    --overrides training.skip_bc=true
```

---

## Reference Papers

| Paper | Venue | Relevance |
|---|---|---|
| Rafailov et al., "Direct Preference Optimization" | NeurIPS 2023 | DPO policy training; β sensitivity expected |
| Ong et al., "RouteLLM" | NeurIPS 2024 | LLM routing; main comparison point |
| Chen et al., "FrugalGPT" | ICML 2023 | Model cascading; cost-accuracy Pareto |
| Chen et al., "EcoAssistant" | ICLR 2024 | Oracle router concept |
| Chow et al., "Risk-Constrained RL" | ICML 2017 | CVaR formulation |
| Rockafellar & Uryasev | Math. Finance 2000 | CVaR foundation |
| Hu et al., "RouterBench" | 2024 | Feature importance for routing |
| Liu et al., "AgentBench" | ICLR 2024 | OOD generalization ablations |
