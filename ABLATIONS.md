# R2V-Agent Ablation Studies

Ablation plan for NeurIPS submission. Organized by priority tier based on which claims each ablation directly validates.

---

## Paper-Based Validation Audit

Cross-check of every design choice against published NeurIPS/ICLR/ICML literature. **Bold = change or addition needed.**

| Design choice | Current setting | Paper evidence | Verdict |
|---|---|---|---|
| CVaR vs worst-case vs EV | CVaR α=0.2 | Chow et al. ICML 2017; IJCAI 2022 "worst-case leads to overly pessimistic policies" | ✓ Correct; worst-case ablation important |
| Lagrange dual LR | 1e-2 (fixed) | Empirical study of Lagrangian methods (arXiv 2510.17564): "effectiveness depends crucially on λ choice" | **Add dual LR sweep** |
| Temperature scaling | post-hoc, scipy | Guo et al. ICML 2017: temperature scaling outperforms all other calibration methods | ✓ Correct design; `no_temp_scaling` ablation confirmed |
| DPO β | 0.1 (fixed) | β-DPO (NeurIPS 2024): β sensitivity documented; rDPO (2024): smaller β → more robust to noisy preferences | ✓ β sweep confirmed; **extend range down to 0.01** |
| DPO divergence (policy constraint) | Reverse KL (default DPO) | f-DPO Beyond Reverse KL (ICLR 2024): JSD and forward KL perform differently per task | **Add f-DPO divergence ablation** |
| DPO preference pair selection | Best vs worst, no margin filter | "Less is More" (NeurIPS 2025): 10% margin-filtered subset beats full dataset by 3–8pp | ✓ Margin filter ablation confirmed; **high priority** |
| BC before DPO | Yes (2-stage) | InstructGPT; general RLHF best practice: SFT warmup stabilizes preference training | ✓ `no_bc_warmup` ablation confirmed |
| BC noisy:clean ratio | 2:1 | No direct paper. PPCL (EACL 2024): consistency training on perturbed data helps | **Keep; add cleaner justification** |
| Consistency divergence | JSD | f-DPO (ICLR 2024) tests JSD in policy space; PPCL (EACL 2024) uses loss-level regularization | ✓ JSD ablation confirmed; **add loss-level regularization variant** |
| Consistency temperature | T=2.0 | No direct paper evidence for T=2.0 choice | **Keep ablation; T=1.0 is the natural baseline** |
| Consistency N positions | 5 | No paper evidence | Keep ablation |
| Verifier step vs outcome weight | w_step=0.3, w_final=1.0 | Lightman et al. "Let's Verify Step by Step" (ICLR 2024): process supervision significantly outperforms outcome supervision | **Add w_step sweep; current 0.3 may be too low** |
| Verifier type | LLM-judge → distilled | PRM literature (ICLR 2024/2025): trained PRMs better than outcome-only | ✓ No-verifier ablation confirmed |
| Self-correction iterations | max=2 | SCoRe (ICLR 2025): iteration count and RL training both matter for self-correction | **Add iteration count ablation (0, 1, 2, 3)** |
| Router under noisy verifier | Not tested | RouterBench (2024): cascading degrades rapidly when scorer error > 0.2 | **Add verifier error rate ablation** |
| LoRA rank | r=64 | LoRA paper (ICLR 2022): r=1 or 2 often suffices; r=64 likely high for 8B model | ✓ Rank sweep confirmed; **extend down to r=4, r=8** |
| Training data for router | Features from data | RouteLLM (2024): small golden-label augmentation (2% of data) dramatically improves routers | **Add router data augmentation ablation** |

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

**Paper evidence on β:** β-DPO (NeurIPS 2024) shows β should adapt to preference data quality. The rDPO paper (2024) proves that smaller β → more robust to noisy labels. Since R2V uses verifier-scored preferences under perturbations, lower β may better handle verifier noise. Extend range down to 0.01.

**DPO β sweep:**

| Ablation key | Override | β | Expected behavior |
|---|---|---|---|
| `dpo_beta_0.01` | `training.preference.beta=0.01` | 0.01 | Near-unconstrained; maximally robust to noisy verifier scores |
| `dpo_beta_0.05` | `training.preference.beta=0.05` | 0.05 | Weak KL constraint |
| *(default)* | — | **0.1** | **Default** |
| `dpo_beta_0.2` | `training.preference.beta=0.2` | 0.2 | Stronger KL constraint |
| `dpo_beta_0.5` | `training.preference.beta=0.5` | 0.5 | Strong; may over-constrain under noisy preferences |

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

**Paper evidence:** The original LoRA paper finds r=1 or r=2 often suffices even for models with d=12,288, because weight updates have low intrinsic rank. For an 8B model with instruction-following already baked in, r=64 is likely unnecessarily large. The ablation may reveal that r=16 or r=32 matches r=64, which strengthens the efficiency narrative.

| Ablation key | Override | r | α | Scale (α/r) |
|---|---|---|---|---|
| `lora_r4` | `training.bc.lora_r=4` | 4 | 8 | 2.0 |
| `lora_r8` | `training.bc.lora_r=8` | 8 | 16 | 2.0 |
| `lora_r16` | `training.bc.lora_r=16` | 16 | 32 | 2.0 |
| `lora_r32` | `training.bc.lora_r=32` | 32 | 64 | 2.0 |
| *(default)* | — | **64** | **128** | **2.0** |
| `lora_r128` | `training.bc.lora_r=128` | 128 | 256 | 2.0 |

**Expected result:** Performance plateau between r=16–64; r=4/r=8 may still be competitive given that Llama-3.1-8B-Instruct already has strong priors. If r=16 matches r=64, report that to strengthen the lightweight-adapter narrative.

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

*Why required:* Three design choices in consistency regularization are unjustified: (1) symmetric JSD vs one-directional KL, (2) temperature T=2.0, (3) N=5 last token positions. Also, a loss-level variant used in PPCL (EACL 2024) is a legitimate alternative to distribution-level JSD.

**Paper evidence on divergence choice:** f-DPO "Beyond Reverse KL" (ICLR 2024) systematically compares reverse KL, forward KL, JSD, and α-divergence as divergence constraints for preference alignment, finding task-dependent differences. PPCL (EACL 2024) uses loss-level regularization (divergence between CE losses on clean vs perturbed inputs) rather than distribution-level JSD, and recovers 59–69% of performance drop with 10× fewer augmented samples than data augmentation alone.

**JSD vs one-directional KL vs loss-level:**

| Ablation key | Override | Divergence | Level | Symmetric? |
|---|---|---|---|---|
| `cons_forward_kl` | `training.consistency.divergence=forward_kl` | KL(P ‖ Q) | Distribution | No |
| `cons_reverse_kl` | `training.consistency.divergence=reverse_kl` | KL(Q ‖ P) | Distribution | No |
| *(default)* | — | **JSD** | **Distribution** | **Yes** |
| `cons_loss_level` | `training.consistency.divergence=loss_level` | |CE(z) − CE(z')| | Loss | Yes |

**Expected result:** JSD ≥ one-directional KL because perturbation pairs are unordered (no canonical "clean" vs "noisy" direction). Loss-level regularization may be a useful ablation since it's cheaper to compute (no vocab-size softmax) but coarser.

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
| Consistency with loss-level regularization | | | |
| LoRA r=8 | | | |
| LoRA r=16 | | | |
| LoRA r=128 | | | |

---

## Non-Router Ablation Evaluations

Policy and verifier ablations (sections I–Q and S) require **two evaluation stages** each, not one. A single end-to-end SR number hides whether a change helped because the policy is better or because the router compensated. Both stages are needed.

---

### Stage A — Component-Level Metrics (Intrinsic Quality)

Run immediately after retraining the ablated component. These do not require routing and run fast.

#### Policy Quality Metrics (after stages 3b or 6)

| Metric | Definition | Direction | When to use |
|---|---|---|---|
| BC held-out perplexity | Cross-entropy loss on clean val trajectories | ↓ | After any BC ablation (N, M, J) |
| DPO preference accuracy | % of val pairs where `log π(a+) > log π(a-)` | ↑ | After DPO ablations (I, P, Q, V) |
| Reward margin | Mean `log π(a+|x) − log π(a−|x)` on val pairs | ↑ | After DPO β sweep, margin filter |
| Consistency gap | Mean JSD between clean/noisy token distributions on val set | ↓ | After consistency ablations (O) |
| KL from reference | Mean `KL(π_θ ‖ π_ref)` on val trajectories | — | After DPO β sweep (high β → high KL) |

**What these tell you:**
- If preference accuracy is low under a β change, the policy is over/under-constrained — SR drop is expected.
- If BC perplexity is unchanged across LoRA ranks (M), the rank ablation is informative (r=16 ≈ r=64).
- If consistency gap does not decrease when consistency is enabled, the regularizer is not taking effect — check λ and temperature settings.

**Where to find these:** The training logs (`outputs/policy/{benchmark}_noisy/final/training_log.jsonl`) should already record per-epoch DPO loss and preference accuracy. BC perplexity is the final validation loss from `bc_trainer.py`. If not logged, add a post-training evaluation pass.

**Quick evaluation command (DPO preference accuracy on val set):**
```bash
python scripts/train_policy.py --stage preference \
    --config configs/humaneval/noisy.yaml \
    --eval-only \
    --checkpoint outputs/policy/ablations/{ABLATION_KEY}/final \
    --output results/ablations/{ABLATION_KEY}/policy_metrics.json
```

---

#### Verifier Quality Metrics (after stage 4)

The verifier trainer (`verifier_trainer.py`) already tracks these — confirm they are written to the output JSONL.

| Metric | Definition | Direction | Notes |
|---|---|---|---|
| PR-AUC | Area under precision-recall curve | ↑ | **Primary metric** (per verifier_trainer.py). Random baseline = 0.22 at 78% success prevalence |
| AUROC | Area under ROC curve | ↑ | Secondary; less sensitive to class imbalance than PR-AUC |
| Balanced accuracy | `(TPR + TNR) / 2` at optimal threshold | ↑ | Used for threshold selection in training |
| Step accuracy | BCE accuracy on step-level labels only | ↑ | Needed for ablation S (w_step sweep) |
| Outcome accuracy | BCE accuracy on final-outcome labels only | ↑ | Needed for ablation S |
| ECE | Expected calibration error of verifier probability outputs | ↓ | Verifier miscalibration propagates into router features |
| Brier score | MSE of verifier probabilities | ↓ | Composite calibration+refinement |

**Critical for ablation S (w_step sweep):** Report step accuracy and outcome accuracy *separately* for each w_step value. If process-heavy training (w_step=3.0) improves step accuracy but hurts outcome accuracy, this explains any mixed downstream result.

**Command:**
```bash
python scripts/train_verifier.py \
    --config configs/humaneval/noisy.yaml \
    --output outputs/verifier/ablations/{ABLATION_KEY} \
    --trajectories data/trajectories/humaneval_noisy/trajectories.jsonl \
    --overrides {OVERRIDE}
# Results written to outputs/verifier/ablations/{ABLATION_KEY}/training_log.jsonl
```

**Table 6 — Verifier Ablation Metrics:**

| System | PR-AUC ↑ | AUROC ↑ | Bal. Acc ↑ | Step Acc ↑ | Outcome Acc ↑ | ECE ↓ | SR ↑ | CVaR-Fail ↓ |
|---|---|---|---|---|---|---|---|---|
| **Default (w_step=0.3, w_final=1.0)** | | | | | | | | |
| Outcome-only (ORM, w_step=0.0) | | | | | | | | |
| Equal weight (w_step=1.0) | | | | | | | | |
| Process-heavy (w_step=3.0) | | | | | | | | |
| Process-only (PRM, w_final=0.0) | | | | | | | | |
| LLM judge verifier | | | | | | | | |
| Heuristic verifier | | | | | | | | |

*The SR and CVaR-Fail columns come from Stage B (see below). The verifier metrics explain why SR changes.*

---

### Stage B — Downstream Evaluation (End-to-End)

After any policy or verifier ablation, you must re-run stages 7→8→9 because router features depend on verifier scores, which in turn depend on the policy's candidate actions.

**Dependency chain:**

```
Policy change (stages 3b/6)
    → New candidate actions (stage 5: generate_candidates.py)
    → New router features (stage 7: generate_router_features.py)
    → Router retrain (stage 8: train_router.py)
    → Evaluate (stage 9: evaluate.py)

Verifier change (stage 4)
    → New router features (stage 7: generate_router_features.py)  ← same verifier scores new
    → Router retrain (stage 8)
    → Evaluate (stage 9)
```

**For policy ablations (I, J, M, N, O, P, Q, V) — resume from stage 5:**
```bash
bash run_pipeline.sh --from 5 \
    --overrides {ABLATION_OVERRIDE}
```

**For verifier ablations (S, U) — resume from stage 7:**
```bash
bash run_pipeline.sh --from 7 \
    --overrides {ABLATION_OVERRIDE}
```

**Metrics to collect from evaluate.py (same as Table 1):**

| Metric | Symbol | Direction |
|---|---|---|
| Mean success rate | SR ↑ | Higher |
| Worst-seed SR | Worst-SR ↑ | Higher |
| CVaR failure rate (α=0.2) | CVaR-Fail ↓ | Lower |
| LLM escalation rate | LLM% ↓ | Lower |

**Expanded Table 5 — SLM Training Ablations:**

| System | BC Perplexity ↓ | DPO Pref Acc ↑ | Cons Gap ↓ | SR ↑ | Worst-SR ↑ | CVaR-Fail ↓ | LLM% |
|---|---|---|---|---|---|---|---|
| **Full R2V (BC→DPO+Cons)** | | | | | | | |
| BC-only (no DPO) | | N/A | | | | | |
| DPO from scratch (no BC) | | | | | | | |
| DPO with base LLM reference | | | | | | | |
| BC clean-only | | | | | | | |
| BC noisy-only | | | | | | | |
| No consistency reg | | | N/A | | | | |
| Consistency: forward KL | | | | | | | |
| Consistency: loss-level | | | | | | | |
| Consistency T=1.0 | | | | | | | |
| Consistency T=4.0 | | | | | | | |
| DPO β=0.01 | | | | | | | |
| DPO β=0.5 | | | | | | | |
| DPO margin filter 0.2 | | | | | | | |
| LoRA r=8 | | | | | | | |
| LoRA r=16 | | | | | | | |
| LoRA r=128 | | | | | | | |

*Stage A metrics (columns 2–4) explain mechanism; Stage B metrics (columns 5–8) measure impact.*

---

### Self-Correction Analysis (Ablation T)

Self-correction is inference-time only (no retraining), but needs iteration-level tracking that `evaluate.py` may not currently produce.

**Metrics to collect per iteration count:**

| Metric | Definition | Notes |
|---|---|---|
| SR per iteration | SR after 0, 1, 2, 3 self-correction iterations | Main result for ablation T |
| Verifier score Δ | Mean change in verifier score after each correction | Confirms verifier is gating corrections |
| Correction acceptance rate | % of steps where self-correction changes the action | High rate + low SR gain = verifier not gating |
| LLM% with self-correction | Does self-correction change escalation rate? | Should not, since correction uses SLM |

**Table format for ablation T:**

| Max iterations | SR ↑ | CVaR-Fail ↓ | Mean verifier Δ | Acceptance rate |
|---|---|---|---|---|
| 0 (no correction) | | | — | 0% |
| 1 | | | | |
| **2 (default)** | | | | |
| 3 | | | | |

**Warning to check (per Huang et al. ICLR 2024):** If SR at iteration 2 < SR at iteration 1, the self-correction loop is hurting. This happens when the verifier does not gate corrections (it accepts all corrections regardless of verifier score). Confirm that `inference.max_self_correct > 0` only triggers a correction when the verifier score *increases* after the corrected action.

---

### Summary: Which Metrics to Report Where

| Ablation group | Stage A metric | Stage B metric | Table |
|---|---|---|---|
| Policy: BC ratio (N) | BC perplexity | SR, Worst-SR, CVaR-Fail | Table 5 |
| Policy: LoRA rank (M) | BC perplexity | SR, CVaR-Fail | Table 5 |
| Policy: DPO β (I) | DPO pref acc, reward margin | SR, CVaR-Fail | Table 5 |
| Policy: DPO reference (P) | DPO pref acc, KL from ref | SR, CVaR-Fail | Table 5 |
| Policy: DPO margin (Q) | Dataset size after filter, pref acc | SR, CVaR-Fail | Table 5 |
| Policy: Consistency design (O) | Consistency gap (JSD) | SR, CVaR-Fail | Table 5 |
| Policy: DPO divergence (V) | Reward margin | SR, CVaR-Fail | Table 5 |
| Verifier: w_step sweep (S) | PR-AUC, step acc, outcome acc, ECE | SR, CVaR-Fail | Table 6 |
| Verifier: noisy verifier (U) | PR-AUC, ECE | SR, CVaR-Fail, LLM% | Table 6 |
| Self-correction (T) | Verifier Δ per iter, acceptance rate | SR per iteration | Inline table |

---

## Paper-Validated New Ablations

These are additions directly motivated by recent literature. None were in the original plan.

---

### R. Lagrange Dual LR Sensitivity

*Paper:* "An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning" (arXiv 2510.17564): "the effectiveness of Lagrangian methods depends crucially on the choice of the Lagrange multiplier λ." Currently R2V uses a fixed dual LR of 1e-2 with no justification.

| Ablation key | Override | Dual LR | Expected behavior |
|---|---|---|---|
| `dual_lr_1e3` | `training.router.lagrangian_lr=0.001` | 1e-3 | Slow λ adaptation; constraint violations persist longer |
| `dual_lr_5e3` | `training.router.lagrangian_lr=0.005` | 5e-3 | |
| *(default)* | — | **1e-2** | **Default** |
| `dual_lr_5e2` | `training.router.lagrangian_lr=0.05` | 5e-2 | Fast λ adaptation; potential oscillation |
| `dual_lr_1e1` | `training.router.lagrangian_lr=0.1` | 1e-1 | Unstable; λ overshoots |

**What to report:** λ trajectory across training epochs for each LR. The paper recommends reporting constraint violation over time, not just final SR.

**Retrain:** Router only (~25s each). Run 5 values.

---

### S. Verifier Step vs Outcome Supervision Weight

*Paper:* Lightman et al. "Let's Verify Step by Step" (ICLR 2024): process (step-level) supervision significantly outperforms outcome supervision. R2V currently uses w_step=0.3, w_final=1.0, which heavily down-weights process supervision — the opposite direction from what the paper recommends.

| Ablation key | Override | w_step | w_final | Ratio |
|---|---|---|---|---|
| `verif_outcome_only` | `training.verifier.w_step=0.0` | 0.0 | 1.0 | Outcome-only (ORM) |
| *(default)* | — | **0.3** | **1.0** | **Process lightly weighted** |
| `verif_equal` | `training.verifier.w_step=1.0` | 1.0 | 1.0 | Equal weight |
| `verif_process_heavy` | `training.verifier.w_step=3.0` | 3.0 | 1.0 | Process-heavy (PRM) |
| `verif_process_only` | `training.verifier.w_final=0.0` | 1.0 | 0.0 | Process-only (pure PRM) |

**Expected result per paper:** w_step > w_final (process-heavy) should outperform outcome-only. If the current 0.3:1.0 setting is suboptimal, this is both an ablation and a performance improvement.

**Retrain:** Verifier only, then router (router features depend on verifier quality).

---

### T. Self-Correction Iteration Count

*Paper:* SCoRe (ICLR 2025): self-correction requires proper multi-turn RL training. Crucially, Stage I (single-turn warmup) prevents collapse when self-correction is added. Removing Stage I causes 2–3% performance drop. Also, intrinsic self-correction without external feedback (Huang et al., ICLR 2024) often *hurts* performance — suggesting the R2V self-correction loop must be carefully validated.

| Ablation key | Override | Max iterations | Tests |
|---|---|---|---|
| `no_self_correction` | `inference.max_self_correct=0` | 0 | No self-correction (existing ablation) |
| `self_correct_1` | `inference.max_self_correct=1` | 1 | Single iteration |
| *(default)* | — | **2** | **Default** |
| `self_correct_3` | `inference.max_self_correct=3` | 3 | Three iterations |

**Critical check:** Per Huang et al. ICLR 2024, if the self-correction loop uses only the SLM's own feedback (no verifier), it may degrade performance. Confirm that the verifier signal gates each correction iteration.

**Retrain:** No retraining; inference-time change only.

---

### U. Router under Noisy Verifier

*Paper:* RouterBench (Hu et al. 2024): cascading router performance degrades "rapidly" when scorer error rate exceeds 0.2. R2V's router is trained on verifier features from a 70B LLM judge — but what happens at inference if the verifier is weaker (e.g., after distillation)?

| Ablation key | Verifier type | Expected error rate | Tests |
|---|---|---|---|
| `router_llm_judge` | 70B LLM judge | ~0.05 | Full-quality verifier signal |
| *(default)* | — | ~0.10 | **Distilled 8B verifier** |
| `router_heuristic_verifier` | `HeuristicVerifier` | ~0.20 | Rule-based verifier |
| `router_random_verifier` | Random scores | ~0.50 | No verifier signal (`no_verifier` existing) |

**Implementation:** Swap the verifier used during `generate_router_features.py`, retrain router on features from each verifier. Compare final evaluation metrics.

**Retrain:** Feature regeneration + router retrain for each condition.

---

### V. DPO Policy Divergence Type

*Paper:* f-DPO "Beyond Reverse KL" (ICLR 2024): JSD, forward KL, and α-divergences produce task-dependent differences as the **policy** divergence constraint. This is separate from the consistency regularization JSD (which is a regularizer over observation pairs). The standard DPO uses reverse KL for the policy constraint implicitly. The f-DPO framing lets you swap it.

| Ablation key | Policy divergence | Expected behavior |
|---|---|---|
| *(default)* | Reverse KL (standard DPO) | Mode-seeking; concentrates on preferred actions |
| `dpo_forward_kl` | Forward KL | Mean-seeking; broader preference coverage |
| `dpo_jsd` | JSD | Symmetric; balanced between chosen and rejected |
| `dpo_alpha_div` | α-divergence (α=0.5) | Interpolates between forward/reverse KL |

**Why this matters for R2V:** Under noisy perturbations, mean-seeking forward KL may generalize better across perturbation seeds than mode-seeking reverse KL. This is a testable hypothesis.

**Retrain:** DPO stage only.

---

### W. Router Training Data Augmentation

*Paper:* RouteLLM (2024): augmenting router training data with a small set of golden-label samples (only ~2% of total data) dramatically improves router performance across all architectures. This suggests that adding a small set of "gold" routing decisions (where we know with certainty the SLM fails) could improve R2V's router.

| Ablation key | Training data | Description |
|---|---|---|
| *(default)* | Verifier-derived labels | Labels from verifier-scored features |
| `router_gold_aug_1pct` | + 1% gold labels | Add small set of oracle routing decisions |
| `router_gold_aug_5pct` | + 5% gold labels | |
| `router_gold_aug_10pct` | + 10% gold labels | |

**Gold label construction:** Steps where the SLM's output definitively fails (test suite fails, reward=0 with certainty) are gold negative labels; steps where teacher and SLM agree are gold positive. These require no additional inference — they come from the existing trajectory store.

**Retrain:** Router only.

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
| + Self-correction disabled | | | | | |
| Static entropy threshold | | | | | |
| Heuristic router | | | | | |
| Oracle router (upper bound) | | | | | |

### Table 2b — New Paper-Validated Ablations (Router)

| Method | SR ↑ | Worst-Seed SR ↑ | CVaR-Fail ↓ | LLM% ↓ |
|---|---|---|---|---|
| **R2V full** | | | | |
| Dual LR = 1e-3 (slow λ) | | | | |
| Dual LR = 1e-1 (fast λ) | | | | |
| Router + gold label aug 5% | | | | |
| Router with heuristic verifier | | | | |
| Router with LLM judge verifier | | | | |

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

| Paper | Venue | Ablation relevance |
|---|---|---|
| Rafailov et al., "Direct Preference Optimization" | NeurIPS 2023 | DPO β sensitivity; reverse KL as policy constraint |
| Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data" | 2024 | Router training data augmentation; threshold calibration |
| Chen et al., "FrugalGPT" | TMLR 2024 | Model cascading; cost-accuracy Pareto; oracle upper bound |
| Chen et al., "EcoAssistant" | ICLR 2024 | Oracle router concept |
| Chow et al., "Risk-Constrained RL" | ICML 2017 | CVaR formulation; α sensitivity |
| Rockafellar & Uryasev | Math. Finance 2000 | CVaR theoretical foundation |
| Hu et al., "RouterBench" | 2024 | Verifier error rate → cascading degradation; MLP vs KNN router |
| Liu et al., "AgentBench" | ICLR 2024 | OOD perturbation generalization ablations |
| Hu et al., "LoRA" | ICLR 2022 | Rank sensitivity; r=1–2 often sufficient |
| Wang et al., "Beyond Reverse KL: f-DPO" | ICLR 2024 | DPO divergence ablation (JSD, forward KL, α-div) for policy constraint |
| NeurIPS 2024, "β-DPO: DPO with Dynamic β" | NeurIPS 2024 | β sensitivity; β should adapt to preference data quality |
| arXiv 2403.00409, "Provably Robust DPO" | 2024 | Smaller β → more robust to noisy preference labels |
| Lightman et al., "Let's Verify Step by Step" | ICLR 2024 | Process (step-level) supervision >> outcome supervision; w_step ablation |
| Guo et al., "On Calibration of Modern Neural Networks" | ICML 2017 | Temperature scaling; `no_temp_scaling` ablation |
| arXiv 2510.17564, "Empirical Study of Lagrangian Methods in Safe RL" | 2025 | Dual LR sensitivity; λ selection is critical |
| SCoRe (ICLR 2025) | ICLR 2025 | Self-correction iteration count; warmup stage importance |
| Huang et al., "Large Language Models Cannot Self-Correct Reasoning" | ICLR 2024 | Intrinsic self-correction without verifier may hurt |
| arXiv 2402.15833, "PPCL: Prompt Perturbation Consistency Learning" | EACL 2024 | Loss-level consistency regularization alternative to JSD |
| arXiv 2502.14560, "Less is More: DPO Data Selection" | NeurIPS 2025 | Margin-filtered 10% subset beats full dataset; `dpo_margin_*` ablations |
| arXiv 2602.11348, "AgentNoiseBench" | 2026 | Tool-noise vs user-noise; per-perturbation-type evaluation |

---

## Results Section Evaluations

These evaluations complete the **main results section** of the paper (distinct from ablations). They produce the primary evidence for all four RQs and must appear before any ablation tables. Every number that appears in the paper narrative should come from one of these evaluations.

---

### Table 1 — Main Results (Cross-Benchmark)

*Required for every ML paper.* Shows R2V vs all baselines across all four benchmarks with consistent metrics.

**Metrics to report per cell:**

| Metric | Symbol | Direction | Implemented in |
|---|---|---|---|
| Mean success rate | SR ↑ | Higher | `evaluate.py` |
| Worst-seed success rate | Worst-SR ↑ | Higher | `robustness.py: compute_worst_seed_sr` |
| CVaR failure rate (α=0.2) | CVaR-Fail ↓ | Lower | `robustness.py: compute_cvar_failure` |
| LLM escalation rate | LLM% ↓ | Lower | `evaluate.py` |
| Normalized cost | Cost ↓ | Lower | `LLM% × c_LLM + (1−LLM%) × 1` (use c_LLM=50) |

**Table format:**

| Method | HumanEval SR ↑ | CVaR-Fail ↓ | LLM% ↓ | GAIA SR ↑ | CVaR-Fail ↓ | LLM% ↓ | ALFWorld SR ↑ | CVaR-Fail ↓ | LLM% ↓ |
|---|---|---|---|---|---|---|---|---|---|
| SLM-only | | | — | | | — | | | — |
| LLM-only | | | 100% | | | 100% | | | 100% |
| Entropy router | | | | | | | | | |
| Heuristic router | | | | | | | | | |
| Verifier router | | | | | | | | | |
| Oracle router | | | | | | | | | |
| **R2V (ours)** | | | | | | | | | |

*Report SR as mean ± 95% bootstrap CI across seeds (e.g., `0.72 ± 0.03`).*

**Commands (run for each benchmark):**
```bash
for BENCH in humaneval gaia alfworld; do
    python scripts/evaluate.py \
        --config configs/${BENCH}/noisy.yaml \
        --features data/router_features/${BENCH}.jsonl \
        --trajectories data/trajectories/${BENCH}_noisy/trajectories.jsonl \
        --router-path outputs/router/${BENCH}_noisy/router_final.pt \
        --output results/${BENCH}_noisy \
        --methods r2v slm_only llm_only entropy_router oracle_router heuristic_router verifier_router \
        --seeds 1 2 3 4 5
done
```

---

### Statistical Significance Protocol

*Required by NeurIPS reviewers for any quantitative claim.* Apply to every number in Table 1.

**Bootstrap CI on SR (already implemented in `statistical.py: bootstrap_ci`):**
- Run 1000 bootstrap resamples per method per benchmark
- Report as `mean ± halfwidth` at 95% confidence
- This is already produced by `evaluate.py` — confirm it is included in the structured JSON output

**McNemar's test for pairwise comparisons (implemented in `statistical.py: paired_mcnemar_test`):**

Run McNemar's test for R2V vs each baseline on per-task binary success:

| Comparison | Null hypothesis | Expected result |
|---|---|---|
| R2V vs SLM-only | No difference in per-task SR | p < 0.01 |
| R2V vs Entropy router | No difference | p < 0.05 |
| R2V vs Heuristic router | No difference | p < 0.05 |
| R2V vs Oracle router | No difference | p > 0.05 (gap exists but not always significant) |

**Multiple comparison correction:** Apply Holm-Bonferroni across the 6 pairwise tests (implemented in `statistical.py`). Report corrected p-values.

**What to include in the paper:**
- Table 1 cells: `SR ± CI` format
- Footnote or appendix: McNemar p-values for R2V vs SLM-only and R2V vs best non-oracle baseline
- Flag any comparison that is NOT statistically significant

**Command to extract per-task outcomes for McNemar:**
```bash
python scripts/evaluate.py \
    --config configs/humaneval/noisy.yaml \
    --features data/router_features/humaneval.jsonl \
    --router-path outputs/router/humaneval_noisy/router_final.pt \
    --output results/humaneval_noisy/stats \
    --methods r2v slm_only entropy_router heuristic_router oracle_router \
    --export-per-task-outcomes  # flag needed in evaluate.py if not already present
```

---

### Table — Per-Perturbation-Type Breakdown

*Required to validate RQ4 (generalization) and respond to "does R2V work on all perturbation types?"*

Report SR and CVaR-Fail for R2V vs top-2 baselines, split by perturbation type:

| Perturbation type | SLM-only SR | Heuristic SR | R2V SR | R2V CVaR-Fail |
|---|---|---|---|---|
| Tool flakiness | | | | |
| Partial observability | | | | |
| Prompt injection | | | | |
| Distractors | | | | |
| Clean (no perturbation) | | | | |

**Key signal:** If R2V's SR drops significantly on one perturbation type relative to its overall SR, that type is a weakness to discuss. The gap between SLM-only and R2V should be largest for prompt injection (hardest for SLM, clearest escalation signal).

**Command:**
```bash
python scripts/evaluate.py \
    --config configs/humaneval/noisy.yaml \
    --features data/router_features/humaneval.jsonl \
    --router-path outputs/router/humaneval_noisy/router_final.pt \
    --output results/humaneval_noisy/per_perturbation \
    --methods r2v slm_only heuristic_router \
    --stratify-by-perturbation-type  # flag needed in evaluate.py if not already present
```

*If `--stratify-by-perturbation-type` is not implemented: filter `data/router_features/humaneval.jsonl` by `perturbation_type` field and run evaluate.py separately for each subset.*

---

### Figure — Reliability Diagram (Calibration)

*Required to support any calibration claim in the paper.* ECE alone is insufficient; reviewers expect a reliability diagram showing predicted probability vs actual frequency across confidence bins.

**What to plot:**
- x-axis: Predicted escalation probability (router output), binned into 15 equal-width bins
- y-axis: Fraction of steps where escalation was actually necessary (oracle label)
- Diagonal = perfect calibration
- Plot R2V (with temperature scaling) vs R2V (no temperature scaling) vs Heuristic router

**Metrics to report alongside the figure:**

| Method | ECE ↓ | Brier ↓ |
|---|---|---|
| R2V + temperature scaling | | |
| R2V (no temperature scaling) | | |
| Entropy router | | |
| Heuristic router | N/A | N/A |

*`calibration.py: compute_calibration_metrics` already implements ECE and Brier. The reliability diagram data (bin boundaries, bin accuracies, bin confidences) needs to be extracted and plotted.*

**Command:**
```bash
python scripts/evaluate.py \
    --config configs/humaneval/noisy.yaml \
    --features data/router_features/humaneval.jsonl \
    --router-path outputs/router/humaneval_noisy/router_final.pt \
    --output results/humaneval_noisy/calibration \
    --methods r2v \
    --export-calibration-bins  # ensures per-bin data is written to CSV for plotting
```

---

### Table — Routing Decision Quality

*Answers "does the router escalate at the right times?" without conflating with downstream SR.*

Report precision/recall of escalation decisions against the oracle (ground-truth: escalate iff SLM fails) for R2V and the best heuristic baseline:

| Method | Escalation Precision ↑ | Escalation Recall ↑ | F1 ↑ | LLM% ↓ |
|---|---|---|---|---|
| Entropy router | | | | |
| Heuristic router | | | | |
| Verifier router | | | | |
| R2V | | | | |
| Oracle | 1.00 | 1.00 | 1.00 | (variable) |

**Definitions:**
- TP: Router escalates AND SLM would have failed
- FP: Router escalates AND SLM would have succeeded (unnecessary cost)
- FN: Router does NOT escalate AND SLM fails (missed escalation = failure)
- Precision = TP / (TP + FP); Recall = TP / (TP + FN)

*These can be computed from the per-step decisions output by `evaluate.py` — check whether the structured JSON output already includes per-step routing decisions.*

---

### Figure — Pareto Frontier with Threshold Sweep (Final Version)

This is already planned as Figure 2, but the **exact commands and what to sweep** are specified here for completeness.

**Threshold sweep for R2V** (vary decision threshold from 0.1 to 0.9):
```bash
python scripts/evaluate.py \
    --config configs/humaneval/noisy.yaml \
    --features data/router_features/humaneval.jsonl \
    --router-path outputs/router/humaneval_noisy/router_final.pt \
    --output results/humaneval_noisy/pareto \
    --methods r2v entropy_router \
    --router-threshold-sweep 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

**Points to overlay on the Pareto plot:**
- R2V threshold sweep → curve (SR vs LLM%)
- Entropy router threshold sweep → comparison curve
- SLM-only → single point (LLM%=0, SR=baseline)
- LLM-only → single point (LLM%=100%, SR=ceiling)
- Oracle router → single point (upper bound)
- Heuristic router → single point
- R2V default (threshold=0.5) → highlighted point on the R2V curve

**Expected result:** R2V curve dominates (higher SR at same LLM%) or Pareto-dominates (higher SR AND lower LLM%) the entropy router curve. Oracle is unachievable but shows headroom.

---

### Suggested Evaluation Run Order

1. **First**: Run Table 1 cross-benchmark evaluation — this is the load-bearing result. If R2V does not outperform baselines here, everything else is secondary.
2. **Second**: Run statistical significance tests on Table 1 numbers.
3. **Third**: Per-perturbation breakdown (Table) — needed for RQ4.
4. **Fourth**: Pareto frontier figure (Figure 2) — needed for RQ3.
5. **Fifth**: Reliability diagram (Figure) — needed for calibration claims.
6. **Sixth**: Routing decision quality (Table) — supporting analysis.
7. **Then**: Run ablations A–W in tier order (Tier 1 → Tier 2 → Tier 3).

The ablation tables (Tables 2–5) should be filled in only after the main results (Table 1) are confirmed, since ablations are meaningless if the full system does not outperform baselines.
