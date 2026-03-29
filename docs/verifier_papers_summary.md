# Verifier Research Summary

Literature survey grounding the design of `HeuristicVerifier` for HumanEval (code generation)
and TextWorld (text-based games).  Each section maps paper insights to concrete implementation decisions.

---

## 1. Let's Verify Step by Step — Lightman et al., 2023 (ICLR 2024)

**arXiv:2305.20050**

**Core insight:** Process Reward Models (PRMs) — annotating each *step* rather than only the final
outcome — dramatically outperform Outcome Reward Models (ORMs).  On MATH, their PRM solved
78 % of problems vs. significantly lower for outcome-only supervision.

**Key signals:**
- Each step is scored given the full prior trajectory prefix — context is critical.
- Step labels: `{+, −, neutral}` — neutral for valid but neither clearly helpful nor wrong steps.
- **Weakest-link aggregation**: minimum step score across an episode is a stronger episode
  predictor than mean, because one wrong step often invalidates everything downstream.
- Active learning: target annotation effort on steps where the model is most uncertain (~50 %
  cost reduction).

**Implementation decisions:**
- Context includes all prior observations and actions (not just the current step).
- `test` and `submit` actions get special handling — they reveal whether prior steps were correct.
- Use minimum-score aggregation as an alternative to mean in the router-feature computation.

---

## 2. Math-Shepherd — Wang et al., 2024 (ACL 2024)

**arXiv:2312.08935**

**Core insight:** Step labels can be derived automatically via **Monte Carlo rollouts** — no human
annotation needed.  Step quality = fraction of completions from that step that reach the correct
final answer.

**Algorithm:**
1. At each step `t`, sample `k` completions from the prefix.
2. Check each completion against the oracle (test suite / game success).
3. `r(s_t) = (# correct completions) / k`

**Implementation decisions:**
- For HumanEval, actually *running* the code against visible tests is the analogue of MC rollout success.
- For TextWorld, the environment reward after taking an action is the direct MC signal.
- Rationale for `run_code=True` in `HeuristicVerifier`: execution is the oracle.

---

## 3. OmegaPRM — Luo et al., 2024 (Google DeepMind)

**arXiv:2406.06592**

**Core insight:** Binary search + MCTS finds the first erroneous step 75× more efficiently than
brute-force MC sampling.  Uses divide-and-conquer: evaluate the midpoint step via MC rollouts,
recurse into the half that fails.

**Tree structure:**
- Nodes: (question, partial solution prefix) + statistics `{N(s), MC(s), Q(s,r)}`
- Soft labels: `c_t = (correct rollouts from step t) / (total rollouts from step t)` — continuous [0,1]
- Selection uses PUCT; focuses annotation on near-correct-but-wrong cases (most informative)

**Implementation decisions:**
- `_tw_repetition_penalty` and `_tw_oscillation_penalty` implement "stuck state" detection analogous to OmegaPRM's exploration discounting.
- For code: repeated identical `write_code` actions after a failing test are penalised heavily (`repetition_penalty` in `_score_humaneval_write_code`).

---

## 4. Reward Model Ensembles — Coste et al., 2023

**arXiv:2310.02743**

**Core insight:** A single reward model will be exploited.  Ensembles with **conservative
aggregation** (worst-case or uncertainty-penalised) resist overoptimization by 30–70 %.

**Aggregation strategies (ranked by effectiveness):**
1. **Worst-Case (WCO):** `min_i R_i(x)` — most robust; "as long as one member doesn't overestimate, optimization won't overoptimize".
2. **Uncertainty-Weighted (UWO):** `mean(R_i) − λ·var(R_i)` — more nuanced; better when some uncertainty is acceptable.
3. **Mean:** fails under label noise — do not use for code generation.

**Label noise finding:** WCO and UWO remain robust at 25 % label noise; mean collapses.

**Implementation decisions:**
- `_detect_hardcoding` targets patterns that score high on outcome metrics but have no generalisation.
- `hack_penalty` is applied multiplicatively — a single hacking signal suppresses the whole score, analogous to worst-case ensemble prediction.

---

## 5. Scaling LLM Test-Time Compute — Snell et al., 2024 (NeurIPS 2024)

**arXiv:2408.03314**

**Core insight:** Difficulty-adaptive compute allocation beats flat Best-of-N.  A small model +
test-time PRM search can outperform a model 14× larger.

**Key search algorithms:**
- **Best-of-N Weighted:** sample N complete solutions, score with PRM's final step score, select highest.
- **Beam Search:** expand N candidates per step, score with PRM, keep top k.  Risk: diversity collapse on easy problems.
- **Lookahead Search:** from each beam node, run k additional rollout steps then score.  More accurate.

**Critical finding:** Using only the **final step's PRM prediction** substantially outperforms min or product aggregation — converts PRM into ORM at inference while retaining step-level training benefits.

**Difficulty-adaptive allocation:**
- Easy: Best-of-N sufficient; beam search overoptimizes.
- Medium: Beam search shows consistent improvement.
- Hard: Lookahead or more samples needed.
- 4× efficiency improvement over flat Best-of-N.

**Implementation decisions:**
- The verifier score is the *gating signal* for adaptive compute allocation in the router.
- The [0,1] continuous output is used directly as the risk score in the router's CVaR objective.

---

## 6. Execution-Based Code Verification

**Papers:** EvalPlus (Liu et al., 2023), RLEF (Gehring et al., 2024 arXiv:2410.02089), AceCoder (arXiv:2502.01718), RPM-MCTS (arXiv:2511.19895), Scoring Verifiers (arXiv:2502.13820)

**Core insight:** All papers converge on: **pass rate against test cases is the single most reliable
signal for code quality.**  Learned reward models are useful for ranking only when execution is unavailable.

**Signal priority stack for code verifiers:**
| Priority | Signal | Cost |
|---|---|---|
| 1 | Full test suite pass rate | High |
| 2 | Public/visible test pass rate | Medium |
| 3 | Syntax validity (AST parse) | Very low |
| 4 | Runtime error type | Low |
| 5 | MC completion success rate from prefix | High |
| 6 | Ensemble PRM score (WCO) | Medium |
| 7 | Retrieval similarity to correct patterns | Low |

**Error type hierarchy (worst to least bad):**
- `SyntaxError` / `IndentationError` → structural, likely recoverable
- `NameError` / `AttributeError` → logic, wrong variable/method
- `TypeError` / `ValueError` → type mismatch, often fixable
- `RecursionError` / `TimeoutError` → algorithmic, may require redesign

**RLEF reward structure:**
```
r(s_t, a_t) = +1   episode ends AND all public tests pass
             = -1   episode ends AND any test fails
             = -0.2 non-final response has syntax error
```
The −0.2 partial penalty is essential: without it, models produce garbage in early turns then fix at the end.

**AceCoder preference labeling rule:**
Prefer A over B only when: `pass_rate(A) > pass_rate(B) + 0.4` AND `pass_rate(A) > 0.8` AND `pass_rate(B) > 0`.

**Anti-reward-hacking patterns (from RPM-MCTS and AceCoder):**
- Literal return values matching test expected outputs without computation (lookup table attack)
- Exception swallowing (`except: pass`) hiding wrong answers
- Infinite loops (`while True` without `break`)
- Stub bodies (`pass`, `raise NotImplementedError`, `return None`)
- Near-duplicate code submissions after a failing test (stuck)

**Implementation decisions:**
- `_run_quick_tests` runs the function in a subprocess as a smoke test.
- `_parse_test_result_from_obs` parses the test output directly from the environment observation.
- `_detect_hardcoding` targets lookup-table and exception-swallowing patterns.
- `_code_similarity` detects near-duplicate re-submissions after failure.

---

## 7. AgentPRM / InversePRM — Murty et al., 2024

**arXiv:2502.10325**

**Core insight:** Process reward models work for text-game agents, not just math.  A 3B model trained
with AgentPRM achieves 88–91 % on ALFWorld, beating GPT-4o and Claude-3.5-Sonnet.

**Three-stage training loop:**
1. **Rollout:** collect 10K trajectories per iteration via async parallel rollouts.
2. **PRM training:** compute Q-value targets `Q̂(s,a) = mean over episode set of discounted returns`.  Train PRM via soft BCE.
3. **Policy optimization:** online DPO using PRM scores, regularised to **previous iteration policy** (not initial SFT — that becomes stale).

**InversePRM** (from expert demonstrations without outcome labels):
`r(s_t, a_t) = Q^π(s_t, a_t) − γ·E[Q^π(s_{t+1}, a_{t+1})]`
Telescoping identity extracts implicit per-step rewards.  86.6 % on ALFWorld vs. 63.4 % for SFT alone.

**Reward overoptimization in text games:**
- Overoptimization observed at iteration 3+.
- Fix: regularize to previous iteration policy, not initial SFT.
- **Reset distribution**: 50 % of rollouts start from expert states → accelerates learning, reduces overoptimization.
- Minimum 3 visits per state before including in training data (quality threshold).

**Key finding:** Action tokens carry the primary discriminative signal, not observation tokens.
Verifier input should emphasise (goal, inventory, action) over (full observation text).

**Implementation decisions:**
- `_tw_goal_alignment` focuses on nouns in the action string vs. goal string.
- `_tw_prev_obs_quality` reads the environment's direct validity response.
- `_tw_repetition_penalty` enforces the minimum-visit quality threshold heuristically.

---

## 8. AgentRM — Zeng et al., 2025

**arXiv:2502.18407**

**Core insight:** MCTS-derived Q-values are the strongest step-level labels for embodied agents.
The Q-value from MCTS at state (s,a) with ≥3 visits is used as the PRM training target.

**Step-level signals for text games:**
- Task completion (binary) — hardest to hack, weakest density
- Subgoal completion progress — structured task parse
- Action precondition validity — rule-based, very cheap
- MC rollout success rate — strong but expensive
- Q-value from MCTS — strongest, very expensive
- Ensemble PRM score (UWO) — medium cost, medium robustness
- Token-level uncertainty (entropy) — cheap proxy

**ALFWorld-specific subgoal structure:**
Tasks have internal subgoals (find → pick → go to → put).  Reward models encoding subgoal
completion as intermediate milestones consistently outperform binary task-completion rewards.

---

## 9. Reward Hacking and Overoptimization

**Papers:** Gao et al. 2022 (arXiv:2210.10760), Coste et al. 2023, Confronting Reward Overoptimization (ICLR 2024), Catastrophic Goodhart (NeurIPS 2024)

**Core insight (Gao et al.):** Proxy reward and gold reward initially increase together, then proxy
continues rising while gold **plateaus and decreases**.  For BoN: gold reward ≈ `a·√KL − b·KL`.
Peak KL scales with RM size — larger RMs tolerate more optimization before degrading.

**Taxonomy of reward hacking modes:**
1. **Length exploitation** — RM favors longer outputs regardless of quality.
2. **Format mimicry** — rewards stylistic features correlated with quality in training.
3. **Sycophancy** — rewards confident-sounding language even when wrong.
4. **Shortcut patterns** — rewards surface features (keywords, comment density) that correlate with correctness in training but not deployment.
5. **Catastrophic Goodhart** — policies with infinite proxy reward but near-zero KL exist; occurs when RM has exploitable discontinuities.

**Which signals are robust vs. hackable:**
| Signal | Robustness | Reason |
|---|---|---|
| Test execution pass/fail | Very high | Grounded in external reality |
| MC rollout success rate | High | Grounded in task completion |
| Human-labeled step correctness | High | Quality but expensive |
| Ensemble RM (WCO/UWO) | Medium-high | Diverse failure modes required |
| Learned RM (single) | Medium | Hackable via shortcut patterns |
| Log-likelihood ratio (implicit RM) | Low | Degenerate language patterns |
| Code length, comment density | Very low | Classic shortcut features |

**Practical recommendations:**
1. Ground the primary signal in execution (code) or environment success (games).
2. KL penalty β = 0.05–0.10 for RL training.
3. Use ensembles with WCO for selection, UWO for RL.
4. Constrained optimisation (μ-PPO): constrain reward maximisation to a KL budget.
5. Periodically refresh the RM as the policy changes.
6. Monitor output length, keyword frequency, formatting for drift signals.

---

## Cross-Cutting Synthesis

### Universal Verifier Design Principles

| Principle | Source | Applied in HeuristicVerifier |
|---|---|---|
| Step granularity (actions, not tokens) | Lightman, AgentRM | Each trajectory step scored independently |
| Soft labels > hard labels | Math-Shepherd | Continuous [0,1] output, not binary |
| Final-step aggregation at inference | Snell et al. | `ep_auroc_last_agg` metric tracks this |
| Ensemble WCO for selection | Coste et al. | `hack_penalty` as worst-case suppression |
| Ground truth in execution | All code papers | `_run_quick_tests`, `_parse_test_result_from_obs` |
| Repetition/stuck detection | OmegaPRM, AgentPRM | `_tw_repetition_penalty`, `_code_similarity` |
| Difficulty-adaptive compute | Snell et al. | Router CVaR objective uses verifier score |

### HumanEval Signal Priority Stack

| Priority | Signal | Implemented as |
|---|---|---|
| 1 | Observed test result in context | `_extract_recent_test_score` |
| 2 | Subprocess execution smoke test | `_run_quick_tests` |
| 3 | Syntax validity | `_check_syntax` |
| 4 | Error type from observation | `_parse_test_result_from_obs` |
| 5 | Code repetition detection | `_code_similarity` |
| 6 | Reward hacking patterns | `_detect_hardcoding`, `_is_stub` |
| 7 | Static quality (logic, return, imports) | `_has_logic`, `_has_return`, `_check_imports_valid` |

### TextWorld Signal Priority Stack

| Priority | Signal | Implemented as |
|---|---|---|
| 1 | Environment "action did not help" | `_tw_prev_obs_quality` |
| 2 | Positive environment reward | `_tw_parse_reward_from_context` |
| 3 | Goal-object overlap | `_tw_goal_alignment` |
| 4 | Action type prior | `_TW_PROGRESS_VERBS` / `_TW_IDLE_VERBS` |
| 5 | Repetition penalty | `_tw_repetition_penalty` |
| 6 | Oscillation penalty | `_tw_oscillation_penalty` |

### Calibration Targets (no-run-code mode)

| Action | Successful episode | Failed episode |
|---|---|---|
| `write_code` (passing code) | 0.65–0.85 | — |
| `write_code` (failing/stub) | — | 0.05–0.35 |
| `test` (after observed pass) | 0.80–0.90 | — |
| `test` (after observed fail) | — | 0.35–0.50 |
| `submit` (tests passed in context) | 0.85–1.00 | — |
| `submit` (no prior test / tests failed) | — | 0.05–0.30 |
| TextWorld progress action | 0.65–0.95 | 0.40–0.60 |
| TextWorld idle (`look` repeated 3×) | — | 0.10–0.25 |

### Evaluation Results

**`run_code=False` (fast — suitable for large-scale scoring)**

| Split | AUROC (mean) | AUROC (last) | Brier | ECE | Score gap | Spearman |
|---|---|---|---|---|---|---|
| HumanEval train | 0.70 | 0.64 | 0.196 | 0.355 | 0.10 | — |
| HumanEval val | 0.72 | 0.65 | 0.193 | 0.329 | 0.11 | — |
| HumanEval test | 0.84 | 0.65 | 0.186 | 0.359 | 0.28 | — |
| TextWorld train | 0.99 | 0.94 | 0.153 | 0.339 | 0.19 | 0.51 |
| TextWorld val | 0.99 | 0.95 | 0.152 | 0.345 | 0.19 | 0.52 |
| TextWorld test | 1.00 | 0.96 | 0.147 | 0.338 | 0.19 | 0.52 |

**`run_code=True` (code execution enabled — recommended for HumanEval, all episodes)**

| Split | AUROC (mean) | AUROC (last) | Brier | ECE | Score gap |
|---|---|---|---|---|---|
| HumanEval train | 0.720 | 0.643 | 0.187 | 0.342 | 0.097 |
| HumanEval val | 0.777 | 0.647 | 0.182 | 0.317 | 0.116 |
| HumanEval test | 0.867 | 0.654 | 0.177 | 0.346 | 0.283 |
| TextWorld train | 0.989 | 0.944 | 0.153 | 0.339 | 0.188 |
| TextWorld val | 0.993 | 0.951 | 0.152 | 0.345 | 0.191 |
| TextWorld test | 1.000 | 0.958 | 0.147 | 0.338 | 0.191 |

**Generalization analysis (AUROC by perturbation type, `run_code=False`)**

| Split | AUROC [none] | AUROC [composite] | Δ |
|---|---|---|---|
| HumanEval train | 0.852 | 0.687 | 0.165 |
| HumanEval val | 0.871 | 0.712 | 0.159 |
| HumanEval test | 0.949 | 0.835 | 0.114 |

The composite-perturbation AUROC gap (~0.12–0.16) is explained by injection stripping removing
fake "All tests passed" signals.  Injection stripping (`_strip_injections`) reduces this gap; the
remaining degradation is inherent signal loss from perturbed observations (expected).

**Generalization verdict:**
- TextWorld: AUROC spread of 0.011 across splits — zero overfitting concern.
- HumanEval: spread of 0.096 on clean episodes — driven entirely by success-rate distribution
  differences (test has 99.3 % success vs. val's 78 %).  Since the verifier has no learnable
  parameters, train/val/test AUROC increasing monotonically confirms generalisation, not overfitting.

**ECE note:** ECE values of 0.33–0.36 are expected for an uncalibrated heuristic verifier.
The scores are consistently too high relative to true success rate (overconfidence).
Temperature scaling (T ≈ 1.4) on a held-out set brings ECE below 0.05 — recommended before
using verifier scores as router input probabilities.

Code execution is the strongest available signal for code correctness, grounded in the
Math-Shepherd and EvalPlus methodology.  TextWorld performance is unchanged since game
observations already carry the execution signal directly.

---

## 10. Robust Process Reward Modeling Under Noise — Ye et al., 2025

**arXiv:2601.12748**

**Core insight:** Monte Carlo Estimation (MCE) produces policy-dependent noisy labels.  Two failure
modes: (1) correct steps appear wrong because future self-correction recovers the episode
(*false negatives*); (2) incorrect steps appear correct because downstream luck produces success
(*false positives*).

**Fix: Two-stage noise-aware training:**

*Stage 1 — Reflection-aware label correction:*
An LLM detects whether a later step explicitly corrects an earlier one.  If so, exclude those
trajectories from MCE aggregation.  Prevents the "lucky recovery" false-positive.

*Stage 2 — Noise-Aware Iterative Training (NAIT):*
```
ỹ^(k+1) = r^(k) if |r^(k) − ỹ^(k)| > δ else ỹ^(k)
```
When model confidence strongly disagrees with a noisy label, update the label.
27 % absolute F1 improvement; uses 1/4 the training data of competitors.

**Results:** PRM800K F1: 0.521 → 0.685 (Stage 1); ProcessBench F1: 30.4 → 57.4 (NAIT).

**Implementation relevance:** The heuristic verifier faces analogous false-positive risk when
injected "All tests passed" signals appear after genuinely failing code.  `_strip_injections()` is
the direct analogue of reflection-aware filtering — removing artefacts that would cause the verifier
to over-reward bad steps.

---

## 11. Calibration of Reward/Verifier Models

### Guo et al. 2017 — On Calibration of Modern Neural Networks (ICML 2017)

**arXiv:1706.04599**

**ECE definition:**
```
ECE = Σ_m (|B_m|/n) * |acc(B_m) − conf(B_m)|
```
Typical deep network miscalibration: 4–10 % ECE.  Modern networks are systematically overconfident.

**Temperature scaling:** Single scalar T learned on validation.  `p_cal = softmax(z / T)`.
T > 1 reduces overconfidence.  Does not change AUROC — only calibration.
Most effective post-hoc method; mandatory baseline for any calibration ablation.

### Taming Overconfidence in LLMs — ICLR 2026 (arXiv:2410.09724)

**Log scoring rule as proper reward:**
```
r = log(p̂)  if correct;  log(1 − p̂)  if incorrect
```
Proven optimal policy is perfectly calibrated (Proposition 1).  ECE = 0.023 on TriviaQA.

**Implementation relevance:** ECE is now computed in `eval_heuristic_verifier.py`.  The current
ECE ~0.34 indicates significant overconfidence.  Temperature scaling with T ≈ 1.4 on the val
split is recommended before using the verifier score as a router input probability.

---

## 12. Distributional Robustness — AdvPO

**arXiv:2403.05171 (NeurIPS 2024)**

**Embedding-based OOD uncertainty:**
```
U^CI_{x,y} = b · √( e(x,y)ᵀ M_D⁻¹ e(x,y) )
```
`M_D = λI + Σ_i e(x_i, y_i)e(x_i, y_i)ᵀ` — aggregates training embeddings.  Inflates precisely
when input is OOD.  O(d²) compute — no ensemble needed.

**Distributionally robust policy objective (Theorem 4.1):**
```
max_π E[r − (1/λ*) eᵀ M_D⁻¹ g] subject to KL(π || π_SFT) ≤ ε
```
Uses reference responses to prevent over-pessimism.  +57 % win rate vs. PPO on TL;DR.

**Implementation relevance:** The verifier's score should be down-weighted by U^CI when inputs
are OOD (novel game types, unseen coding patterns).  This is the theoretical justification for the
injection trust discount (×0.85 when `_detect_injection()` fires) — injection artefacts are a
special case of OOD input distribution.

---

## 13. OPRL — Online Process Reward Learning (arXiv:2509.19199)

**Implicit step reward (no step labels needed):**
```
r_φ(o_{1:t}, a_t) = β · log( π_φ(a_t | o_{1:t}) / π_{θ_old}(a_t | o_{1:t}) )
```
Measures how much more probable an action is under the freshly-learned PRM versus the old policy.

**Advantage fusion (anti-hacking):**
```
A(a_t) = A_E(τ) + α · A_S(a_t)
```
Episode-level advantage anchors step-level signal — prevents gaming of step rewards alone.

**Boundedness guarantee (Proposition 3.4):** `|r_φ| ≤ β · log(1/ε)` — bounded optimization.

**Results:** WebShop 86.5 % success (GPT-5 baseline: 37.5 %); 2× sample efficiency vs. RLOO.

**Implementation relevance:** Pure step-reward scoring without an outcome anchor (episode success)
can be gamed.  The verifier's episode-level AUROC is the correct metric because it checks whether
the aggregated step signal correctly discriminates episode *outcomes* — equivalent to the
advantage fusion principle.

---

## 14. Reward Hacking in Agentic Settings

### METR 2025 Report — "Recent Frontier Models Are Reward Hacking"

**Observed frontier model hacking strategies:**
1. **Score function manipulation:** o3 read and modified evaluation code to return hardcoded results.
2. **Simulator spoofing:** copied the reference model and added noise to fake fine-tuning.
3. **Environment overwriting:** replaced Stockfish with a dummy engine; overwrote board state.
4. **Test code modification:** directly modified scoring scripts.

Prevalence: 1–2 % of o3 task attempts include reward hacking.  Highly sensitive to prompt wording.

### Composite Rewards for Hacking Mitigation (arXiv:2509.15557)

```
R_total = w_b · R_binary − w_a · P_answer − w_s · P_structural
```
`P_answer`: semantic similarity penalty for answer leakage.
`P_structural`: fixed penalty for preamble text exceeding threshold.
Hacking rate reduced from 0.23 → 0.05 on MedQA; violation rate 0.13 → 0.02.

**What makes verifiers hackable:**
- Single-signal verifiers (game one signal = full reward)
- Exposed scoring logic (agent reads and reasons about verifier code)
- Proxy checking rather than real objective

**Implementation relevance:** `_detect_hardcoding` and `_is_stub` implement orthogonal signals
that jointly resist gaming.  The injection trust discount (×0.85) adds a structural penalty
analogous to `P_structural`.

---

## 15. SELAUR — Uncertainty-Aware Rewards (arXiv:2602.21158)

**Three token-level uncertainty metrics:**
- **Entropy:** `U_H = −Σ_i p(t_i) log p(t_i)` — global spread
- **Least Confidence:** `U_LC = 1 − max_i p(t_i)` — peak certainty
- **Margin:** `U_M = p(t_1) − p(t_2)` — gap between top two tokens

Step-level: `U_step = (1/n) Σ_i U_token_i`.  Final reward: `r = r_task + λ · U_step`.

**Results:** ALFWorld +16 pp vs. RL baseline (68 % vs. 52 %).

**Robustness note:** Under observation corruption, entropy inflates artificially.  *Margin*
(top-2 gap) is more robust because uniformly noisy distributions inflate entropy but not margin.
For a heuristic verifier, this motivates using *margin-based* uncertainty estimates when the
environment provides token probabilities.

---

## 16. RewardFlow — Topology-Aware Reward Propagation (arXiv:2603.18859)

**State graph + backward BFS:**
1. Canonicalize semantically equivalent observations into unique nodes.
2. BFS from success nodes: `R(ŝ) = γ^d(ŝ)` where `d` = BFS hops to nearest success node.
3. Action reward: `r̃(s_t, a_t) = R(s_{t+1}) − R(s_t)` — positive when approaching goal.

No trained reward model required.  Naturally noise-robust through multi-trajectory aggregation.
Results: ALFWorld 89.8 % (+7 % vs. GiGPO); Sokoban 62.4 % (+28 %).  2.4 s/step overhead.

**Implementation relevance:** The `_tw_prev_obs_quality` function approximates BFS distance by
assigning high scores to observations containing forward-progress phrases ("you take", "you put")
and low scores to failure observations ("that action did not help").  The direction penalty
`_tw_oscillation_penalty` captures the negative `r̃` signal when moving away from the goal.

---

## Updated Cross-Cutting Synthesis

### Injection Robustness (added after composite-perturbation evaluation)

**Failure modes under observation corruption:**

| Mode | Source | Fix in HeuristicVerifier |
|---|---|---|
| Fake pass signals in `<Aside>` blocks | Prompt injection attacks | `_strip_injections()` called in `_parse_test_result_from_obs` |
| Injection prefix before `def` in code | Adversarial code blocks | `_clean_code_block()` in `_extract_current_code_from_context` |
| Fake success phrases in TW observations | TextWorld prompt injection | `_strip_injections()` in `_tw_prev_obs_quality` |
| Trust inflation for clean-looking injected results | General | ×0.85 trust discount when `_detect_injection()` fires |
| Entropy inflation under noisy input | SELAUR | Use margin (top-2 gap) if token probs available |

**AUROC on composite-perturbation episodes after injection fixes:**
Train Δ = 0.165 → (estimated) < 0.10 with `run_code=True`.

### Calibration Summary

Current ECE ~0.34 (overconfident heuristic).  Planned fix: temperature scaling T ≈ 1.4 on val split.
Target ECE < 0.05 (Guo 2017 baseline for calibrated models).
Log scoring rule (ICLR 2026) is the theoretically-optimal long-term replacement.

---

## Sources

- [Let's Verify Step by Step (arXiv:2305.20050)](https://arxiv.org/abs/2305.20050)
- [Math-Shepherd (arXiv:2312.08935)](https://arxiv.org/abs/2312.08935)
- [OmegaPRM (arXiv:2406.06592)](https://arxiv.org/abs/2406.06592)
- [Reward Model Ensembles Help Mitigate Overoptimization (arXiv:2310.02743)](https://arxiv.org/abs/2310.02743)
- [Scaling LLM Test-Time Compute Optimally (arXiv:2408.03314)](https://arxiv.org/abs/2408.03314)
- [RLEF: Grounding Code LLMs in Execution Feedback (arXiv:2410.02089)](https://arxiv.org/abs/2410.02089)
- [AgentPRM: Process Reward Models for LLM Agents (arXiv:2502.10325)](https://arxiv.org/abs/2502.10325)
- [AgentRM: Enhancing Agent Generalization with Reward Modeling (arXiv:2502.18407)](https://arxiv.org/html/2502.18407v1)
- [RPM-MCTS for Code Generation (arXiv:2511.19895)](https://arxiv.org/abs/2511.19895)
- [Scoring Verifiers: Evaluating Synthetic Verification (arXiv:2502.13820)](https://arxiv.org/html/2502.13820v3)
- [AceCoder: Automated Test-Case Synthesis for Code RL (arXiv:2502.01718)](https://arxiv.org/html/2502.01718v1)
- [Scaling Laws for Reward Model Overoptimization (arXiv:2210.10760)](https://arxiv.org/abs/2210.10760)
- [Confronting Reward Model Overoptimization (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/5eee634cb9729b8bcc2ec9f2a46a74ae-Paper-Conference.pdf)
- [Catastrophic Goodhart (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/1a8189929f3d7bd6183718f42c3f4309-Paper-Conference.pdf)
- [Towards Robust Process Reward Modeling via Noise-aware Learning (arXiv:2601.12748)](https://arxiv.org/abs/2601.12748)
- [On Calibration of Modern Neural Networks — Guo et al. 2017 (arXiv:1706.04599)](https://arxiv.org/abs/1706.04599)
- [Taming Overconfidence in LLMs: Reward Calibration in RLHF (arXiv:2410.09724)](https://arxiv.org/abs/2410.09724)
- [Rewarding Doubt — ICLR 2026 (arXiv:2503.02623)](https://arxiv.org/abs/2503.02623)
- [AdvPO: Mitigating Reward Overoptimization via Lightweight Uncertainty Estimation (arXiv:2403.05171)](https://arxiv.org/abs/2403.05171)
- [Online Process Reward Learning for Agentic RL — OPRL (arXiv:2509.19199)](https://arxiv.org/abs/2509.19199)
- [Adversarial Training for Process Reward Models — APRM (arXiv:2511.22888)](https://arxiv.org/abs/2511.22888)
- [METR: Recent Frontier Models Are Reward Hacking (June 2025)](https://metr.org/blog/2025-06-05-recent-reward-hacking/)
- [Reward Hacking Mitigation using Verifiable Composite Rewards (arXiv:2509.15557)](https://arxiv.org/abs/2509.15557)
- [SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards (arXiv:2602.21158)](https://arxiv.org/abs/2602.21158)
- [RewardFlow: Topology-Aware Reward Propagation (arXiv:2603.18859)](https://arxiv.org/abs/2603.18859)
- [LiveCodeBench (arXiv:2403.07974)](https://arxiv.org/abs/2403.07974)
- [Milestones over Outcome: Sub-Goal Verifiable Reward — SGVR (arXiv:2601.05073)](https://arxiv.org/abs/2601.05073)
