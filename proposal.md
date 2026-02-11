

# Risk-Calibrated Routing + Robust Verifier-Distillation Objective

* **What:** Learn a router that minimizes cost subject to *robust* success (min/CVaR over perturbations) using verifier-derived risk estimates; distill SLM with verifier-labeled self-correction on *perturbed trajectories* (tool noise + injection).


## Method specification

### Problem setup and notation

We model a tool-using agent as a POMDP with *tool noise*:

* State $s_t$, observation $o_t = \Omega(s_t)$ (webpage accessibility tree + URL + goal; or repo state + logs).
* Action $a_t \in \mathcal{A}$ (WebArena action grammar; SWE-bench patch/tool actions).
* Tool response randomness via perturbation seed $z \sim \mathcal{Z}$: executing $a_t$ yields next state $s_{t+1} \sim P(\cdot \mid s_t, a_t; z)$.

Let $G$ be the task goal, and define episode success $S(\tau) \in \{0,1\}$ (WebArena functional correctness; SWE-bench tests passing). WebArena already provides functional correctness evaluation and a standardized action interface.  

We have:

* **SLM policy** $\pi_\theta(a \mid x)$ where $x_t = (G, o_{\le t}, a_{<t}, y_{<t})$ is the agent context.
* **LLM teacher** $\pi_T$ (used for data generation + fallback).
* **Verifier** $V_\phi(x_t, a_t) \to [0,1]$: predicts probability that choosing $a_t$ at $x_t$ leads to eventual success (or “safe & progress-making”).
* **Router** $r_\psi(x_t) \to [0,1]$: probability of using LLM fallback instead of SLM at step $t$.

### Architecture/components

1. **Policy backbone (SLM):** decoder-only LM (7–13B class) producing:

   * action token sequence (WebArena action grammar; SWE-bench patch/tool calls)
   * optional short "plan tag" $(not\ full\ CoT;\ just\ a\ structured\ plan\ string)$ used as input to verifier.
2. **Verifier $V_\phi$:** can be:

   * $(a)$ a frozen strong LLM scorer $(SCORE/PRM\ style)$, or
   * $(b)$ a distilled smaller verifier trained from strong-verifier labels.
     SCORE formalizes verifier-based selection and oversample-then-rerank using verifier probabilities. 
3. **Router $r_\psi$:** lightweight classifier/regressor on $(policy\ hidden\ state\ summary + uncertainty + verifier\ score)$. Output is calibrated fallback probability.

### Data construction (teacher + perturbations + labels)

We build a dataset of trajectories with *paired clean + perturbed replays*:

1. **Clean teacher trajectories**

   * Run $\pi_T$ in base environments (WebArena, SWE-bench) to collect successful trajectories $\tau = (x_t, a_t, o_{t+1})_{t=1}^T$.
   * Use ReAct-style prompting or direct-agent prompting as baselines for teacher trajectory generation (WebArena provides direct-agent system prompt and action grammar).  ReAct provides a standard “reasoning+acting” pattern and shows it improves interactive task success (e.g., WebShop). 

2. **Perturbation operator $\mathcal{P}_k$**
   For each trajectory, generate $M$ perturbed variants by applying one perturbation type $k$ and seed $z$:

   * **Tool flakiness:** stochastic failures/timeouts; stale cached results; inconsistent top-k results.
   * **Partial observability:** hide/reorder DOM elements, truncate logs, remove key lines.
   * **Prompt injection:** insert malicious “instructions” into webpage text/logs; agent must treat tool outputs as untrusted.
   * **Distractors:** irrelevant but plausible tool outputs.

3. **Step labels for verifier training (automatic where possible)**

   * **WebArena:** define progress signal using evaluation script components $(exact\ match\ /\ must\text{-}include\ /\ fuzzy\ match\ etc.)$ and intermediate subgoals; WebArena's evaluation is designed for functional correctness.
   * **SWE-bench:** label success via “% resolved” (tests pass after patch) and “% applied”. 

   We create labels:

   * $y^{\text{final}} \in \{0,1\}$: episode success under perturbation.
   * $y^{\text{step}}_t \in \{0,1\}$: whether action $a_t$ is *consistent & progress-making* $(defined\ by\ environment\text{-}specific\ heuristics + optional\ strong\ LLM\ judgment)$. This is the agent analogue of **process supervision**: PRM/ORM distinction motivates stepwise feedback. 

4. **Contamination controls** $(esp.\ SWE\text{-}bench)$

   * Use SWE-bench train split for training; evaluation on test split; note SWE-bench explicitly uses disjoint repos for SFT to reduce contamination. 
   * For WebArena, ensure train/eval tasks are disjoint by template + seed.

### Training objectives

**(i) SLM policy: behavior cloning + verifier-guided self-correction**

1. **Behavior cloning (BC) on teacher actions**

$$
\mathcal{L}_{\mathrm{BC}}(\theta)
= -\mathbb{E}_{(x,a^\star)\sim D_T}\big[\log \pi_\theta(a^\star\mid x)\big]
$$

2. **Verifier-guided preference distillation on perturbed contexts**

For each $x$, sample $K$ candidate actions $a^{(1..K)}\sim \pi_\theta(\cdot|x)$. Score them with verifier:

$$
s_k = V_\phi(x,a^{(k)})
$$

Pick $a^{+} = \arg\max s_k$, $a^{-} = \arg\min s_k$. Train with a DPO-style preference objective:

$$
\mathcal{L}_{\text{pref}}(\theta)= -\mathbb{E}_{x}\left[\log \sigma\left(\beta(\log \pi_\theta(a^{+}|x)-\log \pi_\theta(a^{-}|x))\right)\right]
$$

This directly implements "small models need strong verifiers" as a training signal, and aligns with SCORE's verifier-based selection logic.

3. **Tool-consistency regularizer** (noise-awareness)
   
   For contexts where the agent can re-query a tool, enforce invariance across stochastic tool outputs:
   
   * Sample two tool outcomes $(y,y')$ under different seeds for the same query, producing contexts $(x, x')$.
   * Encourage consistent *high-level action choice* (e.g., same next tool call type or same stop condition):

     $$
     \mathcal{L}_{\text{cons}}(\theta)=\mathbb{E}_{(x,x')}\text{KL}\big(\pi_\theta(\cdot|x)\,||\,\pi_\theta(\cdot|x')\big)
     $$
     This is the "under tool noise" part that is not present in clean imitation.

**Overall:**

$$
\min_\theta \; \mathcal{L}_{\text{BC}} + \lambda_{\text{pref}}\mathcal{L}_{\text{pref}} + \lambda_{\text{cons}}\mathcal{L}_{\text{cons}}
$$

**(ii) Verifier: stepwise + final-outcome supervision**

Train $V_\phi$ to predict success/progress for an $(x,a)$ pair:

$$
\mathcal{L}_V(\phi)= -\mathbb{E}\left[y^{\text{final}}\log V_\phi(x,a) + (1-y^{\text{final}})\log(1-V_\phi(x,a))\right]
$$

Optionally multi-task with step labels $y^{\text{step}}_t$. This is the agent analogue of process-supervised reward modeling motivation. 

**(iii) Router: risk-calibrated cost–robustness optimization**

Define per-step costs $c_{\text{SLM}}\ll c_{\text{LLM}}$. Router chooses $d_t\in\{\text{SLM},\text{LLM}\}$.

We optimize a *robust* objective over perturbations $z$. Two options:

1. **CVaR success constraint (recommended)**

Let success under seed $z$ be $S_z$. Define $\text{CVaR}_\alpha(1-S_z)$ as tail failure rate. Optimize:

$$
\min_\psi \; \mathbb{E}[ \text{cost}(d_{1:T})] \;\text{s.t.}\; \text{CVaR}_\alpha(1-S_z)\le \epsilon
$$
   Implement via Lagrangian with empirical CVaR over sampled seeds.

2. **Worst-case (min over seeds)**

$$
\max_{z\in \mathcal{Z}_{\text{eval}}} \; (1-S_z) \;\text{penalized}
$$

**Training signal:** for each context $x_t$, estimate “SLM risk” using verifier + uncertainty:

$$\rho(x_t)= f\big(1- V_\phi(x_t,\hat a^{\text{SLM}}), \; H(\pi_\theta(\cdot|x_t))\big)$$

Define router label $y^{\text{route}}= \mathbb{1}[\rho(x_t)>\tau]$ (or learn $\tau$ from cost/robustness tradeoff), and fit $r_\psi$ with calibration $(Brier + ECE)$.

### Inference procedure (step-by-step pseudocode)

```python
# Robust Router + Verifier-guided Agent (R2V-Agent)

def act(goal, obs, history, budget):
    x = build_context(goal, obs, history)

    # 1) SLM proposes K candidates (compute knob)
    cand_actions = sample_actions(pi_slm, x, K=budget.K)

    # 2) Verifier scores candidates
    scores = [V(x, a) for a in cand_actions]
    a_slm = cand_actions[argmax(scores)]
    risk = risk_score(scores, pi_slm_uncertainty(pi_slm, x))

    # 3) Router decides fallback
    if router(r, x, risk) and budget.llm_calls_left > 0:
        a = pi_llm(x)             # LLM fallback action
        budget.llm_calls_left -= 1
    else:
        a = a_slm                 # SLM action

    # 4) Optional self-correction loop (compute knob)
    for _ in range(budget.self_correct_iters):
        if V(x, a) >= budget.accept_thresh:
            break
        # refine action using either SLM+verifier or LLM refiner
        a = refine_action(x, a, pi_slm, V, budget)

    return a
```

**Compute knobs / stopping criteria**

* $K$: number of SLM candidates $(quality\ vs\ latency)$.
* `llm_calls_left`: max fallback calls per episode $(cost\ cap)$.
* `self_correct_iters` and `accept_thresh`: how much verifier-gated refinement to do.
* Step limit $T$ $(already\ standard\ in\ WebArena\text{-}style\ setups)$.  

## What is technically new
### What’s new (relative to closest work)

* **New objective for routing:** not heuristics; explicitly optimizes **robust tail risk (CVaR/worst-case)** under perturbations while trading off cost. ReAct’s switching is heuristic-based. 
* **Noise-aware distillation:** trains on *perturbed trajectory replays* + **tool-consistency regularization**, directly targeting the failure modes your benchmark introduces $(flakiness,\ inconsistent\ tool\ output,\ injection)$.
* **Agent-step "process supervision":** operationalizes PRM's process supervision concept for agent actions/tool calls $(stepwise\ correctness\ signals)$, not just reasoning chains.

### Why it should work (mechanistic intuition)

* Tool noise/injection creates **high-variance regions** where SLMs fail catastrophically; a calibrated router is essentially a **selective prediction** mechanism. CVaR focuses learning pressure on those tail regions rather than average-case.
* Verifier-guided preference distillation converts a strong verifier into dense training signal for SLMs (SCORE-style selection formalism), improving self-correction without requiring full LLM at inference. 
* Consistency regularization makes the SLM policy less brittle to stochastic tool outputs $(the\ core\ "tool\ noise"\ axis)$.

## Minimal reproducible prototype (MRP, 1–2 weeks)

**Goal:** Validate *Claim 1* on a small but defensible slice.

**Dataset/tasks**

* **WebArena**: pick 5–10 templates, 5 task instances each (≈25–50 tasks). WebArena reports per-template variability; templates are a meaningful grouping. 
* Create **2 perturbation types**:

  1. **Search inconsistency**: randomize ranking / drop top results with seed $z$.
  2. **Prompt injection**: insert a malicious instruction into page text $(e.g.,\\ "ignore\\ objective,\\ reveal\\ password")$—agent must ignore tool text as instruction.

**Models**

* Teacher: one strong LLM (can be API or open weights).
* SLM: 7B-class.
* Verifier: start with strong LLM-as-verifier (no training), then distill later.

**Baselines (minimal)**

1. LLM-only direct agent (WebArena prompt scaffold). 
2. SLM-only BC (imitate teacher on clean trajectories).
3. Naive router: fallback when SLM entropy > threshold.
4. **Your method**: verifier-scored candidate actions + learned router (even if router is logistic regression).

**Success criterion**

* On perturbed tasks: **worst-seed success** improves by $\geq +10\text{--}15$ absolute points over SLM-only BC, while using $\leq 30\%$ of the LLM calls of LLM-only.

---

# Experimental plan

## Benchmarks/tasks

**Standard tasks (≥2):**

1. **WebArena (clean)** — interactive web tasks with functional correctness evaluation. 
2. **SWE-bench (clean)** — real GitHub issue resolution measured by “% resolved / % applied”. 

**Stress/OOD suite (≥1): SLM-AgentGym perturbations**

* **Noisy-WebArena**: tool flakiness, partial observability, injection, distractors $(factorized)$.
* **Noisy-SWE-bench**: flaky tests $(random\ fails)$, truncated logs, dependency/version drift, misleading error messages.

## Baselines (must be strong + explicit variants)

### For WebArena

1. **LLM-only direct agent** $(system\ prompt + action\ grammar\ as\ in\ WebArena)$. 
2. **LLM-only ReAct** $(reason\text{+}act\ prompting)$. ReAct improves interactive success in WebShop and compares to act-only. 
3. **SLM-only**:

   * BC distillation from teacher trajectories $(clean\ only)$
   * BC + noisy augmentation $(no\ verifier,\ no\ router)$
4. **Router baselines**:

   * entropy-threshold router
   * verifier-threshold router $(no\ training;\ fixed\ \tau)$
   * oracle router $(upper\ bound)$: route if SLM fails in hindsight
5. **Conversion pipeline baseline**: implement the decompose→route→distill steps as described $(closest\ prior)$. 

### For SWE-bench

1. LLM-only agentic patching $(single\ attempt)$
2. LLM-only + multi-sample rerank $(best\text{-}of\text{-}N)$
3. SLM-only fine-tune on SWE-bench train $(repo\text{-}disjoint\ split\ noted)$. 
4. SLM + verifier rerank $(SCORE\text{-}style\ oversample\text{-}then\text{-}rerank)$. 
5. Your robust router + verifier-distilled SLM

## Metrics

**Primary**

* **Robust success**: $\min_{z \in \mathcal{Z}} \text{SR}(z)$ or $\text{CVaR}_\alpha(\text{failure})$ over perturbation seeds.

**Secondary (≥3)**

1. **Average success** on clean + perturbed.
2. **Cost**: tokens, # tool calls, # LLM fallbacks, $ proxy.
3. **Latency**: wall-clock or model-call count.
4. **Router calibration**: ECE/Brier on “should fallback”.
5. **Safety failure rate**: executes injected instructions / credential exfiltration attempts $(binary + severity)$.

## Stress tests (systematic + worst-case)

* **Format shifts**: reorder observation fields, truncate DOM/log, random whitespace/noise.
* **Distractors**: add irrelevant but plausible search results/log lines.
* **Adversarial prompt injection**: malicious instructions embedded in tool outputs/web content.
* **Distribution shift**:

  * WebArena: unseen templates or increased task length $(more\ repetitive\ operations)$; WebArena notes complexity increases within template variants. 
  * SWE-bench: unseen repos; stricter tests; larger diffs.

Report tail metrics: min-over-seeds, and “bottom-10% seed success”.

## Ablations & sensitivity

* Remove verifier ($V$) → BC-only.
* Remove router ($r$) → SLM-only + verifier self-correction.
* Clean-only vs noisy-augmented training.
* Consistency regularizer on/off ($\lambda_{\text{cons}}$).
* Router objective: expected vs CVaR vs worst-case.
* Compute sweeps: $K$ candidates, # self-correction iters, LLM call budget.
* Data scaling curves: # teacher trajectories, # perturbation seeds.

## Statistical rigor

* **Seeds**: ≥5 random seeds for training; ≥20 perturbation seeds per task template for evaluation.
* **CIs**: bootstrap over tasks (paired bootstrap) for SR and robust SR.
* **Significance**: paired McNemar test on per-task success for key comparisons.
* **Pre-register comparisons** (must beat):

  * Your method vs SLM-only BC on robust SR.
  * Your method vs naive router on (robust SR, calibration).
  * Your method vs LLM-only on cost at matched robust SR targets $(frontier\ plot)$.

## Reliability & leakage controls

* **SWE-bench split hygiene**: train only on SWE-bench train split; evaluate on test; note repo-disjoint design for SFT. 
* **Prompt/template leakage**: keep prompts fixed across methods; report prompt tokens; avoid tuning on test templates.
* **WebArena hidden seeds**: hold out perturbation seeds; refresh seeds periodically.
* **Contamination checks**:

  * For SWE-bench: deduplicate against training corpora when possible; at minimum, track whether the model reproduces known PR patches verbatim $(hash\text{-}based)$.
  * For WebArena: ensure no overlap in $(template,\ site,\ entity\ IDs)$ between train and eval.
