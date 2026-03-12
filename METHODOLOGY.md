# R2V-Agent: Complete Methodology

## Risk-Calibrated Routing with Robust Verifier-Distillation Objective

---

## Table of Contents

1. [Research Hypothesis & Goals](#1-research-hypothesis--goals)
2. [Problem Formulation](#2-problem-formulation)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Component 1: SLM Policy (π_θ)](#4-component-1-slm-policy-π_θ)
5. [Component 2: Teacher LLM (π_T)](#5-component-2-teacher-llm-π_t)
6. [Component 3: Verifier (V_φ)](#6-component-3-verifier-v_φ)
7. [Component 4: Router (r_ψ)](#7-component-4-router-r_ψ)
8. [Training Objective Functions](#8-training-objective-functions)
9. [Data Construction Pipeline](#9-data-construction-pipeline)
10. [Training Pipeline](#10-training-pipeline)
11. [Inference Procedure](#11-inference-procedure)
12. [Compute Infrastructure](#12-compute-infrastructure)
13. [Code-to-Theory Mapping](#13-code-to-theory-mapping)

---

## 1. Research Hypothesis & Goals

### Core Hypothesis

A lightweight router trained with a **CVaR-constrained Lagrangian objective** can learn to **dynamically escalate** from a small language model (SLM) to a large language model (LLM) at the step level, achieving near-LLM task success rates under realistic tool-noise perturbations while using the LLM for only a small fraction of steps, thereby maintaining low inference cost.

### Research Goals

1. **Robustness**: Maintain high task success rate even under the worst-case perturbation conditions (measured via Conditional Value at Risk on the bottom α-fraction of perturbation seeds).
2. **Cost Efficiency**: Minimize the number of expensive LLM calls by routing the majority of steps through the SLM.
3. **Calibration**: Ensure the router's escalation probability is well-calibrated — i.e., when the router says there is a 70% chance the SLM will fail, the SLM should actually fail approximately 70% of the time.
4. **Generalization**: The routing policy should generalize across perturbation types unseen during training.

### Key Research Questions

- **RQ1**: Does CVaR-constrained routing improve worst-case robustness over entropy-based baselines?
- **RQ2**: How does the verifier-distillation objective affect routing quality?
- **RQ3**: What is the cost-accuracy Pareto frontier for the hybrid SLM+LLM system?
- **RQ4**: Does consistency regularization improve performance under varied perturbation seeds?

---

## 2. Problem Formulation

### POMDP with Tool Noise

The agent-environment interaction is modeled as a **Partially Observable Markov Decision Process (POMDP)** augmented with a tool-noise perturbation seed:

$$\mathcal{M}_z = (\mathcal{S}, \mathcal{A}, \mathcal{O}, T, O_z, R, \gamma)$$

where:
- $\mathcal{S}$: State space (true environment state, e.g., repository code, file system)
- $\mathcal{A}$: Action space (tool calls: bash commands, file edits, search queries)
- $\mathcal{O}$: Observation space (tool outputs, error messages, test results)
- $T: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$: Deterministic transition function
- $O_z: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{O})$: **Noisy observation function**, parameterized by perturbation seed $z \sim \mathcal{Z}$
- $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$: Reward function (task completion)
- $\gamma$: Discount factor

The critical insight is that the observation function $O_z$ introduces stochastic noise — the same action in the same state may produce different observations depending on the perturbation seed $z$. This models real-world tool unreliability (flaky tests, truncated logs, network timeouts).

### Notation

| Symbol | Description |
|--------|-------------|
| $\pi_\theta$ | SLM policy parameterized by $\theta$ (Llama-3.1-8B + LoRA) |
| $\pi_T$ | Teacher LLM policy (Gemini 3 Flash) |
| $V_\phi$ | Verifier parameterized by $\phi$ |
| $r_\psi$ | Router parameterized by $\psi$ |
| $x_t$ | Observation context at step $t$ |
| $a_t$ | Action taken at step $t$ |
| $z$ | Perturbation seed |
| $\mathcal{Z}$ | Set of perturbation seeds, $|\mathcal{Z}| = 20$ |
| $d_t$ | Router decision: $d_t = 1$ means escalate to LLM |
| $S_z$ | Task success indicator under seed $z$ |
| $H$ | Episode horizon (current step count) |

### The Routing Decision

At each step $t$, the system must decide:

$$d_t = \begin{cases} 0 & \text{(use SLM)} \\ 1 & \text{(escalate to LLM)} \end{cases}$$

The router outputs a continuous probability $p_t = r_\psi(f_t) \in [0, 1]$ which is thresholded.

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    R2V-Agent System                       │
│                                                          │
│  Observation x_t                                         │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────┐    K candidates    ┌──────────────┐       │
│  │ SLM π_θ  │──────────────────▶│ Verifier V_φ  │       │
│  │ (8B+LoRA)│                    │ (LLM Judge)   │       │
│  └──────────┘                    └──────┬───────┘       │
│                                         │ scores        │
│                                         ▼               │
│                                  ┌─────────────┐        │
│                     features f_t │ Router r_ψ  │        │
│                                  │ (MLP+CVaR)  │        │
│                                  └──────┬──────┘        │
│                                         │ p_t           │
│                           ┌─────────────┴────────────┐  │
│                           │ if risk > threshold       │  │
│                           ▼                          ▼  │
│                    ┌────────────┐          ┌──────────┐ │
│                    │ Teacher π_T │          │Use SLM a_t│ │
│                    │  (Gemini)  │          │ directly  │ │
│                    └────────────┘          └──────────┘ │
│                                                          │
│                    Self-Correction Loop (up to 2 iters)  │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Component 1: SLM Policy (π_θ)

### Architecture

- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct` (8 billion parameters)
- **Adaptation**: LoRA (Low-Rank Adaptation) applied to all major projection layers
- **Quantization**: 4-bit NormalFloat (NF4) quantization via bitsandbytes (disabled for multi-GPU DDP training)
- **Attention**: Flash Attention 2 for efficient training

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank ($r$) | 64 |
| Alpha ($\alpha$) | 128 |
| Dropout | 0.05 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

The effective scaling factor is $\alpha / r = 128 / 64 = 2.0$, applied as:

$$W' = W + \frac{\alpha}{r} \cdot BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ are the low-rank matrices.

### Quantization Details

- **Type**: NF4 (4-bit NormalFloat) via bitsandbytes
- **Compute dtype**: `bfloat16`
- **Behavior**: When `WORLD_SIZE > 1` (multi-GPU), quantization is automatically disabled for DDP compatibility; the model runs in full `bfloat16` precision instead

### Key Capabilities

The `PolicyModel` class (in `r2v/models/policy.py`) provides:

1. **`forward(input_ids, attention_mask, labels)`**: Standard causal LM forward pass for BC training
2. **`compute_log_probs(input_ids, attention_mask)`**: Extracts per-token log probabilities needed for DPO preference training
3. **`compute_action_distribution(input_ids, attention_mask, num_positions)`**: Returns logits over the last $N$ token positions for computing entropy and KL divergence (used by consistency regularization)
4. **`generate_candidates(input_ids, attention_mask, K, temperature, top_p)`**: Generates $K$ candidate action sequences using temperature sampling; used during inference and candidate generation

### Gradient Checkpointing

Gradient checkpointing is enabled via `model.gradient_checkpointing_enable()` to trade compute for memory, allowing training with longer sequences on limited GPU memory.

---

## 5. Component 2: Teacher LLM (π_T)

### Configuration

- **Model**: Gemini 3 Flash (`gemini-3-flash-preview`, via Google API)
- **Provider**: Google (also supports OpenAI GPT-4o, Anthropic Claude, DeepSeek)
- **Max sequence length**: 8,192 tokens
- **Max output tokens**: 4,096 tokens
- **Temperature**: 0.7
- **Top-p**: 0.95

### Role in the System

The teacher LLM serves three distinct purposes:

1. **Trajectory Collection (Training Phase)**: Generates expert demonstrations on training tasks, producing the dataset of successful trajectories that form the BC training data.
2. **LLM Fallback (Inference Phase)**: When the router decides to escalate, the teacher LLM generates the action for the current step.
3. **Verifier Supervision (Optional)**: Can provide LLM-as-judge scores for verifier training data.

### Cost Model

The unified `LLMClient` (in `r2v/models/llm_client.py`) tracks per-call costs:

| Model | Input ($/1K tokens) | Output ($/1K tokens) |
|-------|---------------------|----------------------|
| GPT-4o | $0.005 | $0.015 |
| GPT-4o-mini | $0.00015 | $0.0006 |
| Claude 3.5 Sonnet | $0.003 | $0.015 |
| Gemini 1.5 Pro | $0.00125 | $0.005 |

The client implements retry logic with exponential backoff (3 attempts, base delay 1.0s, factor 2.0).

---

## 6. Component 3: Verifier (V_φ)

### Purpose

The verifier scores the quality of SLM-generated candidate actions, producing $V_\phi(\text{context}, \text{action}) \in [0, 1]$ — a scalar estimate of action quality. These scores serve as the primary signal for the router's escalation decision.

### Mode 1: LLM-as-Judge Verifier (Primary)

In the primary configuration used in experiments:

- **Model**: `meta-llama/Llama-3.1-70B-Instruct` (local inference)
- **Mechanism**: The verifier constructs a prompt containing the observation context and the candidate action, then asks the LLM to evaluate quality on a 0-to-1 scale
- **Output format**: Parses a JSON response containing a `score` field
- **Batched inference**: For local models, uses left-padded batched `generate()` for throughput (implemented in `_score_batch_local()`)
- **Threaded API parallelism**: For API-based providers, uses `ThreadPoolExecutor` with configurable `max_workers` (default: 4)

### Mode 2: Trained Distilled Verifier (Alternative)

A smaller, faster verifier distilled from the LLM judge:

- **Backbone**: `meta-llama/Llama-3.1-8B-Instruct` (frozen)
- **Classification head**: 2-layer MLP with dimensions `[hidden_dim=1024, hidden_dim=1024, 1]`, dropout 0.1
- **Activation**: GELU with sigmoid output
- **Training**: Multi-task loss (see Section 8.3)

### Scoring Protocol

The verifier provides rich statistics over $K$ candidates. Given candidates $\{a^{(1)}, \ldots, a^{(K)}\}$:

- **Best score**: $\max_k V_\phi(x_t, a^{(k)})$
- **Mean score**: $\frac{1}{K} \sum_k V_\phi(x_t, a^{(k)})$
- **Score spread**: $\max_k V_\phi - \min_k V_\phi$
- **Score std**: $\text{std}(\{V_\phi(x_t, a^{(k)})\}_k)$

These four statistics become router input features.

---

## 7. Component 4: Router (r_ψ)

### Architecture

The router is a lightweight **Multi-Layer Perceptron (MLP)** that maps a 13-dimensional feature vector to a scalar escalation probability:

$$r_\psi: \mathbb{R}^{13} \rightarrow [0, 1]$$

#### Network Structure

```
Input (13-dim) → Linear(13, 128) → GELU → BatchNorm → Dropout(0.2)
              → Linear(128, 64) → GELU → BatchNorm → Dropout(0.2)
              → Linear(64, 1) → Temperature-Scaled Sigmoid
```

The output uses a **temperature-scaled sigmoid**:

$$r_\psi(f) = \sigma\left(\frac{\text{logit}(f)}{T}\right)$$

where $T$ is a learnable temperature parameter (initialized to 1.0). Lower $T$ produces sharper (more decisive) routing decisions; higher $T$ produces softer, more calibrated probabilities.

### Input Features (13-dimensional)

The router receives a carefully engineered 13-dimensional feature vector $f_t$ at each step $t$. These features are computed by running the SLM policy and verifier on each trajectory step (see `scripts/generate_router_features.py`):

| Feature Index | Feature Name | Description | Computation |
|:---:|---|---|---|
| 0 | `entropy` | SLM action distribution entropy | $H(\pi_\theta(\cdot \mid x_t)) = -\sum_v p(v) \log p(v)$ over last-token logits |
| 1 | `verifier_score_spread` | Range of verifier scores across candidates | $\max_k V_\phi(x_t, a^{(k)}) - \min_k V_\phi(x_t, a^{(k)})$ |
| 2 | `verifier_score_mean` | Mean verifier score across candidates | $\frac{1}{K}\sum_k V_\phi(x_t, a^{(k)})$ |
| 3 | `verifier_score_std` | Std dev of verifier scores | $\text{std}(\{V_\phi(x_t, a^{(k)})\}_k)$ |
| 4 | `verifier_score_best` | Best verifier score among candidates | $\max_k V_\phi(x_t, a^{(k)})$ |
| 5 | `horizon_fraction` | Fraction of episode budget consumed | $t / T_{\max}$ where $T_{\max} = 30$ |
| 6 | `step_number` | Absolute step index (normalized) | $t / 50$ |
| 7 | `normalized_context_length` | Context length as fraction of max | $\text{len}(x_t)$ / max_seq_len |
| 8 | `perturbation_tool_flakiness` | One-hot: tool flakiness active | $\mathbb{1}[\text{flakiness active}]$ |
| 9 | `perturbation_partial_obs` | One-hot: partial observability active | $\mathbb{1}[\text{partial obs active}]$ |
| 10 | `perturbation_prompt_injection` | One-hot: prompt injection active | $\mathbb{1}[\text{injection active}]$ |
| 11 | `perturbation_distractors` | One-hot: distractors active | $\mathbb{1}[\text{distractors active}]$ |
| 12 | `perturbation_none` | One-hot: no perturbation (clean) | $\mathbb{1}[\text{clean}]$ |

### Routing Labels (Ground Truth for Training)

Each step receives a binary routing label:

$$y_t = \begin{cases} 1 & \text{if SLM action quality is insufficient (should escalate)} \\ 0 & \text{if SLM action is adequate (no escalation needed)} \end{cases}$$

The label is determined by comparing the best verifier score of SLM candidates against the teacher's action score:

$$y_t = \mathbb{1}\left[\max_k V_\phi(x_t, a^{(k)}_{\text{SLM}}) < V_\phi(x_t, a_{\text{teacher}})\right]$$

### Router Objective Function (CVaR-Constrained Lagrangian)

The router's training objective is a **constrained optimization** problem solved via **primal-dual** (Lagrangian relaxation):

#### Primal Problem

$$\min_\psi \; \mathbb{E}_{(f, z)}[\text{cost}(d(f))] \quad \text{subject to} \quad \text{CVaR}_\alpha(1 - S_z) \leq \varepsilon$$

where:
- $\text{cost}(d) = c_{\text{SLM}} \cdot (1 - d) + c_{\text{LLM}} \cdot d$ with $c_{\text{SLM}} = 1, c_{\text{LLM}} = 50$
- $S_z$ is the task success indicator for perturbation seed $z$
- $\alpha = 0.2$ (CVaR focuses on the worst 20% of seeds)
- $\varepsilon = 0.3$ (maximum allowable CVaR failure rate)

#### Lagrangian Relaxation

$$\mathcal{L}(\psi, \lambda) = \underbrace{\mathbb{E}[\text{cost}(d)]}_{\text{Expected routing cost}} + \underbrace{\lambda \cdot \left(\text{CVaR}_\alpha(1 - S_z) - \varepsilon\right)}_{\text{Risk constraint penalty}} + \underbrace{\text{Brier}(r_\psi(f), y)}_{\text{Calibration loss}}$$

where:
- $\lambda \geq 0$ is a **learned Lagrange multiplier** (parameterized as $\lambda = \text{softplus}(\log\lambda_{\text{raw}})$ to enforce non-negativity)
- $\text{Brier}(p, y) = \frac{1}{N}\sum_i (p_i - y_i)^2$ is the Brier score for calibration

#### CVaR Computation

The CVaR (Conditional Value at Risk) of failure is computed by:

1. Group samples by perturbation seed $z$
2. Compute per-seed failure rate: $F_z = 1 - \text{SR}_z$
3. Sort seeds by failure rate (descending)
4. Take the top-$\alpha$ fraction (worst 20% of seeds)
5. Average those failure rates:

$$\text{CVaR}_\alpha(1 - S_z) = \frac{1}{\lceil \alpha \cdot |\mathcal{Z}| \rceil} \sum_{z \in \text{worst-}\alpha} (1 - \text{SR}_z)$$

**Implementation detail**: The code (`_compute_cvar_loss` in `router.py`) also supports a `_compute_worst_case_loss` variant that uses only the single worst seed.

#### Primal-Dual Optimization

The primal (router weights $\psi$) and dual (Lagrange multiplier $\lambda$) are optimized with **separate optimizers**:

- **Router (primal)**: AdamW, lr = $10^{-3}$, weight decay = $10^{-4}$, cosine LR schedule, gradient clip norm = 1.0
- **Lambda (dual)**: Adam, lr = $10^{-2}$ (higher LR for faster constraint adaptation)

At each training step:
1. Sample a batch of (features, labels, perturbation seeds)
2. Compute $\mathcal{L}(\psi, \lambda)$
3. Update $\psi$ via gradient **descent** on $\mathcal{L}$
4. Update $\lambda$ via gradient **ascent** on $\mathcal{L}$ (maximizing the dual)

This ensures the router minimizes cost while the multiplier $\lambda$ grows if the risk constraint is violated, automatically increasing the penalty for unsafe routing until the constraint is satisfied.

### Post-Hoc Temperature Scaling

After training, the router undergoes **post-hoc temperature scaling** on a held-out evaluation set to improve calibration:

1. Collect router logits and true labels on eval set
2. Optimize temperature $T^*$ by minimizing Negative Log-Likelihood (NLL):

$$T^* = \arg\min_T \; -\frac{1}{N}\sum_i \left[y_i \log \sigma\left(\frac{\ell_i}{T}\right) + (1-y_i) \log \left(1 - \sigma\left(\frac{\ell_i}{T}\right)\right)\right]$$

3. Solved via `scipy.optimize.minimize_scalar` on $T \in [0.1, 10.0]$
4. Report post-scaling ECE (Expected Calibration Error)

---

## 8. Training Objective Functions

### 8.1 Behavioral Cloning (BC) Loss

The SLM is first trained to imitate the teacher's actions via standard next-token prediction:

$$\mathcal{L}_{\text{BC}}(\theta) = -\mathbb{E}_{(x, a^*) \sim \mathcal{D}_{\text{teacher}}}\left[\log \pi_\theta(a^* \mid x)\right]$$

where $\mathcal{D}_{\text{teacher}}$ contains (context, action) pairs from successful teacher trajectories.

**Implementation** (`r2v/training/bc_trainer.py`):
- Cross-entropy loss from `model.forward()` with labels
- AdamW optimizer, cosine scheduler with warmup
- Gradient accumulation over 8 micro-batches

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 4 (per GPU) |
| Gradient accumulation steps | 8 |
| Effective batch size | $4 \times 8 \times 4\text{ GPUs} = 128$ |
| Learning rate | $2 \times 10^{-5}$ |
| Weight decay | 0.01 |
| Warmup ratio | 0.05 |
| Max gradient norm | 1.0 |
| LR scheduler | Cosine |
| Max sequence length | 4096 tokens |

### 8.2 Preference Optimization (DPO) Loss

After BC, the policy is refined using **Direct Preference Optimization (DPO)**, which aligns the policy toward actions preferred by the verifier without requiring reward model training:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, a^+, a^-)}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(a^+ \mid x)}{\pi_{\text{ref}}(a^+ \mid x)} - \log \frac{\pi_\theta(a^- \mid x)}{\pi_{\text{ref}}(a^- \mid x)}\right]\right)\right]$$

where:
- $a^+$ is the **chosen** (preferred) action: highest verifier-scored candidate
- $a^-$ is the **rejected** action: lowest verifier-scored candidate
- $\pi_{\text{ref}}$ is a **frozen copy** of the BC-trained policy (prevents drift)
- $\beta = 0.1$ controls the strength of the preference signal

**Implementation detail** (`r2v/training/preference_trainer.py`):
- Chosen and rejected sequences are concatenated into a single forward pass for efficiency
- Log-probabilities are computed via `policy.compute_log_probs()`, then split
- The frozen reference model is loaded as a separate copy

| Hyperparameter | Value |
|---|---|
| Epochs | 2 |
| Batch size | 2 (per GPU) |
| Gradient accumulation steps | 16 |
| Effective batch size | $2 \times 16 \times 4\text{ GPUs} = 128$ |
| Learning rate | $5 \times 10^{-6}$ |
| β (preference strength) | 0.1 |
| Num candidates per step | 8 |

### 8.3 Verifier Multi-Task Loss

The trained verifier (when used) optimizes a **multi-task objective** combining step-level and episode-level prediction:

$$\mathcal{L}_V(\phi) = w_{\text{final}} \cdot \text{BCE}(V_\phi^{\text{final}}, y_{\text{final}}) + w_{\text{step}} \cdot \text{BCE}(V_\phi^{\text{step}}, y_{\text{step}})$$

where:
- $y_{\text{final}} \in \{0, 1\}$: Whether the action led to ultimate episode success
- $y_{\text{step}} \in [0, 1]$: Step-level quality score (from teacher or heuristic)
- $w_{\text{final}} = 1.0$, $w_{\text{step}} = 0.3$

| Hyperparameter | Value |
|---|---|
| Epochs | 5 (LLM-judge mode) / 3 (trained mode, later runs) |
| Batch size | 8 → 16 → 32 (varied across runs) |
| Gradient accumulation steps | 4 |
| Learning rate | $1 \times 10^{-5}$ |
| Weight decay | 0.01 |

### 8.4 Consistency Regularization Loss

To make the policy robust to perturbation-induced observation noise, a **consistency regularizer** penalizes divergence between the policy's action distributions under different perturbation seeds:

$$\mathcal{L}_{\text{cons}}(\theta) = \mathbb{E}_{x, z, z'}\left[\text{JSD}\left(\pi_\theta(\cdot \mid x_z) \;\|\; \pi_\theta(\cdot \mid x_{z'})\right)\right]$$

where:
- $x_z, x_{z'}$ are observations of the same underlying state under different perturbation seeds
- JSD is the Jensen-Shannon Divergence (symmetric KL):

$$\text{JSD}(P \| Q) = \frac{1}{2}\text{KL}(P \| M) + \frac{1}{2}\text{KL}(Q \| M), \quad M = \frac{P + Q}{2}$$

**Implementation** (`r2v/training/consistency.py`):
- Computed over the last $N = 5$ token positions of the action distribution
- Temperature-scaled: logits divided by $T = 2.0$ before softmax (encourages smoother distributions)
- Applied as a regularization term added to the DPO loss:

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{DPO}}(\theta) + \lambda_{\text{cons}} \cdot \mathcal{L}_{\text{cons}}(\theta)$$

with $\lambda_{\text{cons}} = 0.1$

### 8.5 Router Lagrangian Loss

(Detailed in Section 7 above)

$$\mathcal{L}_{\text{router}}(\psi, \lambda) = \mathbb{E}[\text{cost}(d)] + \lambda \cdot (\text{CVaR}_\alpha(1 - S_z) - \varepsilon) + \text{Brier}(r_\psi(f), y)$$

| Hyperparameter | Value |
|---|---|
| Epochs | 20 |
| Batch size | 64 |
| Learning rate (router) | $1 \times 10^{-3}$ |
| Learning rate (λ) | $1 \times 10^{-2}$ |
| Weight decay | $1 \times 10^{-4}$ |
| CVaR α | 0.2 (worst 20% of seeds) |
| CVaR ε | 0.3 (max allowable failure) |
| Cost SLM | 1.0 |
| Cost LLM | 50.0 |

### 8.6 Combined Policy Objective

The overall policy training objective (across BC and preference stages) is:

$$\min_\theta \; \mathcal{L}_{\text{BC}}(\theta) + \lambda_{\text{pref}} \cdot \mathcal{L}_{\text{DPO}}(\theta) + \lambda_{\text{cons}} \cdot \mathcal{L}_{\text{cons}}(\theta)$$

with $\lambda_{\text{pref}} = 0.5$ and $\lambda_{\text{cons}} = 0.1$.

Note: In practice, BC is trained first to convergence, then DPO + consistency are trained jointly (the policy starts from the BC checkpoint). The verifier and router are trained separately afterward.

---

## 9. Data Construction Pipeline

### 9.1 Teacher Trajectory Collection

1. **Dataset**: GAIA (princeton-nlp/GAIA_Verified, test split — 500 human-validated instances)
2. **Agent**: Gemini 3 Flash teacher solves tasks in Docker containers
3. **Dockerized execution**: Each task runs in an isolated container (timeout: 600s, 8GB memory, 4 CPUs)
4. **Output**: 500 teacher trajectories stored as JSONL (each trajectory is a sequence of steps with observations, actions, and metadata)
5. **Filtering**: Only successful trajectories (where the teacher's patch resolves the issue) are retained for BC training

### 9.2 Perturbation Generation

Each trajectory is replayed under 20 different perturbation seeds ($|\mathcal{Z}| = 20$), each applying a random combination of 4 perturbation types:

#### Perturbation Type 1: Tool Flakiness
Simulates unreliable tool outputs in software engineering environments:

| Sub-perturbation | Probability |
|---|---|
| Tool failure (complete) | 0.10 |
| Tool timeout | 0.05 |
| Stale cache results | 0.10 |
| Result shuffling | 0.20 |
| Partial response | 0.10 |
| Result drop (10-50% of content) | 0.10–0.50 |
| Flaky test results | 0.15 |
| Dependency drift | 0.10 |

#### Perturbation Type 2: Partial Observability
Simulates incomplete or corrupted observations:

| Sub-perturbation | Probability |
|---|---|
| DOM element hiding | 0.15 |
| DOM reordering | 0.10 |
| Attribute stripping | 0.10 |
| Log truncation | 0.20 |
| Log line dropping | 0.15 |
| Information masking | 0.05 |
| Stack trace truncation | 0.15 |
| File content truncation | 0.10 |

#### Perturbation Type 3: Prompt Injection
Simulates adversarial or misleading content:

| Sub-perturbation | Probability |
|---|---|
| Direct injection | 0.15 |
| Indirect injection | 0.10 |
| Goal hijacking | 0.05 |
| Exfiltration attempt | 0.05 |
| Role confusion | 0.05 |
| Encoded injection | 0.03 |
| Misleading error messages | 0.15 |
| Misleading code comments | 0.10 |

#### Perturbation Type 4: Distractors
Simulates plausible but incorrect information:

| Sub-perturbation | Probability |
|---|---|
| Semantic distractors | 0.20 |
| Red herring elements | 0.15 |
| Decoy elements | 0.10 |
| Plausible wrong answers | 0.10 |
| Similar filepath (wrong file) | 0.15 |
| Plausible wrong fix | 0.15 |
| Decoy error messages | 0.10 |

### 9.3 Step Labeling

The labeling pipeline (`r2v/data/labeling.py`) assigns ground-truth quality labels:

**GAIA Labeler** checks:
- Whether the action correctly identifies and modifies the gold files
- Whether tests pass after the action
- Pattern matching for common fix strategies
- Falls back to LLM-judge for ambiguous cases

**WebArena Labeler** checks:
- Pattern-based progress detection
- Safety violation checking (unauthorized actions)
- DOM state changes
- LLM-judge fallback

### 9.4 Router Feature Generation

The feature generation pipeline (`scripts/generate_router_features.py`) is the most compute-intensive data preparation step:

1. **Load the BC-trained SLM** and verifier onto GPUs
2. For each trajectory step:
   a. Run SLM to get action distribution → compute **entropy** from last-token logits
   b. Generate $K$ candidate actions via temperature sampling
   c. Score all candidates with the verifier → compute **spread, mean, std, best**
   d. Record **step number**, **horizon fraction**, **context length**
   e. Encode the **perturbation type** as a 5-dimensional one-hot vector
3. **Multi-GPU sharding**: Trajectories are distributed across GPUs via modular assignment
4. **Resume support**: Skips already-processed steps via checkpoint files
5. **GPU keepalive**: Background thread sends periodic dummy tensors to prevent idle GPU timeouts

### 9.5 Dataset Classes

| Dataset Class | Purpose | Source |
|---|---|---|
| `BCDataset` | BC training | Successful teacher trajectories |
| `PreferenceDataset` | DPO training | Verifier-scored candidate pairs ($a^+$, $a^-$) |
| `ConsistencyDataset` | Consistency regularization | Paired observations under different seeds |
| `VerifierDataset` | Verifier training | (context, action, quality_label) tuples |
| `RouterDataset` | Router training | (13-dim features, binary label, seed) tuples |

All datasets use dynamic-padding collation functions for efficient batching (2-3× throughput improvement over fixed-length padding).

---

## 10. Training Pipeline

The full training pipeline consists of **9 sequential stages**, managed by the `JointTrainer` class (`r2v/training/joint_trainer.py`). The pipeline is **checkpoint-based and resumable** — each stage writes a completion marker, and re-runs skip completed stages.

```
Stage 1: collect_trajectories      → Teacher Gemini 3 Flash solves tasks
Stage 2: generate_perturbations    → Apply 20 perturbation seeds
Stage 3: label_steps               → Assign quality labels
Stage 4: train_bc                  → SLM behavioral cloning
Stage 5: train_verifier            → Train/validate verifier
Stage 6: generate_candidates       → SLM generates K candidates per step
Stage 7: train_preference          → DPO with consistency regularization
Stage 8: train_router_features     → Generate 13-dim feature vectors
Stage 9: train_router              → CVaR-constrained Lagrangian training
```

### Stage-by-Stage Details

**Stage 1 – Collect Trajectories**:
- Gemini 3 Flash teacher solves GAIA tasks
- 500 trajectories collected, max 20 steps each
- Stored as JSONL with full observation/action/metadata

**Stage 2 – Generate Perturbations**:
- Each trajectory replayed under 20 random seeds
- Each seed activates a stochastic combination of the 4 perturbation types
- Produces perturbed observations while preserving ground-truth actions

**Stage 3 – Label Steps**:
- GAIA labeler assigns step-level and episode-level quality scores
- Uses gold file matching, test execution detection, and LLM-judge fallback

**Stage 4 – Train BC**:
- Train SLM (Llama-3.1-8B + LoRA) on teacher demonstrations
- 3 epochs, effective batch size 128
- Uses HF Accelerate + DeepSpeed Stage 2

**Stage 5 – Train Verifier**:
- Train or validate the verifier model
- In LLM-judge mode: validates the judge produces meaningful scores
- In trained mode: multi-task BCE training on step + final labels

**Stage 6 – Generate Candidates**:
- BC-trained SLM generates 8 candidates per trajectory step
- Temperature sampling (T=0.7, top_p=0.95)
- Each candidate scored by the verifier

**Stage 7 – Train Preference (DPO + Consistency)**:
- DPO pairs constructed from best/worst candidates (by verifier score)
- Consistency regularization integrated: paired observations under different seeds
- 2 epochs, effective batch size 128

**Stage 8 – Generate Router Features**:
- Run SLM + verifier over all trajectory steps
- Produce 13-dimensional feature vectors
- Multi-GPU compute-intensive step

**Stage 9 – Train Router**:
- 80/20 train/eval split
- 20 epochs of primal-dual optimization
- Post-hoc temperature scaling on eval set
- Save best model by eval Brier score (evaluated every 5 epochs)

---

## 11. Inference Procedure

At inference time, the `R2VAgent` class (`r2v/agent/r2v_agent.py`) executes a **4-step decision loop** at each step $t$:

### Step 1: Generate Candidates

The SLM generates $K = 4$ candidate actions:

$$\{a^{(1)}_t, \ldots, a^{(K)}_t\} \sim \pi_\theta(\cdot \mid x_t)$$

using temperature sampling (T=0.7, top_p=0.95).

### Step 2: Verify Candidates

The verifier scores all candidates:

$$s^{(k)}_t = V_\phi(x_t, a^{(k)}_t), \quad k = 1, \ldots, K$$

Select the best candidate: $a^*_t = \arg\max_k s^{(k)}_t$

### Step 3: Route

Compute the risk signal:

$$\rho(x_t) = 0.7 \cdot (1 - V_\phi(x_t, a^*_t)) + 0.3 \cdot \min\left(\frac{H}{5}, 1\right)$$

where:
- $1 - V_\phi(x_t, a^*_t)$ is the **quality risk** (how uncertain the verifier is about the best candidate)
- $\min(H/5, 1)$ is the **horizon risk** (urgency increases as more steps are consumed)
- The weights (0.7, 0.3) balance quality vs. urgency

The router also computes the full 13-dim feature vector and produces $p_t = r_\psi(f_t)$.

**Escalation decision**: If $\rho(x_t) > \tau_{\text{accept}} = 0.7$ **and** the LLM call budget is not exhausted:

$$d_t = \begin{cases} 1 & \text{if } \rho(x_t) > 0.7 \text{ and LLM calls} < 10 \\ 0 & \text{otherwise} \end{cases}$$

### Step 4: Execute (with Self-Correction)

- If $d_t = 0$: Execute the best SLM candidate $a^*_t$
- If $d_t = 1$: Query the teacher LLM for an action $a_t \sim \pi_T(\cdot \mid x_t)$

**Self-correction loop** (up to 2 iterations):
If the verifier scores the chosen action below the acceptance threshold, the agent:
1. Generates new candidates
2. Re-scores with the verifier
3. Selects the best from the combined pool

### Inference Budget

| Parameter | Value |
|---|---|
| Candidates per step ($K$) | 4 |
| Max LLM calls per episode | 10 |
| Max self-correction iterations | 2 |
| Accept threshold ($\tau$) | 0.7 |
| Step limit per episode | 30 |

### Baseline Agents

Three baseline agents are implemented for comparison:

1. **`LLMOnlyAgent`**: Always uses the teacher LLM (upper bound on quality, maximum cost)
2. **`SLMOnlyAgent`**: Always uses the SLM without routing (lower bound on cost, baseline quality)
3. **`EntropyRouterAgent`**: Routes based solely on SLM output entropy — escalates to LLM when entropy exceeds a threshold (no verifier, no CVaR constraint)

---

## 12. Compute Infrastructure

| Resource | Specification |
|---|---|
| GPUs | 4× (type determined by cluster) |
| Precision | `bfloat16` (bf16) |
| Distributed strategy | DeepSpeed Stage 2 |
| Gradient checkpointing | Enabled |
| Flash Attention | Flash Attention 2 |
| Random seed | 42 |
| Logging | Weights & Biases (wandb) |

### DeepSpeed Stage 2

- **Optimizer state partitioning**: Each GPU holds 1/4 of the optimizer states
- **Gradient partitioning**: Gradients are reduced and partitioned across GPUs
- **Parameters remain replicated**: Full model on each GPU for forward pass

### Memory Optimization Chain

1. LoRA reduces trainable parameters from ~8B to ~67M (0.84%)
2. 4-bit quantization (when available) reduces model memory by ~4×
3. Gradient checkpointing trades ~30% compute for ~60% memory savings
4. Dynamic padding eliminates wasted compute on pad tokens
5. Flash Attention 2 reduces attention memory from O(n²) to O(n)

---

## 13. Code-to-Theory Mapping

| Theoretical Concept | Code Location |
|---|---|
| SLM Policy $\pi_\theta$ | `r2v/models/policy.py` → `PolicyModel` |
| Teacher LLM $\pi_T$ | `r2v/models/llm_client.py` → `LLMClient` |
| Verifier $V_\phi$ | `r2v/models/verifier.py` → `LLMJudgeVerifier`, `TrainedVerifier` |
| Router $r_\psi$ | `r2v/models/router.py` → `Router` |
| Router Loss (CVaR + Lagrangian) | `r2v/models/router.py` → `RouterLoss` |
| Temperature Scaling | `r2v/models/router.py` → `TemperatureScaling` |
| BC Training | `r2v/training/bc_trainer.py` → `BCTrainer` |
| DPO Training | `r2v/training/preference_trainer.py` → `PreferenceTrainer` |
| Consistency Loss | `r2v/training/consistency.py` → `ConsistencyRegularizer` |
| Verifier Training | `r2v/training/verifier_trainer.py` → `VerifierTrainer` |
| Router Training | `r2v/training/router_trainer.py` → `RouterTrainer` |
| 9-Stage Pipeline | `r2v/training/joint_trainer.py` → `JointTrainer` |
| Feature Generation | `scripts/generate_router_features.py` |
| Inference Agent | `r2v/agent/r2v_agent.py` → `R2VAgent` |
| Inference Budget | `r2v/agent/budget.py` → `InferenceBudget` |
| Data Structures | `r2v/data/trajectory.py` → `Episode`, `Step`, `Observation`, `Action` |
| All Datasets | `r2v/data/datasets.py` → `BCDataset`, `PreferenceDataset`, etc. |
| Labeling | `r2v/data/labeling.py` → `GAIALabeler`, `WebArenaLabeler` |
| Evaluation Metrics | `r2v/evaluation/metrics.py` → `R2VEvaluator` |
| Robustness Metrics | `r2v/evaluation/robustness.py` → `compute_cvar_failure`, etc. |
| Calibration | `r2v/evaluation/calibration.py` → `compute_ece`, `compute_brier` |
| Statistical Tests | `r2v/evaluation/statistical.py` → `bootstrap_ci`, `paired_mcnemar_test` |
| Noisy Config | `configs/gaia/noisy.yaml` |
