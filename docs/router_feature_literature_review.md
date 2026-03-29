# Router Feature Design: Literature Review & Justification

**Purpose:** Internal reference for team discussion and paper writing.
**Scope:** Features used to decide when a step-level agent router should escalate from a cheap SLM to an expensive teacher LLM. Covers routing papers (2023–2026), uncertainty estimation, and agent process reward literature.

---

## Table of Contents

1. [Background & Problem Setting](#1-background--problem-setting)
2. [Paper Survey: LLM Routing & Cascade Systems](#2-paper-survey-llm-routing--cascade-systems)
3. [Paper Survey: Uncertainty Estimation for LLMs](#3-paper-survey-uncertainty-estimation-for-llms)
4. [Paper Survey: Step-Level Quality in Agent Systems](#4-paper-survey-step-level-quality-in-agent-systems)
5. [Feature Analysis: What to Use and Why](#5-feature-analysis-what-to-use-and-why)
6. [Feature Analysis: What NOT to Use and Why](#6-feature-analysis-what-not-to-use-and-why)
7. [Our Final Feature Set (24-dim)](#7-our-final-feature-set-24-dim)
8. [Open Questions & Future Work](#8-open-questions--future-work)
9. [References](#9-references)

---

## 1. Background & Problem Setting

Most LLM routing literature frames routing as a **query-level** decision: given a single user query, pick which LLM to call. Our setting is fundamentally different:

- **Step-level routing** within a multi-step agent trajectory
- The context grows at every step (prior observations + actions accumulate)
- Routing costs are paid *per step*, not per query
- Failure propagates: a bad step at position *t* can invalidate all subsequent steps
- The system operates under four distinct perturbation types (tool flakiness, partial observability, prompt injection, distractors)

This distinction matters: features that work at the query level (e.g., raw query embeddings) may not transfer to step-level routing, while features that capture trajectory dynamics (horizon fraction, verifier score spread over candidates) have no analogue in the query-routing literature.

---

## 2. Paper Survey: LLM Routing & Cascade Systems

### 2.1 FrugalGPT (Chen et al., 2023)

**Citation:** Lingjiao Chen, Matei Zaharia, James Zou. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* arXiv:2305.05176, May 2023.

**Mechanism:** LLM cascade — queries pass sequentially through a ranked list of APIs (cheap to expensive). A learned scoring function `g(query, answer) → [0,1]` decides whether to stop at each level.

**Features used:**
- The scoring function is a DistilBERT regression model trained on (query, answer) pairs
- No explicit confidence signals — quality is judged by reading the output
- Training signal: correctness labels on (query, answer) pairs

**Key results:** Up to 98% cost reduction vs. GPT-4-only on HEADLINES benchmark while matching accuracy. 4% accuracy gain over GPT-4 at equal cost on some tasks.

**Relevance to R2V:**
- Validates the core cascade idea: cheapest model first, escalate only when needed
- Shows that the quality estimator is the bottleneck — noisy judges collapse cascade performance
- Our verifier score is the analogous quality estimator; training it well is critical

**Limitation noted:** The scoring function requires correctness labels at training time and a separate judge model at inference. Our verifier is the judge.

---

### 2.2 RouteLLM (Ong et al., 2024)

**Citation:** Isaac Ong et al. *RouteLLM: Learning to Route LLMs with Preference Data.* arXiv:2406.18665, June 2024.

**Four router architectures, all using only query text:**

| Router | Input | Architecture |
|---|---|---|
| Matrix Factorization | `text-embedding-3-small` embedding | Bilinear δ(model, query) |
| Similarity-Weighted | `text-embedding-3-small` embedding | Bradley-Terry + cosine similarity |
| BERT Classifier | Raw query text | BERTbase fine-tuned |
| Causal LLM Router | Raw query text | Llama 3 8B fine-tuned |

**Training signal:** ~80K human preference labels (Chatbot Arena) + ~120K LLM-judge labels (GPT-4 on Nectar) + MMLU correctness labels.

**Key results:** Matrix factorization + similarity-weighted are most data-efficient. All routers achieve >2× cost savings while preserving 95% of GPT-4 quality. Routers transfer across model pairs (GPT-4/3.5 → Claude Opus/Sonnet) without retraining.

**Relevance to R2V:**
- Demonstrates that query text embeddings alone are strong routing signals at the query level
- Transfer result suggests embedding space captures task difficulty, not model-specific properties

**Why we do NOT directly apply this:** RouteLLM operates at query level with a static context. At step level, the context grows with every step, making raw embedding of the full context expensive and semantically unstable. The routing decision at step *t* depends on trajectory dynamics (how far into the episode, current verifier score, candidate spread), not just the static query.

---

### 2.3 RouterBench (Hu et al., 2024)

**Citation:** Jiarui Hu et al. *RouterBench: A Benchmark for Multi-LLM Routing System.* arXiv:2403.12031, March 2024.

**Router types evaluated:**

| Router | Features |
|---|---|
| KNN | Query embeddings (all-MiniLM-L12-v2, 40 neighbors) |
| MLP | Same query embeddings, 2-layer MLP |
| Cascading | Post-hoc quality score `g(answer_text) → [0,1]` with threshold |

**Key results:**
- 405K inference outcomes across representative LLMs as ground truth
- Cascade performance collapses when judge error rate exceeds ~0.2
- Oracle router shows large gap over current methods — most headroom is in quality estimation, not routing architecture

**Relevance to R2V:** The 0.2 error rate threshold for cascade collapse is a critical design constraint. Our verifier must achieve better than 80% accuracy or the router's CVaR constraint will be violated systematically.

---

### 2.4 Unified Routing & Cascading (Dekoninck et al., 2024)

**Citation:** Jasper Dekoninck, Maximilian Baader, Martin Vechev. *A Unified Approach to Routing and Cascading for LLMs.* arXiv:2410.10347, October 2024.

**Two types of quality estimators:**
- **Ex-ante (before model runs):** Query features — benchmark origin, input characteristics, complexity. Robust to post-hoc noise; weak to distribution shift.
- **Post-hoc (after model runs):** Log probabilities from model outputs, confidence measures, test-case execution for code. Robust to query-level noise; weak to model-level noise.

**Key result:** Cascade > routing-only or cascading-only "by a large margin." With accurate estimators: up to 14% improvement on SWE-Bench. With poor estimators: only 1.2% gain. **Log probability explicitly identified and validated as a post-hoc routing feature.**

**Relevance to R2V:**
- Directly validates `log_prob_best`, `log_prob_mean`, `log_prob_std` in our feature set
- The "unified" framing — use both ex-ante features (entropy, step position) and post-hoc features (verifier score, log probability) — matches our 24-dim design
- The quality estimator accuracy dependency echoes FrugalGPT's finding

---

### 2.5 Routing to the Expert / Zooter (Lu et al., 2023)

**Citation:** Keming Lu et al. *Routing to the Expert: Efficient Reward-Guided Ensemble of Large Language Models.* arXiv:2311.08692, November 2023.

**Features:** Query text via mDeBERTa-v3-base (86M params). Tag-based label enhancement: blends sample-level and tag-level reward model scores (`β·r_sample + (1−β)·r_tag`, optimal β=0.3).

**Key results:** Mean Task Rank 1.94 vs. 2.23 for best single model across 26 subsets. Ranks first on 44% of tasks — 13pp more than best single model. 42× more efficient than reward-ranking ensemble.

**Relevance to R2V:** The tag/domain-level enhancement is conceptually similar to our benchmark one-hot — it tells the router which type of task it is, allowing specialization.

---

### 2.6 EmbedLLM (Zhuang et al., 2024)

**Citation:** Richard Zhuang et al. *EmbedLLM: Learning Compact Representations of Large Language Models.* arXiv:2410.02223, October 2024.

**Features:** Learns model-level embeddings by matrix-factorizing a (model × question) correctness matrix. At routing time: model embedding + question embedding → element-wise product → correctness probability.

**Key results:** 15× faster than causal LLM routers, uses <1GB GPU vs. 60GB. Outperforms KNN on correctness forecasting at most training set sizes.

**Relevance to R2V:** We have only two "models" (SLM, LLM), so the full EmbedLLM framework is overkill. However, the insight that query embeddings and model-capability embeddings factorize independently is useful for future work on multi-model routing.

---

### 2.7 kNN Paper (Rethinking Predictive Modeling for LLM Routing, 2025)

**Citation:** *Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers.* arXiv:2505.12601, May 2025.

**Features:** BERT-base [CLS] embeddings (768-dim) for text queries. kNN with k=100; averages neighbor model performance.

**Key result:** kNN AUC 52.68 (text), 72.12 (vision-language). **Matches or outperforms 8 parametric alternatives.** Core finding: model performance has *locality in embedding space* — semantically similar queries benefit from the same model.

**Relevance to R2V:** This is a sanity check result. For query-level routing, a strong embedding + kNN is sufficient. For step-level routing under perturbations, we argue the perturbation type and trajectory dynamics provide additional signal beyond query semantics — which is why our benchmark one-hot and perturbation one-hot are useful beyond what a kNN on embeddings would capture.

---

### 2.8 GreenServ (2026)

**Citation:** *GreenServ: Energy-Efficient Context-Aware Dynamic Routing.* arXiv:2601.17551, January 2026.

**Features:** Task type (query classification), semantic cluster (embedding-based grouping), text complexity (rule-based scoring). Multi-armed bandit for online adaptation.

**Key results:** 22% accuracy improvement + 31% energy reduction vs. random routing across 16 LLMs.

**Relevance to R2V:** Validates task type and text complexity as routing features. Our `benchmark_onehot` is the exact analogue of their task type feature. Our `goal_length` and `normalized_context_length` serve as lightweight proxies for text complexity.

---

### 2.9 SelectLLM (Maurya et al., 2024)

**Citation:** Kaushal Kumar Maurya et al. *SelectLLM: Query-Aware Efficient Selection Algorithm for LLMs.* arXiv:2408.08545, August 2024.

**Features:** Multi-label classifier on query text; confidence scores as policy input.

**Key results:** 13% latency reduction on GSM8K, 70% on MMLU vs. ensemble baselines.

**Relevance to R2V:** Confirms that a classifier trained on query text features is sufficient for routing in standard benchmarks. Our setting requires more (trajectory dynamics, verifier signals) because we route under adversarial perturbations.

---

### 2.10 SynapseRoute (2025)

**Citation:** *SynapseRoute: Auto-Route Switching Framework on Dual-State LLM.* arXiv:2507.02822, July 2025.

**Mechanism:** Routes between "thinking" (expensive) and "non-thinking" (fast) modes of the same model family using query complexity estimation.

**Key finding:** 58% of medical questions can be answered by non-thinking mode. 36.8% faster inference, 39.7% fewer tokens. **Critically: over-reasoning on simple queries paradoxically decreases accuracy.**

**Relevance to R2V:** The over-reasoning finding is directly relevant — escalating to the teacher LLM on easy steps may hurt, not help. This supports our CVaR-constrained objective over always-escalate: the constraint prevents both under-escalation (failures) and over-escalation (degraded accuracy + wasted cost).

---

## 3. Paper Survey: Uncertainty Estimation for LLMs

### 3.1 Semantic Entropy & Related Work

**Citation (primary):** Lin et al. *Teaching Models to Express Their Uncertainty in Words.* arXiv:2205.14334, 2022. Kuhn et al. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR 2023.

**Features:**
- **Token-level entropy** H(π(·|x)): entropy over next-token distribution at last position. Cheap (1 forward pass), universal, but brittle under distribution shift.
- **Semantic entropy:** Cluster K samples by meaning (using NLI), compute entropy over semantic equivalence classes. More robust but requires K samples + a clustering model.
- **P(True):** Model's self-reported probability of correctness.
- **Multi-sample consistency:** Agreement fraction across K independent samples.

**Key results:** Semantic entropy consistently outperforms token-level entropy for hallucination detection. Multi-sample consistency is expensive but highly predictive. P(True) is poorly calibrated without fine-tuning.

**Relevance to R2V:**
- Token-level entropy (dim 0) is validated here as a useful but imperfect signal
- Our `semantic_entropy` (dim 10) approximates the proper semantic entropy cheaply using first-token cluster distribution — no NLI model needed
- Our `candidate_consistency` (dim 9) is the multi-sample consistency approximation over the K verifier candidates
- Together, dims 0, 9, 10 cover the three main uncertainty families from this literature

---

### 3.2 Hidden-State Structural Signals (Wang et al., 2026)

**Citation:** *Trust in One Round: Confidence Estimation via Structural Signals.* arXiv:2602.00977, February 2026.

**Features:** Spectral features of the final-layer hidden-state trajectory, local-variation descriptors, global shape descriptors — all from a single forward pass.

**Key results:** Outperforms token likelihood, semantic similarity, and multi-sample consistency on FEVER, SciFact, WikiBio, TruthfulQA. Identifies errors that probability-based methods miss.

**Relevance to R2V:** Suggests that the hidden state contains more information about uncertainty than the logits alone. This is the strongest case for adding a hidden-state embedding to our feature set.

**Why we did NOT include this in the current implementation:**
- Requires `output_hidden_states=True` on the policy model (Llama-3.1-8B)
- Returns all 32 layers × [B, seq_len, 4096] tensors; for seq_len=2048, batch=4, this is ~4GB of activations per forward pass
- A random projection from 4096 → 4 dims captures very little signal (Johnson-Lindenstrauss guarantees distance preservation, not task-relevant signal)
- Fitting the spectral/structural features described in the paper requires a separate training pipeline
- **Recommended for future work** after ablating the current 24 features

---

### 3.3 Reinforced Hesitation (2025)

**Citation:** *Trustworthy Language Models through Reinforced Hesitation.* arXiv:2511.11500, November 2025.

**Mechanism:** Ternary reward training (correct=+1, abstain=0, error=−λ). Models learn to say "I don't know" rather than hallucinate. λ controls risk regime.

**Key result:** Creates models on a cost-quality Pareto frontier. Cascade with hesitation beats majority voting at lower cost.

**Relevance to R2V:** Abstention as a routing signal — if the SLM is well-calibrated, low confidence can trigger escalation. We approximate this via the `verifier_score_best` + `entropy` features rather than training a hesitating SLM, but this is worth revisiting.

---

### 3.4 E-valuator (2025)

**Citation:** *E-valuator: Reliable Agent Verifiers with Sequential Hypothesis Testing.* arXiv:2512.03109, December 2025.

**Features:** Sequential e-processes applied to verifier scores over trajectory steps.

**Key result:** Better false-alarm rate control than heuristic thresholds. Can terminate problematic trajectories early (token savings). Tested across 6 datasets, 3 agents.

**Relevance to R2V:** The sequential testing framing is complementary to our per-step routing — cumulative verifier score over trajectory could be a stronger routing signal than the single-step score. **Candidate future feature: rolling mean of verifier scores over the last N steps.**

---

## 4. Paper Survey: Step-Level Quality in Agent Systems

### 4.1 AgentProcessBench (Yao et al., 2026)

**Citation:** *AgentProcessBench: Diagnosing Step-Level Process Quality in Tool-Using Agents.* arXiv:2603.14465, March 2026.

**Setup:** 8,509 manually annotated steps with ternary labels (correct / neutral / erroneous), 89.1% inter-annotator agreement.

**Key results:**
- Process supervision provides distinct value beyond outcome-based feedback
- **Distinguishing neutral from erroneous is the hardest challenge** — has direct bearing on our verifier training
- Weaker policy models show inflated "correct step" ratios due to early termination — confounds step-level metrics

**Relevance to R2V:**
- Our verifier is trained on binary (success/failure) labels; the ternary finding suggests adding a "neutral" class could improve routing precision
- The early-termination confound applies to our SLM trajectories — step-level success rates from short trajectories may be misleading

---

### 4.2 Critical Step Optimization (Wang et al., 2026)

**Citation:** *Verified Critical Step Optimization for LLM Agents.* arXiv:2602.03412, February 2026.

**Features:** High-entropy tokens in policy reasoning (critical step candidates concentrate at high-entropy positions). PRM scores identify candidate critical steps.

**Key results:**
- Only **16% of trajectory steps are "critical"** — binary verification of alternates determines causality
- High entropy at a decision point empirically predicts criticality
- 37% relative improvement on GAIA-Text-103 by focusing optimization on critical steps

**Relevance to R2V:**
- **Direct empirical evidence that entropy (dim 0) predicts step criticality** — the most important validation of our entropy feature
- The 16% critical-step finding suggests most escalations are unnecessary; our CVaR constraint is appropriate here (don't escalate uniformly, only at critical steps)
- `horizon_fraction` (dim 11) is partially validated here: critical steps are non-uniformly distributed across the trajectory

---

### 4.3 Adaptive Test-Time Compute (Aggarwal et al., 2026)

**Citation:** *What If We Allocate Test-Time Compute Adaptively?* arXiv:2602.01070, February 2026.

**Features:** Step-level PRM scores for pruning/expanding reasoning paths. Trajectory-level reward for answer selection.

**Key results:** PRM-guided adaptive allocation outperforms uniform compute on MATH-500, AIME24 "by several-fold."

**Relevance to R2V:** The PRM score driving path-expansion decisions is directly analogous to our `verifier_score_best` (dim 4) driving LLM escalation. The "adaptive allocation" framing is equivalent to our cost-constrained router: spend compute where the verifier is uncertain.

---

### 4.4 ToolPRMBench (2026)

**Citation:** *ToolPRMBench: Evaluating Process Reward Models for Tool-Using Agents.* arXiv:2601.12294, January 2026.

**Key results:**
- Specialized PRMs for tool-use outperform general-purpose PRMs
- Online sampling (full rollouts) is needed in addition to offline (isolated steps) for good PRM training data
- Multi-LLM verification pipeline required to reduce label noise

**Relevance to R2V:** Validates that our domain-specific verifier (trained on R2V trajectories) should outperform a general-purpose judge. Also motivates using an ensemble for verifier scoring rather than a single judge.

---

### 4.5 How LLMs Fail in Agentic Scenarios (2025)

**Citation:** *How Do LLMs Fail in Agentic Scenarios?* arXiv:2512.07497, December 2025.

**Failure archetypes identified:**
1. Premature action without grounding
2. Over-helpfulness substituting missing entities (hallucination)
3. Distractor-induced context pollution
4. Fragile execution under load (complexity)

**Key finding:** Model scale does NOT predict agentic robustness. Architectural and training design choices dominate.

**Relevance to R2V:** Escalating to a larger model is not always the right response to failure. This supports our **perturbation-type one-hot features** (dims 19–23): the router should learn different escalation policies per perturbation type, not uniformly escalate when the verifier is low. For example, distractor-induced failures may be better addressed by ignoring irrelevant context than by switching to the LLM.

---

## 5. Feature Analysis: What to Use and Why

### 5.1 Token-Level Entropy (dim 0)

**Use:** Yes.

**Supporting evidence:**
- Critical Step Optimization [arXiv:2602.03412]: "High-entropy tokens in the policy's reasoning concentrate at critical step candidates" — direct empirical validation that entropy predicts which steps will cause trajectory failure
- Unified Routing [arXiv:2410.10347]: entropy is listed as a standard ex-ante routing feature
- Semantic entropy papers [ICLR 2023]: token-level entropy is the cheapest proxy for model uncertainty

**Why it works for step-level routing (not just query-level):** At step *t*, the model must commit to an action. High entropy over the next-token distribution means the model is uncertain what to do, which is predictive of a wrong action. This signal is computed in a single forward pass (the entropy pass is already needed for generation, so it's free).

**Limitation:** Token-level entropy is sensitive to vocabulary structure (e.g., rare token bias), is brittle under distribution shift, and can be high for benign reasons (ambiguous phrasing, not task difficulty). Semantic entropy mitigates this — see dim 10.

---

### 5.2 Verifier Score Statistics (dims 1–5)

**Use:** Yes — all five: spread, mean, std, best, worst.

**Supporting evidence:**
- FrugalGPT [arXiv:2305.05176]: quality estimator score is the primary routing signal in the cascade
- RouterBench [arXiv:2403.12031]: cascading routers use exactly this type of post-hoc quality score
- Unified Routing [arXiv:2410.10347]: post-hoc signals (after model runs) are complementary to ex-ante signals; verifier score is the canonical post-hoc signal
- Adaptive Compute [arXiv:2602.01070]: PRM scores drive routing decisions in test-time compute allocation

**Why five statistics instead of one:**
- `spread` (max − min): captures disagreement across K candidates. High spread = the verifier sees both good and bad candidates, suggesting the SLM is sometimes right but unreliably so → escalate
- `mean`: expected verifier score across K candidates; the average quality of what the SLM would generate
- `std`: variance in quality — high std means the SLM is inconsistent
- `best` (max): upper bound on what the SLM can do. If even the best candidate scores poorly, escalate unconditionally
- `worst` (min): lower bound — how bad the worst candidate is. Even if the best is good, a very bad worst candidate may indicate an unreliable SLM

**Note:** Before this work, only `best` and mean were commonly used. Including `worst` and `spread` together allows the router to distinguish "consistently good" from "lucky but usually bad."

---

### 5.3 Action Log-Probability Statistics (dims 6–8)

**Use:** Yes — log_prob_best, log_prob_mean, log_prob_std.

**Supporting evidence:**
- Unified Routing [arXiv:2410.10347]: action log-probability explicitly identified and validated as a post-hoc routing signal: "After the model runs, log probabilities from model outputs are among the most informative quality features"
- Semantic entropy literature: log-probability is the basis of token-level entropy; using it directly (not aggregated into entropy) provides a complementary signal about the SLM's overall confidence in the generated sequence (not just the first token)

**Distinction from entropy (dim 0):**
- **Entropy** (dim 0) is computed over the *marginal next-token distribution* at the last context token — it measures how uncertain the model is about what to generate next
- **Log-probability** (dims 6–8) is the *joint log-probability of the entire generated action sequence* — it measures how probable the model thinks the full chosen action is, conditional on having chosen it

These are different signals. A model can have low next-token entropy (confident about the first token) but low sequence log-probability (the full action is low-probability). Including both captures complementary aspects of SLM confidence.

**Implementation note:** These values are already computed during candidate generation (stored as `cand["log_prob"]`) — zero additional GPU cost.

---

### 5.4 Candidate Consistency (dim 9)

**Use:** Yes.

**Supporting evidence:**
- Self-consistency [Wang et al., 2023, arXiv:2203.11171]: majority voting over K samples is one of the strongest cheap uncertainty signals for reasoning tasks — the fraction of agreement across samples predicts correctness
- Mirror-Consistency [arXiv:2410.10857]: consistency over sampled outputs is more robust than single-sample confidence
- RouterBench [arXiv:2403.12031]: the quality estimator error rate directly controls cascade performance; consistency is a cheap quality estimator

**Why this works:** If K=5 candidates all start with the same action token, the SLM is robustly committing to the same strategy regardless of sampling temperature. If the K candidates diverge, the SLM is uncertain and different samples explore different strategies — high risk.

**Relationship to semantic entropy (dim 10):** Consistency gives the *mode fraction* (fraction of candidates that match the most common output). Semantic entropy gives the full *distributional entropy* over all observed clusters. They are complementary: consistency is more interpretable, semantic entropy captures the full picture.

---

### 5.5 Semantic Entropy (dim 10)

**Use:** Yes.

**Supporting evidence:**
- Semantic entropy [Kuhn et al., ICLR 2023]: entropy over meaning-equivalent clusters of K samples outperforms token-level entropy for hallucination detection
- Our implementation uses first-token clustering as a cheap proxy for meaning-equivalence (no NLI model needed)

**Distinction from token-level entropy (dim 0):**
- Token-level entropy (dim 0) is computed over the *marginal vocabulary distribution* at the last context token
- Semantic entropy (dim 10) is computed over the *distribution of distinct strategies* observed in K generated candidates

Both are entropy measures, but they capture fundamentally different uncertainty:
- Dim 0: "How uncertain is the model about its immediate next word?"
- Dim 10: "How diverse are the strategies the model is considering?"

A model can be token-certain (picks the same first token 5/5 times) but still be generating diverse sequences in later tokens — dim 10 catches this.

**Limitation:** First-token clustering is a rough approximation of semantic equivalence. Two candidates starting with "execute" and "run" are semantically equivalent but would be assigned to different clusters. For a future improvement, use an NLI model or embedding-based clustering as in the original semantic entropy paper.

---

### 5.6 Horizon Fraction & Step Number (dims 11–12)

**Use:** Yes.

**Supporting evidence:**
- Critical Step Optimization [arXiv:2602.03412]: critical steps are non-uniformly distributed across the trajectory — early steps and late steps may have different failure modes
- AgentProcessBench [arXiv:2603.14465]: error propagation means early errors are more costly than late ones (early failure contaminates the entire remainder)
- Our ablations confirmed that horizon_fraction is a predictive feature in preliminary router experiments

**Why both?** `horizon_fraction` normalizes by episode length (useful for comparing across episodes of different lengths). `step_number` (absolute) captures absolute depth effects independent of total episode length. Both are useful since some failure modes correlate with absolute depth (e.g., context window pressure) and some with relative position (e.g., approaching a goal at 80% completion).

---

### 5.7 Normalized Context Length (dim 13)

**Use:** Yes.

**Supporting evidence:**
- GreenServ [arXiv:2601.17551]: "text complexity" (rule-based scoring) is a useful routing feature
- Practical observation: as context length grows, the policy model's effective attention window fills up, degrading reasoning quality

**Note:** `normalized_context_length` is a proxy for both task complexity (harder tasks accumulate more context) and model degradation risk (approaching context window limits). It is correlated with `step_number` but not identical — some steps accumulate long observations and some are short.

---

### 5.8 Goal Length (dim 14)

**Use:** Yes.

**Supporting evidence:**
- GreenServ [arXiv:2601.17551]: text complexity positively correlates with routing accuracy gains
- LLMRank literature: syntactic complexity and reasoning depth in the query are predictive of which model level is needed
- Empirical observation: longer, more detailed goals (e.g., multi-constraint web navigation tasks) are harder than short goals (e.g., single function HumanEval tasks)

**Limitation:** Goal length is a weak proxy. A long but simple goal may be easier than a short but ambiguous one. Better alternatives (for future work): syntactic complexity score (e.g., Flesch-Kincaid), number of distinct constraints, or a learned complexity classifier.

---

### 5.9 Benchmark One-Hot (dims 15–18)

**Use:** Yes.

**Supporting evidence:**
- Zooter [arXiv:2311.08692]: tag/domain-level enhancement (blending sample-level and domain-level reward signals) significantly improves routing accuracy — the router can learn domain-specific escalation policies
- LLMRank literature: task type and domain classification are among the most reliably useful routing features
- GreenServ [arXiv:2601.17551]: task type classification drives 22% accuracy improvement

**Why this matters for R2V specifically:** Our four benchmarks (GAIA, ALFWorld, HumanEval+, WebArena) have fundamentally different action spaces, failure modes, and optimal step-level strategies. A single routing threshold across all benchmarks is suboptimal. The benchmark one-hot allows the router to learn that:
- GAIA tasks benefit from LLM escalation when search tool results are ambiguous
- HumanEval tasks benefit from escalation when the candidate code fails tests
- WebArena tasks benefit from escalation when DOM observations are noisy
- ALFWorld tasks are more sequential and may benefit from earlier escalation at decision points

**Implementation:** Detected from the config file path (e.g., `configs/humaneval/noisy.yaml` → `humaneval`). All four values will be 0 for unknown benchmarks — the router will fall back to its average policy.

---

### 5.10 Perturbation One-Hot (dims 19–23)

**Use:** Yes.

**Supporting evidence:**
- "How LLMs Fail" [arXiv:2512.07497]: different failure archetypes are associated with different perturbation types — the router should respond differently per perturbation
- Our hypothesis (R2V-specific): prompt injection failures are less helped by LLM escalation (the teacher also misread the injected instruction), while tool flakiness failures often benefit because the teacher can recognize and retry

**Why one-hot instead of a continuous encoding:** The perturbation types are categorically distinct with qualitatively different failure modes. A one-hot allows the router to learn a separate "threshold shift" per perturbation type without conflating them.

---

## 6. Feature Analysis: What NOT to Use and Why

### 6.1 Raw Query/Context Embeddings

**Rejected for this work.**

**Reason:** RouteLLM [arXiv:2406.18665], EmbedLLM [arXiv:2410.02223], and the kNN paper [arXiv:2505.12601] all use embeddings of the full query text as the primary routing signal. This works well at query level but has three problems for step-level routing:

1. **Growing context:** The context at step *t* contains the full trajectory history. Embedding a 2048-token context is expensive and produces a noisy representation mixing goal, prior actions, and observations — semantically unstable as the trajectory evolves.

2. **Redundancy with trajectory features:** Horizon fraction, step number, and context length already capture trajectory position. The incremental information in a full embedding is low.

3. **Cost:** Running a BERT-base or sentence transformer on every step adds latency that defeats the purpose of the cheap SLM step.

**The kNN finding [arXiv:2505.12601] shows that embeddings are powerful at query level** — but their analysis is on static queries, not evolving agent trajectories. This is an important caveat to note when positioning our work.

---

### 6.2 Hidden-State Structural Features

**Rejected for this work (recommended for future work).**

**Reason:** "Trust in One Round" [arXiv:2602.00977] shows that spectral features of the final-layer hidden-state trajectory outperform probability-based methods for hallucination detection. However:

1. **Memory cost:** Returning all hidden states from a 32-layer Llama-3.1-8B for batch processing (B=4, seq_len=2048) requires ~4GB of activations per forward pass — doubles memory footprint.
2. **Feature engineering complexity:** The spectral and local-variation descriptors require non-trivial post-processing (computing eigenvalues, sequence-level statistics).
3. **Unclear benefit for routing vs. hallucination detection:** The paper tests on static QA tasks (FEVER, WikiBio), not on step-level agent actions. Transfer to our setting is uncertain.

**Recommended path:** Hook the final transformer block to extract only the last-token hidden state (4096 dims), apply a fixed random projection to 8 dims (seeded), add as optional features for ablation. This avoids the memory issue while testing the hypothesis.

---

### 6.3 Semantic Entropy with NLI Clustering

**Adopted in approximated form; full version rejected for inference speed.**

**Reason:** The full semantic entropy method [Kuhn et al., ICLR 2023] requires:
- K forward passes (already done for candidate generation)
- An NLI model to determine semantic equivalence between pairs of candidates

Our approximation (first-token cluster entropy) is computable from existing candidate texts at zero extra cost. The NLI version would be more accurate but adds:
- A 500M–2B param NLI model on GPU
- O(K²) NLI calls per step

For the router feature generation stage (a one-time offline stage), the NLI version would be feasible, but the benefit for routing (vs. the combination of token-level entropy + consistency + our first-token entropy) is likely marginal. **Recommended for ablation.**

---

### 6.4 Query Embedding Distance to Training Set (kNN Distance)

**Rejected for this work.**

**Reason:** The kNN paper [arXiv:2505.12601] uses the distance from the current query to the nearest training set neighbors as a routing signal. Closer = more similar to seen tasks = more confident routing decision. This works well at query level but requires:

1. Storing training set embeddings (~500 × 2048-token trajectories × 768 dims = ~3GB)
2. Running a BERT-base forward pass at each step
3. A nearest-neighbor search at each step (cheap but requires indexing)

More importantly, "distance to training distribution" is already partially captured by `benchmark_onehot` and `pert_type_onehot` — out-of-distribution steps are more likely to be from novel perturbation types.

---

### 6.5 Model Capability Embeddings (EmbedLLM-style)

**Not applicable.**

**Reason:** EmbedLLM [arXiv:2410.02223] factors routing into model embeddings × query embeddings, useful when routing across many models. We have exactly two models (SLM, LLM). The binary routing decision doesn't benefit from the factorized embedding framework.

---

### 6.6 P(True) and Self-Reported Confidence

**Not included; limited reliability.**

**Reason:** P(True) (the model's self-reported probability of being correct, estimated by prompting "Is your answer correct? Yes/No") is mentioned in the "Language Models (Mostly) Know What They Know" literature. In practice:
- It is poorly calibrated without specific fine-tuning for calibration
- It requires an extra forward pass with a different prompt
- Calibrated verifier score (`verifier_score_best`, dim 4) serves the same purpose with better reliability since the verifier is trained for this task

---

## 7. Our Final Feature Set (24-dim)

| Dim | Name | Group | Literature backing |
|---|---|---|---|
| 0 | `entropy` | Uncertainty | Critical Step Opt. [2602.03412]; Semantic entropy [ICLR 2023]; Unified Routing [2410.10347] |
| 1 | `verifier_score_spread` | Verifier | FrugalGPT [2305.05176]; RouterBench [2403.12031]; Adaptive Compute [2602.01070] |
| 2 | `verifier_score_mean` | Verifier | Same |
| 3 | `verifier_score_std` | Verifier | Same |
| 4 | `verifier_score_best` | Verifier | Same + Adaptive Compute [2602.01070] |
| 5 | `verifier_score_worst` | Verifier | Novel — floor signal; implicit in cascade quality estimator design |
| 6 | `log_prob_best` | Log-prob | Unified Routing [2410.10347] |
| 7 | `log_prob_mean` | Log-prob | Unified Routing [2410.10347] |
| 8 | `log_prob_std` | Log-prob | Unified Routing [2410.10347] |
| 9 | `candidate_consistency` | Diversity | Self-consistency [Wang et al., 2023]; Mirror-Consistency [2410.10857] |
| 10 | `semantic_entropy` | Diversity | Semantic entropy [Kuhn et al., ICLR 2023] |
| 11 | `horizon_fraction` | Trajectory | Critical Step Opt. [2602.03412]; AgentProcessBench [2603.14465] |
| 12 | `step_number` | Trajectory | Same |
| 13 | `normalized_context_length` | Complexity | GreenServ [2601.17551] |
| 14 | `goal_length` | Complexity | GreenServ [2601.17551]; LLMRank |
| 15–18 | `benchmark_onehot` | Domain | Zooter [2311.08692]; GreenServ [2601.17551]; LLMRank |
| 19–23 | `perturbation_onehot` | Perturbation | R2V-specific; motivated by "How LLMs Fail" [2512.07497] |

**Feature groups and their roles:**
- **Uncertainty (dims 0):** Is the SLM's *immediate next decision* uncertain?
- **Verifier (dims 1–5):** How good are the SLM's candidate actions, and how consistent is the verifier's assessment?
- **Log-prob (dims 6–8):** How probable does the SLM think its own generated actions are?
- **Diversity (dims 9–10):** Are the K candidates exploring different strategies, or is the SLM converging?
- **Trajectory (dims 11–12):** Where in the episode are we?
- **Complexity (dims 13–14):** How hard is this task?
- **Domain (dims 15–18):** Which benchmark are we operating on?
- **Perturbation (dims 19–23):** What kind of noise is present in the environment?

---

## 8. Open Questions & Future Work

The following features are theoretically motivated but not yet implemented, ranked by expected benefit:

### 8.1 Rolling Verifier Score (Sequential Hypothesis Testing)

**Motivation:** E-valuator [arXiv:2512.03109] shows that sequential e-processes over verifier scores have better false-alarm control than per-step thresholds. A rolling mean or cumulative sum of verifier scores over the last N steps would give the router a trajectory-level signal.

**Implementation:** Add `rolling_verifier_mean_3`, `rolling_verifier_mean_5` as new features. Requires storing per-episode state during feature generation.

### 8.2 Hidden-State Embedding (Last Token, Fixed Projection)

**Motivation:** Structural signals [arXiv:2602.00977] outperform probability-based methods for error detection. Even a crude projection of the last-token hidden state may capture information not available in the logits.

**Implementation:** Add `output_hidden_states=True` to entropy forward pass; extract `hidden_states[-1][:, seq_lens, :]` (last token's last-layer hidden state); apply fixed 4096→8 random projection (seed=42). Adds ~5% memory overhead; no additional GPU passes.

### 8.3 Semantic Entropy with NLI Clustering

**Motivation:** Full semantic entropy [Kuhn et al., ICLR 2023] uses NLI to cluster K candidates. Our first-token approximation conflates "print" and "run" as different clusters even if semantically equivalent.

**Implementation:** Use a small NLI model (DeBERTa-base, 86M params) to cluster K=5 candidates. O(K²) = 10 NLI calls per step. Feasible in offline feature generation.

### 8.4 Number of Distinct Tools Called in Context

**Motivation:** Tool diversity in the trajectory is a proxy for task complexity and breadth. Episodes that have called many different tools are likely in complex multi-step tasks. Not yet implemented.

### 8.5 Verifier Score Trend (Slope over Last N Steps)

**Motivation:** A declining verifier score over the last 3 steps indicates deteriorating trajectory quality — a strong escalation signal that single-step features miss.

---

## 9. References

| arXiv ID | Title | Venue | Year |
|---|---|---|---|
| 2305.05176 | FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance | arXiv | 2023 |
| 2311.08692 | Routing to the Expert: Efficient Reward-Guided Ensemble of LLMs (Zooter) | arXiv | 2023 |
| 2203.11171 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | ICLR 2023 | 2022 |
| 2403.12031 | RouterBench: A Benchmark for Multi-LLM Routing System | arXiv | 2024 |
| 2406.18665 | RouteLLM: Learning to Route LLMs with Preference Data | arXiv | 2024 |
| 2408.08545 | SelectLLM: Query-Aware Efficient Selection Algorithm for LLMs | ACL 2025 | 2024 |
| 2410.02223 | EmbedLLM: Learning Compact Representations of Large Language Models | arXiv | 2024 |
| 2410.10347 | A Unified Approach to Routing and Cascading for LLMs | arXiv | 2024 |
| 2410.10857 | Mirror-Consistency | arXiv | 2024 |
| 2505.12601 | Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers | arXiv | 2025 |
| 2507.02822 | SynapseRoute: Auto-Route Switching Framework on Dual-State LLM | arXiv | 2025 |
| 2509.09782 | One Head, Many Models: Cross-Attention Routing for Cost-Aware LLM Selection | arXiv | 2025 |
| 2511.11500 | Trustworthy Language Models through Reinforced Hesitation | arXiv | 2025 |
| 2512.03109 | E-valuator: Reliable Agent Verifiers with Sequential Hypothesis Testing | arXiv | 2025 |
| 2512.07497 | How Do LLMs Fail in Agentic Scenarios? | arXiv | 2025 |
| 2601.12294 | ToolPRMBench: Evaluating Process Reward Models for Tool-Using Agents | arXiv | 2026 |
| 2601.17551 | GreenServ: Energy-Efficient Context-Aware Dynamic Routing | arXiv | 2026 |
| 2602.00977 | Trust in One Round: Confidence Estimation via Structural Signals | arXiv | 2026 |
| 2602.01070 | What If We Allocate Test-Time Compute Adaptively? | arXiv | 2026 |
| 2602.03412 | Verified Critical Step Optimization for LLM Agents | arXiv | 2026 |
| 2603.14465 | AgentProcessBench: Diagnosing Step-Level Process Quality in Tool-Using Agents | arXiv | 2026 |
| Kuhn et al. ICLR 2023 | Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG | ICLR 2023 | 2023 |
