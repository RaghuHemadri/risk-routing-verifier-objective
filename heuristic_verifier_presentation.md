# Detailed Evaluation Report: Heuristic Verifier Methodology & Overfitting Audit

This report maps the conceptual design of our rule-based heuristic verifier (`r2v/models/heuristic_verifier.py`) directly to its code implementation, serving as a comprehensive audit of its generalization capabilities, algorithmic robustness, and overfitting risks. 

---

## 1. Core Architecture & Defensive Mechanisms

The heuristic verifier relies entirely on static analysis, observation parsing, and optional subprocess execution. It avoids neural network inference, making it an efficient, programmatic baseline.

### 1.1. Prompt Injection & Distractor Handling (`_detect_injection`, `_strip_injections`)
Before any scoring occurs, the verifier aggressively filters adversarial text:
*   **Regex Stripping**: Removes XML tags (`<Aside>`), HTML comments, DOM accessibility trees (`[123] button 'Foo'`), and internal chain-of-thought leaks (e.g., `Plugin Execution Mode:`).
*   **Phrase Detection**: Scans for known attack vectors (`"ignore previous instructions"`, `"override goal"`).
*   **Penalty**: If an injection is detected in the current action or recent context, the final score is multiplicatively clamped by **`0.85`** to reflect uncertainty/untrustworthiness.

---

## 2. In-Depth Domain Heuristics

### 2.1. HumanEval Pipeline (`_score_humaneval`)

Actions are parsed into `write_code`, `test`, or `submit`. The scoring formula is heavily weighted towards execution and structural completeness.

**A. Structural Guardrails (Static Analysis):**
*   **Syntax (`_check_syntax`)**: Uses `ast.parse`. If invalid, score drops to near-zero (`0.05`).
*   **Stub Detection (`_is_stub`)**: Penalizes code with `pass`, `...`, `NotImplementedError`, or `# TODO`.
*   **Import Safety (`_check_imports_valid`)**: Checks imports against a whitelist (`_OK_MODULES` like `os`, `math`, `collections`). Unrecognized imports incrementally lower the score.
*   **Reward Hacking Penalties (`_detect_hardcoding`)**: Specifically looks for:
    *   Bare literal returns with no computation (≤3 lines).
    *   `if-else` ladders returning only constants (lookup table pattern).
    *   Swallowing exceptions blindly (`except Exception:`).
    *   Infinite loops (`while True:` without `break`).
*   **Repetition (`_code_similarity`)**: Calculates Jaccard similarity of code lines against the previous submission. If similarity is >0.95 and the previous test *failed*, a massive penalty is applied to prevent the agent from getting stuck in loops.

**B. Subprocess Smoke Test (`_run_quick_tests`):**
If `run_code=True`, the Python code is executed in a protected subprocess with a 5-second timeout to verify it defines the entry point and doesn't crash on import/parsing.
*   Success = `0.85`.
*   Syntax/Name/Import errors scale down from `0.3` to `0.0`.

**C. Observation Extraction (`_parse_test_result_from_obs`):**
When the agent executes tests, the verifier parses the output string directly.
*   "all tests passed" -> `1.0`
*   "M/N tests passed" -> `M/N`
*   Catches exact Python tracebacks (`SyntaxError`, `IndexError`, etc.) and heavily penalizes (`0.0` - `0.3`).

**D. Scoring Formula (`write_code`):**
The final score is a weighted linear combination, bounded by `[0, 1]`, and multiplicatively discounted by the hacking and repetition penalties:
`Score = (0.35 * Exec) + (0.20 * Fn_Def) + (0.15 * Logic) + (0.10 * Ret) + (0.10 * Imports) + (0.10 * Length)`


### 2.2. TextWorld Pipeline (`_score_textworld`)

TextWorld relies on tracking progression through valid verbs, spatial mapping, and observation tone.

**A. Action Value (`_tw_action_verb`):**
Actions must match `_TW_VALID_VERBS`. 
*   **Progress Verbs** (e.g., `take`, `open`, `unlock`) -> `0.75` base.
*   **Movement** (`go`) -> `0.65` base.
*   **Idle** (`look`, `inventory`) -> `0.45` base.

**B. Goal Alignment (`_tw_goal_alignment`):**
Extracts non-stopword tokens from the overall task goal and checks intersection with tokens in the action string.

**C. Environment State Extraction (`_tw_prev_obs_quality`):**
Parses the simulator's immediate string response:
*   **Success**: "you take", "opened", "you go" (-> `0.80` - `0.95`)
*   **Failure/Blocked**: "can't do that", "that action did not help" (-> `0.10`)

**D. Cyclical Penalties:**
*   **Repetition (`_tw_repetition_penalty`)**: Scans the last 8 actions. `1` repeat -> `0.20` penalty; `2+` repeats -> `0.45+` penalty.
*   **Oscillation (`_tw_oscillation_penalty`)**: Detects movement inverses (e.g., `go north` immediately after `go south`). Applies a `0.4` penalty to discourage wandering.

**E. Scoring Formula:**
`Score = (0.25 * Obs) + (0.25 * VerbBase) + (0.25 * GoalAlign) + (0.15 * (1 - Repetition)) + (0.10 * (1 - Oscillation))`
If explicit reward text (`Reward: X`) is detected, it is blended in at 50% weight.

---

## 3. Methodology Overfitting & Generalization Audit

We ran an extensive evaluation to investigate whether this verifier has *memorized* dataset quirks (Overfitting) or *relies too heavily on specific templates* (Methodological Specialization).

### 3.1. Split Leakage Analysis (Passed)
*   **Result**: `0.0%` Leakage.
*   **Validation Check**: Train, Validation, and Test splits group entirely by `task_id` and parent episodes. Task, episode, and goal overlaps are strictly zero.
*   Performance on Test is actually higher than on Train (e.g., HumanEval `+0.14` AUROC), contradicting the standard signature of memorization-based overfitting.

### 3.2. Ablation & Robustness Studies
We executed masked-cue and random-label ablations to test if the heuristic functions are brittle to environment prompt structures (`results/heuristic_verifier_methodology_check.json`).

**HumanEval Methodology Robustness: STRONG**
*   **Baseline Test AUROC**: `0.8449`
*   **Masked Result Cues (removing "tests passed", "error", etc.)**: `0.8589`
*   **Random Label Shuffle AUROC (300 trials)**: Mean `0.5028` (std=`0.0398`, Max=`0.6061`).
*   *Conclusion*: HumanEval's high AUROC (`~0.84`) is a mathematically genuine signal derived largely from *structural/syntax code analysis* and the *subprocess execution framework*. It is **not** overfitted to superficial text parsing strings.

**TextWorld Methodology Robustness: HIGHLY SPECIALIZED**
*   **Baseline Test AUROC**: `1.000`
*   **Masked Reward Cues**: `0.999`
*   *Conclusion*: TextWorld achieves near-oracle performance because the spatial grammar (`go`, `take`) and valid state logic tightly couples with the simulator's output templates. It acts perfectly in this domain but is highly specialized to the specific text-adventure schema used.

## 4. Final Verdict

1. **Information Leakage**: None. The heuristic verifier generalizes perfectly to unseen dataset task distributions.
2. **Textual Overfitting**:
    *   **HumanEval**: Robust. Handles noisy inputs safely, uses execution, and survives text cue ablation.
    *   **TextWorld**: Template-specialized. Near-perfect due to parsing the environment's standardized state grammar but requires rewriting for vastly different NLP simulators.
