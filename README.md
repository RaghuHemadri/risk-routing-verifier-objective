# R2V-Agent: Risk-Calibrated Routing + Robust Verifier-Distillation Objective

> **Risk-calibrated routing and robust verifier-distillation for reliable agentic AI under environmental perturbations.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

R2V-Agent is a framework that makes small language model (SLM) agents reliable under noisy, adversarial environments by combining three key ideas:

1. **Robust Verifier-Distillation** — Train an SLM policy via behavior cloning + DPO preference distillation from a teacher LLM verifier, with consistency regularization
2. **Risk-Calibrated Routing** — A learned router that decides when to escalate from the cheap SLM to an expensive teacher LLM, optimizing a CVaR-based Lagrangian objective
3. **Systematic Perturbation Evaluation** — Four realistic perturbation types (tool flakiness, partial observability, prompt injection, distractors) for rigorous robustness testing

## Project Structure

```
r2v-agent/
├── configs/                     # YAML configurations
│   ├── base.yaml               # Base config (inherited by all)
│   ├── mrp.yaml                # Minimal Reproducible Prototype
│   ├── webarena/               # WebArena-specific
│   │   ├── clean.yaml
│   │   └── noisy.yaml
│   ├── gaia/                   # GAIA (General AI Assistants)
│   │   ├── clean.yaml
│   │   └── noisy.yaml
│   ├── alfworld/               # ALFWorld (embodied text agents)
│   │   ├── clean.yaml
│   │   └── noisy.yaml
│   └── humaneval/              # HumanEval+ (code generation)
│       ├── clean.yaml
│       └── noisy.yaml
├── r2v/                         # Core library
│   ├── data/                   # Data structures & loading
│   │   ├── trajectory.py       # Episode, Step, Observation dataclasses
│   │   ├── datasets.py         # PyTorch datasets (BC, DPO, Router, etc.)
│   │   ├── labeling.py         # Step-level labelers (per-benchmark)
│   │   └── perturbations/      # Perturbation modules
│   │       ├── base.py         # Pipeline & registry
│   │       ├── tool_flakiness.py
│   │       ├── partial_observability.py
│   │       ├── prompt_injection.py
│   │       └── distractors.py
│   ├── models/                 # Model wrappers
│   │   ├── policy.py           # SLM policy (HF + LoRA + 4-bit)
│   │   ├── verifier.py         # LLM-judge & trained verifier
│   │   └── router.py           # Risk-calibrated MLP router
│   ├── training/               # Training pipelines
│   │   ├── bc_trainer.py       # Behavior Cloning
│   │   ├── preference_trainer.py  # DPO preference distillation
│   │   ├── consistency.py      # JSD consistency regularization
│   │   ├── verifier_trainer.py # Verifier training
│   │   ├── router_trainer.py   # Lagrangian CVaR router training
│   │   └── joint_trainer.py    # Full orchestrated pipeline
│   ├── agent/                  # Agent inference
│   │   ├── budget.py           # Inference budget tracking
│   │   └── r2v_agent.py        # R2V-Agent + baseline agents
│   ├── evaluation/             # Evaluation & metrics
│   │   ├── metrics.py          # R2VEvaluator (full eval pipeline)
│   │   ├── robustness.py       # CVaR, worst-seed, bottom-k%
│   │   ├── calibration.py      # ECE, Brier score
│   │   └── statistical.py      # Bootstrap CI, McNemar test
│   └── utils/                  # Utilities
│       ├── logging.py          # Logging + wandb helpers
│       ├── config.py           # OmegaConf config management
│       └── results.py          # Structured results for paper writing
├── scripts/                     # Entry-point scripts
│   ├── collect_trajectories.py
│   ├── generate_perturbations.py
│   ├── generate_candidates.py
│   ├── generate_router_features.py
│   ├── train_policy.py
│   ├── train_verifier.py
│   ├── train_router.py
│   ├── evaluate.py
│   ├── run_ablations.py
│   ├── launch_candidates.sh    # Multi-GPU candidate generation
│   └── launch_router_features.sh  # Multi-GPU router features
├── run_pipeline.sh              # Full pipeline script (--from/--only/--dry-run)
├── RUN.md                       # Pipeline instructions (local execution)
├── pyproject.toml
├── requirements.txt
└── proposal.md                  # Research proposal
```

## Quick Start

```bash
# Clone and install
git clone <this-repo>
cd risk-routing-verifier-objective
pip install -e ".[dev]"

# Set API key + HuggingFace token
export GOOGLE_API_KEY=<your-key>
source exports.sh

# Run full pipeline on a specific benchmark
BENCHMARK=gaia      bash run_pipeline.sh   # GAIA benchmark (default)
BENCHMARK=alfworld  bash run_pipeline.sh   # ALFWorld benchmark
BENCHMARK=humaneval bash run_pipeline.sh   # HumanEval+ benchmark

# Or resume / run a single stage
bash run_pipeline.sh --benchmark gaia --from 5
bash run_pipeline.sh --only 3
bash run_pipeline.sh --dry-run
```

See [RUN.md](RUN.md) for detailed per-stage instructions.

## Benchmarks

| Benchmark | Domain | Tasks | Docker-Free | Key Evaluation |
|-----------|--------|-------|-------------|----------------|
| **GAIA** | General AI assistant | 165 | ✅ | Exact/fuzzy answer match |
| **ALFWorld** | Embodied text agent | 134 | ✅ | Task completion (env reward) |
| **HumanEval+** | Code generation | 164 | ✅ | Extended test suite (EvalPlus) |

All benchmarks support the full perturbation framework (tool flakiness, partial observability, prompt injection, distractors) since perturbations operate on the generic `Observation` representation.

### Benchmark-specific dependencies

```bash
# GAIA: uses `datasets` (already in base) + optional web search API key
export SERPER_API_KEY=<your-key>   # or SERPAPI_KEY

# ALFWorld
pip install "alfworld[full]" textworld

# HumanEval+
pip install evalplus
```

## Key Features

### Training Objective

The policy is trained with a combined loss:

$$\mathcal{L} = \mathcal{L}_{\text{BC}} + \lambda_{\text{pref}} \mathcal{L}_{\text{DPO}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}}$$

Where:
- $\mathcal{L}_{\text{BC}}$: Standard behavior cloning on teacher demonstrations
- $\mathcal{L}_{\text{DPO}}$: DPO preference distillation from verifier-ranked candidates
- $\mathcal{L}_{\text{cons}}$: Jensen-Shannon divergence consistency regularization

### Router Objective (Lagrangian CVaR)

$$\min_\psi \max_\lambda \; \mathbb{E}[\text{cost}(d)] + \lambda(\text{CVaR}_\alpha(1 - S_z) - \varepsilon)$$

### Perturbation Types

| Type | Description | Example |
|------|-------------|---------|
| Tool Flakiness | HTTP errors, timeouts, stale caches | `503 Service Unavailable` |
| Partial Observability | DOM reordering, log truncation | Hidden elements, stripped attributes |
| Prompt Injection | Direct/indirect attacks, goal hijacking | Encoded payloads, role confusion |
| Distractors | False search results, decoy UI | Plausible-but-wrong code suggestions |

## Results Format

All results are saved in structured formats optimized for LLM-assisted paper writing:

- **JSON bundles**: Complete experiment data (`results/*/structured_results/*.json`)
- **CSV tables**: Machine-readable summaries (`main_table.csv`, `comparisons.csv`, `ablations.csv`)
- **LaTeX tables**: Paper-ready (`main_table.tex`)
- **LLM summary**: Optimized for feeding to LLM agents (`llm_summary.json`)

## Citation

```bibtex
@article{r2v-agent-2025,
  title={Risk-Calibrated Routing + Robust Verifier-Distillation Objective},
  year={2025},
}
```

## License

MIT
