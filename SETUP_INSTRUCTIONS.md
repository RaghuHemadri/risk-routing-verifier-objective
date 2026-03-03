# Setup Instructions — R2V-Agent Data Collection (SWE-bench)

This guide walks through the full environment setup, **SWE-bench data
collection using Gemini 3 Flash**, and **perturbation generation** on NYU HPC.

> **Environment:** NYU Greene HPC, Singularity container
>
> **Teacher model:** Google Gemini 3 Flash (`gemini-3-flash-preview`)
>
> **Docker:** Not required — SWE-bench collection is API-only
>
> **Important:** Only **one terminal** can be connected to the container
> at a time (containers are opened in write mode).

---

## Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Environment setup | ✅ Complete |
| 2 | API keys | ✅ Complete |
| 3 | Smoke test | ✅ Complete |
| 4a | SWE-bench clean collection (900 ep) | ✅ Complete (2026-03-03) |
| 4b | Perturbation generation (2700 ep) | ✅ Complete (2026-03-03) |
| 5 | Training (BC, verifier, DPO, router) | ☐ Not started — requires GPU |

---

## 1. Environment Setup

```bash
# 1. Enter the Singularity container
/scratch/rh3884/bin/enter_container.sh

# 2. Navigate to the project directory
cd /scratch/rh3884/risk-routing-verifier-objective

# 3. Upgrade pip and setuptools
pip install --upgrade pip setuptools

# 4. Install the package with all dependencies
pip install -e ".[dev]"

# 5. Verify the installation
python -c "import r2v; print('r2v imported successfully')"
```

> **Note:** The `pyproject.toml` uses `build-backend = "setuptools.build_meta"`.
> Verify the first three lines read:
> ```toml
> [build-system]
> requires = ["setuptools>=45", "wheel"]
> build-backend = "setuptools.build_meta"
> ```

---

## 2. API Keys

We use **Gemini 3 Flash** as the teacher model. Only `GOOGLE_API_KEY` is
required.

```bash
cp .env.example .env
```

Edit `.env` and set:

```dotenv
GOOGLE_API_KEY=<your-google-api-key>
```

Get your key at: https://aistudio.google.com/apikey

Verify the key works:

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.environ.get('GOOGLE_API_KEY', '')
print(f'GOOGLE_API_KEY is set: {bool(key)}  (length={len(key)})')
"
```

---

## 3. Smoke Test — SWE-bench with Gemini 3 Flash

Run this immediately after step 2 to confirm the pipeline works.
Uses 1 task, 3 seeds — cost under **$0.001**.

```bash
python -m scripts.collect_trajectories \
    --config configs/smoke_test_swebench.yaml \
    --output data/smoke_test_swebench \
    --num-episodes 1 \
    --overrides teacher.provider=google teacher.model_name=gemini-3-flash-preview
```

Expected output:
```
[INFO] Teacher: provider=google, model=gemini-3-flash-preview
[INFO] Loaded 1 SWE-bench instances
[INFO] Loaded 1 tasks, collecting with seeds=[1, 2, 3]
[INFO] Collection complete — Episodes: 3
```

Verify output:

```bash
wc -l data/smoke_test_swebench/*/trajectories.jsonl
# Should show 3 lines (1 task × 3 seeds)
```

---

## 4. Full Data Collection (Gemini 3 Flash)

The SWE-bench dataset downloads automatically from HuggingFace on first run.
No Docker required.

### 4a. SWE-bench Clean — 300 tasks × 3 seeds = 900 episodes

> **✅ Completed 2026-03-03** — Run ID: `swebench_google_gemini-3-flash-preview_20260303T004504`
> - 900 episodes, 120 successful (13.3%), cost $0.60, wall time ~2.4h (4 workers)
> - Output: `data/runs/swebench_google_gemini-3-flash-preview_20260303T004504/trajectories.jsonl`

```bash
python -m scripts.collect_trajectories \
    --config configs/swebench/clean.yaml \
    --num-episodes 300 \
    --seeds 1 2 3 \
    --num-workers 4 \
    --overrides teacher.provider=google teacher.model_name=gemini-3-flash-preview
```

### 4b. Generate Perturbations — 900 clean × 3 seeds = 2700 perturbed

> **✅ Completed 2026-03-03** — 2700 perturbed + 900 clean = 3600 total episodes (79 seconds)
> - Perturbation types: tool flakiness, partial observability, prompt injection, distractors
> - Output: `data/trajectories/swebench_noisy/trajectories.jsonl` (36 MB)

Noisy data is **not** collected from the LLM. Instead, perturbations are
applied locally to the clean trajectories using `generate_perturbations.py`:

```bash
python -m scripts.generate_perturbations \
    --config configs/swebench/noisy.yaml \
    --input data/runs/swebench_google_gemini-3-flash-preview_20260303T004504/trajectories.jsonl \
    --output data/trajectories/swebench_noisy/trajectories.jsonl \
    --seeds 1 2 3 \
    --include-clean
```

This applies all four perturbation types (configured in `configs/swebench/noisy.yaml`)
as a composite pipeline to each clean episode, once per seed. The `--include-clean`
flag also copies the original clean episodes into the output for training convenience.

### 4c. Resume a partially-completed run

If collection is interrupted (e.g., job timeout, container restart), re-run
with the same `--run-id` to pick up where you left off:

```bash
python -m scripts.collect_trajectories \
    --config configs/swebench/clean.yaml \
    --output data/trajectories/swebench_clean \
    --num-episodes 300 \
    --seeds 1 2 3 \
    --num-workers 8 \
    --overrides teacher.provider=google teacher.model_name=gemini-3-flash-preview \
    --run-id <run-id-from-previous-output>
```

The run ID is printed at the start of every collection run — copy it for
resume purposes.

---

## 5. Actual Collection Stats (Gemini 3 Flash)

| Step | Episodes | Wall Time | Cost | Workers |
|------|----------|-----------|------|---------|
| SWE-bench clean (collection) | 900 | ~2.4 h | $0.60 | 4 |
| Perturbation generation | 2700 (+900 clean) | ~79 s | $0 (CPU-only) | — |
| **Total** | **3600** | **~2.5 h** | **$0.60** | — |

> The original estimates (~$1.80, ~6h) were conservative. Gemini 3 Flash
> was faster and cheaper than expected. Perturbation generation is purely
> local, CPU-only, and takes under 2 minutes.

---

## 6. Quick Reference — Run Order

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Enter container                          ✅ Done   │
│    /scratch/rh3884/bin/enter_container.sh                   │
│                                                             │
│  STEP 2: cd /scratch/rh3884/risk-routing-verifier-objective │
│                                                    ✅ Done   │
│  STEP 3: Install (first time only)                ✅ Done   │
│    pip install --upgrade pip setuptools                     │
│    pip install -e ".[dev]"                                  │
│                                                             │
│  STEP 4: Set API key (first time only)            ✅ Done   │
│    cp .env.example .env                                     │
│    # edit .env → set GOOGLE_API_KEY                         │
│                                                             │
│  STEP 5: Smoke test (Section 3)                   ✅ Done   │
│  STEP 6: SWE-bench clean collection (Section 4a)  ✅ Done   │
│  STEP 7: Generate perturbations (Section 4b)      ✅ Done   │
│  STEP 8: Training (BC, verifier, DPO, router)     ☐ Next   │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Directory Structure After Collection

```
risk-routing-verifier-objective/
├── .env                               # API keys (GOOGLE_API_KEY)
├── configs/
│   ├── base.yaml
│   ├── smoke_test_swebench.yaml
│   └── swebench/
│       ├── clean.yaml
│       └── noisy.yaml                 # Perturbation config (4 types)
├── data/
│   ├── registry.json                  # Central index of all collection runs
│   ├── smoke_test_eval/               # Smoke test output
│   │   └── swebench_google_gemini-3-flash-preview_20260303T003125/
│   ├── runs/                          # Versioned collection runs
│   │   └── swebench_google_gemini-3-flash-preview_20260303T004504/
│   │       ├── run_manifest.json      # Config snapshot + stats
│   │       ├── trajectories.jsonl     # 900 clean episodes (9.5 MB)
│   │       ├── collection_log.jsonl
│   │       └── worker_*_trajectories.jsonl  # Per-worker shards
│   └── trajectories/
│       └── swebench_noisy/
│           ├── trajectories.jsonl     # 3600 episodes (900 clean + 2700 perturbed, 36 MB)
│           └── perturbation_log.jsonl
├── r2v/                               # Core library
└── scripts/
    ├── collect_trajectories.py        # Step 1: collect from LLM
    └── generate_perturbations.py      # Step 2: apply perturbations locally
```

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `BackendUnavailable: Cannot import 'setuptools.backends._legacy'` | Verify `pyproject.toml` uses `build-backend = "setuptools.build_meta"` and run `pip install --upgrade setuptools` |
| `GOOGLE_API_KEY not found` | Check `.env` exists and has the key set. Run: `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ.get('GOOGLE_API_KEY','NOT SET'))"` |
| `google.generativeai` import error | Run `pip install google-generativeai>=0.4.0` |
| `ModuleNotFoundError: No module named 'r2v'` | Run `pip install -e ".[dev]"` from the project root |
| Container session dropped | Re-enter with `/scratch/rh3884/bin/enter_container.sh`. Resume collection with `--run-id` (see Section 4c) |
| `Cannot connect to container` | Only one terminal can be connected at a time (write mode). Close other sessions first |
| Rate limit errors from Gemini API | Reduce `--num-workers` to 4. The script uses exponential backoff with 6 retries automatically |
| Collection interrupted mid-run | Resume with the same `--run-id` flag (see Section 4c) |
