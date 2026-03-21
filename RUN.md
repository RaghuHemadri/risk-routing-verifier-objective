# R2V-Agent Run Guide (HumanEval + TextWorld)

This repository now supports only two benchmarks:
- `humaneval`
- `textworld`

## 1) End-to-end pipeline

Run full pipeline with default benchmark (`humaneval`):

```bash
bash run_pipeline.sh
```

Run full pipeline for TextWorld:

```bash
BENCHMARK=textworld bash run_pipeline.sh
```

Resume from a stage:

```bash
bash run_pipeline.sh --benchmark textworld --from 3
```

Run a single stage:

```bash
bash run_pipeline.sh --benchmark humaneval --only 5
```

Dry run:

```bash
bash run_pipeline.sh --benchmark humaneval --dry-run
```

## 2) Stage breakdown

- Stage 0: smoke collection (`collect_trajectories.py` with 1 episode)
- Stage 1: clean trajectory collection
- Stage 2: perturbation generation
- Stage 3: BC policy training
- Stage 4: verifier training
- Stage 5: candidate generation
- Stage 6: preference tuning
- Stage 7: router feature generation
- Stage 8: router training
- Stage 9: evaluation
- Stage 10: ablations

## 3) Direct script examples

Collect trajectories:

```bash
python scripts/collect_trajectories.py \
  --config configs/humaneval/clean.yaml \
  --output data/runs \
  --num-episodes 100 \
  --seeds 1 2 3
```

Generate perturbations:

```bash
python -m scripts.generate_perturbations \
  --config configs/humaneval/noisy.yaml \
  --input data/runs/<run_id>/trajectories.jsonl \
  --output data/trajectories/humaneval_noisy/trajectories.jsonl \
  --seeds 1 2 3 \
  --include-clean
```

Train policy (BC):

```bash
python scripts/train_policy.py \
  --config configs/humaneval/noisy.yaml \
  --output outputs/policy/humaneval_noisy \
  --stage bc \
  --trajectories data/trajectories/humaneval_noisy/trajectories.jsonl
```

Train verifier:

```bash
python scripts/train_verifier.py \
  --config configs/humaneval/noisy.yaml \
  --output outputs/verifier/humaneval_noisy \
  --trajectories data/trajectories/humaneval_noisy/trajectories.jsonl
```

## 4) Supported configs

- `configs/humaneval/clean.yaml`
- `configs/humaneval/noisy.yaml`
- `configs/textworld/clean.yaml`
- `configs/textworld/noisy.yaml`

Any config path pointing to removed benchmarks is no longer valid.
