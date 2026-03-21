#!/usr/bin/env bash
set -euo pipefail

# Single entrypoint for Slurm submission + in-job end-to-end pipeline execution.
# - Outside Slurm: submits itself with sbatch.
# - Inside Slurm: runs selected pipeline stages.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/slurm_end_to_end.sh [options]

Core options:
  --benchmark NAME              Benchmark name (humaneval|textworld). Default: humaneval
  --slm-model MODEL             SLM model type or HF ID (e.g., QWEN, LLAMA, Qwen/Qwen2.5-7B-Instruct)
  --verifier-model MODEL        Verifier backbone type or HF ID (e.g., LLAMA, QWEN, meta-llama/Llama-3.1-8B-Instruct)
  --run-id ID                   Optional run id; default auto-generated
  --from-step STEP              Resume from a step number or name
  --only-step STEP              Run exactly one step number or name

Pipeline/data options:
  --config-clean PATH           Clean config path. Default: configs/<benchmark>/clean.yaml
  --config-noisy PATH           Noisy config path. Default: configs/<benchmark>/noisy.yaml
  --clean-trajectories PATH     Existing clean trajectories (required if starting at step >= 2 and skipping collect)
  --noisy-trajectories PATH     Existing noisy trajectories (required if starting at step >= 3 and skipping perturb)
  --num-gpus N                  Number of GPUs for multi-GPU stages. Default: 4
  --num-workers N               Workers for trajectory collection. Default: 4
  --seeds "1 2 3"               Seed list used in collect/perturb/eval. Default: "1 2 3"
  --router-threshold-sweep "..."
                                Threshold sweep list for evaluate.py
  --extra-overrides "k=v ..."   Additional OmegaConf overrides appended to all train/eval commands

Slurm options:
  --account NAME                Slurm account. Default: torch_pr_67_tandon_advanced
  --job-name NAME               Slurm job name prefix. Default: r2v_e2e
  --time HH:MM:SS               Wall time. Default: 24:00:00
  --cpus N                      CPUs per task. Default: 8
  --mem SIZE                    Memory. Default: 256G
  --gpu-type TYPE               GPU type. Default: h200
  --workdir PATH                Working directory for Slurm logs/sbatch. Default: /scratch/$USER
  --project-dir PATH            Repository path on cluster. Default: current directory

Container options (optional):
  --use-container               Run each stage in singularity
  --image PATH                  Singularity image path
  --overlay PATH                Overlay path
  --env-name NAME               Conda env name in container. Default: pt311

Misc:
  --dry-run                     Print commands only
  -h, --help                    Show help

Step map:
  0 smoke
  1 collect_clean
  2 perturb_noisy
  3 train_policy_bc
  4 train_verifier
  5 generate_candidates
  6 train_policy_preference
  7 generate_router_features
  8 train_router
  9 evaluate

Examples:
  Full run:
    bash scripts/slurm_end_to_end.sh --benchmark humaneval --slm-model QWEN --verifier-model LLAMA

  Resume from step 5:
    bash scripts/slurm_end_to_end.sh --benchmark humaneval --slm-model QWEN --verifier-model LLAMA --from-step 5 \
      --noisy-trajectories data/trajectories/humaneval_noisy/trajectories.jsonl

  Run only router training:
    bash scripts/slurm_end_to_end.sh --benchmark humaneval --slm-model QWEN --verifier-model LLAMA --only-step 8 \
      --run-id humaneval_qwen_llama_20260317
EOF
}

BENCHMARK="humaneval"
SLM_MODEL_INPUT="QWEN"
VERIFIER_MODEL_INPUT="LLAMA"
RUN_ID=""
FROM_STEP="0"
ONLY_STEP=""

NUM_GPUS="4"
NUM_WORKERS="4"
SEEDS="1 2 3"
ROUTER_THRESHOLD_SWEEP="0.05 0.1 0.12 0.14 0.16 0.18 0.2 0.25 0.3"
EXTRA_OVERRIDES=""

ACCOUNT="torch_pr_67_tandon_advanced"
JOB_NAME="r2v_e2e"
TIME_LIMIT="24:00:00"
CPUS="8"
MEM="256G"
GPU_TYPE="h200"
WORKDIR="/scratch/${USER}"
PROJECT_DIR="$(pwd)"

USE_CONTAINER="0"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="${WORKDIR}/my_pytorch.ext3"
ENV_NAME="pt311"

DRY_RUN="0"
INSIDE_SLURM="0"

CONFIG_CLEAN=""
CONFIG_NOISY=""
CLEAN_TRAJECTORIES=""
NOISY_TRAJECTORIES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --slm-model) SLM_MODEL_INPUT="$2"; shift 2 ;;
    --verifier-model) VERIFIER_MODEL_INPUT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --from-step) FROM_STEP="$2"; shift 2 ;;
    --only-step) ONLY_STEP="$2"; shift 2 ;;

    --config-clean) CONFIG_CLEAN="$2"; shift 2 ;;
    --config-noisy) CONFIG_NOISY="$2"; shift 2 ;;
    --clean-trajectories) CLEAN_TRAJECTORIES="$2"; shift 2 ;;
    --noisy-trajectories) NOISY_TRAJECTORIES="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --router-threshold-sweep) ROUTER_THRESHOLD_SWEEP="$2"; shift 2 ;;
    --extra-overrides) EXTRA_OVERRIDES="$2"; shift 2 ;;

    --account) ACCOUNT="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;

    --use-container) USE_CONTAINER="1"; shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --overlay) OVERLAY="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;

    --dry-run) DRY_RUN="1"; shift ;;
    --inside-slurm) INSIDE_SLURM="1"; shift ;;

    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${CONFIG_CLEAN}" ]]; then
  CONFIG_CLEAN="configs/${BENCHMARK}/clean.yaml"
fi
if [[ -z "${CONFIG_NOISY}" ]]; then
  CONFIG_NOISY="configs/${BENCHMARK}/noisy.yaml"
fi

norm_upper() {
  tr '[:lower:]' '[:upper:]' <<<"$1"
}

resolve_model() {
  local input="$1"
  local kind="$2"
  local upper
  upper="$(norm_upper "$input")"

  if [[ "$input" == */* ]]; then
    echo "$input"
    return 0
  fi

  case "$kind:$upper" in
    slm:QWEN) echo "Qwen/Qwen2.5-7B-Instruct" ;;
    slm:QWEN_CODER) echo "Qwen/Qwen2.5-Coder-7B-Instruct" ;;
    slm:LLAMA) echo "meta-llama/Llama-3.1-8B-Instruct" ;;
    slm:MISTRAL) echo "mistralai/Mistral-7B-Instruct-v0.3" ;;

    verifier:QWEN) echo "Qwen/Qwen2.5-7B-Instruct" ;;
    verifier:QWEN_CODER) echo "Qwen/Qwen2.5-Coder-7B-Instruct" ;;
    verifier:LLAMA) echo "meta-llama/Llama-3.1-8B-Instruct" ;;
    verifier:MISTRAL) echo "mistralai/Mistral-7B-Instruct-v0.3" ;;
    *)
      # Fallback: accept raw token if user gives a custom symbolic value.
      echo "$input"
      ;;
  esac
}

slugify() {
  local s="$1"
  s="${s,,}"
  s="${s//\//-}"
  s="${s//_/-}"
  s="${s// /-}"
  s="${s//[^a-z0-9.-]/}"
  echo "$s"
}

step_to_num() {
  local s="$1"
  case "$s" in
    0|smoke) echo "0" ;;
    1|collect_clean) echo "1" ;;
    2|perturb_noisy) echo "2" ;;
    3|train_policy_bc) echo "3" ;;
    4|train_verifier) echo "4" ;;
    5|generate_candidates) echo "5" ;;
    6|train_policy_preference) echo "6" ;;
    7|generate_router_features) echo "7" ;;
    8|train_router) echo "8" ;;
    9|evaluate) echo "9" ;;
    *)
      echo "Invalid step: $s"
      exit 1
      ;;
  esac
}

FROM_STEP_NUM="$(step_to_num "$FROM_STEP")"
ONLY_STEP_NUM=""
if [[ -n "$ONLY_STEP" ]]; then
  ONLY_STEP_NUM="$(step_to_num "$ONLY_STEP")"
fi

if [[ -n "$ONLY_STEP_NUM" && "$FROM_STEP_NUM" != "0" ]]; then
  echo "Use either --only-step or --from-step (non-zero), not both."
  exit 1
fi

SLM_MODEL="$(resolve_model "$SLM_MODEL_INPUT" "slm")"
VERIFIER_MODEL="$(resolve_model "$VERIFIER_MODEL_INPUT" "verifier")"

SLM_TAG="$(slugify "$SLM_MODEL_INPUT")"
VER_TAG="$(slugify "$VERIFIER_MODEL_INPUT")"

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="${BENCHMARK}_${SLM_TAG}_${VER_TAG}_$(date +%Y%m%dT%H%M%S)"
fi

LOG_ROOT="logs/slurm/${RUN_ID}"
META_ROOT="outputs/experiments/${RUN_ID}"
mkdir -p "$WORKDIR" "$PROJECT_DIR" "$PROJECT_DIR/$LOG_ROOT" "$PROJECT_DIR/$META_ROOT"

if [[ "$INSIDE_SLURM" == "0" ]]; then
  echo "Submitting Slurm job for run_id=${RUN_ID}"

  SBATCH_OUT="${WORKDIR}/${JOB_NAME}_${RUN_ID}.%j.out"
  SBATCH_ERR="${WORKDIR}/${JOB_NAME}_${RUN_ID}.%j.err"

  submit_args=(
    --job-name "${JOB_NAME}_${BENCHMARK}"
    --account "${ACCOUNT}"
    --gres "gpu:${GPU_TYPE}:${NUM_GPUS}"
    --cpus-per-task "${CPUS}"
    --mem "${MEM}"
    --time "${TIME_LIMIT}"
    --output "${SBATCH_OUT}"
    --error "${SBATCH_ERR}"
  )

  # Reconstruct args for in-job run.
  job_args=(
    --inside-slurm
    --benchmark "$BENCHMARK"
    --slm-model "$SLM_MODEL_INPUT"
    --verifier-model "$VERIFIER_MODEL_INPUT"
    --run-id "$RUN_ID"
    --from-step "$FROM_STEP"
    --config-clean "$CONFIG_CLEAN"
    --config-noisy "$CONFIG_NOISY"
    --num-gpus "$NUM_GPUS"
    --num-workers "$NUM_WORKERS"
    --seeds "$SEEDS"
    --router-threshold-sweep "$ROUTER_THRESHOLD_SWEEP"
    --account "$ACCOUNT"
    --job-name "$JOB_NAME"
    --time "$TIME_LIMIT"
    --cpus "$CPUS"
    --mem "$MEM"
    --gpu-type "$GPU_TYPE"
    --workdir "$WORKDIR"
    --project-dir "$PROJECT_DIR"
    --image "$IMAGE"
    --overlay "$OVERLAY"
    --env-name "$ENV_NAME"
  )

  if [[ -n "$ONLY_STEP" ]]; then
    job_args+=(--only-step "$ONLY_STEP")
  fi
  if [[ -n "$CLEAN_TRAJECTORIES" ]]; then
    job_args+=(--clean-trajectories "$CLEAN_TRAJECTORIES")
  fi
  if [[ -n "$NOISY_TRAJECTORIES" ]]; then
    job_args+=(--noisy-trajectories "$NOISY_TRAJECTORIES")
  fi
  if [[ "$USE_CONTAINER" == "1" ]]; then
    job_args+=(--use-container)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    job_args+=(--dry-run)
  fi
  if [[ -n "$EXTRA_OVERRIDES" ]]; then
    job_args+=(--extra-overrides "$EXTRA_OVERRIDES")
  fi

  jid=$(sbatch --parsable "${submit_args[@]}" "$0" "${job_args[@]}")
  echo "$jid" > "${WORKDIR}/.last_r2v_e2e_jobid"

  echo "Submitted JobID: ${jid}"
  echo "Track: squeue -u $USER -j ${jid}"
  echo "Logs:  ${SBATCH_OUT}"
  echo "       ${SBATCH_ERR}"
  exit 0
fi

# ---------- In-job execution below ----------

cd "$PROJECT_DIR"

should_run() {
  local step="$1"
  if [[ -n "$ONLY_STEP_NUM" ]]; then
    [[ "$step" == "$ONLY_STEP_NUM" ]]
  else
    [[ "$step" -ge "$FROM_STEP_NUM" ]]
  fi
}

split_words() {
  local str="$1"
  local -n out_ref="$2"
  out_ref=()
  read -r -a out_ref <<<"$str"
}

SEED_ARR=()
split_words "$SEEDS" SEED_ARR
THRESHOLD_SWEEP_ARR=()
split_words "$ROUTER_THRESHOLD_SWEEP" THRESHOLD_SWEEP_ARR
EXTRA_OVERRIDES_ARR=()
if [[ -n "$EXTRA_OVERRIDES" ]]; then
  split_words "$EXTRA_OVERRIDES" EXTRA_OVERRIDES_ARR
fi

CLEAN_RUN_DIR="data/runs/${RUN_ID}_clean"
NOISY_OUT_DEFAULT="data/trajectories/${RUN_ID}/trajectories.jsonl"
if [[ -z "$NOISY_TRAJECTORIES" ]]; then
  NOISY_TRAJECTORIES="$NOISY_OUT_DEFAULT"
fi

POLICY_DIR="outputs/policy/${RUN_ID}"
POLICY_FINAL="${POLICY_DIR}/final"
VERIFIER_DIR="outputs/verifier/${RUN_ID}"
VERIFIER_FINAL="${VERIFIER_DIR}/final/verifier.pt"
CANDIDATES_PATH="data/candidates/${RUN_ID}.jsonl"
ROUTER_FEATURES_PATH="data/router_features/${RUN_ID}.jsonl"
ROUTER_DIR="outputs/router/${RUN_ID}"
ROUTER_FINAL="${ROUTER_DIR}/router_final.pt"
EVAL_DIR="outputs/eval/${RUN_ID}"

mkdir -p "$(dirname "$NOISY_TRAJECTORIES")" "$POLICY_DIR" "$VERIFIER_DIR" \
  "$(dirname "$CANDIDATES_PATH")" "$(dirname "$ROUTER_FEATURES_PATH")" "$ROUTER_DIR" "$EVAL_DIR"

if [[ -z "$CLEAN_TRAJECTORIES" ]]; then
  CLEAN_TRAJECTORIES="${CLEAN_RUN_DIR}/trajectories.jsonl"
fi

COMMON_OVERRIDES=(
  "policy.model_name=${SLM_MODEL}"
  "verifier.mode=trained"
  "verifier.trained.backbone=${VERIFIER_MODEL}"
  "policy.quantization.load_in_4bit=false"
  "policy.max_seq_len=2048"
  "logging.wandb_mode=disabled"
)
COMMON_OVERRIDES+=("${EXTRA_OVERRIDES_ARR[@]}")

# Keep logs and run manifest for experiment/version tracking.
GIT_COMMIT="unknown"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_COMMIT="$(git rev-parse HEAD || echo unknown)"
fi

cat > "${META_ROOT}/manifest.env" <<EOF
RUN_ID=${RUN_ID}
DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SLURM_JOB_ID=${SLURM_JOB_ID:-}
BENCHMARK=${BENCHMARK}
SLM_MODEL_INPUT=${SLM_MODEL_INPUT}
SLM_MODEL_RESOLVED=${SLM_MODEL}
VERIFIER_MODEL_INPUT=${VERIFIER_MODEL_INPUT}
VERIFIER_MODEL_RESOLVED=${VERIFIER_MODEL}
CONFIG_CLEAN=${CONFIG_CLEAN}
CONFIG_NOISY=${CONFIG_NOISY}
CLEAN_TRAJECTORIES=${CLEAN_TRAJECTORIES}
NOISY_TRAJECTORIES=${NOISY_TRAJECTORIES}
POLICY_DIR=${POLICY_DIR}
VERIFIER_DIR=${VERIFIER_DIR}
CANDIDATES_PATH=${CANDIDATES_PATH}
ROUTER_FEATURES_PATH=${ROUTER_FEATURES_PATH}
ROUTER_DIR=${ROUTER_DIR}
EVAL_DIR=${EVAL_DIR}
NUM_GPUS=${NUM_GPUS}
SEEDS=${SEEDS}
GIT_COMMIT=${GIT_COMMIT}
EOF

run_array_cmd() {
  local step_name="$1"
  shift
  local -a cmd=("$@")

  local log_file="${LOG_ROOT}/${step_name}.log"
  local cmd_str
  printf -v cmd_str '%q ' "${cmd[@]}"

  echo "[$(date +%F_%T)] ${step_name}: ${cmd_str}" | tee -a "${META_ROOT}/commands.log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN ${step_name}: ${cmd_str}" | tee -a "$log_file"
    return 0
  fi

  if [[ "$USE_CONTAINER" == "1" ]]; then
    if [[ ! -f "$OVERLAY" ]]; then
      echo "Container overlay not found: $OVERLAY"
      return 1
    fi
    if [[ ! -f "$IMAGE" ]]; then
      echo "Container image not found: $IMAGE"
      return 1
    fi

    singularity exec --nv --fakeroot --overlay "${OVERLAY}:ro" "${IMAGE}" \
      /bin/bash -lc "set -euo pipefail; source /ext3/env.sh; conda activate ${ENV_NAME}; cd ${PROJECT_DIR}; source exports.sh; ${cmd_str}" \
      2>&1 | tee "$log_file"
  else
    "${cmd[@]}" 2>&1 | tee "$log_file"
  fi
}

run_stage3_and_4_parallel() {
  local log3="${LOG_ROOT}/step3_train_policy_bc.log"
  local log4="${LOG_ROOT}/step4_train_verifier.log"

  local -a cmd3
  local -a cmd4

  if [[ "$NUM_GPUS" -gt 1 ]]; then
    cmd3=(
      accelerate launch --num_processes="$NUM_GPUS" --multi_gpu
      scripts/train_policy.py
      --config "$CONFIG_NOISY"
      --output "$POLICY_DIR"
      --stage bc
      --trajectories "$NOISY_TRAJECTORIES"
      --overrides
      "${COMMON_OVERRIDES[@]}"
      training.bc.batch_size=8
      training.bc.gradient_accumulation_steps=4
      training.bc.learning_rate=1e-5
      training.bc.num_workers=8
    )

    cmd4=(
      accelerate launch --num_processes="$NUM_GPUS" --multi_gpu
      scripts/train_verifier.py
      --config "$CONFIG_NOISY"
      --output "$VERIFIER_DIR"
      --trajectories "$NOISY_TRAJECTORIES"
      --overrides
      "${COMMON_OVERRIDES[@]}"
      training.verifier.epochs=3
      training.verifier.batch_size=32
      training.verifier.gradient_accumulation_steps=4
    )
  else
    cmd3=(
      python scripts/train_policy.py
      --config "$CONFIG_NOISY"
      --output "$POLICY_DIR"
      --stage bc
      --trajectories "$NOISY_TRAJECTORIES"
      --overrides
      "${COMMON_OVERRIDES[@]}"
      training.bc.batch_size=8
      training.bc.gradient_accumulation_steps=4
      training.bc.learning_rate=1e-5
      training.bc.num_workers=8
    )

    cmd4=(
      python scripts/train_verifier.py
      --config "$CONFIG_NOISY"
      --output "$VERIFIER_DIR"
      --trajectories "$NOISY_TRAJECTORIES"
      --overrides
      "${COMMON_OVERRIDES[@]}"
      training.verifier.epochs=3
      training.verifier.batch_size=8
      training.verifier.gradient_accumulation_steps=8
    )
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    run_array_cmd "step3_train_policy_bc" "${cmd3[@]}"
    run_array_cmd "step4_train_verifier" "${cmd4[@]}"
    return 0
  fi

  local cmd3_str
  local cmd4_str
  printf -v cmd3_str '%q ' "${cmd3[@]}"
  printf -v cmd4_str '%q ' "${cmd4[@]}"

  echo "[$(date +%F_%T)] step3_train_policy_bc(parallel): ${cmd3_str}" | tee -a "${META_ROOT}/commands.log"
  echo "[$(date +%F_%T)] step4_train_verifier(parallel): ${cmd4_str}" | tee -a "${META_ROOT}/commands.log"

  set +e
  if [[ "$USE_CONTAINER" == "1" ]]; then
    singularity exec --nv --fakeroot --overlay "${OVERLAY}:ro" "${IMAGE}" \
      /bin/bash -lc "set -euo pipefail; source /ext3/env.sh; conda activate ${ENV_NAME}; cd ${PROJECT_DIR}; source exports.sh; ${cmd3_str}" \
      >"$log3" 2>&1 &
    pid3=$!

    singularity exec --nv --fakeroot --overlay "${OVERLAY}:ro" "${IMAGE}" \
      /bin/bash -lc "set -euo pipefail; source /ext3/env.sh; conda activate ${ENV_NAME}; cd ${PROJECT_DIR}; source exports.sh; ${cmd4_str}" \
      >"$log4" 2>&1 &
    pid4=$!
  else
    bash -lc "$cmd3_str" >"$log3" 2>&1 &
    pid3=$!
    bash -lc "$cmd4_str" >"$log4" 2>&1 &
    pid4=$!
  fi

  wait "$pid3"; rc3=$?
  wait "$pid4"; rc4=$?
  set -e

  if [[ $rc3 -ne 0 || $rc4 -ne 0 ]]; then
    echo "Parallel stage failure: stage3_rc=$rc3 stage4_rc=$rc4"
    echo "Check logs: $log3 and $log4"
    return 1
  fi

  cat "$log3"
  cat "$log4"
}

# Helpful runtime env defaults from your successful command history.
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12

if should_run 0; then
  run_array_cmd "step0_smoke" \
    python scripts/collect_trajectories.py \
      --config "$CONFIG_CLEAN" \
      --output data/smoke_test_eval \
      --num-episodes 1 \
      --seeds "${SEED_ARR[0]}" \
      --overrides "${COMMON_OVERRIDES[@]}"
fi

if should_run 1; then
  run_array_cmd "step1_collect_clean" \
    python scripts/collect_trajectories.py \
      --config "$CONFIG_CLEAN" \
      --output "$CLEAN_RUN_DIR" \
      --num-episodes 500 \
      --seeds "${SEED_ARR[@]}" \
      --num-workers "$NUM_WORKERS" \
      --overrides "${COMMON_OVERRIDES[@]}"
fi

if should_run 2; then
  if [[ ! -f "$CLEAN_TRAJECTORIES" ]]; then
    echo "Missing clean trajectories: $CLEAN_TRAJECTORIES"
    exit 1
  fi

  run_array_cmd "step2_perturb_noisy" \
    python -m scripts.generate_perturbations \
      --config "$CONFIG_NOISY" \
      --input "$CLEAN_TRAJECTORIES" \
      --output "$NOISY_TRAJECTORIES" \
      --seeds "${SEED_ARR[@]}" \
      --include-clean
fi

if should_run 3 && should_run 4 && [[ -z "$ONLY_STEP_NUM" ]]; then
  if [[ ! -f "$NOISY_TRAJECTORIES" ]]; then
    echo "Missing noisy trajectories: $NOISY_TRAJECTORIES"
    exit 1
  fi
  run_stage3_and_4_parallel
else
  if should_run 3; then
    if [[ ! -f "$NOISY_TRAJECTORIES" ]]; then
      echo "Missing noisy trajectories: $NOISY_TRAJECTORIES"
      exit 1
    fi

    if [[ "$NUM_GPUS" -gt 1 ]]; then
      run_array_cmd "step3_train_policy_bc" \
        accelerate launch --num_processes="$NUM_GPUS" --multi_gpu \
          scripts/train_policy.py \
          --config "$CONFIG_NOISY" \
          --output "$POLICY_DIR" \
          --stage bc \
          --trajectories "$NOISY_TRAJECTORIES" \
          --overrides \
          "${COMMON_OVERRIDES[@]}" \
          training.bc.batch_size=8 \
          training.bc.gradient_accumulation_steps=4 \
          training.bc.learning_rate=1e-5 \
          training.bc.num_workers=8
    else
      run_array_cmd "step3_train_policy_bc" \
        python scripts/train_policy.py \
          --config "$CONFIG_NOISY" \
          --output "$POLICY_DIR" \
          --stage bc \
          --trajectories "$NOISY_TRAJECTORIES" \
          --overrides \
          "${COMMON_OVERRIDES[@]}" \
          training.bc.batch_size=8 \
          training.bc.gradient_accumulation_steps=4 \
          training.bc.learning_rate=1e-5 \
          training.bc.num_workers=8
    fi
  fi

  if should_run 4; then
    if [[ ! -f "$NOISY_TRAJECTORIES" ]]; then
      echo "Missing noisy trajectories: $NOISY_TRAJECTORIES"
      exit 1
    fi

    if [[ "$NUM_GPUS" -gt 1 ]]; then
      run_array_cmd "step4_train_verifier" \
        accelerate launch --num_processes="$NUM_GPUS" --multi_gpu \
          scripts/train_verifier.py \
          --config "$CONFIG_NOISY" \
          --output "$VERIFIER_DIR" \
          --trajectories "$NOISY_TRAJECTORIES" \
          --overrides \
          "${COMMON_OVERRIDES[@]}" \
          training.verifier.epochs=3 \
          training.verifier.batch_size=32 \
          training.verifier.gradient_accumulation_steps=4
    else
      run_array_cmd "step4_train_verifier" \
        python scripts/train_verifier.py \
          --config "$CONFIG_NOISY" \
          --output "$VERIFIER_DIR" \
          --trajectories "$NOISY_TRAJECTORIES" \
          --overrides \
          "${COMMON_OVERRIDES[@]}" \
          training.verifier.epochs=3 \
          training.verifier.batch_size=8 \
          training.verifier.gradient_accumulation_steps=8
    fi
  fi
fi

if should_run 5; then
  run_array_cmd "step5_generate_candidates" \
    bash scripts/launch_candidates.sh "$NUM_GPUS" \
      --config "$CONFIG_NOISY" \
      --policy-path "$POLICY_FINAL" \
      --verifier-path "$VERIFIER_FINAL" \
      --trajectories "$NOISY_TRAJECTORIES" \
      --output "$CANDIDATES_PATH" \
      --no-resume \
      --K 5 \
      --batch-size 8 \
      --gen-micro-batch-size 2 \
      --verifier-batch-size 2 \
      --max-new-tokens 96 \
      --tokenize-cache-size 0 \
      --prefetch-depth 2 \
      --pipeline-depth 2 \
      --gpu-keepalive-interval 0.1 \
      --write-queue-size 2000 \
      --write-flush-every 128 \
      --overrides "${COMMON_OVERRIDES[@]}" \
      inference.temperature=0.95
fi

if should_run 6; then
  if [[ "$NUM_GPUS" -gt 1 ]]; then
    run_array_cmd "step6_train_policy_preference" \
      accelerate launch --num_processes="$NUM_GPUS" --multi_gpu \
        scripts/train_policy.py \
        --config "$CONFIG_NOISY" \
        --output "$POLICY_DIR" \
        --stage preference \
        --preference-data "$CANDIDATES_PATH" \
        --resume "$POLICY_FINAL" \
        --overrides \
        "${COMMON_OVERRIDES[@]}" \
        training.preference.batch_size=2 \
        training.preference.gradient_accumulation_steps=16 \
        training.preference.concat_pairs=false
  else
    run_array_cmd "step6_train_policy_preference" \
      python scripts/train_policy.py \
        --config "$CONFIG_NOISY" \
        --output "$POLICY_DIR" \
        --stage preference \
        --preference-data "$CANDIDATES_PATH" \
        --resume "$POLICY_FINAL" \
        --overrides \
        "${COMMON_OVERRIDES[@]}" \
        training.preference.batch_size=2 \
        training.preference.gradient_accumulation_steps=16 \
        training.preference.concat_pairs=false
  fi
fi

if should_run 7; then
  run_array_cmd "step7_generate_router_features" \
    bash scripts/launch_router_features.sh "$NUM_GPUS" \
      --config "$CONFIG_NOISY" \
      --policy-path "$POLICY_FINAL" \
      --trajectories "$NOISY_TRAJECTORIES" \
      --output "$ROUTER_FEATURES_PATH" \
      --K 5 \
      --batch-size 4 \
      --overrides "${COMMON_OVERRIDES[@]}"
fi

if should_run 8; then
  run_array_cmd "step8_train_router" \
    python scripts/train_router.py \
      --config "$CONFIG_NOISY" \
      --features "$ROUTER_FEATURES_PATH" \
      --output "$ROUTER_DIR" \
      --overrides \
      "${COMMON_OVERRIDES[@]}" \
      training.router.cost_llm=20 \
      training.router.cvar_epsilon=0.1
fi

if should_run 9; then
  run_array_cmd "step9_evaluate" \
    python scripts/evaluate.py \
      --config "$CONFIG_NOISY" \
      --features "$ROUTER_FEATURES_PATH" \
      --trajectories "$NOISY_TRAJECTORIES" \
      --router-path "$ROUTER_FINAL" \
      --output "$EVAL_DIR" \
      --methods r2v slm_only llm_only entropy_router \
      --router-threshold-sweep "${THRESHOLD_SWEEP_ARR[@]}" \
      --overrides "${COMMON_OVERRIDES[@]}"
fi

echo "Completed run_id=${RUN_ID}"
echo "Manifest: ${META_ROOT}/manifest.env"
echo "Logs: ${LOG_ROOT}"
