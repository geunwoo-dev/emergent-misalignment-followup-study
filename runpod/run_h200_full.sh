#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GENERATED_ROOT="${GENERATED_ROOT:-$ROOT/experiment/followup_study/generated_runpod}"
MAIN_VENV="${MAIN_VENV:-/workspace/venvs/emergent-misalignment}"
BENCH_VENV="${BENCH_VENV:-/workspace/venvs/emergent-benchmarks}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/workspace/benchmarks}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
LOG_DIR="${FULL_LOG_DIR:-$ROOT/logs/h200_full}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-850}"
BACKUP_AFTER_EACH_STAGE="${BACKUP_AFTER_EACH_STAGE:-0}"
AUTO_BACKUP_AFTER_FULL="${AUTO_BACKUP_AFTER_FULL:-0}"
MIN_H200_GPUS="${MIN_H200_GPUS:-2}"
MAX_PARALLEL_GPUS="${MAX_PARALLEL_GPUS:-3}"
AVAILABLE_GPU_COUNT=0

ALIASES=(
  "llama_3_1_8b_instruct"
  "gemma_2_9b_it"
  "qwen_2_5_7b_instruct"
)

mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/full_$(date -u +%Y%m%dT%H%M%SZ).log"
status_file="$LOG_DIR/latest_status.txt"
current_phase="initializing"
current_stage="none"
active_pids=()

exec > >(tee -a "$log_file") 2>&1

finalize() {
  status=$?
  {
    echo "timestamp=$(date -u +%FT%TZ)"
    echo "phase=$current_phase"
    echo "stage=$current_stage"
    echo "exit_code=$status"
    echo "log_file=$log_file"
  } > "$status_file"
}
trap finalize EXIT

handle_signal() {
  current_phase="interrupted"
  for pid in "${active_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${active_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit 130
}
trap handle_signal INT TERM

export ROOT GENERATED_ROOT MAIN_VENV BENCH_VENV BENCHMARK_ROOT HF_HOME
export EM_RESUME_TRAINING="${EM_RESUME_TRAINING:-1}"
export EM_EVAL_LOAD_IN_4BIT="${EM_EVAL_LOAD_IN_4BIT:-1}"
export EM_EVAL_MERGE_LORA="${EM_EVAL_MERGE_LORA:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# The coordinator performs one backup after all model-family waves finish.
export AUTO_BACKUP_AFTER_STAGE=0

require_environment() {
  if [ ! -x "$MAIN_VENV/bin/python" ]; then
    echo "Main environment not found: $MAIN_VENV"
    exit 1
  fi
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is required."
    exit 1
  fi
  local visible_gpu_count
  visible_gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  if [ "$visible_gpu_count" -lt "$MIN_H200_GPUS" ]; then
    echo "At least $MIN_H200_GPUS visible GPUs are required; found $visible_gpu_count."
    exit 1
  fi
  AVAILABLE_GPU_COUNT="$visible_gpu_count"
  if [ "$AVAILABLE_GPU_COUNT" -gt "$MAX_PARALLEL_GPUS" ]; then
    AVAILABLE_GPU_COUNT="$MAX_PARALLEL_GPUS"
  fi
  echo "Using $AVAILABLE_GPU_COUNT GPUs for ${#ALIASES[@]} model families."
  if [ "$AVAILABLE_GPU_COUNT" -eq 2 ]; then
    echo "Two-GPU mode: Llama and Gemma run first; Qwen follows on GPU 0."
  fi
  if [ "$BACKUP_AFTER_EACH_STAGE" = "1" ] || [ "$AUTO_BACKUP_AFTER_FULL" = "1" ]; then
    : "${BACKUP_DEST:?Set BACKUP_DEST when automatic backup is enabled}"
  fi
}

check_disk() {
  local available_kb required_kb
  available_kb=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
  required_kb=$((MIN_FREE_DISK_GB * 1024 * 1024))
  if [ "$available_kb" -lt "$required_kb" ]; then
    echo "Insufficient free disk: require at least ${MIN_FREE_DISK_GB} GiB."
    df -h "$ROOT"
    exit 1
  fi
}

backup_if_requested() {
  if [ "$BACKUP_AFTER_EACH_STAGE" = "1" ]; then
    current_phase="backup"
    bash "$ROOT/runpod/backup.sh"
  fi
}

run_global_stage() {
  local stage="$1"
  local force="${2:-0}"
  current_phase="global_stage"
  current_stage="$stage"
  check_disk
  echo "===== GLOBAL STAGE $stage ====="
  if [ "$force" = "1" ]; then
    FORCE=1 bash "$ROOT/runpod/run_stage.sh" "$stage"
  else
    bash "$ROOT/runpod/run_stage.sh" "$stage"
  fi
  backup_if_requested
}

run_family_stage() {
  local stage="$1"
  local wave_start=0

  current_phase="family_stage"
  current_stage="$stage"
  check_disk
  echo "===== FAMILY STAGE $stage ($AVAILABLE_GPU_COUNT GPUs) ====="

  while [ "$wave_start" -lt "${#ALIASES[@]}" ]; do
    local pids=()
    local labels=()
    local failed=0
    local gpu_index=0
    local alias_index

    while [ "$gpu_index" -lt "$AVAILABLE_GPU_COUNT" ]; do
      alias_index=$((wave_start + gpu_index))
      if [ "$alias_index" -ge "${#ALIASES[@]}" ]; then
        break
      fi
      local alias="${ALIASES[$alias_index]}"
      echo "[launch] stage=$stage model=$alias gpu=$gpu_index"
      bash "$ROOT/runpod/run_family_worker.sh" "$stage" "$alias" "$gpu_index" &
      pids+=("$!")
      labels+=("$alias")
      gpu_index=$((gpu_index + 1))
    done
    active_pids=("${pids[@]}")

    for index in "${!pids[@]}"; do
      if wait "${pids[$index]}"; then
        echo "[worker-complete] stage=$stage model=${labels[$index]}"
      else
        echo "[worker-failed] stage=$stage model=${labels[$index]}"
        failed=1
      fi
    done
    active_pids=()

    if [ "$failed" -ne 0 ]; then
      echo "At least one worker failed in stage $stage. Re-run this script after fixing the error."
      exit 1
    fi
    wave_start=$((wave_start + AVAILABLE_GPU_COUNT))
  done
  backup_if_requested
}

prepare_runtime() {
  current_phase="preflight"
  current_stage="preflight"
  source "$MAIN_VENV/bin/activate"
  cd "$ROOT"
  python experiment/followup_study/generate_assets.py \
    --spec_path experiment/followup_study/study_spec_runpod.json
  python runpod/verify_assets.py
  python runpod/preflight.py --output "$LOG_DIR/preflight.json"
}

prepare_benchmarks() {
  current_phase="benchmark_setup"
  current_stage="benchmark_setup"
  if [ ! -x "$BENCH_VENV/bin/python" ] \
    || [ ! -f "$BENCHMARK_ROOT/HaluEval/data/qa_data.json" ] \
    || [ ! -f "$BENCHMARK_ROOT/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv" ]; then
    bash "$ROOT/runpod/bootstrap_benchmarks.sh"
  else
    echo "[skip] benchmark environment and pinned datasets already exist."
  fi
}

echo "H200 full pipeline started: $(date -u +%FT%TZ)"
echo "root=$ROOT generated_root=$GENERATED_ROOT"
echo "main_venv=$MAIN_VENV bench_venv=$BENCH_VENV benchmark_root=$BENCHMARK_ROOT"
nvidia-smi
df -h "$ROOT"

require_environment
prepare_runtime

# Quality gates and all core confirmatory compute.
run_global_stage 00
# Calibration is cheap and must be revalidated against the exact checked-out prompts.
run_global_stage 25 1
run_family_stage 30
run_family_stage 35
run_family_stage 20
run_family_stage 40
run_family_stage 41
run_family_stage 44
run_global_stage 55
run_global_stage 56
run_global_stage 58
run_family_stage 70

prepare_benchmarks
run_family_stage 48
run_family_stage 49

if [ "$AUTO_BACKUP_AFTER_FULL" = "1" ]; then
  current_phase="final_backup"
  current_stage="backup"
  bash "$ROOT/runpod/backup.sh"
fi

current_phase="complete_core_pipeline"
current_stage="complete"
echo "H200 core pipeline completed: $(date -u +%FT%TZ)"
echo "Next gated work: review Tier-1 results, then explicitly approve stages 32/42."
echo "Stages 90-92 remain blocked on a locked claim manifest and API credentials."
echo "Status: $status_file"
