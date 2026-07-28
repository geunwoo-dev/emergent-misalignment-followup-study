#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GENERATED_ROOT="${GENERATED_ROOT:-$ROOT/experiment/followup_study/generated_runpod}"
MAIN_VENV="${MAIN_VENV:-/workspace/venvs/emergent-misalignment}"
STAGE="${1:?Usage: runpod/run_stage.sh <stage-number>}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/runpod}"
STATE_DIR="${STATE_DIR:-$GENERATED_ROOT/stage_state}"

declare -A STAGES=(
  [00]="00_prepare_experiment_data.sh"
  [20]="20_eval_baselines.sh"
  [25]="25_judge_calibration.sh"
  [30]="30_train_models.sh"
  [32]="32_train_method_ablations.sh"
  [35]="35_train_matched_controls.sh"
  [40]="40_eval_finetuned_models.sh"
  [41]="41_eval_matched_controls.sh"
  [42]="42_eval_method_ablations.sh"
  [44]="44_eval_checkpoint_grid.sh"
  [48]="48_eval_held_out_suite.sh"
  [49]="49_eval_custom_held_out.sh"
  [55]="55_detect_critical_points.sh"
  [56]="56_evaluate_temporal_detector.sh"
  [58]="58_compare_matched_controls.sh"
  [70]="70_intervention_early_stop.sh"
  [90]="90_rejudge_claim_validation.sh"
  [91]="91_prepare_human_validation.sh"
  [92]="92_score_human_validation.sh"
)

SCRIPT_NAME="${STAGES[$STAGE]:-}"
if [ -z "$SCRIPT_NAME" ]; then
  echo "Unknown stage: $STAGE"
  printf 'Available stages: %s\n' "${!STAGES[*]}"
  exit 2
fi

if [ ! -f "$MAIN_VENV/bin/activate" ]; then
  echo "Main environment not found: $MAIN_VENV"
  echo "Run: bash runpod/bootstrap.sh"
  exit 1
fi
source "$MAIN_VENV/bin/activate"

mkdir -p "$LOG_DIR" "$STATE_DIR"
WORKER_ID="${WORKER_ID:-${MODEL_ALIAS_FILTER:-all}}"
SAFE_WORKER_ID="${WORKER_ID//[^a-zA-Z0-9_.-]/_}"
STATE_KEY="${STAGE}_${SAFE_WORKER_ID}"
LOCK_DIR="$STATE_DIR/$STATE_KEY.lock"
DONE_FILE="$STATE_DIR/$STATE_KEY.done"
FAILED_FILE="$STATE_DIR/$STATE_KEY.failed"
LOG_FILE="$LOG_DIR/${STATE_KEY}_$(date -u +%Y%m%dT%H%M%SZ).log"

if [ -f "$DONE_FILE" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "Stage $STAGE is already complete. Set FORCE=1 to rerun."
  exit 0
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  lock_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
  if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
    echo "Stage $STAGE is already running with PID $lock_pid."
    exit 1
  fi
  echo "Removing stale lock: $LOCK_DIR"
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi
printf '%s\n' "$$" > "$LOCK_DIR/pid"

cleanup() {
  status=$?
  rm -rf "$LOCK_DIR"
  if [ "$status" -eq 0 ]; then
    date -u +%FT%TZ > "$DONE_FILE"
    rm -f "$FAILED_FILE"
  else
    printf '%s exit=%s log=%s\n' "$(date -u +%FT%TZ)" "$status" "$LOG_FILE" > "$FAILED_FILE"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

export EM_RESUME_TRAINING="${EM_RESUME_TRAINING:-1}"
export EM_EVAL_LOAD_IN_4BIT="${EM_EVAL_LOAD_IN_4BIT:-1}"
export EM_EVAL_MERGE_LORA="${EM_EVAL_MERGE_LORA:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

cd "$ROOT"
echo "stage=$STAGE worker=$WORKER_ID script=$SCRIPT_NAME log=$LOG_FILE"
if [ -n "${RUNPOD_BUDGET_USD:-}" ] && [ -n "${RUNPOD_PRICE_PER_GPU_HOUR:-}" ]; then
  max_seconds=$(python3 "$ROOT/runpod/cost_guard.py" \
    --budget-usd "$RUNPOD_BUDGET_USD" \
    --price-per-gpu-hour "$RUNPOD_PRICE_PER_GPU_HOUR" \
    --gpu-count 1 \
    --print-max-seconds)
  timeout --signal=TERM --kill-after=10m "$max_seconds" \
    bash "$GENERATED_ROOT/runbooks/$SCRIPT_NAME" 2>&1 | tee "$LOG_FILE"
else
  bash "$GENERATED_ROOT/runbooks/$SCRIPT_NAME" 2>&1 | tee "$LOG_FILE"
fi

if [ "${AUTO_BACKUP_AFTER_STAGE:-0}" = "1" ]; then
  : "${BACKUP_DEST:?Set BACKUP_DEST when AUTO_BACKUP_AFTER_STAGE=1}"
  bash "$ROOT/runpod/backup.sh"
fi
