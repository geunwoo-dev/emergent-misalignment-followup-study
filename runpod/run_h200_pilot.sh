#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GENERATED_ROOT="${GENERATED_ROOT:-$ROOT/experiment/followup_study/generated_runpod}"
MAIN_VENV="${MAIN_VENV:-/workspace/venvs/emergent-misalignment}"
GPU="${GPU:-0}"
PILOT_TIMEOUT="${PILOT_TIMEOUT:-3h}"
PILOT_CONFIG="${PILOT_CONFIG:-$GENERATED_ROOT/train_configs/llama_3_1_8b_instruct__mistake_medical__misaligned_2__seed_0.json}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/h200_pilot}"

mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/pilot_$(date -u +%Y%m%dT%H%M%SZ).log"
status_file="$LOG_DIR/latest_status.txt"
current_phase="initializing"

exec > >(tee -a "$log_file") 2>&1

finalize() {
  status=$?
  {
    echo "timestamp=$(date -u +%FT%TZ)"
    echo "phase=$current_phase"
    echo "exit_code=$status"
    echo "log_file=$log_file"
  } > "$status_file"
  if [ "${AUTO_BACKUP_AFTER_PILOT:-0}" = "1" ]; then
    if [ -z "${BACKUP_DEST:-}" ]; then
      echo "AUTO_BACKUP_AFTER_PILOT=1 but BACKUP_DEST is unset."
    elif ! bash "$ROOT/runpod/backup.sh"; then
      echo "Warning: pilot backup failed. Results remain on persistent storage."
    fi
  fi
}
trap finalize EXIT

echo "H200 pilot started: $(date -u +%FT%TZ)"
echo "root=$ROOT generated_root=$GENERATED_ROOT gpu=$GPU timeout=$PILOT_TIMEOUT"
nvidia-smi
df -h "$ROOT"
python3 --version

source "$MAIN_VENV/bin/activate"
cd "$ROOT"
current_phase="asset_validation"
python experiment/followup_study/generate_assets.py \
  --spec_path experiment/followup_study/study_spec_runpod.json
python runpod/verify_assets.py
python runpod/preflight.py --output "$LOG_DIR/preflight.json"

current_phase="judge_calibration"
echo "[pilot 1/2] local judge calibration"
bash runpod/run_stage.sh 25

current_phase="representative_training"
echo "[pilot 2/2] representative confirmatory training run"
if timeout --signal=TERM --kill-after=10m "$PILOT_TIMEOUT" \
  bash runpod/run_single_training.sh "$PILOT_CONFIG"; then
  echo "Representative training run completed."
else
  status=$?
  if [ "$status" -eq 124 ]; then
    current_phase="training_timed_out_resumable"
    echo "Pilot training window expired. The run is incomplete but resumable."
  else
    echo "Pilot training failed with exit code $status."
    exit "$status"
  fi
fi

current_phase="complete"
echo "H200 pilot completed: $(date -u +%FT%TZ)"
echo "Send this log to the experiment owner: $log_file"
