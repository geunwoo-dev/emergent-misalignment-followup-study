#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAIN_VENV="${MAIN_VENV:-/workspace/venvs/emergent-misalignment}"
CONFIG_PATH="${1:?Usage: runpod/run_single_training.sh <training-config>}"
GPU="${GPU:-0}"

if [ ! -f "$MAIN_VENV/bin/activate" ]; then
  echo "Main environment not found: $MAIN_VENV"
  echo "Run runpod/bootstrap.sh with VENV_DIR and MAIN_VENV on persistent storage."
  exit 1
fi
source "$MAIN_VENV/bin/activate"

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$ROOT/$CONFIG_PATH"
fi
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Training config not found: $CONFIG_PATH"
  exit 1
fi

output_dir=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' \
  "$CONFIG_PATH")
if [ -f "$output_dir/.training_complete" ]; then
  echo "[skip] completed: $CONFIG_PATH"
  exit 0
fi

cd "$ROOT"
echo "[training] config=$CONFIG_PATH gpu=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" python "$ROOT/experiment/training.py" "$CONFIG_PATH"
python "$ROOT/experiment/followup_study/slim_checkpoints.py" \
  --run-dir "$output_dir" \
  --drop-training-state
touch "$output_dir/.training_complete"
echo "[complete] $output_dir"
