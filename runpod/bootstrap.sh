#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/emergent-misalignment}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

mkdir -p "$VENV_DIR" "$PIP_CACHE_DIR" "$HF_HOME"
export PIP_CACHE_DIR HF_HOME

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install \
  -c "$ROOT/runpod/constraints.txt" \
  -r "$ROOT/experiment/requirements_colab.txt"

cd "$ROOT"
python experiment/followup_study/generate_assets.py \
  --spec_path experiment/followup_study/study_spec_runpod.json
python runpod/verify_assets.py
python -m compileall -q experiment
python -m unittest discover \
  -s experiment/followup_study/tests \
  -p 'test_*.py'

echo "Bootstrap complete."
echo "Activate with: source $VENV_DIR/bin/activate"
