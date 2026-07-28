#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAIN_VENV="${MAIN_VENV:-/workspace/venvs/emergent-misalignment}"

source "$MAIN_VENV/bin/activate"
cd "$ROOT"
python experiment/followup_study/activate_generality_extension.py \
  --spec_path experiment/followup_study/study_spec_runpod.json \
  --dataset_dir experiment/dataset/mistake_commonsenseqa
python experiment/followup_study/generate_assets.py \
  --spec_path experiment/followup_study/study_spec_runpod.json

python - <<'PY'
import json
from pathlib import Path

root = Path("experiment/followup_study/generated_runpod/manifests")
runs = json.loads((root / "train_runs.json").read_text())
extension = [run for run in runs if run["dataset"] == "mistake_commonsenseqa"]
if len(extension) != 6:
    raise SystemExit(f"Expected 6 extension runs, found {len(extension)}")
print("Generality extension active: 6 training runs")
PY
