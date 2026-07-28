#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAIN_VENV="${MAIN_VENV:-/workspace/venvs/emergent-misalignment}"
COMMONSENSEQA_REVISION="94630fe30dad47192a8546eb75f094926d47e155"

source "$MAIN_VENV/bin/activate"
cd "$ROOT"
python experiment/followup_study/build_commonsenseqa_dataset.py \
  --dataset_id tau/commonsense_qa \
  --revision "$COMMONSENSEQA_REVISION" \
  --split train \
  --output_dir experiment/dataset/mistake_commonsenseqa \
  --seed 2026 \
  --minimum_examples 5000 \
  --audit_size 100

echo
echo "Dataset files were built but the extension is NOT active."
echo "Complete experiment/dataset/mistake_commonsenseqa/manual_audit.csv,"
echo "then approve audit_approval.json before running activate_generality_extension.sh."
