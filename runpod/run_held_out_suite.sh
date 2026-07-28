#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GENERATED_ROOT="${GENERATED_ROOT:-$ROOT/experiment/followup_study/generated_runpod}"
BENCH_VENV="${BENCH_VENV:-/workspace/venvs/emergent-benchmarks}"
MODEL_ALIAS="${MODEL_ALIAS_FILTER:?Set MODEL_ALIAS_FILTER}"
MANIFEST="$GENERATED_ROOT/manifests/held_out_models_${MODEL_ALIAS}.json"

source "$BENCH_VENV/bin/activate"
cd "$ROOT"
python experiment/followup_study/prepare_held_out_manifest.py \
  --spec experiment/followup_study/study_spec_runpod.json \
  --train-runs "$GENERATED_ROOT/manifests/train_runs.json" \
  --control-runs "$GENERATED_ROOT/manifests/matched_control_runs.json" \
  --checkpoint-root "$GENERATED_ROOT/ckpt" \
  --output "$MANIFEST" \
  --model-alias "$MODEL_ALIAS" \
  --require-checkpoints

python experiment/followup_study/run_lm_eval_suite.py \
  --manifest "$MANIFEST" \
  --output-root "$GENERATED_ROOT/held_out/lm_eval" \
  --model-alias "$MODEL_ALIAS" \
  --allow-code-execution \
  "$@"
