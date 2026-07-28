#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GENERATED_ROOT="${GENERATED_ROOT:-$ROOT/experiment/followup_study/generated_runpod}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/workspace/benchmarks}"
MAIN_VENV="${MAIN_VENV:-/workspace/venvs/emergent-misalignment}"
MODEL_ALIAS="${MODEL_ALIAS_FILTER:?Set MODEL_ALIAS_FILTER}"
MANIFEST="$GENERATED_ROOT/manifests/held_out_models_${MODEL_ALIAS}.json"

source "$MAIN_VENV/bin/activate"
cd "$ROOT"
python experiment/followup_study/prepare_held_out_manifest.py \
  --spec experiment/followup_study/study_spec_runpod.json \
  --train-runs "$GENERATED_ROOT/manifests/train_runs.json" \
  --control-runs "$GENERATED_ROOT/manifests/matched_control_runs.json" \
  --checkpoint-root "$GENERATED_ROOT/ckpt" \
  --output "$MANIFEST" \
  --model-alias "$MODEL_ALIAS" \
  --require-checkpoints

python experiment/followup_study/run_custom_held_out.py \
  --repo-root "$ROOT" \
  --manifest "$MANIFEST" \
  --halueval-qa "$BENCHMARK_ROOT/HaluEval/data/qa_data.json" \
  --halueval-instruction "$BENCHMARK_ROOT/HaluEval/evaluation/qa/qa_evaluation_instruction.txt" \
  --harmbench-behaviors "$BENCHMARK_ROOT/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv" \
  --output-root "$GENERATED_ROOT/held_out/custom" \
  --model-alias "$MODEL_ALIAS" \
  "$@"
