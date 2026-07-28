#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?Usage: runpod/run_family_worker.sh <stage> <model-alias> [gpu-index]}"
MODEL_ALIAS="${2:?Usage: runpod/run_family_worker.sh <stage> <model-alias> [gpu-index]}"
GPU_INDEX="${3:-0}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export MODEL_ALIAS_FILTER="$MODEL_ALIAS"
export WORKER_ID="$MODEL_ALIAS"
export GPU="$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
exec bash "$ROOT/runpod/run_stage.sh" "$STAGE"
