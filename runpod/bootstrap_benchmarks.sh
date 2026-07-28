#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BENCH_VENV="${BENCH_VENV:-/workspace/venvs/emergent-benchmarks}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
LM_EVAL_COMMIT="6d642546f4688648fced259eb3302efd36ece5af"

mkdir -p "$BENCH_VENV" "$PIP_CACHE_DIR"
export PIP_CACHE_DIR
if [ ! -x "$BENCH_VENV/bin/python" ]; then
  python3 -m venv "$BENCH_VENV"
fi

source "$BENCH_VENV/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install \
  "lm_eval[hf] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@$LM_EVAL_COMMIT" \
  "accelerate==1.7.0" \
  "datasets==3.6.0" \
  "peft==0.15.1" \
  "bitsandbytes==0.45.5" \
  "transformers==4.52.3"

lm_eval --version
lm_eval ls tasks | grep -E 'truthfulqa_mc1|truthfulqa_mc2|medqa_4options|gsm8k_cot|mbpp'
bash "$ROOT/runpod/prepare_official_benchmarks.sh"
echo "Benchmark environment ready: $BENCH_VENV"
