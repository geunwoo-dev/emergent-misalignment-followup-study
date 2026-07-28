#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?Usage: runpod/run_three_workers.sh <stage>}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SESSION_PREFIX="${SESSION_PREFIX:-em_${STAGE}}"
ALIASES=(
  "llama_3_1_8b_instruct"
  "gemma_2_9b_it"
  "qwen_2_5_7b_instruct"
)

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [ "$gpu_count" -lt 3 ]; then
  echo "run_three_workers.sh requires three visible GPUs; found $gpu_count."
  echo "For three single-GPU pods, run run_family_worker.sh once on each pod."
  exit 1
fi

for index in 0 1 2; do
  alias="${ALIASES[$index]}"
  session="${SESSION_PREFIX}_${alias}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session exists: $session"
    continue
  fi
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && bash runpod/run_family_worker.sh '$STAGE' '$alias' '$index'"
  echo "[started] $session on GPU $index"
done

tmux list-sessions
