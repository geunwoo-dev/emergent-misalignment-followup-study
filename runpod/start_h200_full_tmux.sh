#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SESSION_NAME="${SESSION_NAME:-h200-full}"
RUNTIME_ENV=""

cleanup() {
  if [ -n "$RUNTIME_ENV" ] && [ -e "$RUNTIME_ENV" ]; then
    rm -f "$RUNTIME_ENV"
  fi
}
trap cleanup EXIT

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required."
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

: "${HF_TOKEN:?Export HF_TOKEN before launching the full pipeline}"
RUNTIME_ENV=$(mktemp "${TMPDIR:-/tmp}/h200-full-env.XXXXXX")
chmod 600 "$RUNTIME_ENV"

for name in \
  ROOT GENERATED_ROOT MAIN_VENV BENCH_VENV BENCHMARK_ROOT HF_HOME PIP_CACHE_DIR \
  HF_TOKEN BACKUP_DEST BACKUP_AFTER_EACH_STAGE AUTO_BACKUP_AFTER_FULL \
  MIN_FREE_DISK_GB; do
  if [ -n "${!name:-}" ]; then
    printf 'export %s=%q\n' "$name" "${!name}" >> "$RUNTIME_ENV"
  fi
done

tmux new-session -d -s "$SESSION_NAME" \
  "source '$RUNTIME_ENV' && rm -f '$RUNTIME_ENV' && cd '$ROOT' && exec bash runpod/run_h200_full.sh"

for _ in {1..50}; do
  if [ ! -e "$RUNTIME_ENV" ]; then
    break
  fi
  sleep 0.1
done
if [ -e "$RUNTIME_ENV" ]; then
  echo "tmux session did not consume its runtime environment."
  exit 1
fi
RUNTIME_ENV=""

echo "Started three-GPU H200 pipeline in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Status file: $ROOT/logs/h200_full/latest_status.txt"
