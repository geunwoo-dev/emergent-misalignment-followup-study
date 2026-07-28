#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="${BENCHMARK_ROOT:-/workspace/benchmarks}"
HALU_COMMIT="b7253db3cdaa0ab2c382f92b26b390109174f77e"
HARM_COMMIT="8e1604d1171fe8a48d8febecd22f600e462bdcdd"

checkout_repo() {
  local url="$1"
  local directory="$2"
  local commit="$3"
  if [ ! -d "$directory/.git" ]; then
    git clone "$url" "$directory"
  fi
  git -C "$directory" fetch origin "$commit"
  git -C "$directory" checkout --detach "$commit"
  actual=$(git -C "$directory" rev-parse HEAD)
  if [ "$actual" != "$commit" ]; then
    echo "Commit mismatch for $directory: $actual"
    exit 1
  fi
}

mkdir -p "$BENCHMARK_ROOT"
checkout_repo \
  https://github.com/RUCAIBox/HaluEval.git \
  "$BENCHMARK_ROOT/HaluEval" \
  "$HALU_COMMIT"
checkout_repo \
  https://github.com/centerforaisafety/HarmBench.git \
  "$BENCHMARK_ROOT/HarmBench" \
  "$HARM_COMMIT"

echo "Official benchmark data ready: $BENCHMARK_ROOT"
