#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BACKUP_DEST="${BACKUP_DEST:?Set BACKUP_DEST, for example gdrive:emergent-study/runpod}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is required for remote backup."
  exit 1
fi

rclone copy "$ROOT" "$BACKUP_DEST/repository" \
  --transfers "${RCLONE_TRANSFERS:-8}" \
  --checkers "${RCLONE_CHECKERS:-16}" \
  --create-empty-src-dirs \
  --exclude '.git/**' \
  --exclude '__pycache__/**' \
  --exclude '*.pyc' \
  --progress
