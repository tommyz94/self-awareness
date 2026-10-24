#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set." >&2
  echo "Export it before launching the matrix." >&2
  exit 1
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec ./.venv/bin/python scripts/dev/openrouter_matrix.py \
    --config scripts/dev/openrouter_matrix.gemma426b_3x7.json \
    "$@"
fi

exec ./.venv/bin/python scripts/dev/openrouter_matrix.py \
  --config scripts/dev/openrouter_matrix.gemma426b_3x7.json \
  --launch \
  "$@"
