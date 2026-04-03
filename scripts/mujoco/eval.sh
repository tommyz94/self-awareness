#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

DEFAULT_CLIENT="${DEFAULT_CLIENT:-ollama}"
DEFAULT_NAV_MODEL="${DEFAULT_NAV_MODEL:-qwen3.5:37b}"
DEFAULT_EXPERIMENT="${DEFAULT_EXPERIMENT:-e1}"
DEFAULT_RUN_ID="${DEFAULT_RUN_ID:-01}"

slugify_model() {
  local value="$1"
  value="${value//:/_}"
  value="${value//\//_}"
  value="${value//\\/_}"
  printf '%s\n' "${value}"
}

NAV_SLUG="$(slugify_model "${DEFAULT_NAV_MODEL}")"
DEFAULT_RUN_JSON="outputs/mujoco/${DEFAULT_CLIENT}/${NAV_SLUG}/runs/${DEFAULT_EXPERIMENT}/run${DEFAULT_RUN_ID}.json"
RUN_JSON="${DEFAULT_RUN_JSON}"

if [[ $# -gt 0 && "${1}" != --* ]]; then
  RUN_JSON="$1"
  shift
fi

SAVE_PATH="${RUN_JSON%.json}_judge.json"

python -m experiments.evaluation.judge \
  "${RUN_JSON}" \
  --model gpt-oss:20b \
  --save "${SAVE_PATH}" \
  --metric M1 M2 M3 M4 \
  --print_reasoning \
  --include-inner-reasoning \
  "$@"
