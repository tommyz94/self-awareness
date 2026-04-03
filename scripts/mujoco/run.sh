#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

python main.py \
  env=mujoco \
  nav=qwen3.5:9b \
  mujoco.headless=false \
  max_steps=12 \
  history_steps=2 \
  third_person=true \
  outputs.heatmap=true \
  print_reasoning=true \
  "$@"
