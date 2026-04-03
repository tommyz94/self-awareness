#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

python -u main.py \
  env=mujoco \
  client=ollama \
  nav=qwen3.5:37b \
  mujoco.headless=true \
  +experiment=e1 \
  max_steps=15 \
  print_reasoning=true \
  history_steps=6 \
  num_runs=2 \
  third_person=false \
  outputs.heatmap=true \
  "$@"
