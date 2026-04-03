#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

NAV_MODEL="${NAV_MODEL:-qwen3.5:37b}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss:20b}"
EXPERIMENT="${EXPERIMENT:-e1}"
NUM_RUNS="${NUM_RUNS:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-self-aware}"

slugify_model() {
  local value="$1"
  value="${value//:/_}"
  value="${value//\//_}"
  value="${value//\\/_}"
  printf '%s\n' "${value}"
}

NAV_SLUG="$(slugify_model "${NAV_MODEL}")"

# Experiment outputs live under outputs/<env>/<client>/<nav_slug>/runs/<experiment_type>/.
RUNS_DIR="${REPO_ROOT}/outputs/mujoco/ollama/${NAV_SLUG}/runs/${EXPERIMENT}"

echo "Running ${NUM_RUNS} MuJoCo headless ${EXPERIMENT} runs with ${NAV_MODEL}..."

python -u main.py \
  env=mujoco \
  client=ollama \
  nav="${NAV_MODEL}" \
  mujoco.headless=true \
  +experiment="${EXPERIMENT}" \
  num_runs="${NUM_RUNS}" \
  max_steps=15 \
  history_steps=6 \
  third_person=false \
  outputs.heatmap=true \
  print_reasoning=true

run_jsons=( "${RUNS_DIR}"/run[0-9][0-9].json )
if [[ ${#run_jsons[@]} -eq 0 ]]; then
  echo "No run JSON files found under ${RUNS_DIR}" >&2
  exit 1
fi

echo "Evaluating ${#run_jsons[@]} runs with judge ${JUDGE_MODEL} and logging to W&B project ${WANDB_PROJECT}..."

for run_json in "${run_jsons[@]}"; do
  save_path="${run_json%.json}_judge.json"
  python -m experiments.evaluation.judge \
    "${run_json}" \
    --model "${JUDGE_MODEL}" \
    --save "${save_path}" \
    --metric M1 M2 M3 M4 \
    --include-inner-reasoning \
    --wandb \
    --wandb-project "${WANDB_PROJECT}"
done
