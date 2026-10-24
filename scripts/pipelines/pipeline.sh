#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Optional reasoning-judge pipeline; benchmark scores use evaluate_selected_runs.py.
: "${NAV_MODEL:?Set NAV_MODEL to the navigation model ID.}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to an installed Ollama judge model.}"
EXPERIMENT="${EXPERIMENT:-e1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper}"
mkdir -p "${OUTPUT_ROOT}"
RUN_ROOT="$(mktemp -d "${OUTPUT_ROOT%/}/pipeline.XXXXXX")"

python -u main.py experiment="${EXPERIMENT}" client="${CLIENT:-ollama}" \
  nav="${NAV_MODEL}" mujoco.headless=true "$@" outputs.dir="${RUN_ROOT}"

run_jsons=()
while IFS= read -r run_json; do
  run_jsons+=("${run_json}")
done < <(find "${RUN_ROOT}" -type f -name 'run[0-9][0-9].json' | sort)
if [[ ${#run_jsons[@]} -eq 0 ]]; then
  echo "No run JSON files found under ${RUN_ROOT}" >&2
  exit 1
fi

for run_json in "${run_jsons[@]}"; do
  python -m experiments.evaluation.judge "${run_json}" \
    --model "${JUDGE_MODEL}" --save "${run_json%.json}_judge.json" \
    --include-inner-reasoning
done
