#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

PROJECT_ID="${PROJECT_ID:-gemini-self-perspetion}"
LOCATION="${LOCATION:-us-central1}"
MODEL="${MODEL:-gemini-2.5-flash}"
EXPERIMENT_ID="${EXPERIMENT_ID:-experimental/direct_perception}"
NUM_RUNS="${NUM_RUNS:-1}"
MAX_STEPS="${MAX_STEPS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/google_vertex_tiny}"

if [[ $# -gt 0 ]]; then
  EXPERIMENT_ID="$1"
fi

case "$EXPERIMENT_ID" in
  experimental/direct_perception)
    # e0 is already the cheapest end-to-end sanity check; keep its 1-step nature.
    MAX_STEPS=1
    ;;
  e1)
    # e1 exercises the standard mirror-room task path but stays intentionally tiny.
    ;;
  *)
    echo "Unsupported experiment '$EXPERIMENT_ID'. Use experimental/direct_perception or e1." >&2
    exit 1
    ;;
esac

echo "Smoke check: Vertex AI Gemini"
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev/google_vertex_smoke.py" \
  --project "$PROJECT_ID" \
  --location "$LOCATION" \
  --model "$MODEL" \
  --no-image

echo
echo "Launching tiny experiment"
echo "  experiment=$EXPERIMENT_ID"
echo "  model=$MODEL"
echo "  project=$PROJECT_ID"
echo "  location=$LOCATION"
echo "  num_runs=$NUM_RUNS"
echo "  max_steps=$MAX_STEPS"
echo "  output_dir=$OUTPUT_DIR"

cd "$ROOT_DIR"
"$PYTHON_BIN" main.py \
  experiment="$EXPERIMENT_ID" \
  client=google \
  nav="$MODEL" \
  num_runs="$NUM_RUNS" \
  max_steps="$MAX_STEPS" \
  mujoco.headless=true \
  outputs.dir="$OUTPUT_DIR" \
  outputs.heatmap=false \
  outputs.save_obs=false \
  google.use_vertexai=true \
  google.project="$PROJECT_ID" \
  google.location="$LOCATION"
