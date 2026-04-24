#!/bin/bash
set -euo pipefail

# ============================================================
# Usage examples:
#
# Direct host, full e0/e8 overnight sweep:
#   ./overnight_runs.sh --host http://127.0.0.1:11559
#
# Direct host, custom models:
#   ./overnight_runs.sh --host http://127.0.0.1:11559 \
#       --models "llava-phi3:3.8b ministral-3:8b ministral-3:14b gemma4:26b gemma4:31b"
#
# GPU-mapped host, single model:
#   ./overnight_runs.sh --gpu 5 --model gemma4:31b
# ============================================================

# Defaults
GPU=""
HOST=""
MODEL=""
MODELS=("llava-phi3:3.8b" "ministral-3:8b" "ministral-3:14b" "gemma4:26b" "gemma4:31b")
NUM_RUNS=10
EXPERIMENTS=("e0" "e8")
MSR_LEVEL=2
PYTHON_BIN="./.venv/bin/python"
SEED=""
E0_DEFAULT_BODY_STYLE="default"
E0_DEFAULT_SELF_PERCEPTION_MODE="color"
E0_CLOTHES_BODY_STYLE="clothes"
E0_CLOTHES_SELF_PERCEPTION_MODE="description"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --models)
            read -r -a MODELS <<< "$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --experiments)
            read -r -a EXPERIMENTS <<< "$2"
            shift 2
            ;;
        --msr-level)
            MSR_LEVEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --e0-default-body-style)
            E0_DEFAULT_BODY_STYLE="$2"
            shift 2
            ;;
        --e0-default-mode)
            E0_DEFAULT_SELF_PERCEPTION_MODE="$2"
            shift 2
            ;;
        --e0-clothes-body-style)
            E0_CLOTHES_BODY_STYLE="$2"
            shift 2
            ;;
        --e0-clothes-mode)
            E0_CLOTHES_SELF_PERCEPTION_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Normalize single-model usage
if [[ -n "$MODEL" ]]; then
    MODELS=("$MODEL")
fi

# Validate host selection
if [[ -z "$HOST" && -z "$GPU" ]]; then
    echo "ERROR: provide either --host or --gpu"
    echo "Example: --host http://127.0.0.1:11559"
    echo "Allowed GPUs: 2, 4, 5, 6"
    exit 1
fi

if [[ -n "$HOST" && -n "$GPU" ]]; then
    echo "ERROR: use either --host or --gpu, not both"
    exit 1
fi

if [[ -z "$HOST" ]]; then
    case "$GPU" in
        2)
            OLLAMA_HOST="http://127.0.0.1:11556"
            ;;
        4)
            OLLAMA_HOST="http://127.0.0.1:11558"
            ;;
        5)
            OLLAMA_HOST="http://127.0.0.1:11559"
            ;;
        6)
            OLLAMA_HOST="http://127.0.0.1:11560"
            ;;
        *)
            echo "ERROR: unsupported GPU '$GPU'"
            echo "Allowed GPUs: 2, 4, 5, 6"
            exit 1
            ;;
    esac
else
    OLLAMA_HOST="$HOST"
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: no models configured"
    echo "Example: --models \"llava-phi3:3.8b ministral-3:8b\""
    exit 1
fi

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    echo "ERROR: no experiments configured"
    echo "Example: --experiments \"e0 e8\""
    exit 1
fi

if [[ "$E0_DEFAULT_SELF_PERCEPTION_MODE" != "color" && "$E0_DEFAULT_SELF_PERCEPTION_MODE" != "description" ]]; then
    echo "ERROR: --e0-default-mode must be 'color' or 'description'"
    exit 1
fi

if [[ "$E0_CLOTHES_SELF_PERCEPTION_MODE" != "color" && "$E0_CLOTHES_SELF_PERCEPTION_MODE" != "description" ]]; then
    echo "ERROR: --e0-clothes-mode must be 'color' or 'description'"
    exit 1
fi

if [[ "$E0_DEFAULT_BODY_STYLE" != "default" && "$E0_DEFAULT_BODY_STYLE" != "clothes" ]]; then
    echo "ERROR: --e0-default-body-style must be 'default' or 'clothes'"
    exit 1
fi

if [[ "$E0_CLOTHES_BODY_STYLE" != "default" && "$E0_CLOTHES_BODY_STYLE" != "clothes" ]]; then
    echo "ERROR: --e0-clothes-body-style must be 'default' or 'clothes'"
    exit 1
fi

if [[ -n "$GPU" ]]; then
    GPU_LABEL="$GPU"
else
    GPU_LABEL="host"
fi

# Validate GPU arg if provided via mapped-host mode
if [[ -n "$GPU" ]]; then
    case "$GPU" in
        2|4|5|6)
            ;;
        *)
            echo "ERROR: unsupported GPU '$GPU'"
            echo "Allowed GPUs: 2, 4, 5, 6"
            exit 1
            ;;
    esac
fi

# Shared seed for comparability
if [[ -z "$SEED" ]]; then
    SEED=$RANDOM
fi

# Logs
mkdir -p logs
LOG_FILE="logs/overnight_batch_${GPU_LABEL}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================" | tee -a "$LOG_FILE"
echo " Starting Experiment Batch                  " | tee -a "$LOG_FILE"
if [[ -n "$GPU" ]]; then
    echo " GPU backend:  $GPU" | tee -a "$LOG_FILE"
else
    echo " GPU backend:  (direct host override)" | tee -a "$LOG_FILE"
fi
echo " Models:       ${MODELS[*]}" | tee -a "$LOG_FILE"
echo " Experiments:  ${EXPERIMENTS[*]}" | tee -a "$LOG_FILE"
echo " MSR Level:    $MSR_LEVEL" | tee -a "$LOG_FILE"
echo " Runs:         $NUM_RUNS per experiment" | tee -a "$LOG_FILE"
echo " Seed:         $SEED" | tee -a "$LOG_FILE"
echo " Ollama:       $OLLAMA_HOST" | tee -a "$LOG_FILE"
echo " e0 default:   body=$E0_DEFAULT_BODY_STYLE mode=$E0_DEFAULT_SELF_PERCEPTION_MODE" | tee -a "$LOG_FILE"
echo " e0 clothes:   body=$E0_CLOTHES_BODY_STYLE mode=$E0_CLOTHES_SELF_PERCEPTION_MODE" | tee -a "$LOG_FILE"
echo " Log file:     $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Check tunnel/server availability
echo "" | tee -a "$LOG_FILE"
echo "Checking Ollama endpoint..." | tee -a "$LOG_FILE"

if ! curl -s "$OLLAMA_HOST/api/tags" > "/tmp/ollama_tags_${GPU_LABEL}.json"; then
    echo "ERROR: Cannot reach Ollama at $OLLAMA_HOST" | tee -a "$LOG_FILE"
    if [[ -n "$GPU" ]]; then
        echo "Check that the tunnel for GPU $GPU is open." | tee -a "$LOG_FILE"
    fi
    exit 1
fi

echo "Ollama endpoint is reachable." | tee -a "$LOG_FILE"

# Pull/check models on selected backend
for ACTIVE_MODEL in "${MODELS[@]}"; do
    SAFE_MODEL_NAME=$(echo "$ACTIVE_MODEL" | tr ':/' '__')
    echo "" | tee -a "$LOG_FILE"
    echo "Checking/Pulling model: $ACTIVE_MODEL" | tee -a "$LOG_FILE"
    curl -s "$OLLAMA_HOST/api/pull" \
        -d "{\"name\":\"$ACTIVE_MODEL\",\"stream\":false}" \
        > "/tmp/ollama_pull_${GPU_LABEL}_${SAFE_MODEL_NAME}.json"
    echo "Model ready: $ACTIVE_MODEL" | tee -a "$LOG_FILE"
done

# Helper to run one configuration
run_one() {
    local exp="$1"
    local model="$2"
    local variant_label="$3"
    shift 3
    local extra_count=$#
    local -a extra_args=()
    if [[ $extra_count -gt 0 ]]; then
        extra_args=("$@")
    fi
    local -a cmd=(
        "$PYTHON_BIN"
        main.py
        "experiment=$exp"
        "nav=$model"
        client=ollama
        "num_runs=$NUM_RUNS"
        "msr_level=$MSR_LEVEL"
        "seed=$SEED"
        print_reasoning=true
        mujoco.headless=true
        "ollama.host=$OLLAMA_HOST"
    )
    if [[ $extra_count -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "--------------------------------------------" | tee -a "$LOG_FILE"
    echo " RUNNING: MODEL=$model | EXP=$exp | VARIANT=$variant_label | LEVEL=$MSR_LEVEL | SEED=$SEED" | tee -a "$LOG_FILE"
    echo "--------------------------------------------" | tee -a "$LOG_FILE"

    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"

    echo "Completed: MODEL=$model | EXP=$exp | VARIANT=$variant_label" | tee -a "$LOG_FILE"
}

# Run all experiments
for ACTIVE_MODEL in "${MODELS[@]}"; do
    for EXP in "${EXPERIMENTS[@]}"; do
        if [[ "$EXP" == "e0" ]]; then
            run_one \
                "$EXP" \
                "$ACTIVE_MODEL" \
                "default_${E0_DEFAULT_SELF_PERCEPTION_MODE}" \
                "body_style=$E0_DEFAULT_BODY_STYLE" \
                "self_perception_mode=$E0_DEFAULT_SELF_PERCEPTION_MODE"

            run_one \
                "$EXP" \
                "$ACTIVE_MODEL" \
                "clothes_${E0_CLOTHES_SELF_PERCEPTION_MODE}" \
                "body_style=$E0_CLOTHES_BODY_STYLE" \
                "self_perception_mode=$E0_CLOTHES_SELF_PERCEPTION_MODE"
        else
            run_one "$EXP" "$ACTIVE_MODEL" "preset"
        fi
    done
done

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo " Batch Completed" | tee -a "$LOG_FILE"
echo " Models:    ${MODELS[*]}" | tee -a "$LOG_FILE"
echo " Seed:      $SEED" | tee -a "$LOG_FILE"
echo " Log file:  $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
