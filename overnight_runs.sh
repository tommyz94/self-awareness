#!/bin/bash

# Configuration
OLLAMA_HOST="http://127.0.0.1:11556"
NUM_RUNS=10
MODELS=("gemini-flash-latest")
EXPERIMENTS=("e3")
PYTHON_BIN="./.venv/bin/python"

echo "============================================"
echo " Starting Overnight Experiment Batch       "
echo " Models: ${MODELS[*]}"
echo " Port:   ${OLLAMA_HOST}"
echo " Runs:   ${NUM_RUNS} per experiment"
echo "============================================"

# Ensure models are pulled
for MODEL in "${MODELS[@]}"; do
    if [[ "$MODEL" == gemini-* ]]; then
        echo "Skip pulling for API model: $MODEL"
    else
        echo "Checking/Pulling model: $MODEL..."
        curl -s $OLLAMA_HOST/api/pull -d "{\"name\":\"$MODEL\",\"stream\":false}" > /dev/null
    fi
done

# Run Experiments
for MODEL in "${MODELS[@]}"; do
    for EXP in "${EXPERIMENTS[@]}"; do
        echo ""
        echo "--------------------------------------------"
        echo " RUNNING: $MODEL | Experiment: $EXP"
        echo "--------------------------------------------"
        
        # Determine the client based on model name
        CLIENT="ollama"
        if [[ "$MODEL" == gemini-* ]]; then
            CLIENT="google"
        fi
        
        $PYTHON_BIN main.py \
            experiment=$EXP \
            nav=$MODEL \
            client=$CLIENT \
            num_runs=$NUM_RUNS \
            mujoco.headless=true \
            ollama.host=$OLLAMA_HOST
            
        echo "Completed: $MODEL | Experiment: $EXP"
    done
done

echo ""
echo "============================================"
echo " Batch Completed! Check outputs/ directory. "
echo "============================================"
