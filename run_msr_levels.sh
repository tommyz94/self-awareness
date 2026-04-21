#!/bin/bash

# Configuration
OLLAMA_HOST="http://127.0.0.1:11556"
NUM_RUNS=10
MODELS=("gemma4:31b" "llava-phi3:3.8b" "ministral-3:8b" "ministral-3:14b")
LEVELS=(0 1 2)
EXPERIMENT="e1"
PYTHON_BIN="./.venv/bin/python"

echo "============================================"
echo " Starting MSR Levels Benchmark              "
echo " Models: ${MODELS[*]}"
echo " Levels: ${LEVELS[*]}"
echo " Runs:   ${NUM_RUNS} per configuration"
echo "============================================"

# Ensure models are pulled
for MODEL in "${MODELS[@]}"; do
    echo "Checking/Pulling model: $MODEL..."
    curl -s $OLLAMA_HOST/api/pull -d "{\"name\":\"$MODEL\",\"stream\":false}" > /dev/null
done

# Run Experiments
for LEVEL in "${LEVELS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "--------------------------------------------"
        echo " RUNNING: $MODEL | Experiment: $EXPERIMENT | MSR_LEVEL: $LEVEL"
        echo "--------------------------------------------"
        
        $PYTHON_BIN main.py \
            experiment=$EXPERIMENT \
            nav=$MODEL \
            client="ollama" \
            num_runs=$NUM_RUNS \
            msr_level=$LEVEL \
            mujoco.headless=true \
            ollama.host=$OLLAMA_HOST
            
        echo "Completed: $MODEL | Level: $LEVEL"
    done
done

echo ""
echo "============================================"
echo " Batch Completed! Check outputs/ directory. "
echo "============================================"
