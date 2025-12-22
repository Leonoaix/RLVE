#!/bin/bash

set -e

EVAL_SCRIPT="scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval_Math-500.sh"
LOG_FILE="eval_time.txt"

cd /root/RLVE

echo "Evaluation start: $(date)" >> "${LOG_FILE}"
echo "----------------------------------------" >> "${LOG_FILE}"

run_eval () {
    model_path="$1"

    echo "Start eval: ${model_path}"
    start_time=$(date +%s)

    bash "${EVAL_SCRIPT}" "${model_path}"

    end_time=$(date +%s)
    cost=$((end_time - start_time))

    echo "$(date '+%F %T') | ${model_path} | ${cost}s" >> "${LOG_FILE}"
}

run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-4-comp2-100-hf"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-4-comp1-100-hf"

echo "----------------------------------------" >> "${LOG_FILE}"
echo "Evaluation end: $(date)" >> "${LOG_FILE}"
