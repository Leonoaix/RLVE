#!/bin/bash

set -e

EVAL_SCRIPT="scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval_BENCHMARKS.sh"
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

run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-256-400-hf"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-256-800-hf"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-16-kl-coef0.005-400-hf/"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-16-kl-coef0.01-400-hf/"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-rand16_2-400-hf"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-rand16_1-400-hf"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-256-300-hf"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-16-300-hf/"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-4-300-hf/"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-256-100-hf"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-16-100-hf/"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-4-100-hf/"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-256-200-hf"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-16-200-hf/"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-4-200-hf/"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-16-400-hf/"
run_eval "../models/Deepseek-R1-Distill-Qwen-1.5B-env-4-400-hf/"

echo "----------------------------------------" >> "${LOG_FILE}"
echo "Evaluation end: $(date)" >> "${LOG_FILE}"
