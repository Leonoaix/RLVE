#!/bin/bash
set -e

# ===== config =====
ROOT_DIR="/root/RLVE"
LOG_FILE="eval_time_all.txt"

EVAL_BENCH="scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval_BENCHMARKS.sh"
EVAL_LIVE="scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval_LiveCodeBench.sh"
EVAL_OOD="scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval_HELD-OUT_ENVIRONMENTS.sh"

cd "${ROOT_DIR}"

echo "Evaluation start: $(date)" >> "${LOG_FILE}"
echo "----------------------------------------" >> "${LOG_FILE}"

# ===== utils =====
run_one () {
    eval_script="$1"
    eval_name="$2"
    model_path="$3"

    echo "Start ${eval_name}: ${model_path}"
    start_time=$(date +%s)

    bash "${eval_script}" "${model_path}"

    end_time=$(date +%s)
    cost=$((end_time - start_time))

    echo "$(date '+%F %T') | ${eval_name} | ${model_path} | ${cost}s" >> "${LOG_FILE}"
}

run_all_for_model () {
    model_path="$1"

    run_one "${EVAL_BENCH}" "BENCHMARKS"        "${model_path}"
    run_one "${EVAL_LIVE}"  "LiveCodeBench"     "${model_path}"
    run_one "${EVAL_OOD}"   "HELD-OUT-ENV (OOD)" "${model_path}"
}

# ===== model list =====
MODELS=(
  "../models/Deepseek-R1-Distill-Qwen-1.5B-env-4-comp2-100-hf"
  "../models/Deepseek-R1-Distill-Qwen-1.5B-env-4-comp1-100-hf"
)

# ===== run =====
for m in "${MODELS[@]}"; do
  run_all_for_model "${m}"
done

echo "----------------------------------------" >> "${LOG_FILE}"
echo "Evaluation end: $(date)" >> "${LOG_FILE}"
