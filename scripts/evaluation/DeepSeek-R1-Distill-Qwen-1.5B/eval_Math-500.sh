path=$1
eval_data="Math-500 data/BENCHMARKS/Math-500/math-500.json data/BENCHMARKS/Math-500/evaluation_config.json"

bash scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval.sh "${path}" "${eval_data}"
