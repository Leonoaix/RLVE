path=$1
eval_data="AIME24 data/BENCHMARKS/AIME/AIME24.json data/BENCHMARKS/AIME/evaluation_config.json     AIME25 data/BENCHMARKS/AIME/AIME25.json data/BENCHMARKS/AIME/evaluation_config.json     OlympiadBench data/BENCHMARKS/OlympiadBench/olympiad_bench.json data/BENCHMARKS/OlympiadBench/evaluation_config.json     BIG-Bench_Extra_Hard data/BENCHMARKS/BIG-Bench_Extra_Hard/bbeh.json data/BENCHMARKS/BIG-Bench_Extra_Hard/evaluation_config.json    Omega-500 data/BENCHMARKS/Omega-500/omega-500.json data/BENCHMARKS/Omega-500/evaluation_config.json"

bash scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval.sh "${path}" "${eval_data}"
