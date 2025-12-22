path=$1

wget -O data/BENCHMARKS/LiveCodeBench/test6.jsonl "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl?download=true"
eval_data="LiveCodeBench-v6 data/BENCHMARKS/LiveCodeBench/test6.jsonl data/BENCHMARKS/LiveCodeBench/evaluation_config.json"

bash scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval.sh "${path}" "${eval_data}"

python evaluation/pass_k.py --path "${path}" --eval-name "LiveCodeBench-v6" --eval-config "data/BENCHMARKS/LiveCodeBench/evaluation_config.json"

if [ -f data/BENCHMARKS/LiveCodeBench/test6.jsonl ]; then
    echo "Removing data/BENCHMARKS/LiveCodeBench/test6.jsonl"
    rm -r data/BENCHMARKS/LiveCodeBench/test6.jsonl
else
    echo "File data/BENCHMARKS/LiveCodeBench/test6.jsonl does not exist."
fi