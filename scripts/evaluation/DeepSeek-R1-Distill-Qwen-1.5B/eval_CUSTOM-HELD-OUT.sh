path=$1
eval_data="DCS-HELD-OUT data/HELD-OUT_ENVIRONMENTS/DCS-HELD-OUT.json data/HELD-OUT_ENVIRONMENTS/evaluation_config.json"

bash scripts/evaluation/DeepSeek-R1-Distill-Qwen-1.5B/eval.sh "${path}" "${eval_data}"
