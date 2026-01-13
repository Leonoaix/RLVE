#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 WANDB_PROJECT RUN_NAME ENVIRONMENT_LIST"
    exit 1
fi

WANDB_PROJECT=$1
RUN_NAME=$2
ENVIRONMENT_LIST=$3

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# fix tokenizers parallelism warning when using multiprocessing
export TOKENIZERS_PARALLELISM=false

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source scripts/models/deepseek-r1-distill-qwen-1.5B.sh

CKPT_ARGS=(
   --hf-checkpoint ../models/DeepSeek-R1-Distill-Qwen-1.5B
   --ref-load ../models/DeepSeek-R1-Distill-Qwen-1.5B_torch_dist
   --load ../logs/${RUN_NAME}/
   --save ../logs/${RUN_NAME}/
   --save-interval 25
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rlve
   --environment-list "${ENVIRONMENT_LIST}"

   --custom-prompt-preprocessor ChatTemplate_NoSystemPrompt
   --apply-chat-template

   --rm-type rlve
   --reward-key reward

   --num-rollout 400
   --rollout-batch-size 128
   --n-samples-per-prompt 16
   --rollout-max-response-len 24576
   --rollout-temperature 1.0

   --over-sampling-batch-size 384
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --partial-rollout

   --num-steps-per-rollout 1
   --wandb-always-use-train-step
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data HELD-OUT_ENVIRONMENTS_128 data/HELD-OUT_ENVIRONMENTS/test_128.json
   --n-samples-per-eval-prompt 1
   --eval-top-p 0.7

   --eval-input-key user_prompt
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   # --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 8
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 3072 # If you are concerned about OOM, you can initially set this value to rollout_max_response_len / cp_size and then increase it later to improve training efficiency.

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   # --kl-loss-coef 0.00
   # --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --override-opt-param-scheduler
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${RUN_NAME}"
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-server-concurrency 256
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:1024\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}