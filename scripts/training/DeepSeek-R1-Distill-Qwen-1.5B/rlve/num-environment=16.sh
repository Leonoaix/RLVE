#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT"
    exit 1
fi

WANDB_PROJECT=$1

bash scripts/training/DeepSeek-R1-Distill-Qwen-1.5B/rlve.sh "${WANDB_PROJECT}" \
    "[DeepSeek-R1-Distill-Qwen-1.5B]_[num-environment=16]_GRPO" \
    "Division EuclidGame GCDOne_Counting HamiltonianPath LampChanging LargestConvexPolygon Multiplication PCPPermutation Path_NoGoingBack_Counting SAT ShortestPath Sorting SpiralMatrix SubsequenceReversalLNDS UndamagedSubmatrixCounting WYRLevelingGround"
