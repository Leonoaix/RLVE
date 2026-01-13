#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT"
    exit 1
fi

WANDB_PROJECT=$1

bash scripts/training/DeepSeek-R1-Distill-Qwen-1.5B/rlve.sh "${WANDB_PROJECT}" \
    "[DeepSeek-R1-Distill-Qwen-1.5B]_high-structural-load-env-16" \
    "TwoSAT CampsitePuzzle CampfireParty TwiddlePuzzle CycleCounting ThreeVertexCycleCounting AlmostCompleteGraphCycleCounting DegreeFixed_SpanningTree FixedOneEdgeNum_SpanningTree Tournament_LongestPath Path_NoGoingBack_Counting AdditionTable AndOr_Sequence_Counting DistinctEdgeColoredCompleteGraphCounting SubsetSumSequence SubgraphIsomorphism"
