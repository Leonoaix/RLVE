#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT"
    exit 1
fi

WANDB_PROJECT=$1

bash scripts/training/DeepSeek-R1-Distill-Qwen-1.5B/rlve.sh "${WANDB_PROJECT}" \
    "[DeepSeek-R1-Distill-Qwen-1.5B]_low-structural-load-env-16" \
    "Division BucketSorting DiscreteLogarithm GCDOne_Counting GCDPrime_Counting AddMultiple_Divisible_Counting BinaryLinearEquation_SolutionCounting LCM CongruentEquation PanSolarPanels Fibonacci Differentiate Integral MaxMultiplicationFixedSum PrefixConcatenation PolynomialInterpolation"
