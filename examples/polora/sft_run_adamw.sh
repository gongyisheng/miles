#!/bin/bash
# AdamW SFT baseline on GPUs 0-3 (DP=4, TP=PP=1).

RUN_TAG=sft-adamw
NUM_EPOCH=1

# 1 to log to wandb; needs credentials on the box (see sft_common.sh). Keep this
# in step with sft_run_polora.sh.
USE_WANDB=${USE_WANDB:-1}

GPU_IDS=0,1,2,3
RAY_PORT_OFFSET=0

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/sft_common.sh"
