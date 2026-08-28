#!/bin/bash
# AdamW baseline on GPUs 0-3.

RUN_TAG=adamw
NUM_ROLLOUT=100

# 1 to log to wandb; needs credentials on the box (see common.sh). Keep this in
# step with run-polora.sh.
USE_WANDB=${USE_WANDB:-1}

GPU_IDS=0,1,2,3
RAY_PORT_OFFSET=0

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
