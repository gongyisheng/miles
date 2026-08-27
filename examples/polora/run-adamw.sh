#!/bin/bash
# Baseline arm: LoRA RL on Qwen3.5-4B with AdamW.
# Runs on the lower half of the box (GPUs 0-3) so it can share the node with
# run-polora.sh. Everything except the optimizer and the placement is shared
# via common.sh.

RUN_TAG=adamw

TRAIN_GPU_IDS=0,1
ROLLOUT_GPU_IDS=2,3
RAY_PORT_OFFSET=0

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5                         # higher LR than full-FT; typical for LoRA
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
