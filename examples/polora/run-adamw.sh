#!/bin/bash
# Baseline arm: LoRA RL on Qwen3.5-4B with AdamW.
# Everything except OPTIMIZER_ARGS is shared with run-polora.sh via common.sh.

RUN_TAG=adamw

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
