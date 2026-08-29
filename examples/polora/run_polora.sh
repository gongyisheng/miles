#!/bin/bash
# Polora treatment on GPUs 4-7.

RUN_TAG=polora
NUM_ROLLOUT=100
USE_WANDB=1

# 1 to log to wandb; needs WANDB_API_KEY in the environment. Keep this in step
# with run-adamw.sh.
USE_WANDB=${USE_WANDB:-1}

GPU_IDS=4,5,6,7
RAY_PORT_OFFSET=1

OPTIMIZER_ARGS=(
   --optimizer polora
   # Polora rescales each factor update to a spectral norm derived from lr.
   --lr 1e-3 # !!! 2e-4 in original experiment, test 1e-3 just for interest
   --lr-decay-style constant
   --weight-decay 0.0
   --polora-beta1 0.9
   --polora-curvature-beta 0.99
   --polora-ns-steps 8
   --polora-higham-iters 8
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
