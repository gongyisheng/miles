#!/bin/bash
# Polora SFT treatment on GPUs 4-7 (DP=4, TP=PP=1 — polora requires TP=PP=1).

RUN_TAG=sft-polora
NUM_EPOCH=1

# 1 to log to wandb; needs WANDB_API_KEY in the environment. Keep this in step
# with sft_run_adamw.sh.
USE_WANDB=${USE_WANDB:-1}

GPU_IDS=4,5,6,7
RAY_PORT_OFFSET=1

OPTIMIZER_ARGS=(
   --optimizer polora
   # Polora rescales each factor update to a spectral norm derived from lr, so
   # its lr lives on a different scale than adam's. Schedule shape is kept
   # identical to the adamw arm.
   --lr 2e-3
   --lr-decay-style cosine
   --min-lr 2e-4
   --lr-warmup-fraction 0.1
   --weight-decay 0.0
   --polora-beta1 0.9
   --polora-curvature-beta 0.99
   --polora-ns-steps 8
   --polora-higham-iters 8
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/sft_common.sh"
