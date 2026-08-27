#!/bin/bash
# Treatment arm: LoRA RL on Qwen3.5-4B with PoLoRA.
# Everything except OPTIMIZER_ARGS is shared with run-adamw.sh via common.sh.

RUN_TAG=polora

OPTIMIZER_ARGS=(
   --optimizer polora
   # polora's lr is not an AdamW lr: each factor update is rescaled to spectral
   # norm rho = lr / (sigma_max(A) + sigma_max(B)), so it sets a step *size* in
   # weight space directly. 2e-4 is the upstream default; sweep this before
   # concluding anything about the AdamW comparison.
   --lr 2e-4
   --lr-decay-style constant
   # polora applies no weight decay -- set to 0 so the run log does not imply it.
   --weight-decay 0.0
   --polora-beta1 0.9
   --polora-curvature-beta 0.99
   --polora-ns-steps 8
   --polora-higham-iters 8
   # --polora-compile                 # compiles the two spectral kernels; update unchanged
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
