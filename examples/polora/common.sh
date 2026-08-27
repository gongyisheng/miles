#!/bin/bash
# Shared configuration for the AdamW-vs-PoLoRA LoRA RL comparison.
#
# Sourced by run-adamw.sh and run-polora.sh. Everything that must be identical
# between the two arms lives here; each runner only sets OPTIMIZER_ARGS and the
# wandb group. Do not put optimizer-specific settings in this file.
#
# The two arms are sized to run side by side on one 8-GPU box, each on its own
# 4 GPUs and its own ray cluster, so they can be launched concurrently.
#
# Required from the caller before sourcing:
#   OPTIMIZER_ARGS   - array with the optimizer flags for this arm
#   RUN_TAG          - short label for this arm; wandb group suffix and ray temp dir
#   TRAIN_GPU_IDS    - comma-separated physical GPU ids for the actor
#   ROLLOUT_GPU_IDS  - comma-separated physical GPU ids for the sglang engines
#   RAY_PORT_OFFSET  - integer added to every fixed ray port so the two arms'
#                      clusters do not collide (0 for one arm, 1 for the other)

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HF_CHECKPOINT=${HF_CHECKPOINT:-/root/models/Qwen3.5-4B}
TRAIN_DATA=${TRAIN_DATA:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}
EVAL_DATA_AIME24=${EVAL_DATA_AIME24:-/root/datasets/aime-2024/aime-2024.jsonl}
EVAL_DATA_AIME25=${EVAL_DATA_AIME25:-/root/datasets/aime-2025/aime-2025.jsonl}

# NOTE: /root/models/Qwen3.5-4B currently holds a *Qwen3-4B* checkpoint
# (config.json: model_type=qwen3, 36 layers, hidden 2560, vocab 151936). The
# repo's `qwen3.5-4B` model args describe a different architecture (32 layers,
# vocab 248320, --attention-output-gate), so loading it against this checkpoint
# would fail. MODEL_NAME therefore defaults to the arch that matches what is on
# disk. Set MODEL_NAME=qwen3.5-4B once a real Qwen3.5-4B checkpoint is in place.
MODEL_NAME=${MODEL_NAME:-qwen3-4B}

# --------------------------------------------------------------------------
# Cluster -- disaggregated: training and rollout hold disjoint GPUs, so the
# actor never sleeps/wakes the sglang engine and the two phases overlap.
# The runner supplies the ids; this arm sees no other GPU on the box.
# --------------------------------------------------------------------------
: "${TRAIN_GPU_IDS:?set TRAIN_GPU_IDS before sourcing common.sh}"
: "${ROLLOUT_GPU_IDS:?set ROLLOUT_GPU_IDS before sourcing common.sh}"

# This ordering is what pins the split. miles' _create_placement_group sorts its
# bundles by (node, gpu id) ascending and hands the actor the first
# actor_num_gpus of them, rollout the rest (miles/ray/placement_group.py). Ray's
# gpu ids index into CUDA_VISIBLE_DEVICES, so listing the train devices first
# puts the actor on TRAIN_GPU_IDS and the sglang engines on ROLLOUT_GPU_IDS --
# in that order, whatever the physical ids are. The two lists must be disjoint.
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS},${ROLLOUT_GPU_IDS}"

IFS=',' read -ra _train_gpu_ids <<< "${TRAIN_GPU_IDS}"
IFS=',' read -ra _rollout_gpu_ids <<< "${ROLLOUT_GPU_IDS}"
TRAIN_GPUS=${#_train_gpu_ids[@]}
ROLLOUT_GPUS=${#_rollout_gpu_ids[@]}
TOTAL_GPUS=$((TRAIN_GPUS + ROLLOUT_GPUS))

export GPUS_PER_NODE=${TRAIN_GPUS}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export PYTHONUNBUFFERED=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

# One ray cluster per arm. Every port ray defaults to a fixed value gets the
# arm's offset, and each cluster keeps its own temp dir, so two heads coexist on
# the same node. Ports left at auto (node manager, object manager, worker range)
# are chosen free and need no offset.
: "${RAY_PORT_OFFSET:?set RAY_PORT_OFFSET before sourcing common.sh}"
RAY_GCS_PORT=$((6379 + RAY_PORT_OFFSET))
RAY_DASHBOARD_PORT=$((8265 + RAY_PORT_OFFSET))
RAY_AGENT_PORT=$((52365 + RAY_PORT_OFFSET))
RAY_CLIENT_PORT=$((10001 + RAY_PORT_OFFSET))
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray-${RUN_TAG}}
# An inherited RAY_ADDRESS would point this arm's CLI at the other arm's cluster.
unset RAY_ADDRESS

# No global process cleanup here: `ray stop --force` and `pkill python` are
# node-wide and would kill the other arm mid-run. Set CLEANUP=1 to reclaim a
# wedged machine -- that kills BOTH arms.
if [[ "${CLEANUP:-0}" == "1" ]]; then
   pkill -9 sglang || true
   ray stop --force || true
   sleep 5
   pkill -9 ray || true
   pkill -9 python || true
   sleep 3
fi

MODEL_ARGS_LINE="$(python3 "${REPO_ROOT}/miles/utils/external_utils/model_args_utils.py" "${MODEL_NAME}")" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
)

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0                # 0.0 for RL training
   --target-modules "all-linear"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 5
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 32
   --balance-data

   # Left off on purpose: dynamic sampling resamples prompts until the group
   # reward has nonzero std, which makes the two arms see different data.
   # Enable on both arms together if you want the DAPO recipe proper.
   # --over-sampling-batch-size 64
   # --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data
      aime24 "${EVAL_DATA_AIME24}"
      aime25 "${EVAL_DATA_AIME25}"
   --n-samples-per-eval-prompt 8       # 30 prompts each; average over 8 to cut variance
   --eval-max-response-len 16384
   --eval-top-k 1
)

PERF_ARGS=(
   # polora requires TP=1 and PP=1 (its update needs whole LoRA factors), so
   # both arms run pure data parallel to keep the comparison apples-to-apples.
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss # if use kl loss, should use --ref-load
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host https://wandb.ai/
   --wandb-project miles-polora-vs-adamw
   --wandb-group "qwen3.5-4b-lora-${RUN_TAG}"
)

SGLANG_ARGS=(
   --rollout-num-gpus "${ROLLOUT_GPUS}"
   --rollout-num-gpus-per-engine 1
   # Rollout GPUs are dedicated in disaggregated mode -- nothing else competes
   # for their memory, so sglang can take most of it.
   --sglang-mem-fraction-static 0.8
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   # same data order and init on both arms
   --seed 42
)

# ray owns every device in CUDA_VISIBLE_DEVICES; the actor and the rollout
# engines each claim their share out of that pool.
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${TOTAL_GPUS}" \
   --port "${RAY_GCS_PORT}" \
   --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" \
   --dashboard-agent-listen-port "${RAY_AGENT_PORT}" \
   --ray-client-server-port "${RAY_CLIENT_PORT}" \
   --temp-dir "${RAY_TEMP_DIR}" \
   --disable-usage-stats

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${TRAIN_GPUS}" \
   --calculate-per-token-loss \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}"
