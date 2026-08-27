#!/bin/bash
# Shared configuration for the AdamW-vs-Polora comparison.
#
# Sourced by both runners so their shared settings cannot drift.
#
# Required from the caller before sourcing:
#   OPTIMIZER_ARGS, RUN_TAG, TRAIN_GPU_IDS, ROLLOUT_GPU_IDS, RAY_PORT_OFFSET

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Paths
HF_CHECKPOINT=${HF_CHECKPOINT:-/root/models/Qwen3.5-4B}
TRAIN_DATA=${TRAIN_DATA:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}
EVAL_DATA_AIME24=${EVAL_DATA_AIME24:-/root/datasets/aime-2024/aime-2024.jsonl}
EVAL_DATA_AIME25=${EVAL_DATA_AIME25:-/root/datasets/aime-2025/aime-2025.jsonl}

# This path currently contains a Qwen3-4B checkpoint, not Qwen3.5-4B.
MODEL_NAME=${MODEL_NAME:-qwen3-4B}

# Cluster
: "${TRAIN_GPU_IDS:?set TRAIN_GPU_IDS before sourcing common.sh}"
: "${ROLLOUT_GPU_IDS:?set ROLLOUT_GPU_IDS before sourcing common.sh}"

# Placement assigns the first visible GPUs to training and the rest to rollout.
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

# Offset Ray's fixed ports so both arms can run on one node.
: "${RAY_PORT_OFFSET:?set RAY_PORT_OFFSET before sourcing common.sh}"
RAY_GCS_PORT=$((6379 + RAY_PORT_OFFSET))
RAY_DASHBOARD_PORT=$((8265 + RAY_PORT_OFFSET))
RAY_AGENT_PORT=$((52365 + RAY_PORT_OFFSET))
RAY_CLIENT_PORT=$((10001 + RAY_PORT_OFFSET))
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray-${RUN_TAG}}
unset RAY_ADDRESS

# CLEANUP=1 stops both arms because these commands are node-wide.
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
   --lora-dropout 0.0
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

   # Dynamic sampling would give the two arms different prompts.
   # --over-sampling-batch-size 64
   # --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data
      aime24 "${EVAL_DATA_AIME24}"
      aime25 "${EVAL_DATA_AIME25}"
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-k 1
)

PERF_ARGS=(
   # Polora requires complete LoRA factors, so both arms use TP=PP=1.
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
   # --use-kl-loss  # Also requires --ref-load.
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
   # Rollout GPUs are dedicated to SGLang.
   --sglang-mem-fraction-static 0.8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # MLA models need a different attention backend.
   --attention-backend flash
   --seed 42
)

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
