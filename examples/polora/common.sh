#!/bin/bash

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
: "${GPU_IDS:?set GPU_IDS before sourcing common.sh}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

IFS=',' read -ra _gpu_ids <<< "${GPU_IDS}"
TOTAL_GPUS=${#_gpu_ids[@]}
TRAIN_GPUS=${TOTAL_GPUS}

export GPUS_PER_NODE=${TRAIN_GPUS}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export PYTHONUNBUFFERED=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

: "${RAY_PORT_OFFSET:?set RAY_PORT_OFFSET before sourcing common.sh}"
RAY_GCS_PORT=$((6379 + RAY_PORT_OFFSET))
RAY_DASHBOARD_PORT=$((8265 + RAY_PORT_OFFSET))
RAY_AGENT_PORT=$((52365 + RAY_PORT_OFFSET))
RAY_CLIENT_PORT=$((20001 + RAY_PORT_OFFSET))
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray-${RUN_TAG}}
unset RAY_ADDRESS

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

SAVE_DIR=${SAVE_DIR:-/root/checkpoints/${RUN_TAG}}

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-10}"
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
   --num-rollout "${NUM_ROLLOUT:-5}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data

   # DAPO configs
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data
      aime24 "${EVAL_DATA_AIME24}"
      aime25 "${EVAL_DATA_AIME25}"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 32768
   --eval-top-k 1
)

if [[ "${SKIP_EVAL_BEFORE_TRAIN:-0}" == "1" ]]; then
   EVAL_ARGS+=(--skip-eval-before-train)
fi

PERF_ARGS=(
   # Polora requires TP=PP=1.
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

if [[ "${USE_WANDB:-0}" == "1" ]]; then
   if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q 'api\.wandb\.ai' "${HOME}/.netrc" 2>/dev/null; then
      echo "USE_WANDB=1 but no credentials: export WANDB_API_KEY or run 'wandb login'" >&2
      exit 1
   fi

   WANDB_ARGS=(
      --use-wandb
      --wandb-project miles-polora-vs-adamw
      --wandb-group "qwen3.5-4b-lora-${RUN_TAG}"
   )
   if [[ -n "${WANDB_API_KEY:-}" ]]; then
      WANDB_ARGS+=(--wandb-host https://wandb.ai/ --wandb-key "${WANDB_API_KEY}")
   fi
else
   WANDB_ARGS=()
fi

SGLANG_ARGS=(
   --num-gpus-per-node "${TOTAL_GPUS}"
   --colocate
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.9
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

# Reuse a head left behind by an earlier attempt
if ray status --address "${MASTER_ADDR}:${RAY_GCS_PORT}" &>/dev/null; then
   echo "Arm '${RUN_TAG}': reusing the ray head already listening on ${RAY_GCS_PORT}"
else
   ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${TOTAL_GPUS}" \
      --port "${RAY_GCS_PORT}" \
      --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" \
      --dashboard-agent-listen-port "${RAY_AGENT_PORT}" \
      --ray-client-server-port "${RAY_CLIENT_PORT}" \
      --temp-dir "${RAY_TEMP_DIR}" \
      --disable-usage-stats
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
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
