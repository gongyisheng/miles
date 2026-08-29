#!/bin/bash
# Shared configuration and launch for the two SFT arms (sft_run_adamw.sh,
# sft_run_polora.sh). Ported from scripts/run_qwen3_sft.py (Qwen3-4B-Base
# recipe), with LoRA added because polora only steps LoRA (A, B) pairs.
#
# Pure SFT: --debug-train-only, so no sglang engine, no generation, no eval.

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Paths
HF_CHECKPOINT=${HF_CHECKPOINT:-/root/models/Qwen3-4B}
TRAIN_DATA=${TRAIN_DATA:-/root/datasets/openhermes2_5.parquet}
MODEL_NAME=${MODEL_NAME:-qwen3-4B}

# OpenHermes-2.5 has ~1M rows; a full epoch is ~7.8k steps at batch 128, far more
# than this A/B needs. Slice it at load time instead (miles' `path@[start:end]`).
# Set TRAIN_ROWS= (empty) to train on the whole file.
TRAIN_ROWS=${TRAIN_ROWS-20000}
if [[ -n "${TRAIN_ROWS}" ]]; then
   PROMPT_DATA="${TRAIN_DATA}@[:${TRAIN_ROWS}]"
else
   PROMPT_DATA="${TRAIN_DATA}"
fi

if [[ ! -f "${TRAIN_DATA}" ]]; then
   echo "Missing SFT dataset ${TRAIN_DATA}; run ${SCRIPT_DIR}/sft_prepare_data.sh first" >&2
   exit 1
fi

# Cluster
: "${GPU_IDS:?set GPU_IDS before sourcing sft_common.sh}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

IFS=',' read -ra _gpu_ids <<< "${GPU_IDS}"
TOTAL_GPUS=${#_gpu_ids[@]}
# TP=PP=CP=1, so data parallelism is the full GPU count of the arm (4 -> DP=4).
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

: "${RAY_PORT_OFFSET:?set RAY_PORT_OFFSET before sourcing sft_common.sh}"
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
   --save-interval "${SAVE_INTERVAL:-50}"
)

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   --target-modules "all-linear"
)

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data "${PROMPT_DATA}"
   --input-key messages
   # no --apply-chat-template: sft_rollout renders the raw messages itself,
   # together with the per-token loss mask
   --loss-mask-type "${LOSS_MASK_TYPE:-qwen}"
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH:-1}"
   --rollout-batch-size 128
   --global-batch-size 128

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   # no rollout generation at all, hence no sglang engine
   --debug-train-only
)

PERF_ARGS=(
   # Polora requires TP=PP=1; keep the adamw arm identical.
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
)

if [[ "${USE_WANDB:-0}" == "1" ]]; then
   if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q 'api\.wandb\.ai' "${HOME}/.netrc" 2>/dev/null; then
      echo "USE_WANDB=1 but no credentials: export WANDB_API_KEY or run 'wandb login'" >&2
      exit 1
   fi

   WANDB_ARGS=(
      --use-wandb
      --wandb-project miles-polora-vs-adamw-sft
      --wandb-group "qwen3-4b-lora-sft-${RUN_TAG}"
   )
   if [[ -n "${WANDB_API_KEY:-}" ]]; then
      WANDB_ARGS+=(--wandb-host https://wandb.ai/ --wandb-key "${WANDB_API_KEY}")
   fi
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --num-gpus-per-node "${TOTAL_GPUS}"
   --seed 42
)

TRAIN_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${TRAIN_GPUS}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${LORA_ARGS[@]}"
   "${SFT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

# Print the train.py argv and stop, without touching ray or the GPUs.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
   printf '%s\n' "${TRAIN_ARGS[@]}"
   exit 0
fi

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
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py "${TRAIN_ARGS[@]}"
