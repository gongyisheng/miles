"""Unified LoRA GRPO launcher (colocated) for miles.

One entry point; `--model` selects the recipe. The shared GRPO / optimizer /
misc / LoRA arg blocks live in helpers; each recipe only carries the
model-specific values (checkpoint, parallelism, sglang).

All recipes use the Megatron backend with `--megatron-to-hf-mode bridge` (no
offline torch_dist conversion) and colocated rollout/train. Every recipe trains
on DAPO-Math-17k and evaluates on AIME-2024.

LoRA hyperparameters fall back to each recipe's default but can be overridden
from the CLI (`--lora-rank`, `--lora-alpha`, `--lora-dropout`, `--target-modules`).

Examples:
  python scripts/run_lora_colocate.py --model qwen2.5-3b
  python scripts/run_lora_colocate.py --model qwen3-4b
  python scripts/run_lora_colocate.py --model gpt-oss-20b
  python scripts/run_lora_colocate.py --model kimi-k25 --num-nodes 16
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

ModelName = Literal["qwen2.5-3b", "qwen3-4b", "gpt-oss-20b", "kimi-k25"]


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model: ModelName = "qwen3-4b"
    run_id: str = U.create_run_id()
    num_gpus_per_node: int | None = None
    enable_eval: bool = True
    # LoRA overrides; None -> use the per-recipe default
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    target_modules: str | None = None
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"


def _lora_args(
    args: ScriptArgs, *, rank: int, alpha: int, dropout: float, target_modules: str, extra: str = ""
) -> str:
    return (
        f"--lora-rank {args.lora_rank if args.lora_rank is not None else rank} "
        f"--lora-alpha {args.lora_alpha if args.lora_alpha is not None else alpha} "
        f"--lora-dropout {args.lora_dropout if args.lora_dropout is not None else dropout} "
        f'--target-modules "{args.target_modules or target_modules}" '
        f"{extra}"
    )


def _rollout_args(
    *,
    prompt_data: str,
    input_key: str,
    rm_type: str,
    num_rollout: int,
    rollout_batch_size: int,
    max_response_len: int,
    global_batch_size: int,
    n_samples: int = 8,
    temperature: str = "1",
    extra: str = "",
) -> str:
    return (
        f"--prompt-data {prompt_data} "
        f"--input-key {input_key} "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--rm-type {rm_type} "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_samples} "
        f"--rollout-max-response-len {max_response_len} "
        f"--rollout-temperature {temperature} "
        f"--global-batch-size {global_batch_size} "
        f"{extra}"
    )


def _ckpt_args(
    hf_checkpoint: str,
    *,
    ref_load: str | None = None,
    model_name: str | None = None,
    save: str | None = None,
    save_interval: int | None = None,
) -> str:
    # Every recipe loads HF weights directly via the bridge (no offline torch_dist conversion).
    s = f"--hf-checkpoint {hf_checkpoint} --megatron-to-hf-mode bridge "
    if ref_load:
        s += f"--ref-load {ref_load} "
    if model_name:
        s += f"--model-name {model_name} "
    if save:
        s += f"--save {save} "
    if save_interval is not None:
        s += f"--save-interval {save_interval} "
    return s


def _grpo_args(extra: str = "") -> str:
    return "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 " + extra


def _optimizer_args(lr: str, *, extra: str = "") -> str:
    return (
        "--optimizer adam "
        f"--lr {lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        f"{extra}"
    )


def _misc_args(args: ScriptArgs, num_gpus_per_node: int, extra: str = "") -> str:
    return (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {num_gpus_per_node} "
        "--colocate "
        f"{extra}"
    )


# Megatron-specific knobs shared by every dense / MLA recipe (gpt-oss uses bshd/fused instead).
_MEGATRON_FP32 = "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "

# Off-engine determinism env vars set by the original shell launchers.
_DETERMINISM_ENV = {
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

# Every recipe trains on DAPO-Math-17k and evaluates on AIME-2024.
_DATASETS = ["zhuzilin/dapo-math-17k", "zhuzilin/aime-2024"]
_DAPO_MATH = "dapo-math-17k/dapo-math-17k.jsonl"
_AIME = "aime-2024/aime-2024.jsonl"


def _aime_eval_args(args: ScriptArgs, *, interval: int) -> str:
    if not args.enable_eval:
        return ""
    return (
        f"--eval-interval {interval} "
        f"--eval-prompt-data aime24 {args.data_dir}/{_AIME} "
        "--n-samples-per-eval-prompt 16 "
        "--eval-max-response-len 16384 "
        "--eval-top-p 1 "
    )


def _download_model(args: ScriptArgs, hf_repo: str, model_name: str) -> None:
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download {hf_repo} --local-dir {args.model_dir}/{model_name}")


def _download_datasets(args: ScriptArgs, datasets: list[str]) -> None:
    for full_name in datasets:
        U.hf_download_dataset(full_name, data_dir=args.data_dir)


def _launch(
    args: ScriptArgs,
    *,
    megatron_model_type: str,
    num_gpus_per_node: int,
    wandb_prefix: str,
    ckpt_args: str,
    lora_args: str,
    rollout_args: str,
    eval_args: str,
    perf_args: str,
    grpo_args: str,
    optimizer_args: str,
    sglang_args: str,
    misc_args: str,
    env_vars: dict | None = None,
) -> None:
    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_name_prefix=wandb_prefix, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        extra_env_vars=env_vars or {},
        megatron_path=args.megatron_path,
    )


# ----------------------------------------------------------------------------- qwen2.5-3b


def _prepare_qwen2_5_3b(args: ScriptArgs) -> None:
    _download_model(args, "Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct")
    _download_datasets(args, _DATASETS)


def _execute_qwen2_5_3b(args: ScriptArgs) -> None:
    """Qwen2.5-3B-Instruct LoRA on DAPO-Math (single node)."""
    model_name = "Qwen2.5-3B-Instruct"
    num_gpus_per_node = args.num_gpus_per_node or 8

    _launch(
        args,
        megatron_model_type="qwen2.5-3B",
        num_gpus_per_node=num_gpus_per_node,
        wandb_prefix="qwen2.5-3B-lora",
        ckpt_args=_ckpt_args(f"{args.model_dir}/{model_name}"),
        lora_args=_lora_args(args, rank=32, alpha=32, dropout=0.0, target_modules="all-linear"),
        rollout_args=_rollout_args(
            prompt_data=f"{args.data_dir}/{_DAPO_MATH}",
            input_key="prompt",
            rm_type="deepscaler",
            num_rollout=100,
            rollout_batch_size=32,
            max_response_len=4096,
            global_batch_size=256,
            extra="--balance-data ",
        ),
        eval_args=_aime_eval_args(args, interval=10),
        perf_args=(
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 9216 "
        ),
        grpo_args=_grpo_args("--kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "),
        optimizer_args=_optimizer_args("1e-5"),
        sglang_args="--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.4 ",
        misc_args=_misc_args(
            args, num_gpus_per_node, _MEGATRON_FP32 + "--calculate-per-token-loss --use-miles-router "
        ),
        env_vars=_DETERMINISM_ENV,
    )


# ----------------------------------------------------------------------------- qwen3-4b


def _prepare_qwen3_4b(args: ScriptArgs) -> None:
    _download_model(args, "Qwen/Qwen3-4B", "Qwen3-4B")
    _download_datasets(args, _DATASETS)


def _execute_qwen3_4b(args: ScriptArgs) -> None:
    """Qwen3-4B LoRA (production): rank 64, lr 2e-5, saves the adapter checkpoint."""
    model_name = "Qwen3-4B"
    num_gpus_per_node = args.num_gpus_per_node or 4

    _launch(
        args,
        megatron_model_type="qwen3-4B",
        num_gpus_per_node=num_gpus_per_node,
        wandb_prefix="qwen3-4B-lora",
        ckpt_args=_ckpt_args(
            f"{args.model_dir}/{model_name}",
            save=f"{args.model_dir}/{model_name}-lora-ckpt",
            save_interval=50,
        ),
        lora_args=_lora_args(args, rank=64, alpha=32, dropout=0.0, target_modules="all-linear"),
        rollout_args=_rollout_args(
            prompt_data=f"{args.data_dir}/{_DAPO_MATH}",
            input_key="prompt",
            rm_type="deepscaler",
            num_rollout=100,
            rollout_batch_size=8,
            max_response_len=4096,
            global_batch_size=64,
            extra="--balance-data ",
        ),
        eval_args=_aime_eval_args(args, interval=20),
        # PERF block left at megatron defaults (TP=PP=CP=1) as in the source recipe.
        perf_args="",
        grpo_args=_grpo_args("--kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "),
        optimizer_args=_optimizer_args("2e-5"),
        sglang_args=(
            "--rollout-num-gpus-per-engine 1 "
            "--sglang-decode-log-interval 1000 "
            "--sglang-mem-fraction-static 0.4 "
            "--sglang-chunked-prefill-size 4096 "
        ),
        misc_args=_misc_args(
            args,
            num_gpus_per_node,
            "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
            "--calculate-per-token-loss --use-miles-router ",
        ),
        env_vars={"NCCL_ALGO": "Ring", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )


# ----------------------------------------------------------------------------- gpt-oss-20b


def _prepare_gpt_oss_20b(args: ScriptArgs) -> None:
    # triton MoE LoRA backend requires a BF16 checkpoint (mxfp4 not supported).
    _download_model(args, "lmsys/gpt-oss-20b-bf16", "gpt-oss-20b")
    _download_datasets(args, _DATASETS)


def _execute_gpt_oss_20b(args: ScriptArgs) -> None:
    """gpt-oss-20b MoE LoRA (BF16 checkpoint, triton MoE LoRA backend)."""
    model_name = "gpt-oss-20b"
    num_gpus_per_node = args.num_gpus_per_node or 4

    _launch(
        args,
        megatron_model_type="gpt-oss-20b",
        num_gpus_per_node=num_gpus_per_node,
        wandb_prefix="gpt-oss-20b-moe-lora",
        ckpt_args=_ckpt_args(f"{args.model_dir}/{model_name}"),
        lora_args=_lora_args(
            args,
            rank=32,
            alpha=32,
            dropout=0.0,
            target_modules="gate_proj,up_proj,down_proj",
            # required for MoE LoRA, else sglang skips MoE layers
            extra="--sglang-lora-backend triton ",
        ),
        rollout_args=_rollout_args(
            prompt_data=f"{args.data_dir}/{_DAPO_MATH}",
            input_key="prompt",
            rm_type="math",
            num_rollout=1,
            rollout_batch_size=32,
            max_response_len=4096,
            global_batch_size=8,
        ),
        eval_args=_aime_eval_args(args, interval=10),
        perf_args=(
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
            # --use-dynamic-batch-size is not supported with --qkv-format bshd
            "--micro-batch-size 1 "
            "--max-tokens-per-gpu 4096 "
        ),
        # TODO: enable KL once gpt-oss ckpt conversion is available.
        grpo_args=_grpo_args(),
        optimizer_args=_optimizer_args(
            "1e-5",
            extra="--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer ",
        ),
        sglang_args=(
            "--rollout-num-gpus-per-engine 4 "
            "--sglang-dtype bfloat16 "
            "--sglang-decode-log-interval 1000 "
            "--sglang-mem-fraction-static 0.2 "
            "--sglang-moe-runner-backend triton "
        ),
        misc_args=_misc_args(
            args,
            num_gpus_per_node,
            # Sink attention (sliding window + learnable softmax) in TE needs BSHD/SBHD, not THD.
            "--qkv-format bshd --attention-backend fused " f"--update-weight-buffer-size {512 * 1024 * 1024} ",
        ),
    )


# ----------------------------------------------------------------------------- kimi-k25


def _prepare_kimi_k25(args: ScriptArgs) -> None:
    # Download INT4 checkpoint and dequantize a BF16 reference for the Megatron bridge.
    _download_model(args, "moonshotai/Kimi-K2.5", "Kimi-K2.5-int4")
    U.exec_command(
        f"python {U.repo_base_dir}/tools/convert_kimi_int4_to_bf16.py "
        f"--model-dir {args.model_dir}/Kimi-K2.5-int4 --output-dir {args.model_dir}/Kimi-K2.5-bf16 "
    )
    _download_datasets(args, _DATASETS)


def _execute_kimi_k25(args: ScriptArgs) -> None:
    """Kimi-K2.5 MoE+MLA LoRA (INT4 rollout + BF16 ref, shared-outer-loras). Use --num-nodes 16."""
    int4_dir = f"{args.model_dir}/Kimi-K2.5-int4"
    bf16_dir = f"{args.model_dir}/Kimi-K2.5-bf16"
    num_gpus_per_node = args.num_gpus_per_node or 8

    _launch(
        args,
        megatron_model_type="kimi-k2-thinking",
        num_gpus_per_node=num_gpus_per_node,
        wandb_prefix="kimi-k25-lora",
        ckpt_args=_ckpt_args(int4_dir, ref_load=bf16_dir, model_name="kimi_k25"),
        lora_args=_lora_args(
            args,
            rank=32,
            alpha=32,
            dropout=0.0,
            target_modules="q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj",
            extra=(
                "--experts-shared-outer-loras "  # shared A on fc1 / shared B on fc2 across experts
                "--lora-base-cpu-backup "  # keep frozen base on CPU to free GPU
                "--no-gradient-accumulation-fusion "
                "--sglang-lora-backend triton "  # required for MoE LoRA
                "--sglang-lora-use-virtual-experts "
            ),
        ),
        rollout_args=_rollout_args(
            prompt_data=f"{args.data_dir}/{_DAPO_MATH}",
            input_key="prompt",
            rm_type="deepscaler",
            num_rollout=20,
            rollout_batch_size=32,
            max_response_len=16384,
            global_batch_size=256,
            extra="--balance-data --filter-zero-reward-samples --use-dynamic-global-batch-size ",
        ),
        eval_args=_aime_eval_args(args, interval=20),
        perf_args=(
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 2 "
            "--context-parallel-size 8 "
            "--expert-model-parallel-size 64 "
            "--expert-tensor-parallel-size 1 "
            "--decoder-last-pipeline-num-layers 30 "
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 4096 "
        ),
        # TIS clamps the cross-engine (sglang INT4 vs Megatron fake-QAT BF16) ratio.
        grpo_args=_grpo_args("--kl-loss-coef 0.00 --kl-loss-type low_var_kl --use-tis "),
        optimizer_args=_optimizer_args(
            "1e-5",
            extra=(
                "--optimizer-cpu-offload "
                "--overlap-cpu-optimizer-d2h-h2d "
                "--use-precision-aware-optimizer "
                "--use-distributed-optimizer "
            ),
        ),
        sglang_args=(
            f"--rollout-num-gpus-per-engine {num_gpus_per_node} "
            "--sglang-mem-fraction-static 0.7 "
            "--sglang-ep-size 8 "
            "--sglang-server-concurrency 1024 "
            "--sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 "
            "--use-rollout-routing-replay "
        ),
        misc_args=_misc_args(
            args,
            num_gpus_per_node,
            _MEGATRON_FP32 + "--no-check-for-nan-in-loss-and-grad --use-miles-router "
            f"--update-weight-buffer-size {4 * 512 * 1024 * 1024} ",
        ),
        env_vars={
            "NCCL_TIMEOUT": "3600",
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": "32",
        },
    )


_RECIPES = {
    "qwen2.5-3b": (_prepare_qwen2_5_3b, _execute_qwen2_5_3b),
    "qwen3-4b": (_prepare_qwen3_4b, _execute_qwen3_4b),
    "gpt-oss-20b": (_prepare_gpt_oss_20b, _execute_gpt_oss_20b),
    "kimi-k25": (_prepare_kimi_k25, _execute_kimi_k25),
}


def prepare(args: ScriptArgs) -> None:
    _RECIPES[args.model][0](args)


def execute(args: ScriptArgs) -> None:
    _RECIPES[args.model][1](args)


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
