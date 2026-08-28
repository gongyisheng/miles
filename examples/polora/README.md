# PoLoRA vs AdamW — LoRA RL on Qwen3.5-4B

A/B comparison of the two optimizers on the same LoRA RL recipe: GRPO on
DAPO-Math-17k, evaluated on AIME 2024 and AIME 2025.

| | |
| --- | --- |
| Model | `/root/models/Qwen3.5-4B` (see caveat below) |
| Train data | DAPO-Math-17k, `deepscaler` rule-based reward |
| Eval data | AIME 2024 + AIME 2025, every 10 rollouts, 1 sample/prompt |
| Adapter | LoRA rank 32, alpha 32, dropout 0, `all-linear` |
| Parallelism | colocated, 4 GPUs per arm — train and rollout share all 4, TP=1 / PP=1 (polora requires both) |

## Run

The two arms split one 8-GPU box and run **at the same time**:

```bash
cd /root/miles
bash examples/polora/run-adamw.sh   &   # GPUs 0-3
bash examples/polora/run-polora.sh  &   # GPUs 4-7
wait
```

`USE_WANDB=1` (in the runner or the environment) logs to wandb project
`miles-polora-vs-adamw`, groups `qwen3.5-4b-lora-adamw` and
`qwen3.5-4b-lora-polora`. Credentials come from either an exported
`WANDB_API_KEY` or a prior `wandb login` (`~/.netrc`) — `common.sh` checks for
one of the two and refuses to launch otherwise, since `--use-wandb`
unauthenticated aborts the run before the first rollout. The adamw arm defaults
to `1`, polora to `0`; set both the same way, or you lose half the comparison.

Neither script kills stray processes, because a node-wide `ray stop --force` or
`pkill python` would take out the other arm — `ray stop` is node-wide and takes
no address, so there is no way to stop one arm's head alone. A head left behind
by a failed attempt is therefore *reused* on the next run of that arm rather
than restarted. To reclaim a wedged machine, run either script with `CLEANUP=1`
— that kills **both** arms.

## Layout

`common.sh` holds every setting that must be identical across the two arms and
performs the launch. Each runner sets only `OPTIMIZER_ARGS`, `RUN_TAG`, its GPU
ids and its ray port offset, then sources it. Keep it that way — a config that
drifts between arms invalidates the comparison.

Overridable via environment: `HF_CHECKPOINT`, `TRAIN_DATA`, `EVAL_DATA_AIME24`,
`EVAL_DATA_AIME25`, `MODEL_NAME`, `GPU_IDS`, `RAY_PORT_OFFSET`, `RAY_TEMP_DIR`,
`MEGATRON_PATH`, `USE_WANDB`, `CLEANUP`, `NUM_ROLLOUT`,
`SKIP_EVAL_BEFORE_TRAIN`.

## Requirements

The repo tracks a Megatron-LM pin (`miles-main`, 2026-08-19 or newer) — older
checkouts fail at import with `No module named
'megatron.core.tokenizers.utils'`. `MEGATRON_PATH` points at the checkout and
defaults to `/root/Megatron-LM`.

The job's ray runtime env mirrors what `command_utils.execute_train` builds:
`NCCL_NVLS_ENABLE` follows an NVLink probe rather than being pinned, since
forcing `NCCL_ALGO=Ring` would disable NVLS on an NVLink box.

## Placement

Each arm owns 4 GPUs and sees no others: `CUDA_VISIBLE_DEVICES` is set to
`${GPU_IDS}` before ray starts.

| Arm | GPUs | ray GCS / dashboard |
| --- | --- | --- |
| adamw | 0,1,2,3 | 6379 / 8265 |
| polora | 4,5,6,7 | 6380 / 8266 |

Within an arm the split is colocated: `--colocate` puts the sglang engines on
the same 4 GPUs the actor trains on, so each role gets four devices instead of
two. Disaggregating them left half the arm idle at any moment, which is what
made it slow. The cost is that `arguments.py` then defaults `--offload-train`
and `--offload-rollout` on, so the two swap in and out of HBM between phases,
and sglang drops to `--sglang-mem-fraction-static 0.4` since it no longer has
the devices to itself — the value the validated LoRA+colocate recipe in
`tests/e2e/lora/test_lora_qwen2.5_0.5B.py` uses.

`--rollout-num-gpus` is not passed: `arguments.py` overrides it to
`actor_num_gpus_per_node * actor_num_nodes` under colocate.

Each arm also runs its own ray head. `RAY_PORT_OFFSET` (0 and 1) shifts every
port ray defaults to a fixed value — GCS, dashboard, dashboard agent, client
server — and each cluster gets its own `--temp-dir` (`/tmp/ray-<tag>`). Ports ray
picks automatically need no offset.

The sglang engine ports are not covered by `RAY_PORT_OFFSET`.
`allocate_rollout_engine_addr_and_ports_normal` scans upward from a hardcoded
`base_port=15000` for the first free port, and nothing exposes that base as a
flag, so two arms that reach engine startup at the same moment can each be
handed the same port before either binds it. The window is narrow — port
allocation happens minutes into startup, well after launch — but if an arm dies
with a bind error on a 15xxx port, stagger the two launches instead of starting
them in the same instant.

To run one arm alone on all 8 GPUs, set `GPU_IDS=0,1,2,3,4,5,6,7`.

## Checkpoint caveat

`/root/models/Qwen3.5-4B` currently contains a **Qwen3-4B** checkpoint —
`config.json` reports `model_type: qwen3`, 36 layers, hidden 2560, vocab 151936.
The repo's `scripts/models/qwen3.5-4B.py` describes a different architecture
(32 layers, vocab 248320, `--attention-output-gate`), so pairing the two would
fail to load. `MODEL_NAME` therefore defaults to `qwen3-4B`, matching what is
actually on disk. Once a genuine Qwen3.5-4B checkpoint is downloaded, run with
`MODEL_NAME=qwen3.5-4B`.

## Comparing fairly

- The learning rates are not on the same scale. AdamW's `1e-5` is a per-element
  step; polora's `2e-4` sets the spectral norm of the whole factor update
  (`rho = lr / (sigma_max(A) + sigma_max(B))`). Sweep each arm's LR before
  reading anything into a single pair of curves.
- polora applies no weight decay; AdamW here uses `0.1`. If you want to isolate
  the preconditioner, set AdamW's `--weight-decay 0.0` too.
- Dynamic sampling is commented out in `common.sh` on purpose: it would give the
  two arms different training data. Enable it on both or neither.
- polora rejects TP>1, PP>1, multi-LoRA, fp16, `--optimizer-cpu-offload`, and
  `--use-precision-aware-optimizer` (see `_validate_polora_args` in
  `miles/utils/arguments.py`). The distributed optimizer is disabled
  automatically for polora runs.
