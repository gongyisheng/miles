# PoLoRA vs AdamW — LoRA RL on Qwen3.5-4B

A/B comparison of the two optimizers on the same LoRA RL recipe: GRPO on
DAPO-Math-17k, evaluated on AIME 2024 and AIME 2025.

| | |
| --- | --- |
| Model | `/root/models/Qwen3.5-4B` (see caveat below) |
| Train data | DAPO-Math-17k, `deepscaler` rule-based reward |
| Eval data | AIME 2024 + AIME 2025, every 10 rollouts, 8 samples/prompt |
| Adapter | LoRA rank 32, alpha 32, dropout 0, `all-linear` |
| Parallelism | disaggregated, 4 GPUs per arm — 2 train + 2 rollout, TP=1 / PP=1 (polora requires both) |

## Run

The two arms split one 8-GPU box and run **at the same time**:

```bash
cd /root/miles
bash examples/polora/run-adamw.sh   &   # GPUs 0,1 train / 2,3 rollout
bash examples/polora/run-polora.sh  &   # GPUs 4,5 train / 6,7 rollout
wait
```

They log to wandb project `miles-polora-vs-adamw`, groups
`qwen3.5-4b-lora-adamw` and `qwen3.5-4b-lora-polora`.

Neither script kills stray processes, because a node-wide `ray stop --force` or
`pkill python` would take out the other arm. To reclaim a wedged machine, run
either script with `CLEANUP=1` — that kills **both** arms.

## Layout

`common.sh` holds every setting that must be identical across the two arms and
performs the launch. Each runner sets only `OPTIMIZER_ARGS`, `RUN_TAG`, its GPU
ids and its ray port offset, then sources it. Keep it that way — a config that
drifts between arms invalidates the comparison.

Overridable via environment: `HF_CHECKPOINT`, `TRAIN_DATA`, `EVAL_DATA_AIME24`,
`EVAL_DATA_AIME25`, `MODEL_NAME`, `TRAIN_GPU_IDS`, `ROLLOUT_GPU_IDS`,
`RAY_PORT_OFFSET`, `RAY_TEMP_DIR`, `CLEANUP`.

## Placement

Each arm owns 4 GPUs and sees no others: `CUDA_VISIBLE_DEVICES` is set to
`${TRAIN_GPU_IDS},${ROLLOUT_GPU_IDS}` before ray starts.

| Arm | Train | Rollout | ray GCS / dashboard |
| --- | --- | --- | --- |
| adamw | 0,1 | 2,3 | 6379 / 8265 |
| polora | 4,5 | 6,7 | 6380 / 8266 |

Within an arm the split is disaggregated: training and rollout hold disjoint
GPUs, there is no `--colocate`, so weights are never offloaded to make room for
the inference engine, and sglang gets `--sglang-mem-fraction-static 0.8` on its
dedicated devices.

The train/rollout pin comes from the *order* of `CUDA_VISIBLE_DEVICES`.
`_create_placement_group` in `miles/ray/placement_group.py` sorts its bundles by
`(node, gpu id)` ascending and gives the actor the first `actor_num_gpus`,
rollout the remainder; ray's gpu ids index into `CUDA_VISIBLE_DEVICES`, so the
train devices being listed first is what lands the actor on them. Swap the two
variables to flip the halves; keep the lists disjoint.

Each arm also runs its own ray head. `RAY_PORT_OFFSET` (0 and 1) shifts every
port ray defaults to a fixed value — GCS, dashboard, dashboard agent, client
server — and each cluster gets its own `--temp-dir` (`/tmp/ray-<tag>`). Ports ray
picks automatically need no offset.

To run one arm alone on all 8 GPUs, colocated instead: set
`TRAIN_GPU_IDS=0,1,2,3,4,5,6,7`, drop `--rollout-num-gpus` (ignored under
colocate), add `--colocate`, and lower the memory fraction.

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
