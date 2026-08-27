# PoLoRA vs AdamW — LoRA RL on Qwen3.5-4B

A/B comparison of the two optimizers on the same LoRA RL recipe: GRPO on
DAPO-Math-17k, evaluated on AIME 2024 and AIME 2025.

| | |
| --- | --- |
| Model | `/root/models/Qwen3.5-4B` (see caveat below) |
| Train data | DAPO-Math-17k, `deepscaler` rule-based reward |
| Eval data | AIME 2024 + AIME 2025, every 10 rollouts, 8 samples/prompt |
| Adapter | LoRA rank 32, alpha 32, dropout 0, `all-linear` |
| Parallelism | 8×H200, colocated, TP=1 / PP=1 (polora requires both) |

## Run

```bash
cd /root/miles
bash examples/polora/run-adamw.sh     # baseline
bash examples/polora/run-polora.sh    # treatment
```

Both scripts kill leftover ray/sglang processes and start their own ray head, so
run them one at a time. They log to wandb project `miles-polora-vs-adamw`, groups
`qwen3.5-4b-lora-adamw` and `qwen3.5-4b-lora-polora`.

## Layout

`common.sh` holds every setting that must be identical across the two arms and
performs the launch. Each runner sets only `OPTIMIZER_ARGS` and `RUN_TAG`, then
sources it. Keep it that way — a config that drifts between arms invalidates the
comparison.

Overridable via environment: `HF_CHECKPOINT`, `TRAIN_DATA`, `EVAL_DATA_AIME24`,
`EVAL_DATA_AIME25`, `MODEL_NAME`, `GPUS_PER_NODE`.

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
