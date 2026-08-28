# Can PoLoRA Improve LoRA RL?

## 1. Introduction

*PoLoRA: A Preconditioned Orthogonalized LoRA Optimizer* [1] introduces PoLoRA,
an optimizer designed specifically for the product structure of LoRA. It
combines a product-aware spectral update direction, curvature
preconditioning motivated by controlling per-sample loss changes, and a
magnitude rule that constrains both factor-level and merged-weight updates.
Across several models and instruction-tuning datasets for code, mathematics,
and language, PoLoRA reaches Adam's final held-out loss in 1.2–1.7× fewer
optimization steps.

However, the original study evaluates PoLoRA only in supervised fine-tuning. In
this work, we investigate whether its optimization advantages transfer to
LoRA-based reinforcement learning by comparing PoLoRA with AdamW under the same
GRPO training setup.

## 2. Experimental Setup

We compare PoLoRA with an AdamW baseline using LoRA-GRPO runs on mathematical
reasoning problems. All RL settings are held constant across the two runs; only
the optimizer configurations differ.

### Training stack

Training used the `polora` branch of a
[forked Miles repository](https://github.com/gongyisheng/miles) at commit
`b3c94ca9`, together with Megatron Core 0.16.0rc0 at commit `235952df6`. All
runs used CUDA 13.0.

### Model

We use Qwen3-4B as the base model. LoRA is applied to all linear layers with rank 32, alpha 32, and no
dropout.

### Data

Training uses DAPO-Math-17k with the `deepscaler` rule-based reward. Evaluation
uses AIME 2024 and AIME 2025, with one sample per prompt and
top-k 1 decoding.

### RL configuration

- We conduct two rounds of GRPO experiments using a clipping range of
  [0.20, 0.28]. The first uses standard sampling without oversampling or
  zero-standard-deviation filtering. The second enables DAPO oversampling and
  discards prompt groups whose rewards have zero standard deviation. In both
  rounds, the KL-loss, KL-reward, and entropy coefficients are set to zero.

- Each run was configured for 100 rollout steps but was stopped early after 24
  steps. Responses are generated at temperature 1. The standard GRPO round uses
  a maximum response length of 16,384 tokens, while the DAPO dynamic-sampling
  round uses 32,768 tokens.

- Each rollout batch contains 32 prompts, with eight samples generated per
  prompt, giving a global batch size of 256. In the second round, DAPO dynamic
  sampling draws an oversampling batch of 64 prompts before applying the
  reward-variation filter. Dynamic batching allows up to 32,768 tokens per GPU.

- We use colocated RL, where training and rollout share the same four GPUs for
  each optimizer arm on a single node. Data parallelism is 4, while tensor,
  pipeline, context, expert, and expert tensor parallelism are all 1. Each
  rollout engine uses one GPU.

### Optimizer

- For AdamW, we use a learning rate of 1e-5 with a constant schedule, weight
  decay of 0.1, beta1 of 0.9, and beta2 of 0.98.

- For PoLoRA, we use a learning rate of 2e-4 with a constant schedule, no
  weight decay, beta1 of 0.9, and curvature beta of 0.99. The original paper
  observes that PoLoRA's optimal learning rate is 20–100 times that of AdamW
  and that PoLoRA is less sensitive to the choice of learning rate. For our RL
  experiments, we therefore started at the lower end of this range by setting
  PoLoRA's learning rate to 20 times that of AdamW. We verified that this
  learning rate produced a healthy gradient norm comparable to AdamW's. We use
  8 PolarExpress matrix-sign iterations and 8 inverse-square-root iterations
  for the spectral updates.

### Hardware

Each experiment was run on four NVIDIA H200 GPUs

## 3. Results

### 3.1 GRPO run

| Metric | AdamW | PoLoRA |
| --- | ---: | ---: |
| Mean rollout reward (steps 20–24) | 0.797 | 0.794 |
| Mean rollout truncation ratio (steps 20–24) | 0.141 | 0.148 |
| AIME 2024 accuracy (step 20)| 63.3% | 63.3% |
| AIME 2025 accuracy (step 20) | 46.7% | 46.7% |
| Mean eval truncation ratio (step 20) | 0.433 | 0.433 |
| Mean gradient norm (steps 20–24) | 0.00997 | 0.00975 |
| Mean train–rollout KL (steps 20–24) | 0.000625 | 0.000624 |

AdamW and PoLoRA show nearly identical rollout, evaluation,
and optimization metrics; this short GRPO run therefore shows no clear
advantage for PoLoRA.

### 3.2 DAPO run

- We enable oversampling and filter prompt groups with zero reward standard
  deviation. Rewards in the GRPO run were already relatively high (about
  0.65–0.80), making all-correct groups more likely. Because equal
  rewards within a group produce no useful relative learning signal, filtering
  these groups focuses training on informative samples, while oversampling
  maintains the effective batch size.

- We increase the maximum rollout and evaluation response length from 16,384 to
  32,768 tokens. Under the original GRPO run settings, the evaluation truncation
  ratio was about 0.4. After enabling zero-standard-deviation filtering,
  rollout responses also became longer and the rollout truncation ratio
  increased from about 0.15 to above 0.3, motivating the larger rollout limit.

| Metric | AdamW | PoLoRA |
| --- | ---: | ---: |
| Mean rollout reward (steps 20–24) | 0.642 | 0.604 |
| Mean rollout truncation ratio (steps 20–24) | 0.034 | 0.034 |
| AIME 2024 accuracy (step 20) | 70.0% | 63.3% |
| AIME 2025 accuracy (step 20) | 56.7% | 63.3% |
| Mean eval truncation ratio (step 20) | 0.167 | 0.200 |
| Mean gradient norm (steps 20–24) | 0.01188 | 0.01177 |
| Mean train–rollout KL (steps 20–24) | 0.000572 | 0.000567 |

AdamW has higher rollout reward and AIME 2024 accuracy, while PoLoRA performs
better on AIME 2025. Their truncation ratios and optimization metrics remain
similar, so the DAPO run shows no consistent advantage for either optimizer.

## 4. Discussion

Our experiments show no consistent advantage for PoLoRA over AdamW. One
possible explanation is that Muon-style orthogonalization is better suited to
the dense, stable gradients of pretraining and supervised fine-tuning. RL
gradients are noisier because of stochastic rollouts, importance sampling, and
sparse or variable rewards. When update directions are poorly estimated,
normalizing their scales may give noisy directions too much influence and
amplify estimation errors. This is a hypothesis rather than a causal conclusion.

Future work can test whether larger effective batches reduce gradient
variance enough to make orthogonalized updates more reliable.

## 5. Limitations

- We evaluate only one relatively small model, Qwen3-4B, so the findings may
  not hold at larger model scales.

- Each run was early stopped after 24 rollout steps, so these short runs cannot
  characterize long-horizon convergence or determine whether the optimizers
  diverge later in training.

- Training and evaluation are restricted to mathematical reasoning, so the
  results may not generalize to other RL tasks, domains, or reward sources.

## References

[1] N. Ghosh, T. Parshakova, and R. M. Gower, “[PoLoRA: A Preconditioned
Orthogonalized LoRA Optimizer](https://arxiv.org/pdf/2607.17620),” arXiv
preprint arXiv:2607.17620, 2026.

## Appendix: W&B Report

The complete run dashboards and logged metrics are available in the
[AdamW vs. PoLoRA W&B report](https://wandb.ai/gongyisheng/miles-polora-vs-adamw/reports/AdamW-vs-Polora--VmlldzoxNzgyMTMwOA).
