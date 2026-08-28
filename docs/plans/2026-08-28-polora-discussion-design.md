# PoLoRA Discussion and Limitations Design

## Goal

Complete the Discussion and Limitations section of
`examples/polora/report.md` with a cautious interpretation of why the Muon
optimizer family may be less well suited to reinforcement-learning gradients,
followed by the study's experimental limitations.

## Discussion structure

The discussion will first connect the empirical result—no clear advantage for
PoLoRA in these runs—to a plausible optimizer–gradient mismatch. Muon-family
methods orthogonalize matrix updates and normalize their scale across singular
directions. This can be useful when gradients are dense and stable, as in
pretraining or supervised fine-tuning, because the dominant update directions
are comparatively well estimated.

The report will then contrast this setting with reinforcement learning, where
the estimator may have high variance because of sparse rewards, finite rollout
sampling, importance-sampling corrections, and reward variability. When the
directions are poorly estimated, orthogonalization and scale equalization may
give noisy directions more influence than their evidence warrants. This is a
hypothesis consistent with the experiment, not a causal conclusion established
by it.

Finally, the discussion will state that a substantially larger effective batch,
potentially on the order of 10 times larger, could improve direction estimates
and make this optimizer family more effective in RL. The factor of 10 will be
presented as a future experimental hypothesis rather than a measured threshold.

## Limitations

The limitations will explicitly cover three scope restrictions:

- the study evaluates only one relatively small model, Qwen3-4B;
- the runs stop after 24 rollout steps, so they do not characterize long-horizon
  convergence or late-training behavior;
- both training and evaluation focus on mathematical reasoning, so the findings
  may not generalize to other RL tasks or reward sources.

The prose will avoid claiming that all RL signals are intrinsically sparse or
that the proposed mechanism has been directly measured.

## Verification

Review the completed section for consistency with the reported experiment,
clear separation between evidence and hypothesis, correct optimizer
terminology, and valid Markdown structure.
