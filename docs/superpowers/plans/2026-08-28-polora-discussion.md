# PoLoRA Discussion and Limitations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the PoLoRA report with an evidence-calibrated discussion of Muon-family optimizers in RL and a concise limitations subsection.

**Architecture:** Make one focused prose edit in the existing report. Separate the proposed mechanism from the directly observed results, and separate both from the experiment's scope limitations.

**Tech Stack:** Markdown, Git text-diff inspection

**Spec:** `docs/plans/2026-08-28-polora-discussion-design.md`

## Global Constraints

- Describe the optimizer–gradient mismatch as a hypothesis rather than a demonstrated causal mechanism.
- Present a 10× larger effective batch as future work, not a measured requirement.
- Cover only the requested limitations: small model, 24-step early-stopped training, and math-only scope.
- Preserve all other report content.

---

### Task 1: Write and verify the discussion and limitations

**Files:**
- Modify: `examples/polora/report.md`
- Test: Markdown structure and textual-claim review of `examples/polora/report.md`

**Interfaces:**
- Consumes: the results in Sections 3.1 and 3.2 and the approved writing design
- Produces: a complete Section 4 that Section 5 can follow without changing the experimental record

- [x] **Step 1: Replace the empty Section 4 with the discussion**

Write prose that connects the absence of a clear PoLoRA advantage to the
hypothesis that Muon-style orthogonalization works best when update directions
are well estimated. Contrast dense, comparatively stable pretraining/SFT
gradients with high-variance RL estimates affected by finite rollouts,
importance sampling, and reward variability. Explain that scale equalization
may amplify poorly estimated directions.

- [x] **Step 2: Add the batch-size hypothesis**

State that a much larger effective batch—potentially around 10× larger—could
reduce variance enough to make orthogonalized updates more reliable, while
making clear that this experiment does not validate that value.

- [x] **Step 3: Add a distinct limitations subsection**

State that the evidence comes from Qwen3-4B, only 24 rollout steps with early
stopping, and mathematical reasoning data and evaluation. Explain the resulting
limits on claims about scale, convergence, and task generality.

- [x] **Step 4: Inspect the edited section**

Run:

```bash
sed -n '/^## 4\./,/^## 5\./p' examples/polora/report.md
```

Expected: Section 4 contains the discussion and a limitations subsection;
Section 5 remains present and unchanged.

- [x] **Step 5: Review the diff and terminology**

Run:

```bash
git diff --check -- examples/polora/report.md
git diff -- examples/polora/report.md
```

Expected: no whitespace errors; the diff changes only Section 4; all causal and
10× batch-size statements are qualified as hypotheses.

### Task 2: Simplify the discussion

**Files:**
- Modify: `examples/polora/report.md:130`
- Test: textual and Markdown review of `examples/polora/report.md`

**Interfaces:**
- Consumes: the completed discussion from Task 1 and the simplification revision in the spec
- Produces: a two-paragraph discussion followed by the unchanged limitations subsection

- [x] **Step 1: Condense the optimizer hypothesis**

Replace the first two discussion paragraphs with one short paragraph that
contrasts well-estimated pretraining/SFT gradients with noisy RL gradients and
briefly explains why orthogonalization may amplify estimation errors.

- [x] **Step 2: Mark larger-batch evaluation as future work**

Replace the batch-size paragraph with a short future-work statement. Retain the
potential 10× scale but explicitly state that it has not been tested.

- [x] **Step 3: Verify the revision**

Run:

```bash
sed -n '/^## 4\./,/^### Limitations/p' examples/polora/report.md
pre-commit run --files examples/polora/report.md
```

Expected: exactly two discussion paragraphs appear before `### Limitations`,
the second marks larger-batch evaluation as future work, and repository hooks
pass.
