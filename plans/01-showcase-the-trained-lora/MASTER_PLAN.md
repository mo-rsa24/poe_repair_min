# Showcase the Trained LoRA

## The overall claim
**The paper's results figures get carried by the adapter's own measured output, under one
standard a reader can trace, instead of by the cached true correction.**

## What is this plan
This scope turns the trained pooled adapter `phase1_r8_100k` into the paper's figure program:
the causal evidence from the run across correction amounts moved off the cached true correction
(the correction computed from the joined prompt, saved once and read back) and onto the adapter,
the cheap decisive tests of what the adapter learned, the three re-scoped experiments with the
softness question measured rather than assumed, the mechanism read live during training, and the
assembled showcase set. Every choice that could
have gone another way is recorded with its reason in
[decisions-taken-here.md](decisions-taken-here.md), this scope's source of truth; plans quote
it rather than re-derive it.

## Why this plan exists
The figure register is imbalanced: roughly twenty figures measure the cached correction and
one measured the adapter, while the adapter is the artifact the paper ships. Two design walks
(diffusion-researcher role) settled the measurement protocol, the caption tiers, the
experiment set and the tests; this scope executes them.

## What this scope actually does (visual)
```
        ┌────────────────┐  cached r_t (amber)  ┌───────────────────┐
        │ training cache │─────────────────────▶│ adapter (amber    │── checkpoints
        │ 58 pairs       │                      │ coil in LoRA box) │   every 5k
        └───────┬────────┘                      └─────────┬─────────┘
                │ blue: plain PoE path                    │ λ·learned correction, steps 0-10
                ▼                                         ▼
        ┌─────────────────────────────────────────────────────┐
        │ ONE RUNNER: PoE + injected correction               │
        │ A (r8 100k→200k) · B (r16/r32 fresh→100k) ·         │
        │ C (λ×window) · amount series (4 controls) · dog×dog │
        └───────┬──────────────────────────┬──────────────────┘
   tracking set │                          │ the checkpoint watcher
   every 10k    │                          │ (h-space + Jacobian, off-device)
                ▼                          ▼
        ┌──────────────┐            ┌──────────────────┐
        │ scorer:      │            │ intervention:     │
        │ count ≥ 2    │            │ direction → does  │
        └───────┬──────┘            │ it compose?       │
                ▼                   └────────┬─────────┘
        ┌─────────────────────────────────────────────┐
        │ THE FIGURE WALL  (paper/iclr/figures/)      │
        │ amount · window overlay · transfer matrix · │
        │ counted joint-prompt · rank ablation        │
        └─────────────────────────────────────────────┘
```
The length-and-rank grid behind A and B: rank 8 exists at 100k (the shared anchor); A extends
that row to 200k (length axis); B fills the 100k column at ranks 16 and 32 (rank axis);
claims read one line of the grid, never a diagonal. Extending B to 200k is an optional later
resume once A's endpoint exists.

## High-level overview
Five subsystems: the runner (`run_lora_residual_inject` and its window and amount variants,
shared by A, B, C, the run across correction amounts and the dog×dog test); the tracking set (the
live-curve measuring tool built in `instrument-02`, extended with four curves and frozen before
any launch); the checkpoint watcher (a process on a non-training device that reads checkpoints while a run is
going); the scorer and counted figures (instance-count scorer, already validated for two-animal
scenes); and the figure wall with its sidecars.

## Purpose and goals
Purpose: a standalone scope under the [root master plan](../../MASTER_PLAN.md); feeds
writing-the-paper's results section and figure register.
Goals: the numbered list under Goals below.

## Where this scope sits in the order
This scope's steps interleave with the other scopes', so ordering lives in the one running
order at the [repo root MASTER_PLAN.md](../../MASTER_PLAN.md); steps get numbers when
`sync-plan-tree` next recomputes it. **Next in this scope:** reading where the training curve
stops rising (no GPU), then the dog×dog test and the tracking-set extension.

## Mission
Turn the trained pooled adapter `phase1_r8_100k` into the paper's figure program, with the
causal story carried by the adapter's own output rather than the cached true correction,
every figure obeying the ledger's one standard.

## Objectives
1. Move the causal evidence from the cached true correction to the shipped adapter.
2. Establish what the adapter learned (a rule, not a stored vector) by the cheapest decisive
   tests.
3. Run the three re-scoped experiments (length, capacity, injection) with the softness
   question measured rather than argued.
4. Read the mechanism live during training, each read ending in an intervention.
5. Assemble the showcase figures under the ledger's standard.

## Goals
1. Verdict recorded on where the training curve stops rising (ceiling or waypoint) from logged
   curves, in its review file.
2. Dog x dog test judged against its pre-registration.
3. Tracking set extended, and a first short run proving the wiring, before any launch.
4. A, B, C run; verdicts against their pre-registered thresholds in review files.
5. Compose rate against the multiplier on the adapter's correction, with the four control rows,
   AUC beside the cached correction's 0.387-vs-0.023.
6. Counted joint-prompt figure: three bars per pair, a strip of repaired renders as anecdote.
7. Transfer matrix at the group-pooled, concept-disjoint tier.
8. Mechanism interventions' verdict on the causal caption.
9. 70k-100k per-epoch samples scored; the panel showing what the cached correction can reach at
   best, built and read as B's key.

> AUC here is the area under the curve of [compose rate](../../context/world/compose-rate.md)
> against λ, on a 0-to-1 scale: 1.0 would mean
> every generation composed at every λ, 0.0 that none did.

## Expected Outcome
A results section whose adapter claims are carried by adapter-measured figures, a dog×dog
verdict that sharpens or honestly kills the "learned a rule" caption, and the softness
question answered by measurement rather than by re-training folklore.

## Definition of Done
1. The nine Goals above, each with its artifact or review-file verdict.
2. Every showcase figure in `paper/iclr/figures/` carries a sidecar naming space, metric and
   mode per the ledger; wording rules obeyed (no "outperforms SDXL").
3. ⚠️ W&B panels captured during runs via the wandb and Playwright MCPs and filed into the places
   `runbook/reading-a-training-run.md` keeps for screenshots, with healthy-shape captions
   [inferred, owner: session at launch/harvest time]
4. Non-animal scorer re-validation verdict recorded; tier-three captions opened or declined.
5. The scope has a recall gallery: run `/recap-plan-tree @plans/01-showcase-the-trained-lora/MASTER_PLAN.md`
   once every plan above is ✅, and record the Artifact URL it publishes.

## Sub-Scopes
(none yet)

## Plans
Numbering is per-folder, not the running order (that lives in the root table, rows 31 to 43).
All thirteen plans are written. Assembly (05) runs last despite its number.

| # | Plan | What it does | Status |
|---|---|---|---|
| 01 | [01-read-where-the-curves-flatten](plans/reading/01-read-where-the-curves-flatten.md) | the free read that re-scopes A and B; it informs them, nothing waits on it | ⚠️ written |
| 02 | [02-the-dog-x-dog-same-prompt-check](plans/tests/02-the-dog-x-dog-same-prompt-check.md) | the pre-registered same-prompt check | ⚠️ written |
| 03 | [03-the-correction-amount-series](plans/tests/03-the-correction-amount-series.md) | lambda on the learned correction, four controls; shares one runner with 07 and 02 | ⚠️ written |
| 04 | [04-the-transfer-matrix-figure](plans/figures/04-the-transfer-matrix-figure.md) | group-pooled, concept-disjoint tier | ⚠️ written |
| 05 | [05-assemble-the-showcase-figures](plans/figures/05-assemble-the-showcase-figures.md) | the wall, under the standard; closes the scope | ⚠️ written |
| 06 | [06-extend-the-tracking-set](plans/tools/06-extend-the-tracking-set.md) | the `instrument-02` curves plus four more, before any launch | ⚠️ written |
| 07 | [07-experiment-c-lambda-window](plans/experiments/07-experiment-c-lambda-window.md) | injection run across values, on existing checkpoints, in-session | ⚠️ written |
| 08 | [08-experiment-a-resume-to-200k](plans/experiments/08-experiment-a-resume-to-200k.md) | length axis: rank 8 from 100k to 200k, sbatch on biggpu | ⚠️ written |
| 09 | [09-experiment-b-rank-16-32](plans/experiments/09-experiment-b-rank-16-32.md) | rank axis at 100k, two idle nodes over SSH | ⚠️ written |
| 10 | [10-the-checkpoint-watcher](plans/tests/10-the-checkpoint-watcher.md) | the checkpoint watcher: h-space + Jacobian per broad checkpoint, off-device, ends in interventions | ⚠️ written |
| 11 | [11-the-counted-joint-prompt-figure](plans/figures/11-the-counted-joint-prompt-figure.md) | score mono renders; three bars per pair + repair strip | ⚠️ written |
| 12 | [12-close-f8a-and-the-best-case-panel](plans/figures/12-close-f8a-and-the-best-case-panel.md) | score 70k-100k samples; the panel showing what the cached correction can reach at best (reconciles with figure-01 next door) | ⚠️ written |
| 13 | [13-revalidate-the-scorer-off-animals](plans/tools/13-revalidate-the-scorer-off-animals.md) | opens tier-three captions | ⚠️ written |

## Environment Context
`environment/00-INDEX.md`: the execution protocol (idle-node order, shared-device path, admin
node-cap fallback), the storage split, the fp16 upcast rule, the co3/co3_bw python paths.

## Diagram Prompts
The scope's illustrated map is [diagram-prompts.md](diagram-prompts.md): subject lane and
capstone seeded at scope birth, all pieces [planned]; the process lane is authored by
`populate-plans` once the plan files exist.
