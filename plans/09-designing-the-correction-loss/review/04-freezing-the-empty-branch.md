# 🧪 Review: does freezing the empty branch reach the same composed predictions?

Nothing has run. This file judges [the design](../plans/hypothesis/04-freezing-the-empty-branch.md), which reads the empty branch from the cache instead of the model, closing the direction the loss cannot see exactly rather than approximately.

Its questions are written before the run, which is the ordering the conventions ask for and which [the V0a read](03-the-v0a-read.md) could not have.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/04-freezing-the-empty-branch.md) | the agreement check, the trainer and sampler change, the 50,000-step run, the read |
| **this file** | **the verdict: inside or outside `00`'s band at matched steps, and whether training and sampling agreed** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **Freezing a branch**: not running it through the adapted model and reading the cached frozen tensor instead. It is not freezing weights; the adapter sits on shared weights and changes all three branches at once.
- **The exact closure**: with the empty branch cached, the guidance weight factors out of the loss entirely, leaving a constant the learning rate absorbs. Nothing is left free to move in the invisible direction.
- **The sampler mismatch**: training freezes a branch and sampling does not. The two then describe different models and every number from the run is about neither.
- **The seed-noise band**: computed in plan 02 from the definition fixed in scope 01's `review/09-experiment-b-rank-16-32.md`. Used here as written.
- **Matched steps**: this run at a given optimizer step against the objective running today at the same step.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the hypothesis** (may change the claim, and only by failing the question written before it ran). The claim it can change is the objectives note's central derivation, that freeing the empty branch buys no predictions, which variations `02`, `03` and `04` all rest on.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the agreement check, one seed rendered two ways | builds a measuring tool | | | `agreement-check.png`, a test in `tests/` | not started |
| the two-epoch smoke on the changed trainer | builds a measuring tool | | | a log line naming the `2K` batch | not started |
| the 50,000-step training | tests the hypothesis | | | 20 checkpoints, 20 render sets, one W&B run | not started |
| the scoring pass at matched steps | tests the hypothesis | | | `01-per-checkpoint.json`, 8 blind strips | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **At matched optimizer steps, does freezing the empty branch land inside `00`'s held-out seed-noise band on compose rate?**
  - **The threshold, fixed before the run:** inside the band is a confirmation. The objective is settled in favour of freezing, which is exact where the penalty is approximate and trains on two thirds of the batch. Outside the band and worse falsifies the derivation that freeing the branch buys no predictions, and `02`, `03` and `04` must be reconsidered before any of them is planned. Outside and better is not predicted by the derivation and needs its own explanation before it is believed.
  - **This is the only question here whose failure may move a plan.** It is also the only question in this scope that can move three plans at once, because three unwritten variations assume its answer.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Does the agreement check fail when it is supposed to?**
  - Rendering with the sampler's empty-prompt pass adapted and detached must give visibly different pictures. If it does not, the check measures nothing and every assertion built on it is worthless. This is answered before any training starts.
- [ ] ⚠️ **Does `train/loss_undialled` track `train/loss_fit` under the frozen branch?**
  - Plan 01 added the panel. Under the exact closure the guidance weight has factored out, so the two should now move together. Divergence as large as `00`'s means the substitution did not take and the compose-rate read means nothing.
- [ ] ⚠️ **Is the forward pass actually `2K` wide?**
  - Read it off the log line, not off the intention. A reshape that silently kept three rows would train the objective running today under a new name.
- [ ] ⚠️ **What did it cost per step, against the `3K` baseline?**
  - The "a third cheaper" claim becomes a measurement here or it is dropped.
- [ ] ⚠️ **Does the blind read agree with the compose count?**
  - The scorer counts instances and cannot tell a cat beside a dog from two dogs. Where the two disagree, the eye is the record and the disagreement is reported rather than resolved.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?**
  - Only the frozen-branch switch differs from the run it is read against. Diff the two `config.json` files and say so. Rank, alpha, learning rate, weight decay, pool, cells, step range and schedule all identical.
- [ ] ⚠️ **Was the measuring tool sound?**
  - The same scorer, the same sampler settings, the same eight held-out seeds, the same λ. The agreement check passing on the changed sampler, not only on the old one.
- [ ] ⚠️ **Did the run respect the environment?**
  - The node recorded with its GPU model and the python used. On a Blackwell it must be `co3_bw`: under `co3` a CUDA operation there produces no output rather than an error. Checkpoints on `/datasets`, disk guard naming the filesystem actually written to.
- [ ] ⚠️ **Did training and sampling read the same branch?**
  - The question this whole plan is gated on. A run where they disagree measures neither objective, and the failure is silent: the loss falls, the renders look plausible, and the numbers describe a model that does not exist. On 2026-09-05 a windowed sampler left the adapter attached during plain references and cost a re-render.
- [ ] ⚠️ **Was the band recomputed after seeing this run?**
  - It must not be. Plan 02's band is used as written.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#what-the-run-cost-and-what-it-bought) ➡️

`01`'s page owes the verdict in its first line, the band drawn on the compose-rate figure rather than described, the measured cost per step beside the arithmetic claim, and the agreement check's two renders side by side, because the check is the reason the rest of the page can be believed.

If the answer is a falsification, the objectives note owes an edit: its V1 block currently reads "Tradeoff: none found."

## What the run cost, and what it bought

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Estimated at about 20 hours on an RTX 3090 for 50,000 steps, less if the narrower batch pays off. What it buys either way is a settled objective: a confirmation makes every later variation cheaper and removes the guidance weight from the problem, and a falsification stops three unwritten plans resting on a derivation that does not hold.

## Still open

Navigation: ⬅️ [What the run cost](#what-the-run-cost-and-what-it-bought) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Nothing in this scope yet. This plan's close-out authors `02`, anchoring the empty branch to two animals, which the objectives note says should be compared against this result rather than against the objective running today.
