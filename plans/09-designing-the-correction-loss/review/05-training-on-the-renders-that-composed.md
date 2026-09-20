# 🧪 Review: is a corpus of 72 hand-picked renders enough to train against?

Nothing has run. This file judges [the design](../plans/hypothesis/05-training-on-the-renders-that-composed.md), which builds a second data path and trains against the noise added to renders the model itself got right.

Its questions are written before the runs, which is the ordering the conventions ask for.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/05-training-on-the-renders-that-composed.md) | the second data path, the four nested runs, the curve, the read |
| **this file** | **the verdict: whether compose rate rises with cell count, and if it does, whether this target beats `01`'s** |

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

- **The teacher**: the model's own joint-prompt prediction, which every objective from `00` to `04` trains against. It is wrong on some cells, which is the whole reason for this variation.
- **A render that composed**: an image the model drew that a person judged, by eye, to show both concepts separately. 43 of them, in `cells_v57.json`.
- **Nested subsets**: 10 cells inside 20 inside 40 inside 72. Independently drawn subsets would confound how many cells with which cells.
- **The scaling curve**: compose rate on held-out pairs against cell count, four points.
- **A high-variance target**: the noise drawn for one image on one pass. The prediction it replaces is already an average, so this trades a biased target for a noisy one.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the hypothesis** (may change the claim, and only by failing the question written before it ran), with a **measuring tool** built first: the second data path is an instrument and is judged by whether it can fail, not by what it finds.

The claim it can change is that a non-model target beats the model's own answer. It cannot change any earlier variation's verdict, because it moves a different axis.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the two-epoch smoke on 10 cells | builds a measuring tool | | | a log line naming the path and the count | not started |
| 10 cells, 50,000 steps | tests the hypothesis | | | 20 checkpoints, one W&B run | not started |
| 20 cells, 50,000 steps | tests the hypothesis | | | 20 checkpoints, one W&B run | not started |
| 40 cells, 50,000 steps | tests the hypothesis | | | 20 checkpoints, one W&B run | not started |
| 43 cells, 50,000 steps | tests the hypothesis | | | 20 checkpoints, one W&B run | not started |
| the read against `01` at matched steps | tests the hypothesis | | | `06-per-checkpoint.json`, blind strips | not started, and conditional |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does compose rate on held-out pairs rise with cell count across 10, 20, 30 and 43?**
  - **The threshold, fixed before the runs and held in source, not here:** `MIN_SCALING_SLOPE` and `MIN_COMPOSE_AT_FULL` in the scaling script. Prose can be softened after the fact; a constant in a file shows up in a diff.
  - **Rising and clearing the bar at 72** means the corpus is enough, and only then is the comparison against `01` read.
  - **Flat** is a null and closes the plan. It says either 43 cells is too few or the noisy target costs more than the biased one did, and the next question separates them.
  - **Rising but short of the bar at 72** closes the plan with a request rather than a verdict: the corpus works, and it needs more cells picked by hand.
  - **This is the only question here whose failure may move a plan.** A flat curve ends the variation without any comparison against `01` being read, because a comparison drawn from four runs that never learned means nothing.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **If the curve is flat, is the corpus the limit or is the target too noisy?**
  - The loss separates them. A loss that falls normally while compose rate does not move points at the corpus. A loss visibly noisier than the earlier variations' at matched steps points at the target.
- [ ] ⚠️ **Did all 43 cells resolve to a render?**
  - A missing render is skipped silently and the run trains on fewer cells than its name claims, which puts every point of the curve at the wrong x. The printed count is the check.
- [ ] ⚠️ **Are the four subsets actually nested?**
  - Asserted in code and read again by eye. Four independent draws would confound how many cells with which cells, and nothing in any log would show it.
- [ ] ⚠️ **Does the VAE encode round-trip?**
  - One decode-and-look on the first cell. A wrong-precision encode gives latents that are subtly off and a loss that still falls, so the run succeeds on a corrupted target.
- [ ] ⚠️ **Does the new data path appear in `config.json`?**
  - Plan 01 made objective flags recordable. A run whose saved config cannot say which target it trained against is unattributable later.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?**
  - Against `01`, only the target differs. Rank, alpha, learning rate, schedule, held-out seeds and sampler settings identical. Diff the two configs and say so.
- [ ] ⚠️ **Was the measuring tool sound?**
  - The same scorer and the same sampler settings for every run. The four curves read at matched optimizer steps.
- [ ] ⚠️ **Did the run respect the environment?**
  - Four cards on `bigbatch`, pinned from a live idle probe rather than `sinfo`'s idle list. Latents and checkpoints on `/datasets`. The disk guard naming the filesystem actually written to.
- [ ] ⚠️ **Is the corpus what it claims to be?**
  - 43 cells over 29 pairs, none of them cat × dog. Cat × dog is the held-out test pair and never trains; a cell pool that leaked it would make every held-out number meaningless.
- [ ] ⚠️ **Was the bar changed after the curve was seen?**
  - The two constants live in the scaling script. `git log` on that file against the run dates answers it.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#what-the-run-cost-and-what-it-bought) ➡️

The curve with its bar drawn on it rather than described, the four cell counts with their real resolved sizes, and, if the answer is flat, which of the two causes the loss evidence supports.

The page also owes the sentence that the corpus cannot grow without a person picking cells by eye, because a reader will otherwise ask why the largest point is 72.

## What the run cost, and what it bought

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Four runs at 50,000 steps is roughly 80 GPU-hours, one night on four cards. What it buys either way is the sizing answer: whether a hand-picked corpus of this size can carry a training objective at all. That answer outlives this variation, because every future non-model target faces the same question.

## Still open

Navigation: ⬅️ [What the run cost](#what-the-run-cost-and-what-it-bought) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Nothing follows this plan in the scope. `02` is authored by plan 04's close-out.
