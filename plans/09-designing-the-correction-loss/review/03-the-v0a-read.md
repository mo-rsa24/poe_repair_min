# ⚖️ Review: does fining the empty branch for moving make a better picture?

Nothing has been read. This file judges [the design](../plans/hypothesis/03-the-v0a-read.md), which scores four matched experiments that finished on 2026-09-12 and are recorded nowhere in this repository.

**The ordering this file cannot honour, stated plainly.** This project's conventions say review questions are written before the run they judge. These four runs finished four days before this file existed. The bars below were fixed before the **read**, not before the **run**, and that is a weaker guarantee. Nothing can repair it retroactively. It is written here, and on the page this plan produces, so nobody reads the verdict as a pre-registered one.

## Recommended prompt (when the run lands)

```
/analyze-run 72as6yk3
```
(The other three run ids are `tqs7qf95`, `mw1cqrpk` and `pckb2za7`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/03-the-v0a-read.md) | the render pass, the scoring pass, the blind read, the two pages |
| **this file** | **the verdict against both bars, and whether the comparison between the four is clean** |

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

- **The penalty**: `μ‖u − ū‖²` at `μ = 10`, added to the loss. Selected by `--null-anchor 10`.
- **The control**: the same objective with `--null-anchor 0`, trained in the same hour on the same pool by the same launcher. One axis differs.
- **The mechanism bar**: did the empty branch stop drifting. Free, off W&B. It only says the penalty did what it was told.
- **The outcome bar**: did the picture get better. Compose count plus a blind read by eye. This one decides.
- **Blind**: condition names replaced by ids and the order shuffled per seed before anyone looks, with the shuffle key saved separately so the blinding is checkable afterwards.
- **A sound cell**: a held-out pair and seed whose joint-prompt reference actually shows one of each animal. Two of the four are not sound and are excluded by name, not by judgement at read time.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Ablation over the penalty axis** (may not change the claim; can simplify the method or bound it). The three conditions besides the control each remove or change one thing, and the comparison between them is clean because one launcher pinned everything else.

It is **not** a hypothesis run, despite testing something, because its questions were written after its runs finished. Treating it as one would claim a guarantee it does not have.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| `r16_s0_25_plain`, the control, W&B `tqs7qf95` | ablation | 2026-09-12 12:07 UTC, mscluster46, RTX 3090 | 12.40 h | 3 checkpoints, 12 render sets | finished, unread |
| `r16_s0_25_anchor`, `--null-anchor 10`, W&B `72as6yk3` | ablation | 2026-09-12 12:07 UTC, mscluster49, RTX 3090 | 11.03 h | 3 checkpoints, 12 render sets | finished, unread |
| `r16_s0_25_plurality`, prompts rewritten, W&B `mw1cqrpk` | ablation | 2026-09-12 12:07 UTC, mscluster52, RTX 3090 | 11.62 h | 3 checkpoints, 12 render sets | finished, unread |
| `r16_s0_25_connective`, prompts rewritten, W&B `pckb2za7` | ablation | 2026-09-12 12:07 UTC, mscluster53, RTX 3090 | 13.51 h | 3 checkpoints, 12 render sets | finished, unread |
| the render and scoring pass over all twelve checkpoints | ablation | | | `00b-per-checkpoint.json`, 8 blind strips | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does the penalised experiment produce better pictures than its control on held-out seeds?**
  - **The threshold, fixed before the read:** it composes at least as many of the eight held-out cat × dog seeds as the control, **and** wins the blind by-eye read on at least 5 of 8. Below either is a null. A tie on compose count with 4 of 8 by eye is inconclusive and is written as inconclusive.
  - **This is the only question here whose failure may move a plan.** A null says a fine is not a ban, which is the tradeoff the objectives note names, and it strengthens the case for plan 04 rather than weakening the scope.
  - **Read the caveat at the top of this file before citing this verdict.** The threshold predates the read and not the run.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Is the drift under the penalty at least ten times smaller than the control's at matched steps?**
  - The mechanism bar. Already met off W&B at about 41 times on medians and 33 on means, recorded here so the read is not renegotiated later. If the scored read disagrees with the W&B read, the scoring is wrong, not the mechanism.
- [ ] ⚠️ **What did the penalty cost in fit loss?**
  - Measured at 23% higher on medians over the last 5,000 steps. The scene artifact currently says the fit error "nearly doubled", which is wrong and is corrected by this answer.
- [ ] ⚠️ **Do the two prompt-rewrite experiments differ from the control by more than the spread between them?**
  - The theory says renaming a prompt renames a free tensor, so all three should reach the same set of composed predictions. Their fit losses came out within 20% of each other, which is what that predicts. A visible difference by eye would overturn it, and that is worth more than this file's own question.
- [ ] ⚠️ **Had the fit loss flattened by 30,000 steps in all four?**
  - It fell 3 to 8 percent over the last 5,000, so yes for the fit. That does not locate the best checkpoint: in the scope-01 runs the loss flattened while renders kept getting worse, and only three checkpoints were saved here.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?**
  - Diff the four `config.json` files and say so. Only `null_anchor` differs between the control and the penalised experiment; only `branch_prompt_style` differs for the other two. Everything else is pinned by one launcher.
- [ ] ⚠️ **Was the measuring tool sound?**
  - The same scorer and the same sampler settings for all four. `XFORMERS_DISABLED=1` on the CPU DINOv2 pass. The blind read's shuffle key was not opened before the labels were recorded.
- [ ] ⚠️ **Did the run respect the environment?**
  - All four on `bigbatch` 3090s, checkpoints on `/datasets`, four separate nodes with no contention between them.
- [ ] ⚠️ **Were the broken reference cells excluded by name rather than by judgement?**
  - Seed 10 of cat × dog draws three dogs; seed 9 of elephant × penguin draws an elephant alone. `scripts/correction_loss_variants/arm_grid.py` names them. Excluding them after seeing which condition they favour would be choosing the answer.
- [ ] ⚠️ **Can this result be carried to the runs in plan 02?**
  - No. These four train on denoising steps 0 to 25 of 50, carry `ema_decay 0.999`, and use a different pool. They compare to each other only, and the write-up says so rather than leaving a reader to assume otherwise.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#what-the-run-cost-and-what-it-bought) ➡️

`00b`'s page owes the verdict in its first line, both bars in one table, the blind labels beside the strips, and the ordering caveat at the top rather than in a footnote.

The prompt-rewrite finding owes its number: the spread between the three fit losses, with its units.

The scene artifact `artifacts/drips/seeing-the-correction-loss/` owes a correction to its "nearly doubled" claim.

## What the run cost, and what it bought

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

The four trainings cost 48.6 GPU-hours across four RTX 3090s, spent before this file existed. The read costs about four hours of rendering and no training at all, which is the entire argument for doing it before plan 04.

## Still open

Navigation: ⬅️ [What the run cost](#what-the-run-cost-and-what-it-bought) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

[04: freezing the empty branch](../plans/hypothesis/04-freezing-the-empty-branch.md).
