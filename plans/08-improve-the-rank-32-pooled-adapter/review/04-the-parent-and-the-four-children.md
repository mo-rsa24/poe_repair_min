# 🧪 Review: did the five changes give a clean cat and dog, and which one did it?

Nothing has run. This file judges [the design](../plans/hypothesis/04-the-parent-and-the-four-children.md): one training run carrying five changes at once, four more each undoing one of them, all five rendered under both samplers and labelled blind. It holds two kinds of question, and they answer to different rules. The first asks whether the package works at all and may move the scope. The four that follow are an ablation and may only simplify the method or bound what it claims.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(For a run that died and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/04-the-parent-and-the-four-children.md) | the five configurations, the chain, the rule in source |
| **this file** | **the verdict: whether the parent beat the baseline, and which switch carried it** |
| [procedure: launching](../procedures/hypothesis-04-launch-the-chain.md) | claiming the device, the smoke, the chain, the harvest |
| [procedure: labelling](../procedures/tools-02-run-the-blind-label-pass.md) | taking the labels |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [What the run cost, and what it bought](#what-the-run-cost-and-what-it-bought)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The parent, P**: rank 32, 30,000 steps, with all five changes on. Fourteen pairs rather than eleven; training cells whose joint target shows the wrong animals dropped; training steps sampled from 0 to 24 only; the part of the error outside the two experts' span charged three times; weight decay 1e-2 with an averaged copy of the weights at decay 0.999.
- **The four children**: **C1** keeps every cell. **C2** samples all 50 steps. **C3** uses the plain error. **C4** uses the eleven original pairs. Each undoes exactly one of the parent's changes. None undoes the optimizer setting, which all five share.
- **The baseline, B**: checkpoint 30050, no training, under whichever sampler [plan 03's verdict](03-what-the-stacked-sampler-alone-does.md) named.
- **Better, under the same sampler**: more seeds labelled clean, and none the other side had clean falling out of that label. The thresholds are `MIN_CLEAN_GAIN = 1` and `MAX_CLEAN_LOST = 0`, constants in the scoring script.
- **A switch mattered**: the parent is better than the child that undoes it, by that same rule. A switch that did not matter is a finding, because it simplifies the method.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim, then ablates it.** The parent against the baseline is a claim-testing run and is the only question here that may move the scope: failing its bar closes the data-side and loss-side route as a package. The four children are ablation runs. They may not move the claim; they simplify the method by naming a change that earns nothing, or bound it by naming the one it rests on.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| `v54_smoke`, 200 steps of the filtered pool on the early steps | proves the process | 2026-09-08 17:49, mscluster108 gpu 0 over SSH, `co3` | 7 min | `improve_r32/v54_smoke/`, W&B `s1x1xior` | done, exit 0: 54 cells, 1350 steps, a checkpoint on `/datasets`, the late bucket flat at zero as a step range of [0, 25) predicts |
| `v54_P`: the 54 filtered cells, steps [0, 25), plain loss | tests the claim, minus the orthogonal weight | 2026-09-08 19:44, mscluster109 gpu 1 over SSH, `co3` | about 6.3 h at 0.76 s/step, no gradient checkpointing | `improve_r32/v54_P/`, W&B run in `prime_lab/poe-repair-animals-compose` | running. Carries four of the parent's five changes; the orthogonal weight had not been built when it launched, so this is the child that undoes it |
| `v54_C1_allseeds`: the same 24 pairs, every seed on disk, 182 cells, steps [0, 25) | ablation: undoes the filtering | 2026-09-08 19:44, mscluster108 gpu 0, `co3` | about 10.8 h | `improve_r32/v54_C1_allseeds/` | running |
| `v54_C2_allsteps`: the 54 filtered cells, all fifty steps | ablation: undoes the step range | 2026-09-08 19:44, mscluster108 gpu 1, `co3` | about 10.5 h | `improve_r32/v54_C2_allsteps/` | running; its late bucket reads 0.047 at epoch 1 where the two early-step runs read 0.000, which is the range doing its job |
| `v54_smoke_orth`, 100 steps with `--orth-weight 3.0` | proves the process | 2026-09-08 19:53, mscluster106 gpu 1, `co3` | 4 min | `improve_r32/v54_smoke_orth/` | done, exit 0. Loss early/commit 0.0104/0.0527 against the plain-loss smoke's 0.0060/0.0336 on the same cells, which is the weight charging the out-of-plane part harder; late bucket zero, so the step range held |
| `v54_P_orth3`: the 54 filtered cells, steps [0, 25), `--orth-weight 3.0` | **tests the claim: this is the parent** | 2026-09-08 19:58, mscluster106 gpu 1 over SSH, `co3` | about 12.4 h by its own first-epoch estimate: an RTX 8000, slower than the A6000 the plain-loss run sits on, plus the projection's cost per step | `improve_r32/v54_P_orth3/`, W&B `1z3dey7n` | running. All five changes on: eye-filtered pool, wider pair set, early steps only, orthogonal weight 3, weight decay 1e-2 with EMA 0.999 |
| all four of the above | | | | | **aborted at step 6409 of 30000 by the trainer's own kill rule**, each writing `verdict: aborted`. The rule snapshots the commit-bucket loss at step 200 and, from step 5000, kills on any single step whose running commit loss exceeds half of it. It is checked every step rather than every epoch. Over steps 5000 to 6409 the loss was below the threshold on 1409 steps and above it on 1, and that one killed the run. `v54_P` had fallen from 0.0149 to a low of 0.0015, a tenfold drop, before dying on a spike to 0.0097. All three early-step runs died at exactly step 6409, because the pair-and-cell sampler is seeded identically, so each met the same hard cell at the same step |
| `v54b_P_orth3`, `v54b_P`, `v54b_C2_allsteps`, `v54b_C1_allseeds` | as their `v54_` namesakes | 2026-09-09 02:16, on mscluster108 gpu 0 and 1, mscluster109 gpu 1, mscluster106 gpu 1 | | `improve_r32/v54b_*/` | running: the same four arms relaunched with `--kill-halve-after-steps 1000000000`, which disables that rule. Nothing else changed |

**The pool is not the one the design named.** It is 54 cells across 24 pairs, every cell judged good by eye, from `artifacts/_shared/cross_pair_pool_configs/cells_v54.json`, rather than 14 pairs minus an exclusion list. The five changes hold as designed, but the runs are named for what each one varies rather than P, C1 to C4, because the arms were launched as their switches became available rather than as one chain. The orthogonal weight landed last, so the run first called the parent is the plain-loss child, and the true parent is the run that adds `--orth-weight 3.0` on top of it.

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does the parent give more clean cat × dog seeds than the baseline under the same sampler, and lose none the baseline had clean?**
      This is the bar, and it is the only question here that may move the scope. **Support**: at least one more clean seed and none lost, so a better checkpoint exists and the four children are read to find out which change made it. **Null**: no gain, or a gain paid for with a loss. A null closes the data-side and loss-side route as a package, which is a real result: it says the remaining options are a better teacher than the joint prompt or an inference-time fix, both more expensive, and it says so before either is attempted. The children are still read after a null, because a child beating the parent would say one of the five changes actively hurt.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

**One question per removed component.** Each asks whether the parent beats that child by the rule above. A parent that does not beat a child means the change that child undoes earned nothing, which simplifies the method rather than embarrassing it.

- [ ] ⚠️ **C1, the dropped cells: did excluding training cells whose joint target shows the wrong animals earn its place, and can that be told apart from simply having four times the data?**
      C1 keeps every cell and is otherwise the parent. If the parent beats C1, filtering the targets helped. If C1 matches or beats the parent, the filter cost more good training data than it saved, and the filter comes out of the method.
      **This child's axis is not clean either, and by a wide margin.** The exclusion was computed before the chain was built, over the fourteen pool pairs on training seeds 1 to 8: it drops 80 of 108 cells. The parent trains on 28 cells and C1 trains on 108, so P against C1 is a comparison between a quarter of the data and all of it, not between clean targets and dirty ones. Four pairs lose every cell (rabbit and hare, horse and zebra, donkey and pony, cheetah and cougar), so the parent also trains on ten pairs where C1 trains on fourteen.
      Write this child's answer as a gap between two runs that differ in data volume and target quality together. C1 beating the parent says the filter is not worth its cost at this severity, which is a usable finding. The parent beating C1 is the stronger result, because it would mean 28 clean cells beat 108 mixed ones.
      The severity is a property of the pool rather than of the filter: loosening the rule to drop only a clear same-animal-twice recovers 6 cells and no pairs. On look-alike pairs the joint prompt fails to draw two distinct animals about three quarters of the time.

- [ ] ⚠️ **C2, the late steps: did restricting training to steps 0 to 24 earn its place?**
      C2 trains on all 50. If the parent beats C2, the softness really is learned from the late steps and the mask is load-bearing. **C2 beating the parent is the result that would surprise**: it would say training on the late steps helps even though running the adapter there hurts, and the late-step question would become the finding rather than a settled assumption.

- [ ] ⚠️ **C3, the orthogonal charge: did weighting the part the two experts cannot supply earn its place?**
      C3 uses the plain error. If the parent beats C3, spending the adapter's capacity on the new direction is what improved the render, which is the most method-shaped of the five changes and the one most worth a sentence in the paper. If not, the weighting is a complication with nothing behind it.

- [ ] ⚠️ **C4, the three extra pairs: did widening the pool earn its place, and can that even be told apart from the kind of pair added?**
      C4 trains on the eleven original pairs. If the parent beats C4, more pairs helped, and since none of the three added pairs contains a cat or a dog, that is transfer rather than memorisation.
      **This child's axis is not clean, and the answer has to be written knowing that.** All eleven original pairs are look-alike species pairs: wolf and husky, turtle and tortoise, rabbit and hare, lion and tiger, horse and zebra, gorilla and chimpanzee, donkey and pony, dolphin and porpoise, crow and raven, crocodile and alligator, cheetah and cougar. The three added pairs are lion and horse, wolf and horse, bear and salmon, which are pairs of plainly different animals. So P against C4 moves two things at once: how many pairs the run trains on, and whether the pool contains any pair whose two animals are easy to tell apart. A gap here may be either, and no result from this child alone separates them. Report the gap and state this boundary rather than attributing it to pool size.
      What would separate them is a fifth child training on the eleven originals plus three further look-alike pairs, holding the kind fixed while the count moves. It is not in this chain and is named in [Still open](#still-open).

- [ ] ⚠️ **Does elephant × penguin hold under every run?**
      Report either way. A run that gains cat × dog seeds and loses elephant × penguin ones has traded pairs rather than improved, and the totals on one pair would hide it.

- [ ] ⚠️ **What is the correction's size per pair, and does it order anything?**
      Measured and reported beside the labels. No claim attached until clean, unclear and not-two examples have been laid against the numbers, which happens in [plan 05](../plans/figures/05-the-comparison-sheet.md), not here.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the results raise. Nothing here may become the question above, because it would be written with the answer already visible.

- [ ] ⚠️ **Why does the orthogonal share of the error rise during the first hundred steps of the parent, when the weight on it is three?** (raised 2026-09-08 20:05 from W&B `1z3dey7n`, steps 4 to 126)
      `train/orth_frac`, the fraction of each batch's error lying outside the two experts' own plane, reads 0.15 at step 4 and 0.6 to 0.99 by steps 100 to 126, with the weight confirmed at 3 on every row. A weight of three should make the optimiser attack that part harder, so a rising share is the opposite of the naive expectation. The likely reading is that the in-plane part is the cheap part, a re-weighting of two experts the adapter fits within a few dozen steps, after which nearly all remaining error is the part no re-weighting can supply. That is consistent with the earlier finding that the adapter gets the in-span part and only half the new direction. It is 126 steps of 30,000 and a diagnostic, not a verdict: check where the share settles at the end of the run, and compare it against the plain-loss child's, which logs the same key at weight 1 and so shows what the share does when nothing pushes on it.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Each child differs from the parent by exactly one line of configuration, checked as a diff, and the printed composition per run confirms only the intended thing changed. **C4 is the known exception and is not fair in the strict sense**: its one line changes the pool from fourteen pairs to eleven, and because the three removed pairs are the only ones whose two animals are easy to tell apart, the count and the kind move together. That is recorded as a boundary on C4's answer above, not fixed by the diff being one line long. This is the check that matters most here: turning one knob can move another through the data, and a child whose cell count differs from the parent's for any reason other than the switch it undoes has already mixed its comparison. The parent against the baseline additionally differs in the sampler unless both are read under the same one, which is why the sampler is held fixed within every comparison.
- [ ] ⚠️ **Was the measuring tool sound?** The labels came through the blind pass whose known-example smoke passed, over each run's own tiles, with the mapping untouched during labelling and each run labelled in its own sitting.
- [ ] ⚠️ **Did the run respect the environment?** Every flag selected a non-empty group, the checkpoints landed on `/datasets` rather than `/home-mscluster`, the runs used the Blackwell's own python rather than the one that silently produces no output there, and the harvest was read on the node rather than through the session node's lagging view.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#what-the-run-cost-and-what-it-bought) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| any claim that the adapter improved | which sampler, which baseline column, and that the judgement is one person's blind read rather than a metric |
| any claim naming a change as the cause | that it comes from a single leave-one-out child, not a sweep, so it says the change mattered in this configuration and not that it matters alone |
| any claim of transfer from the three added pairs | that none of them contains a cat or a dog, and the clean-seed counts on both judged pairs |
| a null on the parent | that the five changes were tested as a package, so a null does not rule out any one of them applied differently |

## What the run cost, and what it bought

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Filled in after the chain lands: failed launches and what fixed them, whether the estimated two hours per run at 0.24 s a step held on the day, and any constraint discovered by hitting it. Where a lesson generalises, it is written into the environment folder or the failure catalog and linked from here rather than restated.

## Still open

Navigation: ⬅️ [What the run cost, and what it bought](#what-the-run-cost-and-what-it-bought) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether any change helps on its own rather than only in the package | four add-one-in runs, another twelve hours | nothing yet; it becomes the next scope only if the parent wins and no child is worse than it |
| whether a better teacher than the joint prompt would lift the 26% of targets that currently survive the filter, since the shortage of clean targets is now the pool's binding constraint rather than a nuisance | a cache built under an attention-guided joint sampler, plus a check that it composes on look-alike pairs where the plain prompt does not | the parent's data volume, and therefore the interpretation of every run in this chain |
| whether a gap between the parent and C4 is the number of pairs or the kind of pair, since the three added pairs are the only ones in the whole pool whose two animals are easy to tell apart | a fifth child trained on the eleven originals plus three further look-alike pairs, holding the kind fixed while the count moves; about two hours | C4's answer, which can otherwise only be reported as a gap with its cause unresolved |
| whether a better teacher than the joint prompt would beat all of this | a new cache under an attention-guided joint sampler, plus a validation that it composes where the plain one does not | out of this scope by design; it is the follow-on if C1 shows the wrong targets matter |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's task 1.1](../plans/hypothesis/04-the-parent-and-the-four-children.md#1--fix-the-verdict-rule-in-source-before-launch), which puts the better-than rule in source before the chain is launched.
