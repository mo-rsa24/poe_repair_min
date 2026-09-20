# 🧪 Review: does handing the tail back to the frozen model make the render crisp?

Nothing has run. This file judges [the design](../plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md), which sweeps the step after which the adapter is detached, on checkpoints that already exist, and trains a new objective only if the sweep falls short.

Its questions and its bars are written before any render.

## Recommended prompt (when the sweep lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md) | the shrinkage check, the hand-off sweep, the training run behind it |
| **this file** | **the verdict: whether any `k` clears the fidelity bar on five or more seeds with composition held** |

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

- **The correction, `r`**: `eps_J - eps_PoE`, what the adapter is trying to supply.
- **What the adapter supplies, `r_hat`**: `eps_poe_lora - eps_poe_frozen`, formed at `trainer.py:588`.
- **How big the push is**: `‖r_hat‖ / ‖r‖`, where 1.0 means the adapter applied exactly what the picture needed.
- **Whether the push points the right way**: the cosine between `r_hat` and `r`, where 1.0 is the right direction and 0 is unrelated.
- **The hand-off step `k`**: the denoising step after which the adapter is detached. `k = 50` is the adapter on the whole path, and it is the control.
- **Fidelity distance**: cosine distance between a render and that seed's own joint-prompt render in the compose scorer's DINOv2 embedding. Lower is nearer the picture the adapter is trying to reach.

## Run kind

Navigation: ⬅️ [Previous](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

Stage A is a render sweep over existing checkpoints and trains nothing. Stage B is one training run and happens only on a ⚪ verdict at stage A.

## Runs

Navigation: ⬅️ [Previous](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | What it is for | When, where, W&B | Outcome |
|---|---|---|---|
| Task 1.1, `scripts/shrinkage_per_step.py`, both checkpoints, eight held-out seeds, all 50 steps | tests the premise the plan was built on | 2026-09-20 07:11 to 07:18, `mscluster106` device 1, Quadro RTX 8000, no W&B | 800 rows into `shrinkage-per-step.json`. The premise as first written is falsified and a different failure is measured in its place |

## The question written before the run

Navigation: ⬅️ [Previous](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**Does detaching the adapter for the last stretch of denoising bring a render nearer its own joint-prompt target without losing the second animal?**

**Support**: some `k` below 50 clears a fidelity gain of at least `0.05` against the same checkpoint at `k = 50`, on five or more of the eight held-out `a_cat__x__a_dog` seeds, with the instance count within one seed of `k = 50`.

**Null**: no `k` clears it, or one clears it on fewer than five seeds.

**Composition broken**: two or more composed seeds lost at the `k` that wins on fidelity. A fidelity gain bought by deleting an animal is not a result.

**Inconclusive**: `an_elephant__x__a_penguin` seeds 09 and 10 disagree in the same direction as they disagree on style, which would mean the measure is reading the seed's rendering and not the hand-off.

The `0.05` is `CLEAN_MIN_MONO_GAIN`, the constant already used for this question in [can a corrector or a clean tail sharpen the adapter's renders](../../../report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md). It is inherited rather than chosen after seeing these pictures, which is the point of naming it here.

## Written before the run, answered after

Navigation: ⬅️ [Previous](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] **Does the push get smaller in the late steps?** Asked because the plan's first premise said the haze was the adapter applying too little near the end. **Answer: no.** Mean size per band on `pool43-all50` is 1.08, 0.87, 0.88, 0.95, 0.94 across steps 0-9, 10-19, 20-29, 30-39 and 40-49. It is close to full size throughout and slightly larger at the end than in the middle. `v1` behaves the same way. That premise is falsified.
- [x] **Does the push point the right way in the late steps?** Not asked before the run; the measurement recorded it and it is what replaced the premise above. **Answer: no.** Direction on `pool43-all50` is 0.77, 0.85, 0.91, 0.88, 0.78 across the same bands, and on `v1` it ends at 0.63. Best in the middle, worst at the two ends, and the late end is where fine detail is drawn. Stage A tests whether removing that off-target push there makes the pictures crisper.
- [ ] **Which `k` wins, and on how many seeds?** Answer:
- [ ] **Did composition hold at that `k`?** Answer:
- [ ] **Did your eye and the fidelity number pick the same `k`?** Recorded from instruction 5.1 before the numbers were read. Answer:
- [ ] **If stage B ran, did it beat the best `k` found on the existing checkpoints?** Answer:

## Asked after the result

Navigation: ⬅️ [Previous](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Left empty on purpose. Questions raised by the result go here, marked as such, and never above.

## Could the answer be an artefact

Navigation: ⬅️ [Previous](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- **The seed carries the style, not the run.** `an_elephant__x__a_penguin` seed 09 is an engraved plate at every checkpoint of `pool43-all50` and seed 10 is photographic at every one. If a `k` wins only on the photographic seeds, the measure is reading rendering style. This is why every read is per seed.
- **Two of the three checkpoints are rank 32 and one is rank 16.** They are not compared against each other, only each against its own `k = 50`, so the rank difference does not enter the verdict. Stating it because a reader will notice the mixture.
- **The `k = 50` control and the earlier finding's control are not the same picture.** That finding cut at step 29 of 50 and measured against the adapter alone; the numbers here are not comparable to its 0.038 and should not be quoted beside it.
- **Laplacian variance will disagree with the eye.** It counts edges and prefers drawn fur. It is reported and it is not a bar.

## What the write-up owes

Navigation: ⬅️ [Previous](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

`report/designing-the-correction-loss/08-fitting-the-whole-path-and-handing-back-the-tail.md`, carrying the verdict against the bar above, the per-seed table, F1 and F2 from the plan's Figure Catalog, and the sentence saying whether stage B was needed.

## Still open

Navigation: ⬅️ [Previous](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [ ] ⚠️ **Why did three rank-32 runs die within five minutes of each other on 2026-09-17?** Nothing has read those logs. It blocks any rank-32 confirmation run, here and elsewhere in the scope.
- [ ] ⚠️ **What measure separates a well-formed pair of animals from two animal-shaped things attached to one body?** The instance count reads 1.000 through `a_cat__x__a_dog` seed 09's collapse at step 60,000. Not this plan's question, and it limits what any verdict here can claim.

## Next step

Navigation: ⬅️ [Previous](#still-open) | 📋 [TOC](#table-of-contents)

If stage A clears the bar, every other variation in this scope should be rendered the same way, and the report page is what they read.
