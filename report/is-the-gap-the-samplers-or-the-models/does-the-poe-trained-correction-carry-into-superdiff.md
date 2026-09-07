# Does the PoE-trained correction carry into SuperDiff?   ⚪ null, it hurts · verified 2026-09-05

**The claim**

The adapters this project trained against the product-of-experts residual, injected into
SuperDiff on every one of its 200 steps, do not restore the second concept at any strength.
Every seed degrades from λ 0.25 and washes out from λ 0.5, on both test pairs and both ranks
rendered, where SuperDiff alone reaches two animals by λ 0.75 once its own residual is added
back. The correction is the right size and points the wrong way: on one cell its per-step
cosine against SuperDiff's missing residual never rises above 0.66 and sits near 0.1 for most
of the run. The PoE residual and the SuperDiff residual are different vectors at the same
state.

**What would have counted**

Pre-registered in [the plan](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md)
before the set ran: per pair and rank, the first λ at which each seed shows two separate
concepts, beside the same number from the no-adapter sheet. Transfers if the adapter's λ is
lower on more seeds than not, indistinguishable if equal, hurts if higher. Judged in
[the review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/07-does-the-poe-trained-correction-reach-superdiff.md).

## 1. No seed separates at any strength

![The rank-8 PoE adapter inside SuperDiff on cat × dog, rows seeds 9 to 12, columns λ](../../artifacts/results/is-the-gap-the-samplers-or-the-models/poe-adapter-rank8-in-superdiff-cat-dog-lambda-sheet.png)
*Read beside the no-adapter sheet in the previous finding, which has the same seeds, κ and
noise. What to notice: every tile right of λ 0 is softer than the one left of it, and no tile
in any row reaches a cat beside a dog.*

![The same adapter on butterfly × meadow](../../artifacts/results/is-the-gap-the-samplers-or-the-models/poe-adapter-rank8-in-superdiff-butterfly-meadow-lambda-sheet.png)
*What to notice: the butterfly the no-adapter sheet pulls forward by λ 0.25 never appears; the
meadow itself loses contrast instead.*

**The number.** First separating λ, eye read: never, on 12 of 12 seeds rendered (rank 8 on both
pairs, rank 16 on cat × dog), against 0.5 to 0.75 for the no-adapter cat × dog sheet and 0.25
for butterfly × meadow. The set was stopped at 49 of 96 cells with the verdict fixed; rank 16
on butterfly × meadow and rank 32 were not rendered. Checkpoints were the latest at launch, not
the finals: rank 8 at 410k of 450k, rank 16 at 85k of 100k. From the three sheets and their
sidecars under `paper/iclr/figures/how-much-is-added/across-composition-rules/`, renders under
`corrector/superdiff/transfer/grid/`.

## 2. Right size, wrong direction

![Per-step cosine between the adapter's correction and SuperDiff's missing residual, and both norms](../../artifacts/results/is-the-gap-the-samplers-or-the-models/poe-adapter-in-superdiff-cosine-profile.png)
*Left: y is the cosine between the adapter's correction Δ̂ and `r_t^SD = ε_J − ε_M` at the same
state, x is the SuperDiff step from noise to image, orange bars are the median per step bucket.
Right: the two norms on a log axis. What to notice: the cosine hugs zero outside the middle of
the run, and the norms are the same order of size throughout.*

**The number.** Median cosine by step bucket, cat × dog seed 9, rank 8 at `lora_step_420000.pt`,
λ 1, κ 0.5: 0.10 over steps 0 to 19, 0.34 over 20 to 99, 0.18 over 100 to 179, 0.06 over 180
to 199; per-step minimum −0.40, maximum 0.66. Median ‖Δ̂‖ against median ‖r_t^SD‖ by the same
buckets: 10.7 vs 8.6, 20.2 vs 15.6, 27.1 vs 17.1, 47.2 vs 15.3, in L2 units of the ε
prediction. Read this session from the sidecar
`corrector/superdiff/transfer_diag/cosine/pairs/a_cat__x__a_dog/seed_9/superdiff_200steps_kappa0.50_with_rt_lorar8_lv1.00/*.json`,
fields `delta_hat_cos_r_t`, `delta_hat_norms`, `r_t_sd_norms`, and written to
`superdiff-report-figures.json` beside the figure.

## 3. The off pass is the baseline

**The number.** With the adapter attached and λ 0, a 20-step render of cat × dog seed 9 has the
same sha256 as the no-adapter render (`79238174f0b3e244`), and at λ 1 the correction is non-zero
on all 20 steps, 10.0 to 78.3. Attaching matched 210 modules for each of rank 8, 16 and 32.
From the review file's run table (attach check and inertness check on mscluster109 device 1).
This rung has no picture; it is the check that rung 1 measures the adapter and not a wiring bug.

## What this cannot tell you

Whether the finals would have changed the read: the sheets used mid-run checkpoints (rank 8 at
410k of 450k, rank 16 at 85k of 100k), by the user's call not to wait. The mechanism in rung 2
is one cell, one seed, one rank; it explains the sheet it sits beside and is not a population
statistic. The eye read has no detector behind it, because the degraded tiles have nothing an
instance count could mean. "Wrong direction" says the two residuals differ at the same state;
it does not say what the SuperDiff residual would need to be learned from, which is the next
finding's question.

## Where this came from

| What | Source | Mark |
|---|---|---|
| The cosine and norm profiles | the `transfer_diag` sidecar named above, read 2026-09-05 by `scripts/superdiff/report_figures.py` | verified |
| The first-separating λ on 12 seeds | three sheets under `paper/iclr/figures/how-much-is-added/across-composition-rules/`, eye read 2026-09-03 and recorded in the review file | stated |
| The sha and attach counts | review file run table, 2026-09-03 | inferred |
| The runs | attach and inertness checks in session on mscluster109 device 1; the transfer set by `nohup` on mscluster109 device 1, 49 cells at about 121 s each before the stop; the cosine cell on mscluster109 device 0; all 2026-09-03 | verified |
| Regenerate with | [runbook § inject a PoE adapter into SuperDiff](../../runbook/running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#7-inject-a-poe-trained-adapter-into-superdiff-and-measure-its-direction) | |
| The plan | [does the PoE-trained correction reach SuperDiff](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md), root step 49 | |

## Depends on

- The injection grammar `ε(off) + λ · (ε(on) − ε(off))` on every step, shared with the PoE-side dose sweep: [context index](../../context/00-INDEX.md)
- Which PoE adapters these are (pooled rank 8, 16, 32 under `outputs/showcase/phase1_r*`): [the pooled-adapter finding](../does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md)

## Still open

- [ ] The sha and attach counts are lifted from the review; re-running the inertness check makes them `verified`.
- [ ] The 47 unrendered cells (rank 16 butterfly × meadow, rank 32 both pairs) if a reviewer asks for the full set.
