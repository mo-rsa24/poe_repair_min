# The figure register: every figure the paper will carry

One row per figure, including figures whose experiment has not run yet. A figure not in this
register is incidental by definition, however good it looks: this file is what separates the
handful of load-bearing images from the 1600 figure files in the repo.

**A reserved slot is a figure planned before its result exists.** It states the claim the figure
will make, the layout that would show it, and the review question that must be answered before
the caption may make that claim. `plan_pulse` watches the slots: when a slot's review question is
answered, it reports the slot as fillable at session start; if the question is answered against
the slot's claim, it reports the slot as broken, which is a contradiction arriving through the
figure door and routes to the diagnosis procedure in `docs/RESEARCH_GUIDELINES.md`.

Status: `reserved` (waiting on its run), `fillable` (question answered, figure not built),
`built` (file exists and is referenced from the tex), `broken` (the result does not fit the
claim; do not build, diagnose).

| # | Status | The claim the caption will make | Layout | Answered by | File |
|---|---|---|---|---|---|
| F1 | reserved | PoE composition fails in a specific way: one blended animal, not two | three-panel density diagram over real λ=0 cells | no run: drawn from settled λ=0 cells. Panels owed by [figure-01](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/plans/figure-01-the-seven-paper-figures.md) | |
| F2 | fillable | the correction is what is missing: more of it composes more, and two size-matched fakes stay flat | image strip above the three curves, shared λ axis, controls gray | [../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-02-more-correction-more-composition.md](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-02-more-correction-more-composition.md), bar answered ✅. Re-scored on the pinned seeds (commit dcca290): 32 cells per row-and-strength, 6% to 94%, AUC 0.422 against 0.059 and 0.070. **The percentages are still pre-cutoff-choice, so the caption may state the direction but not yet the numbers.** | `dose_curves.png` is stale: the re-score ran with --no-figure |
| F3 | reserved | the correction's size follows noise level, one shared curve across pairs | normalized ‖r_t‖ vs log-SNR, thin line per pair, one bold mean | [hypothesis-04 review](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-04-what-the-cached-runs-already-show.md), answered 🟡: the collapse holds only to 19.7% spread, so the caption claims it that narrowly | |
| F4 | reserved | the correction matters in a specific window of the denoising run | timing curves with the fork elbow as a vertical band | [hypothesis-03 review](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md), bar written, run not started | |
| F5 | reserved | one dial moves three spaces together | manifold walk, caption crossover, density climb, shared λ colorbar | [hypothesis-04 review](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-04-what-the-cached-runs-already-show.md), judged per question rather than by one bar | |
| F6 | reserved | the correction is low-rank enough to learn | spectrum with the random floor shaded, held-out projection inset | [hypothesis-04 review](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-04-what-the-cached-runs-already-show.md), judged per question rather than by one bar | |
| F7 | fillable | the adapter changes what a word paints more than where it looks, on pairs it never trained on | mechanism panel beside the transfer table | [../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-01-what-the-fix-changes-inside-the-model.md](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-01-what-the-fix-changes-inside-the-model.md), bar answered ✅ (median 1.52 over 64 cells). **Caption capped by the caveat: the effect also shows on the control pair, so the sentence is what the adapter does to any pair it touches, not why the fix works.** | |
| F8 | reserved | transfer holds on held-out pairs and degrades gracefully as fewer pairs train | leaderboard plus degradation curve | [hypothesis-02 review](../../plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/review/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md) plus [baseline-01 review](../../plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/review/baseline-01-the-size-matched-control-pool.md), both bars written, runs not started | |

Owed by [../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/plans/figure-01-the-seven-paper-figures.md](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/plans/figure-01-the-seven-paper-figures.md)
(F1 to F7) and [../../plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/plans/figure-01-the-transfer-figures.md](../../plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/plans/figure-01-the-transfer-figures.md)
(F8). The manuscript currently includes no figures; the `\includegraphics` lines in the tex are
the ICLR template's own examples.

**What a caption may never claim:** anything its review question does not say. A bounded failure
is written as the boundary it found ("holds for X, not for Y"), not softened and not omitted.
