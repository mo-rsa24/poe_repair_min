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
| F1 | reserved | PoE composition fails in a specific way: one blended animal, not two | three-panel density diagram over real λ=0 cells | settled results; illustrative panels have ChatGPT prompts in plan 10 | |
| F2 | fillable | the correction is what is missing: more of it composes more, and two size-matched fakes stay flat | image strip above the three curves, shared λ axis, controls gray | [../../plans/closing-the-compositional-gap/plans/interaction-term/review/03-dose-response.md](../../plans/closing-the-compositional-gap/plans/interaction-term/review/03-dose-response.md), bar answered ✅, re-score owed before the exact percentages are quoted | `dose_curves.png` exists, pre-re-score |
| F3 | reserved | the correction's size follows noise level, one shared curve across pairs | normalized ‖r_t‖ vs log-SNR, thin line per pair, one bold mean | interaction-term plan 05, cache analyses | |
| F4 | reserved | the correction matters in a specific window of the denoising run | timing curves with the fork elbow as a vertical band | interaction-term plan 04 review, not yet written | |
| F5 | reserved | one dial moves three spaces together | manifold walk, caption crossover, density climb, shared λ colorbar | interaction-term plan 05 review | |
| F6 | reserved | the correction is low-rank enough to learn | spectrum with the random floor shaded, held-out projection inset | interaction-term plan 05 review | |
| F7 | reserved | the learned version works where it was never trained | mechanism panel beside the transfer table | interaction-term plan 02 review (mechanism, answered: 1.52x over 64 cells) plus animals-compose-transfer plan 03a read | |
| F8 | reserved | transfer holds on held-out pairs and degrades gracefully as fewer pairs train | leaderboard plus degradation curve | animals-compose-transfer plans 03/04 reviews, not yet written | |

Owed by [../../plans/closing-the-compositional-gap/plans/interaction-term/plans/10-figures.md](../../plans/closing-the-compositional-gap/plans/interaction-term/plans/10-figures.md)
(F1 to F7) and [../../plans/closing-the-compositional-gap/plans/animals-compose-transfer/plans/05-figures.md](../../plans/closing-the-compositional-gap/plans/animals-compose-transfer/plans/05-figures.md)
(F8). The manuscript currently includes no figures; the `\includegraphics` lines in the tex are
the ICLR template's own examples.

**What a caption may never claim:** anything its review question does not say. A bounded failure
is written as the boundary it found ("holds for X, not for Y"), not softened and not omitted.
