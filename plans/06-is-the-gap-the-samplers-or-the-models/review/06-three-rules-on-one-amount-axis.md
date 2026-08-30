# 📊 Review: what do four composition rules do when they travel the same amount axis?

**Nothing has run yet.** Every question below was written before any grid rendered. This file
judges [the amount-axis design](../plans/baselines/06-three-rules-on-one-amount-axis.md). Its cheapest
real finding is the `λ=1` column, which classifies for free which rules genuinely combine scores
and which do something else.

## Recommended prompt (when the run lands)

```
/analyze-run the two grids across composition rules, lambda 0 to 1 at seed 9
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/06-three-rules-on-one-amount-axis.md) | the generalised injection, the two grids, and the two traps handled in advance |
| **this file** | **the verdict: not yet run** |
| [the SuperDiff verdict](05-superdiff-at-this-repos-fifty-steps.md) | whether the SuperDiff row is a fair one |
| [the corrector's verdict](02-the-corrector-and-the-step-size-it-runs-at.md) | the `k` and `c` the two corrector rows run at |
| [the step 26 verdict](03-what-is-left-once-the-chain-settles.md) | whether these grids illustrate a diagnosis or are a baselines table |

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

- **The generalised amount axis**: for any rule `M` with a per-step prediction `eps_M`, the
  correction relative to that rule is `r_t^M = eps_J - eps_M`, and the injection is
  `eps_M + λ·r_t^M`. At `λ=0` the rule runs alone; at `λ=1` the prediction is `eps_J` exactly.
- **The four rows**: plain product-of-experts with `r_t`, SuperDiff with `r_t^SD`, the corrector at
  `k=1`, and the corrector at `k=5`.
- **Not-an-identity**: a label on a render where the rule cannot reach the joint render at `λ=1`
  even though the arithmetic says the prediction is `eps_J`. Expected for the corrector rows,
  because the chain has already moved the latent off the joint trajectory.
- **Matching the amounts**: running so every row delivers the same absolute amount rather than the
  same fraction of its own correction. Implemented for this project once already, at
  [interaction_term_dose_matched.py](../../../scripts/interaction_term_dose_matched.py).

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Establishes a baseline.** Missing the threshold means the comparison has no fair starting point,
because the axis stops meaning the same thing across rows. Per this project's run conventions this
may not change a claim, and it freezes on landing. A striking row earns the right to propose an
experiment and nothing more.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Grid one: 4 rules × seeds 9 to 12, at `λ=0.75` and `λ=0` | Establishes a baseline | not launched | 32 renders | `corrector/dose_across_rules/grid_one/`, figure `across-composition-rules/rules-at-one-dose.png` | ⚠️ not run |
| Grid two: 4 rules × `λ ∈ {0, 0.25, 0.5, 0.75, 1}` at seed 9 | Establishes a baseline | not launched | 20 renders | `corrector/dose_across_rules/grid_two/`, figure `across-composition-rules/rules-as-the-dose-rises.png` | ⚠️ not run |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Is each row byte-identical to its own rule run alone at `λ=0`?** This is the threshold
      because it is the condition under which a `λ` column means the same thing across rows. A row
      that leaks at `λ=0` makes the whole axis meaningless, and no comparison drawn from either
      grid would be worth anything. Checked the same way the
      corrector's own leak check is checked, at
      [step 25](02-the-corrector-and-the-step-size-it-runs-at.md).

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ At `λ=1`, which of the four rules reproduce the joint render exactly? The two
      product-of-experts-family rows should. The corrector rows should not, since the chain has
      already left the joint trajectory. A row that fails to converge there is doing something
      beyond combining scores, which makes that column a free classifier.
- [ ] ⚠️ How were the different absolute amounts per row handled, by putting the delivered total on
      each row's label or by matching the amounts? The choice was made and written down **before**
      any grid rendered, because it changes what the pictures are. Which one, and why?
- [ ] ⚠️ How many renders compose per row in grid one, by the detector and by eye, with the
      disagreements named render by render? A summary count hides exactly the renders a reviewer
      asks about.
- [ ] ⚠️ Do the two corrector rows differ from each other? `k=1` and `k=5` are two different
      trajectories, so a difference between them is about how far the chain settled and not about
      how much correction was added.
- [ ] ⚠️ Did both figures land in `paper/iclr/figures/how-much-is-added/across-composition-rules/`
      rather than in the timing folder? These figures are about how much is added, and the timing
      folder's name is a question about timing.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible. Empty until the grids render.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Two things could break it and both are known in advance.
      Rows deliver different absolute amounts at the same `λ` because `‖r_t^M‖` differs per row,
      which is answered by the label or by matching the amounts. And the SuperDiff row is only fair
      if [its parity check](05-superdiff-at-this-repos-fifty-steps.md) passed; if it did
      not, that sentence is in the caption.
- [ ] ⚠️ **Was the measuring tool sound?** The `λ=0` identity check per row, and the validated
      instance-count detector read over these renders and no others, with the eye count beside it.
- [ ] ⚠️ **Did the run respect the environment?** All 52 renders present across the two grids, saved
      under `/datasets` with only the finished figures and sidecars in the repo, norms upcast to
      fp32 from fp16, launched under `nohup` outside Slurm and harvested by `pgrep`.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| any column read across rows | whether the rows delivered matched absolute amounts or matched fractions, since `λ` is a fraction of each row's own correction |
| the not-an-identity renders | that they are expected rather than failed, because the chain has moved off the joint trajectory, and that this is what makes the `λ=1` column a classifier |
| the corrector rows | the `k` and the `c` they ran at, both of which are compute-budget choices rather than properties of the method |
| the SuperDiff row | the step-count parity verdict from [its own review](05-superdiff-at-this-repos-fifty-steps.md) |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open. This file has not been run against.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Form `r_t^M` on one render for each of the four rows and print its per-step norm, so a missing hook
is found before either grid is planned.
