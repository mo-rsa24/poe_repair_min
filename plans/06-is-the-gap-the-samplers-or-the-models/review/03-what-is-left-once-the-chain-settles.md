# 🧪 Review: does the corrector remove part of the correction and leave part of it?

**The grid has run at three step sizes (`c = 30`, the search's pick; then `3` and `0.3`), and every
one fires the inconclusive branch. The chain moves and, on the composing pair, settles; on cat × dog
the `k = 100` and `k = 200` read-zone ratios never agree within the 5% bar (they differ by 7%, 21%
and 38%), and the composing pair's ratio rises with `k` at two of the three step sizes. The cause is
named below: one seed gives one trajectory per `k`, and the read-zone ratio of one trajectory
scatters by 15% across devices and by 30% to 70% across `k`, so the 5% bars cannot be met at one
seed whatever the answer is. The support-shaped numbers exist (cat × dog drops 18% to 36% at
`k = 200` at every `c`) and are not licensed.** Every question below was written before any
corrector existed, so no answer here can be chosen after the fact. This file judges
[the design for this measurement](../plans/hypothesis/03-what-is-left-once-the-chain-settles.md),
and its answer
decides whether section 7 of the paper carries the corrector as a limitation, as an alternative, or
as a two-sentence note. Step 21 of the running order cannot be written honestly until it is
answered.

## Recommended prompt (when the run lands)

```
/analyze-run the corrector residual curve, k sweep against denoising step
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/03-what-is-left-once-the-chain-settles.md) | the grid, the thresholds in source, and the section on where the two errors separate |
| **this file** | **the verdict: not yet run** |
| [the corrector's verdict](02-the-corrector-and-the-step-size-it-runs-at.md) | the composer and the step size this grid runs at |
| [the free bound's verdict](01-the-free-bound-on-the-models-share.md) | the lower bound this grid's answer is read against |
| [the timing verdict](../../03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md) | the result this grid puts under threat, and the numbers it quotes |
| [the two literature checks before print](../../03-does-the-correction-cause-composition/plans/checks/09-two-literature-checks-before-print.md) | cites Soiffer et al. for the claim this grid turns into a number |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [Where the two errors can be told apart, and where they cannot](#where-the-two-errors-can-be-told-apart-and-where-they-cannot)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

The full vocabulary is in the design file's `## Words this plan uses` and is not copied here. The
four that every answer below turns on:

- **The sampler's share**, the part a corrector can remove, which goes to zero as noise goes to
  zero.
- **The model's share**, the part no corrector touches, which does not.
- **`k`**, how many Langevin steps run at each of the 50 noise levels before the reverse step.
- **The read zone**, the last five denoising steps, where the sampler's share has vanished by
  construction and anything left is the model's. The early steps give a size rather than an
  attribution, and [the section below](#where-the-two-errors-can-be-told-apart-and-where-they-cannot)
  says why.

> A Langevin corrector is a small repeated random walk that nudges the latent along the score and
> adds a little noise each time. Run for long enough at a fixed noise level it forgets where it
> started and settles wherever the score says the probability actually is.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** Missing the threshold below does not close the plan. It selects which of three
paragraphs section 7 carries, and all three are written in the design file's
`## Why this plan exists`.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| The first short run, one pair, one seed, `k ∈ {0,1}` and then `k=5` | Tests the claim | 2026-09-05 16:52 to 17:02, in-session on mscluster85 device 0 (RTX 3090), pids 258072 and 260090, commit 72b8826, at `c=0.035` for timing only | 8.5 min in all | `corrector/logs/smoke.log`; wall time per noise level 1.5 s (`k=0`, 5 UNet evaluations), 2.3 s (`k=1`, 8), 5.3 s (`k=5`, 20) | ✅ within the bar: measured ratios 1.5× and 3.5× against the cost table's 1.6× and 4×, so the branch count is right; one Langevin step costs about 0.76 s on the 3090, so `k=200` on one pair is about 2.1 h there and the whole two-pair grid about 7 h on one 3090 |
| The `k` grid at `c = 30`, 2 pairs × 6 `k` × 50 steps | Tests the claim | 2026-09-05 19:04 to 20:11, `nohup` on mscluster110 device 0 (RTX PRO 6000 Blackwell, `co3_bw`), pids 433699 and 434481, commit 0150704; the composing pair first, then cat × dog, one process each | 67 min for both pairs on that card (`k = 200` takes 16 to 26 min per pair) | `corrector/curves/*__c30__k*.json` (12 cells), aggregated into `corrector/residual_curves.json` with `c = 30` on every row; `corrector/verdict_c30.txt`; figure `mcmc/how-much-of-the-correction-a-corrector-removes-c30.png` | ❓ inconclusive on four guards, see the question below; the composer's final images at this `c` are texture noise from `k = 1` |
| The `k` grid at `c = 0.3`, same cells | Tests the claim | 2026-09-05 19:12 to 23:16, mscluster108 device 1 (RTX 8000, `co3`): the composing pair (pid 300147) then cat × dog (pid 302364), commit 0150704 | 4 h on that card | `corrector/curves/*__c0p3__k*.json`, `verdict_c0p3.txt`, figure `mcmc/how-much-of-the-correction-a-corrector-removes-c0p3.png` | ❓ inconclusive: cat × dog's `k = 100` and `k = 200` differ by 38%; the composing pair rises 13% |
| The `k` grid at `c = 3`, same cells | Tests the claim | 2026-09-05 20:14 to 2026-09-06 00:01, mscluster85 device 0 (RTX 3090, `co3`): the composing pair (pid 297556) then cat × dog (pid 309606); the Blackwell card was taken by another user before its queue reached these | 3.8 h on that card | `corrector/curves/*__c3__k*.json`, `verdict_c3.txt`, figure `mcmc/how-much-of-the-correction-a-corrector-removes-c3.png` | ❓ inconclusive: cat × dog's `k = 100` and `k = 200` differ by 21%; the composing pair passes (falls 10%) |

## Where the two errors can be told apart, and where they cannot

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

**This is why the measurement is read at the end of the run rather than across it, and it is
repeated here in full rather than compressed into a caveat line.** Getting it wrong would put a
number in the paper that means something other than what the caption says.

**What the corrector converges to.** Langevin driven by the product-of-experts score at noise level
`t` has stationary distribution `q_t = p_t(cat)·p_t(dog)`, the product of the two diffused
marginals. That is the distribution the summed score is the exact score of. At `t = 0` the noise is
gone and `q_0 = p(cat)·p(dog)`, which is the product of experts the method asked for in the first
place. So a converged corrector delivers the product exactly, and that is the whole of Du et al.'s
claim.

**What is left at the end of the run is the model's share, cleanly.** At the last steps the latent
is drawn from `p(cat)·p(dog)`. The residual `eps_J - eps_PoE` is then the joint model disagreeing
with the product on the product's own support. There is no sampler error left to contaminate it,
because that error is defined to vanish there. This number is the paper's estimate of the model's
share.

**What is left at the start of the run is still both.** At `t > 0` the corrector has settled the
latent into `q_t`, the product of the diffused marginals. The thing the reverse process would need
is the diffusion of the product, and those two differ precisely because noising and multiplying do
not commute. So what remains at high noise is the non-commutation gap plus the diffused model gap,
and no amount of `k` separates them. **The early part of the curve gives a size rather than an
attribution.**

**Which lands on the timing result.** The compose-decisive window is steps 0 to 10, the high-noise
end, which is exactly the region where this measurement cannot attribute. That limit belongs in the
caption and in this file, and it is never argued away. What this scope can say about the early
window is behavioural rather than attributional, and it is
[step 27](04-does-the-corrector-compose-in-the-same-window.md).

**Why the residual norm is a proxy and not the gap itself.** The corrector does not change the
function `eps_J - eps_PoE`; it changes where that function is evaluated. A falling curve says the
settled latents sit where the two networks agree better, which is evidence about the distributional
gap rather than a measurement of it.

**What the composing pair controls.** On `a_butterfly__x__a_flower_meadow`, which composes under
plain product-of-experts, `q_t` is already close to the joint and `eps_J` is evaluated somewhere it
has seen. Its curve should be low at `k=0` and should not rise with `k`. If it rises the same way
the failing pair does, the rise is this measurement walking the joint branch off-distribution and no
reading of either curve is licensed. `an_elephant__x__a_penguin` is not used for this, because
whether it composes by default is
[an open question in another review file](../../04-does-the-fix-reach-unseen-pairs/review/01-the-clean-pair-pool.md),
and a control whose own behaviour is unsettled controls nothing.

## The question written before the run

Navigation: ⬅️ [Where the two errors can be told apart](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

The thresholds live in source as constants in `scripts/corrector_residual_curve.py`, so moving one
after the answer is visible shows up in a diff.

- [ ] ⚠️ **Does the corrector remove part of the correction and leave part of it?** Support if,
      over the last five steps, the ratio at the `k` where the curve has flattened has fallen by at
      least 0.20 of its `k=0` value and at least 0.20 of it is still there. Null if the change is
      under 0.05 at every step while the chain provably moved. Inconclusive if `k=100` and `k=200`
      still differ by more than 0.05, if the median displacement is under 0.05, or if the composing
      pair behaves like the failing pair.

      > A null here means the correction's size came out the same with the corrector running as
      > without it.

      **The value at any single `k` is a statement about the compute budget rather than about the
      problem. Only the trend across `k` is a result, and only once the curve has flattened.** That
      rule is enforced by `MAX_K_INSTABILITY` in source, which refuses to license a reading, rather
      than by a caption asking the reader to remember it.

      **At `c = 30`, the step size the search picked: ❓ inconclusive**, from `verdict_c30.txt`.
      Read-zone means over the last five steps, one seed, one trajectory per `k`. Cat × dog:
      0.1685 at `k = 0`, then 0.1137, 0.1209, 0.1111, 0.1205, 0.1124 at `k = 1, 5, 20, 100, 200`;
      the numerator fell from 37.9 to 27.9 while the denominator rose from 225 to 278. The
      composing pair: 0.0896 at `k = 0`, then 0.1071, 0.1342, 0.1692, 0.1343, 0.1110. Four guards
      failed: `k = 100` against `k = 200` differ by 0.067 (cat × dog) and 0.173 (composing pair) of
      the `k = 100` value against the 0.05 bar; the latent norm reached 1.52× and 1.66× its start
      at `k = 200` against the 1.5× bar; and the composing pair's read-zone ratio rose by 24%
      against the 5% drift bar. The chain moved (median displacement 1.26 on both pairs). The
      images behind the rows say why: at this step size every `k ≥ 1` render is texture noise
      (`corrector/search_renders/butterfly_c30_by_k.png`), so the joint branch is evaluated far
      off the data and its disagreement with the product means nothing. Per the branch's own
      instruction the step size goes back to step 25, and the grid is repeated at `c = 3` and
      `c = 0.3`.

      **At `c = 3` and `c = 0.3`: ❓ inconclusive both times**, from `verdict_c3.txt` and
      `verdict_c0p3.txt`. Read-zone means over the last five steps, one seed, one trajectory
      per `k`, every cell of one pair and `c` on one device:

      | `c` | pair | `k = 0` | 1 | 5 | 20 | 100 | 200 | `k`=100 vs 200 | control at 200 vs 0 | median chain displacement at 200 |
      |---|---|---|---|---|---|---|---|---|---|---|
      | 0.3 | cat × dog | 0.180 | 0.145 | 0.125 | 0.141 | 0.186 | 0.115 | 38% | | 0.66 |
      | 0.3 | butterfly × meadow | 0.092 | 0.094 | 0.092 | 0.093 | 0.105 | 0.104 | 1.1% | +13% | 0.64 |
      | 3 | cat × dog | 0.153 | 0.138 | 0.261 | 0.203 | 0.103 | 0.125 | 21% | | 1.16 |
      | 3 | butterfly × meadow | 0.097 | 0.095 | 0.099 | 0.105 | 0.090 | 0.086 | 3.5% | −10% | 1.02 |
      | 30 | cat × dog | 0.169 | 0.114 | 0.121 | 0.111 | 0.121 | 0.112 | 6.7% | | 1.26 |
      | 30 | butterfly × meadow | 0.090 | 0.107 | 0.134 | 0.169 | 0.134 | 0.111 | 17% | +24% | 1.25 |

      The guards that fired: at `c = 0.3`, cat × dog not settled (38%) and the control rising
      (13%); at `c = 3`, cat × dog not settled (21%) with the control passing; at `c = 30`, both
      pairs not settled, both past the 1.5× norm bound at `k = 200`, and the control rising. The
      latent norm stayed within 1.02× at `c = 3` and 1.01× at `c = 0.3`.

      **What the numbers would have said if licensed.** At `k = 200` cat × dog's read-zone
      ratio sits 36%, 18% and 33% under its `k = 0` value at `c = 0.3`, `3` and `30`, with 64%,
      82% and 67% still there: the support shape (a drop of at least 20% with at least 20% left)
      at every step size. At `c = 0.3` the drop is the numerator's: `‖eps_J − eps_PoE‖` falls from
      40.8 to 23.9 while `‖eps_PoE‖` moves from 226 to 211. None of it is a result, because the
      same pair's `k = 100` value is 0.186, above `k = 0`, and one trajectory decides each number.

      **Why one seed cannot meet these bars.** The `k = 0` cell alone, the plain PoE run with no
      chain, reads 0.153 on the RTX 3090, 0.169 on the RTX PRO 6000 and 0.180 on the RTX 8000 for
      cat × dog: a 15% spread from fp16 arithmetic across cards over 50 steps. Across `k` at one
      `c` the same pair swings by 30% to 70% between neighbouring counts. The bars are 5% (drift,
      instability) and 20% (split). A pre-registered rule with bars inside the instrument's own
      scatter returns inconclusive whatever the answer, and that is the finding of this run: the
      residual ratio at the settled point is a per-trajectory quantity and needs seeds before a
      trend across `k` can be read. The composing pair scatters less (its ratio is 0.09 to 0.17
      and its curve across `k` is flatter), which is why it settles at `c ≤ 3` and cat × dog does
      not.

      **What the pictures add.** Every measured chain saved its final image
      (`corrector/pairs/<pair>/seed_9/poe_langevin_k<k>_c<c>/`, strips in
      `corrector/search_renders/`). At `c ≤ 3` both pairs keep a coherent photograph through
      `k = 20`; from `k = 100` the settled sample drifts from a photograph toward an illustration
      (a painted butterfly, a dog in sunglasses, then a flat graphic at `c = 3`), with the latent
      norm never above 1.02×. At `c = 30` every `k ≥ 1` image is texture noise. On cat × dog the
      corrector never produces two animals at any `k` or `c`. So what the chain settles into is
      a different kind of image from the joint prompt's, and the joint branch's disagreement
      with that image is what a rising ratio on the composing pair reads. The early window
      (steps 0 to 10) is where all six curves of a panel lie on top of each other, and it is the
      window this measurement cannot attribute by construction.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] 🟡 Did the numerator `‖eps_J - eps_PoE‖` fall, or did the denominator `‖eps_PoE‖` rise? On
      cat × dog the numerator fell at every `c` (read-zone means, `k = 0` to `k = 200`: 40.8 to
      23.9 at `c = 0.3`, 34.3 to 28.1 at `c = 3`, 37.9 to 27.9 at `c = 30`), and the denominator
      held at `c ≤ 3` (226 to 211, 224 to 230) but rose 23% at `c = 30` (225 to 278), so at the
      search's step size a quarter of the ratio's fall is the denominator. On the composing pair
      both rose at `c = 0.3` and `c = 30`. 🟡 because none of these trajectories is licensed.
- [x] ✅ Did the chain actually move? Yes at every `c`: median relative displacement within a
      level at `k = 200` is 0.64 to 0.66 at `c = 0.3`, 1.02 to 1.16 at `c = 3`, 1.25 to 1.26 at
      `c = 30`, all far above the 0.05 floor. The instrument moves; that is not the problem.
- [x] ❌ Did it equilibrate? The composing pair did at `c ≤ 3` (`k = 100` against `k = 200` differ
      by 1.1% and 3.5%). Cat × dog did not at any `c` (38%, 21%, 6.7% against the 5% bar), and its
      curve across `k` is not monotone at any `c`, so the disagreement is scatter rather than a
      chain still falling. Curve on curve, the `k = 100` and `k = 200` lines cross each other all
      along the run on the cat × dog panels.
- [x] 🟡 Does the pair that composes by default behave differently from the pair that blends? At
      `k = 200` cat × dog falls at every `c` (−36%, −18%, −33%) while the composing pair rises at
      `c = 0.3` (+13%) and `c = 30` (+24%) and falls at `c = 3` (−10%). Different at `c = 3`,
      opposite at the other two, and in every case within the scatter one seed produces. The
      composing pair's images say what its ratio is measuring: at `k ≥ 100` the settled sample is
      an illustration rather than a photograph, and the joint branch disagrees with that.
- [x] ⚠️ How does the measured remainder at the last steps compare with
      [the free bound](01-the-free-bound-on-the-models-share.md) read off the cached
      renders? Unanswerable this sitting: step 24 has not run, and no remainder here is licensed
      to compare against it. Two independent routes to the same quantity that disagree is a
      finding about one of the two measuring tools.
- [x] ✅ Was the first short run's measured wall time per noise level within 2× of the cost table?
      Yes. Per noise level on the 3090: 1.5 s at `k=0`, 2.3 s at `k=1`, 5.3 s at `k=5`, against
      5, 8 and 20 UNet evaluations. The measured ratios to `k=0` are 1.5× and 3.5× where the table
      predicts 1.6× and 4×, so the evaluation count per level is right and the grid was not
      re-planned. The absolute cost is higher than the table's plain-render equivalents suggest,
      about 0.76 s per Langevin step, which puts the two-pair grid at roughly 7 hours on one 3090
      rather than the 670 render-equivalents' 4.5 hours; within the 2× bar.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- The read-zone ratio of a single trajectory scatters by more than the bars. Three devices give
  three `k = 0` values for cat × dog (0.153, 0.169, 0.180), so a per-pair comparison across `k`
  is only fair on one device, which every grid above respected, and a 5% bar needs seeds.
- On cat × dog the mid-run ratio (steps 10 to 45) at `c = 3`, `k = 100` and `k = 200` sits near
  0.1, far under `k = 0`'s 0.2 to 0.4, and the curves rejoin in the read zone. The chain changes
  the middle of the run more than its end, which is the opposite of what the design expected
  (the sampler's share was to vanish at low noise). Read as a hint, not a result.
- The settled sample's drift from photograph to illustration with `k` is the one observation
  every `c` and both pairs agree on, and no bar was written for it. It says the product of the
  two diffused marginals has a different typical image from the joint prompt's, in style before
  content, which is a statement about the model's share that the ratio was never going to make.
- The images behind the numbers are the check the code cannot make. Every `k ≥ 1` image at
  `c = 30` is texture noise, while its numbers are as tidy as the others'.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Within one `c` and one pair only `k` varies: same seed, same
      50 DDIM steps, same guidance 7.5, same step size, the same cached starting latent, the same
      Langevin noise stream, and one device for every `k`. Across pairs the device differs at
      `c = 0.3` (both on the RTX 8000) and `c = 3` (both on the 3090) only by the order they ran.
- [x] ❌ **Was the measuring tool sound?** The leak checks pass and the displacement column shows
      the chain moving; the flattening in `k` fails on cat × dog at every `c`. So the instrument
      is inert when off and moves when on, and its read-out at one seed is too noisy for the bars
      written for it. That is the reason the branch is inconclusive and not a null.
- [x] ✅ **Is the quantity what the caption says it is?** Every figure's sidecar carries the three
      sentences: the axis is the correction along the `k`-corrected path; `eps_J` is evaluated
      off-distribution at the settled point on purpose; the residual norm is a proxy for the
      distributional gap, since the corrector moves where `eps_J − eps_PoE` is evaluated and not
      the function. The images add the fourth: what the settled point looks like.
- [x] ✅ **Did the run respect the environment?** 600 rows per `c`, three times, counted by
      `--verdict`; every file under `corrector/` on `/datasets`; every norm float32; every run
      under `nohup` on a shared device with node, device and PID in its log header, harvested by
      `pgrep`; the launch script refused two devices that another user's process took between
      the check and the launch (the 3090 at 20:12, the Blackwell card at 20:14), which is the
      guard doing its job.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Three rows are owed whatever the result, because they are limits of the design rather than
properties of the answer.

| What the paper says | What it owes alongside it |
|---|---|
| the size of the model's share | that it is measured at the last five denoising steps only, because that is the one place the two errors separate |
| anything about the early window | that steps 0 to 10 are where the [compose rate](../../../context/world/compose-rate.md) is decided and where this measurement cannot attribute, so this plan bounds the timing result rather than explaining it |
| the corrected curve at any `k` | that `k=5` and `k=20` are two different trajectories rather than one point wiggled twice, and that `eps_J` is evaluated where the joint model would never go |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether the early window's correction is the sampler's or the model's | nothing in this scope. The two are inseparable at high noise by construction. A different design would be needed, and none is currently known | the strongest version of the timing paragraph. The weaker version, which this scope supports, is that a corrector does or does not reproduce the timing behaviour |
| whether one seed is enough for this curve | it is not: the scatter of one trajectory's read-zone ratio (15% across devices at `k = 0`, 30% to 70% across `k`) is larger than every bar. The named next action is seeds 10, 11 and 12 at `c = 3` (the one step size whose composing-pair control passes) and `k ∈ {0, 20, 100, 200}`, both pairs: about 110 min per seed and pair on a 3090, 11 hours in all, or under 2 hours on the RTX PRO 6000 when it is free. Then the same three-way threshold on the seed-mean curve, bars unchanged | step 26's verdict, step 21's mechanism paragraph, and steps 28 to 30 as a diagnosis rather than a baselines table |
| which step size the scope proceeds at | `c = 3`, by the rule written in [step 25's review](02-the-corrector-and-the-step-size-it-runs-at.md): the largest `c` whose composing-pair curve does not rise past 5% of its `k = 0` value at `k = 200` (30 rises 24%, 0.3 rises 13%, 3 falls 10%). Its settled samples at `k ≥ 100` are flat graphics, so plan 27 runs it at `k = 20`, the last count whose images are photographs on both pairs, and says so | plan 27's runs, which use it |
| a figure under the canonical name `how-much-of-the-correction-a-corrector-removes.png` | a licensed reading. Until then the three `c`-suffixed figures stand, none in the main text | the Figure Catalog row in the design |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run seeds 10 to 12 at `c = 3` when a fast device is free, then `--verdict --c 3` on the seed-mean
curve with the bars unchanged. Plan 27 runs in the meantime at `c = 3`, `k = 20`, recorded there as
run against an inconclusive curve.

## Cross-references

- The finding built from this verdict: [does a Langevin corrector remove part of the correction](../../../report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md).
- The recipe that reruns the grid and prints the branch: [running the Langevin corrector](../../../runbook/running-things-on-the-cluster/running-the-langevin-corrector.md).
