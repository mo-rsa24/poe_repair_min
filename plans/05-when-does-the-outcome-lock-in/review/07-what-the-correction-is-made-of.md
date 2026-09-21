# 🧪 Review: could product-of-experts have supplied the correction by re-weighting its own predictions?

All four rungs have run (2026-09-05) and the deciding question is answered: inconclusive by its
own bar. This file judges [the design](../plans/tests/07-what-the-correction-is-made-of.md); the
finding is [what is the correction made of](../../../report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md).
The questions below were written before any number existed, and the three constants they turn
on were written into `scripts/showcase/correction_span_common.py` in the same sitting.

## Recommended prompt (when the run lands)

```
/analyze-run yb933cr6
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tests/07-what-the-correction-is-made-of.md) | the rule the cache used, the four rungs, the bar and where it lives in source |
| **this file** | **the verdict: what the four reads answered, and what they could not** |

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

- **the correction (r_t)**: at one cached state of the plain product-of-experts run, what the
  joint prompt "a cat and a dog" would have added to the prediction: the guided joint prediction
  minus the product-of-experts prediction. Closed form 7.5 (ε_j − ε_a − ε_b + ε_∅) over the four
  raw predictions the cache stores.
- **the span**: every prediction reachable by re-weighting the three vectors PoE already has,
  α ε_∅ + β (ε_a − ε_∅) + γ (ε_b − ε_∅). A per-step change of guidance scale on either expert
  stays inside it.
- **orthogonal share**: the fraction of the correction's squared norm that lies outside the
  span, one minus the in-span share. It is what no re-weighting can supply.
- **the early window**: steps 0 to 10 inclusive, the window where injecting the correction
  changes the outcome, per the window experiments recorded in
  [the interaction term entry](../../../context/world/interaction-term.md).
- **the adapter's output (δ̂)**: the product-of-experts prediction with the rank-32 adapter
  attached minus the same with it detached, at the same cached state, read at λ 1.
- **same-place share**: the fraction of the 16,384 latent positions where both experts' departure
  from the unconditional estimate is above that expert's own median; chance level 0.25.
- **kinetic energy**: sum over the saved segments of a run's frames of the squared distance its
  DINOv2 embedding moved; **its floor** is the squared straight-line distance divided by the
  number of segments.
- **which-animal score**: cosine to the cat-alone centroid minus cosine to the dog-alone
  centroid; a sign change is the running estimate switching animal.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** A missed bar on rung 2 closes the question "could a re-weighting have done
it" in the sampler's favour and hands the parallel sampler-side sessions their opening; a met
bar closes it against them. Rungs 1, 3 and 4 explore and carry no bar.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| rung 2, in-span share of r_t, seeds 9 to 16, 50 steps | CPU, session node mscluster85 | 2026-09-05 16:30 | 9 seconds | `artifacts/results/what-the-correction-is-made-of/orthogonal-share-over-steps.{png,json}` | done, 400 rows |
| rung 4, track energy and which-animal | CPU, mscluster85 | 2026-09-05 16:35 | seconds | `.../track-kinetic-energy.png`, `.../which-animal-over-steps.png`, `.../track-energy-and-which-animal.json` | done, 48 rows |
| rungs 1 and 3 chained, first launch | `nohup`, mscluster85 | 2026-09-05 16:45 | nothing | none | never started: the launching shell stalled on a foreground CUDA check and died before writing the chain script |
| rung 1, seed 15 decoded estimates and the same-place share | GPU, mscluster85 device 0 (RTX 3090, shared with another session's process holding 7.5 GB), PID 257033, `nohup` | 2026-09-05 16:50 | 51 seconds, 35 decodes | `/datasets/mmolefe/poe_repair_min/outputs/showcase/what_the_correction_is_made_of/tweedie/seed_15/`, `.../seed-15-experts-tweedie-strip.png`, `.../same-place-share.json` | done |
| rung 3, the adapter at 400 cached states | GPU, same device, PID 257603, `nohup` | 2026-09-05 16:51 | 9.5 minutes, 800 UNet forwards | `.../adapter-span-share.json`, `.../adapter-span-share-over-steps.png`, `.../seed-15-adapter-vs-correction-norm-maps.png` | done, sanity 0.9996 |
| the W&B log | `correction_log_wandb.py` | 2026-09-05 17:02 | seconds | run `yb933cr6`, `prime_lab/poe-repair-animals-compose`, six images and one artifact | done |

Logs under `/datasets/mmolefe/poe_repair_min/outputs/showcase/what_the_correction_is_made_of/logs/`.

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] 🟡 Over steps 0 to 10, is the orthogonal share of the correction's squared norm, mean over
      seeds 9 to 16 of each seed's mean over those steps, above `ORTHO_SHARE_REWEIGHT_IMPOSSIBLE`
      (0.5: support, no re-weighting of the two experts could have supplied the correction), below
      `ORTHO_SHARE_REWEIGHT_CANDIDATE` (0.25: null, a per-step guidance re-weighting is a candidate
      fix), or between (inconclusive)? This is the bar because it is the cheapest upper bound on
      what any method that only re-mixes the product's own proposals can reach, and three parallel
      sessions are about to spend GPU time on such methods. The constants live in
      `scripts/showcase/correction_span_common.py`.
      **Answer: inconclusive. The statistic is 0.374**, per seed 0.16 (seed 9), 0.21, 0.45, 0.40,
      0.42, 0.37, 0.33, 0.65 (seed 16); per step it is 0.30 at step 0, 0.51 at step 5, 0.29 at
      step 10. From `orthogonal-share-over-steps.json`, `summary.early_window_orthogonal_share_mean_over_seeds`
      and `summary.verdict`. What the answer means for the sampler-side sessions: about a third
      of the early correction is out of reach of any re-weighting, and the reachable two thirds is
      mostly "turn both experts' guidance down" (next section), so a re-weighting alone is neither
      ruled out nor made a candidate by this number.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ Does the orthogonal share rise, fall or stay flat with step? Rising is what the claim
      expects; falling would surprise.
      **Answer: it rises to step 5, then falls, then rises again at the end.** 0.30 at step 0,
      0.51 at step 5, 0.29 at step 10, 0.09 at step 20, 0.08 at step 30, 0.27 at step 49. The
      joint prompt's own guidance direction becomes more its own over the run (its orthogonal
      share 0.31 at step 0, 0.71 at step 30), so the correction's fall is the correction growing
      inside the span, not the joint direction becoming more expert-like. Neither the expected
      rise nor the surprise: the shape is a peak in the window.
- [x] ✅ What are the least-squares coefficients on ε_∅, ε_a − ε_∅ and ε_b − ε_∅ that best
      approximate r_t inside the span, per step?
      **Answer: near −7.5 plus a small positive amount, so the in-span part is mostly "undo the
      experts' guidance".** The weight the joint prompt's guided prediction puts on the cat
      expert's direction is 2.5 at step 0, 3.3 at step 5, 2.9 at step 10 and 1.3 at step 30; on
      the dog expert's direction 5.3, 3.8, 1.1 and 0.9; PoE puts 7.5 on each. The coefficient on
      ε_∅ is 0.00 throughout. From `summary.joint_guided_in_span_coef_on_*_mean_per_step`.
- [x] ✅ Rung 1: at steps 5 and 10, is the same-place share above chance (0.25), and is the mean
      cosine between the experts' departures on shared positions positive or negative?
      **Answer: above chance and negative.** Same-place share, mean over seeds, 0.35 at step 5
      and 0.40 at step 10 (0.43 by step 30); cosine on the shared positions −0.16 at step 5 and
      −0.41 at step 10, negative on 62% and 77% of those positions. Positive at step 0 (+0.77)
      and at step 49 (+0.44). The chimera is built in one place by two experts pushing opposite
      ways there. From `same-place-share.json`, `summary`.
- [x] 🟡 Rung 3: is the adapter's orthogonal share within 0.1 of r_t's at the same step, and is
      the cosine of its orthogonal part against r_t's above 0.5 over steps 0 to 10?
      **Answer: the second half yes, the first half no.** Cosine between the orthogonal parts,
      early-window mean 0.55 (0.61 at step 0, 0.50 at step 10). Orthogonal share of the adapter's
      output, early-window mean 0.19 against 0.37 for r_t, a gap of 0.18; the two agree within
      0.1 from step 12 on. In-span cosine 0.91 to 0.99 at every step. So the adapter learned the
      re-weightable part almost exactly and the new direction about half, and puts less of its
      energy there than the target does. From `adapter-span-share.json`, `summary.early_window`.
- [x] ✅ Rung 3 sanity: is the cosine between the live adapter-off PoE prediction and the cached
      one above 0.99 on every row?
      **Answer: yes, minimum 0.9996 over 400 rows.** `summary.sanity_cos_live_vs_cached_poe_min`.
- [x] ✅ Rung 4: is the plain PoE run's kinetic energy further above its floor than the joint
      run's and the corrected run's, on the medians over seeds?
      **Answer: above the joint run's, yes; above the corrected run's, no.** From step 10 to 50,
      median kinetic energy PoE 2.79, joint prompt 0.78, PoE plus λ 1.2 correction 2.71, PoE plus
      λ 1.0 3.13, single prompts 0.53 and 0.62; floors 0.06 to 0.15. The correction changes where
      a run ends and not how much it moves. From `track-energy-and-which-animal.json`, `summary`.
- [x] ✅ Rung 4: on how many of the eight PoE runs does the which-animal score change sign after
      step 10, and at which step does seed 15's?
      **Answer: 5 of 8 by the strict rule (both sides beyond ±0.1, `FLIP_MIN_MAGNITUDE`), 7 of 8
      by any sign change; corrected λ 1.2 runs 2 of 8 strict, joint prompt 0 of 8.** Seed 15's
      PoE score is +0.21 at step 20, −0.06 at 25, +0.01 at 30, −0.18 at 35, −0.30 at 40, so the
      crossing happens between steps 20 and 35 through two near-zero frames; it counts under the
      loose rule and not the strict one. This matches plan 06's strip (a cat until step 10, a
      dog by step 40).

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was
written with the answer already visible.

- [ ] ⚠️ Would a run that re-weights both experts down to the joint prompt's own weights (about
      1 to 3 instead of 7.5, per step, from the sidecar's coefficients) compose, with no learned
      term? Raised by the coefficient read. This is a sampler-side idea for scope 06, not a
      question this plan can answer.
- [ ] ⚠️ Is the adapter's shortfall on the orthogonal part (0.19 against 0.37) a held-out effect
      or an architecture effect? Raised by rung 3; settled by the same read on a training pair.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

Three fixed checks. Each is answered or explicitly marked not applicable; none is dropped.

- [x] ✅ **Was the comparison fair?** Every quantity in rungs 1 to 3 is computed at the same
      cached state from the same four raw predictions, upcast to fp32. Rung 3's adapter-on and
      adapter-off forwards ran in one process on the same input, and the adapter-off prediction
      matches the cache at cosine 0.9996 or better on every row. Rung 4 reads the frames plan 06
      rendered under one sampler and one initial noise per seed.
- [x] ✅ **Was the measuring tool sound?** The projection is exact least squares; the PoE
      prediction rebuilt from the raw vectors returns in-span share 1.0000 on all 400 rows, so
      the basis and the rule agree. The same-place share's threshold is each expert's own median,
      fixed at design time, with chance 0.25 by construction. The seed-15 strip was read by eye
      (instruction 6.1): the two experts put their animal in the same head position at steps 5
      and 10 and the heat map is brightest there, so the same-place number has a picture behind
      it; recorded as agrees.
- [x] ✅ **Did the run respect the environment?** Decoded pictures and logs on `/datasets`; every
      cached tensor upcast before use; CUDA confirmed available inside both GPU scripts before
      any work (the scripts refuse to run on CPU); node, device and PID in the Runs table. The
      first chain launch never started and is in the table.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#what-the-run-cost-and-what-it-bought) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a third of the early correction lies outside what the experts span | that the span is three vectors at one state, so it bounds per-step re-weighting only, not a method that changes the state; and that the bar was inconclusive, not support |
| the reachable part is mostly turning both experts' guidance down | the coefficients per step and that they are read off-policy, at states the joint prompt never steered |
| the adapter learned the re-weightable part and half of the new direction | read at λ 1 on cached states, not at λ 1.2 on its own trajectory |
| product-based runs keep moving after step 10 and the plain one changes animal | 14 saved frames at uneven spacing; the flip rule's threshold; that energy does not separate composing from not |

## What the run cost, and what it bought

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Eleven minutes of one shared RTX 3090 and under a minute of CPU. It bought the deciding
question's answer, the four figures, and the number the sampler-side sessions asked for. One
launch was lost to a foreground CUDA check that stalled the launching shell; the fix was to
launch with `nohup` first and let the script check CUDA itself, which both GPU scripts now do.

## Still open

Navigation: ⬅️ [What the run cost](#what-the-run-cost-and-what-it-bought) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| Whether the orthogonal part of r_t is the sampler's error or the model's (claim 2 of [the idea map](../../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md)) | scope 06's sampler-share measurement, which this plan does not touch | the framing of the paper's section 7 |
| The inconclusive bar | a second pair and a training pair through the same script, so the 0.37 has company; and the damping-only run named above | the sampler-side sessions' reading of their own results |
| The adapter's orthogonal shortfall, held-out or architectural | rung 3 on a training pair | the claim about what the adapter learned |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run the plan's close-out: `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md — four rungs done, bar inconclusive at 0.374`.
