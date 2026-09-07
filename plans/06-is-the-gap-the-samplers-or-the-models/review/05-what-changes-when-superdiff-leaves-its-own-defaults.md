# 🔌 Review: what changes when SuperDiff leaves its own defaults?

**Nothing has run yet.** Every question below was written before the pipeline was wired. This file
judges [the SuperDiff design](../plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md). Its
answer decides whether every later comparison against SuperDiff is a fair one, or a comparison
between a working rule and one run outside its intended settings.

## Recommended prompt (when the run lands)

```
/analyze-run the SuperDiff eight-cell grid, 200 against 50 steps, kappa clamped against unclamped
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md) | the wiring, the `kappa` clamp, the eight-cell grid, and the per-step prediction hook |
| **this file** | **the verdict: not yet run** |
| [the amount-axis verdict](06-three-rules-on-one-amount-axis.md) | where this row joins this project's own shared `λ` axis, using this plan's `eps_M` hook |
| [the step 26 verdict](03-what-is-left-once-the-chain-settles.md) | whether this half of the scope is a diagnosis or a baselines table |

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

- **SuperDiff**: a composition rule derived from the continuity equation rather than from adding
  scores. Skreta et al., [arXiv 2412.17762](https://arxiv.org/abs/2412.17762), ICLR 2025 Spotlight.
- **The step count**: how many denoising steps a render takes. `SuperDiffSDXLPipeline` defaults to
  200; everything else measured in this project runs at 50. This pipeline has no DDIM mode, so
  step count is the only sampling setting it exposes.
- **`kappa`**: the pipeline's per-step blending weight between the two prompts. Computed with no
  clamp anywhere in the shipped source, confirmed by reading `pipeline.py` in full.
- **The `kappa` clamp**: a cap this plan adds (roughly [−0.5, 1.5], holding `kappa` at 0.5 for the
  first ~10% of steps), toggleable so the grid can measure whether it changes the composes verdict.
- **The per-step prediction, `eps_M`**: what a composition rule predicts at each step. Exposing it
  is what lets `r_t^SD = eps_J - eps_M` be formed, which the shared amount axis needs.
- **The eight-cell grid**: `a_cat__x__a_dog` and `a_butterfly__x__a_flower_meadow` (seed 9, this
  scope's standard pair of pairs), each rendered at 200 and 50 steps, each with the `kappa` clamp
  on and off.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Establishes a baseline.** Missing the threshold means the comparison has no fair starting point.
Every figure comparing against SuperDiff would then compare a working rule against a crippled one.
It does not close the plan; it adds a sentence to every caption that uses this row.

Per this project's run conventions a baseline may not change a claim, and it freezes on landing.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Task 1.1: wiring verification, `a_cat__x__a_dog` seed 9, 50 steps, clamped | In-session, mscluster85 | 2026-09-03 | 1 render | `corrector/superdiff/parity/grid/pairs/a_cat__x__a_dog/seed_9/superdiff_50steps_clamped/` | ✅ composes, real image not noise |
| Task 1.2: `r_t^SD` hook verification, same pair/seed/steps, 4th joint-prompt row added | In-session, mscluster85 | 2026-09-03 | 1 render | `.../superdiff_50steps_clamped_with_rt/` | ✅ 50 non-zero, non-NaN values, 8.3 → 17.6 |
| Task 1.3: `kappa`-clamp bound verification, synthetic out-of-range input | In-session, mscluster85 | 2026-09-03 | none (no render, direct check of the clamp expression) | — | ✅ 2.3→1.5, −1.8→−0.5, in-range value untouched |
| The eight-cell grid (2 pairs × 2 step counts × 2 `kappa`-clamp settings) | Establishes a baseline | 2026-09-03 | 8 renders | `corrector/superdiff/parity/grid/`, all eight scored (detector + eye) | ✅ done, reversed result (see below) |
| The per-step prediction hook, verified on one render | Establishes a baseline | 2026-09-03 | 1 render (shared with task 1.2, above) | one norm per step of `eps_M`, written to that render's `.json` sidecar | ✅ 50 values recorded |
| The κ sweep at 50 steps (task 4 as first written; superseded by the row below) | Supplementary figure | 2026-09-03, killed at 30 of 80 | 30 renders kept | `corrector/superdiff/kappa_sweep/`, cat×dog complete (20), butterfly×meadow 10 of 20 | ⏹ stopped on purpose: 50-step figures dropped from the design |
| The κ × λ figure set, 200 steps: cat×dog at six κ settings, butterfly×meadow at κ=0.5; each figure seeds 9–12 by λ ∈ {0, 0.25, 0.5, 0.75, 1} | Supplementary figures | 2026-09-03, `nohup` pid 2505673 on mscluster85, log `corrector/superdiff/lambda_sweep.log` | 140 renders, ~5h estimated (107s per λ=0 cell, ~136s per λ>0 cell) | `corrector/superdiff/lambda_sweep/` (140 renders, `sweep_summary.json`), seven sheets under `paper/iclr/figures/how-much-is-added/across-composition-rules/`, each with a sidecar and a README entry | ✅ complete, 4.94 h, mean 127 s per cell |

The finding this review feeds: [does superdiff compose at its own defaults](../../../report/is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md).

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ⚠️ **Does SuperDiff still compose at 50 steps, and does clamping `kappa` change the answer?**
      Read across each pair's row: does cutting from 200 to 50 steps break it? Read clamped against
      unclamped at 50 steps: was any failure the step count, or the estimator's unclamped `kappa`?
      Answered by the detector and by eye, on both pairs, at both step counts, both clamp settings.

      **Reversed from what was expected: SuperDiff composes better at 50 steps than at 200,
      for both pairs.** The detector's validated contract only covers animal pairs
      (`scorer_validated.json`), so only the `a_cat__x__a_dog` row below is a validated read;
      the `a_butterfly__x__a_flower_meadow` row is the detector's raw output, reported but not
      treated as ground truth (see the artefact check below).

      | Pair | Steps | `kappa` | Detector (n instances, verdict) | Eye |
      |---|---|---|---|---|
      | cat×dog | 200 | clamped | 1, blend | ❌ only a cat, no dog visible |
      | cat×dog | 200 | unclamped | 1, blend | ❌ only a cat, no dog visible |
      | cat×dog | 50 | clamped | 2, **compose** | ✅ dog-shaped silhouette filled with tiny cat figures |
      | cat×dog | 50 | unclamped | 1, blend | ✅ same dog-silhouette/cat-texture pattern as the clamped cell above — **detector and eye disagree here** |
      | butterfly×meadow | 200 | clamped | 1, blend (query mismatch, not validated) | 〰️ photorealistic meadow, one faint blurred butterfly-like shape |
      | butterfly×meadow | 200 | unclamped | 1, blend (query mismatch, not validated) | 〰️ same, faint blur |
      | butterfly×meadow | 50 | clamped | 1, blend (query mismatch, not validated) | ✅ meadow with a clearly distinct butterfly shape |
      | butterfly×meadow | 50 | unclamped | 1, blend (query mismatch, not validated) | ✅ meadow with clearly distinct butterfly shapes (two visible) |

      Per this project's practice, the eye is cited where the two disagree (cat×dog, 50,
      unclamped): recorded as composing.

      **The `kappa` clamp made no visible or detected difference in any of the 8 cells.**
      Consistent with the earlier finding that its [−0.5, 1.5] bound never actually triggered
      for these two pairs at these step counts — the clamp exists and is verified (see task 1.3
      above), it just wasn't the thing controlling the outcome here.

      **What actually changed the outcome was step count, in the opposite direction the plan's
      falsify condition anticipated.** At 200 steps (the pipeline's own default, the "safer"
      setting), both pairs collapsed toward a single dominant concept: the cat×dog renders show
      only a cat, the butterfly×meadow renders show a photorealistic meadow with the butterfly
      nearly invisible. At 50 steps, both pairs show a clear two-concept blend. This falls
      outside the plan's pre-registered Pass/Fail/Inconclusive categories, which all assumed
      cutting steps could only hurt composition, never help it — recorded here as its own
      category: **reversed**.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ⚠️ Is `eps_M` available at every step of at least one render, verified by forming `r_t^SD`
      and printing its per-step norm? A wrapper that returns only a finished image cannot supply
      the shared amount axis, and finding that out inside the grid is expensive.
      **Yes.** A 4th conditioning row (a single CFG branch on the literal joint prompt) rides
      in the same batched UNet call at every step; `r_t^SD = eps_J - eps_M` formed on the actual
      trajectory, 50 values, none zero or NaN, rising from 8.3 to 17.6 across the render.
      `kappa`'s bound was checked separately, directly against the clamp expression on a
      synthetic out-of-range tensor (2.3 → 1.5, −1.8 → −0.5, an in-range value passed through
      unchanged), since neither test pair naturally produced a raw `kappa` outside [−0.5, 1.5]
      even down to 3 steps (checked at 3, 5, and 10 steps, both pairs, unclamped: min −0.432,
      max 1.373 across the sweep) — worth noting as its own finding: this estimator may simply
      not spike for these two pairs at these seeds, which the clamped/unclamped grid cells will
      settle either way. The warmup half of the clamp (forcing `kappa`=0.5 for the first ~10% of
      steps) is verified organically: it fires in every clamped render.
- [x] ⚠️ Does either pair compose at 200 steps at all? If neither pair composes at its easiest
      setting (200 steps, clamped), the pairs are wrong for this check rather than the method being
      at fault, and one more pair is tried before anything is recorded.
      **No, by the validated detector on cat×dog (n=1, blend), and no by eye on either pair**
      (cat×dog shows only a cat; butterfly×meadow shows the butterfly nearly invisible). This is
      not read as "the pairs are wrong": both pairs compose cleanly at 50 steps on the same seed,
      so the pairs themselves are fine, it is specifically the 200-step setting that collapses.
- [x] ⚠️ Did the checkpoint download land under `/datasets` rather than `/home-mscluster`? The home
      filesystem has hit 100% once here and silently killed checkpointing.
      **Yes.** `SUPERDIFF_HF_CACHE` defaults to
      `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/hf_cache`;
      confirmed nothing landed under `$HOME`.
- [ ] ⚠️ Which branch did
      [the measurement at step 26](03-what-is-left-once-the-chain-settles.md) fire? The
      runs in this plan are the same either way, though the sentences around them are not.
- [x] ⚠️ How long did the first render take? No wall-clock cost for this pipeline was measured
      anywhere in this repo before this plan; recorded here so the remaining 7 renders can be
      costed against it.
      **26.7s for the sampling loop itself at 50 steps** (measured inside the render, excludes
      model load), on mscluster85's RTX 3090 (fp16, 1024²), once the checkpoint is cached.
      First-ever call (cold cache) took 267s, almost entirely the one-time ~7GB download. At this
      rate, 200 steps should cost roughly 4× the loop time (~107s), and the remaining 7 grid cells
      (4 at 50 steps, 3 at 200) fit comfortably in-session; no `nohup`-outside-Slurm launch needed
      for this plan.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- [ ] 🟡 Why does more denoising steps make this pipeline *less* compositional? One candidate:
      the estimator settles onto whichever concept the model finds a locally-stronger score
      basin near, and more steps gives it more chances to fully commit, while a coarser 50-step
      discretization never fully escapes the shared region between the two concepts. Untested
      here; would need `eps_M`'s per-step norm trend (already recorded, 200-step vs 50-step)
      read against where in the run each render's image stopped changing.
- [ ] 🟡 Is the butterfly×meadow row's real compose behaviour being hidden by the detector's
      animal-only query? A validated detector for that pair type (querying "butterfly" and
      "flowers" or similar, rather than "animal") would settle whether the eye read there
      (50-step composes, 200-step nearly doesn't) is real or an artefact of only one pair type
      having a validated scorer.
- [x] 🟡 Only one seed (9) was tested per pair. Does the 200-steps-collapses pattern hold on
      other seeds, or was seed 9 specifically unlucky for the 200-step setting on both pairs?
      **It holds on all four seeds for cat×dog.** The first κ × λ sheet
      (`cat_dog_grid_200_steps_kappa_balanced`, seeds 9–12, eye read only, sheet designed after
      the grid result so nothing here was pre-registered): at λ=0, SuperDiff alone at its own
      κ, every seed is a single blended animal. At λ=0.25 still one animal on all four. At
      λ=0.5 all four seeds show two separate animals. At λ=0.75 and λ=1 two animals, and those
      two columns are near-identical, so the trajectory has reached the joint prediction by
      0.75. The one-to-two switch sits between λ=0.25 and 0.5 on every seed, the same shape as
      this project's own PoE strength grid
      (`paper/iclr/figures/how-much-is-added/strength-grid-with-three-controls-and-compose-curve.png`).
      **Forcing κ=0.5 (`cat_dog_grid_200_steps_kappa_050`) changes little.** λ=0 and λ=0.25
      are one animal on all four seeds. Seeds 11 and 12 switch to two animals at λ=0.5 as
      before; seeds 9 and 10 switch one column later, at λ=0.75 (seed 9's λ=0.5 tile is two
      heads pressed into one body). Consistent with the pipeline's own κ sitting near 0.5 for
      most of the run anyway (its per-step values in the sidecars run roughly 0.4 to 0.7 after
      the first steps), so forcing it there is close to a no-op.

      **The easy pair separates one column earlier
      (`butterfly_meadow_grid_200_steps_kappa_050`, seeds 9–12, eye read).** At λ=0 the
      butterfly is present on every seed but dissolved into the meadow: small, camouflaged,
      seed 10 a smear. At λ=0.25 it is a distinct butterfly on all four seeds. From λ=0.5 it is
      a large, sharp, foreground butterfly, and the 0.5, 0.75 and 1 columns are near-identical.
      So SuperDiff-alone at 200 steps does not lose the second concept here, it shrinks it into
      the background; a quarter of the residual pulls it forward. This matches the 8-cell grid's
      200-step eye read for this pair (butterfly nearly invisible) and extends it to four seeds.
      **κ forced to 0 (`cat_dog_grid_200_steps_kappa_000`) is the pure prompt-2 ("a dog")
      blend, and it separates earlier but with the wrong identity first.** At λ=0 every seed is
      a plain dog, no cat anywhere. At λ=0.25 seeds 9, 10 and 12 show a second animal that is
      itself a dog (seed 11 still one dog). At λ=0.5 all four seeds have two animals and the
      second has turned cat-like; at 0.75 and 1 it is unmistakably a cat beside a dog. So the
      residual adds a second subject before it fixes that subject's identity, and this κ reaches
      "two subjects" a column earlier than balanced κ does.

      **κ forced to 0.25 (`cat_dog_grid_200_steps_kappa_025`) sits between κ=0 and balanced.**
      λ=0 is a plain dog on all four seeds. At λ=0.25 seed 10 already shows a clear cat beside
      the dog and seed 12 a faint second animal; seeds 9 and 11 are still one dog. At λ=0.5
      seeds 10, 11 and 12 have two animals; seed 9 needs λ=0.75. From 0.75 all four are cat
      beside dog.

      **κ forced to 0.75 (`cat_dog_grid_200_steps_kappa_075`) mirrors κ=0.25.** λ=0 is a plain
      cat on all four seeds (at high κ the blend leans to prompt 1, at low κ to prompt 2), still
      one cat at λ=0.25, two animals on seeds 9, 11 and 12 at λ=0.5, all four by 0.75. Across
      every κ so far the residual is what separates the two animals; κ only decides which single
      animal appears when there is one.

      **κ forced to 1 (`cat_dog_grid_200_steps_kappa_100`) mirrors κ=0.** Plain cat at λ=0
      and 0.25 on all four seeds. At λ=0.5 seeds 9, 11 and 12 show two animals, seed 11's
      second one still cat-like (it becomes a dog by 0.75), seed 10 still one cat. From 0.75 all
      four are cat beside dog. Same "second subject first, identity second" order as κ=0, from
      the other end.

      **All seven sheets together (140 renders, 4.9 h on mscluster85):** at 200 steps SuperDiff
      alone (λ=0) never composes cat×dog on any of the 4 seeds at any of the six κ settings,
      and shrinks the butterfly into the meadow on the easy pair. Adding back its own missing
      residual `r_t^SD` separates the two concepts on every seed by λ=0.75 for cat×dog and by
      λ=0.25 for butterfly×meadow, at every κ. κ moves which single animal appears at λ=0 (dog
      at κ≤0.25, cat at κ≥0.75, a blend in between) and shifts the separating λ by at most one
      column; it never substitutes for the residual.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ⚠️ **Was the comparison fair?** Only step count and the `kappa` clamp differ across the eight
      renders. Same pairs, same seed, same model, same guidance, same prompt handling.
      **Yes.** Same `cell.seed=9` and `torch.cuda.manual_seed(9)` reseeding before every render,
      same `guidance_scale=7.5`, same pipeline instance (module-cached), same `_prepare_prompt_input`
      path. Determinism was checked once, separately, before the grid ran: the smoke-test call
      through `_run_sampling_loop` directly and the full `run()` composer call, same pair/seed/
      steps, produced the same image. The grid script itself reused task 1.1's cached
      `a_cat__x__a_dog`/50-step/clamped file rather than re-rendering it (by design: `run()`
      skips a cell whose image and sidecar already exist), so that particular cell is the same
      file counted twice, not an independent repeat.
- [x] ⚠️ **Was the measuring tool sound?** The scorer is the validated instance-count detector, read
      over all eight renders and no others, with the eye verdict recorded beside it.
      **Only for the cat×dog row.** The detector's validated contract
      (`scorer_validated.json`) queries a fixed generic `"animal"` term and was built and checked
      only on animal pairs. For `a_butterfly__x__a_flower_meadow` this is a query/subject
      mismatch (a flower meadow is not an animal), so those four `n=1, blend` readings are
      reported as the scorer's raw output, not treated as validated ground truth. The eye read
      is what's actually cited for that pair, per this project's practice.
- [x] ⚠️ **Did the run respect the environment?** Renders and weights under `/datasets` with the
      disk guard on the filesystem actually written to, and no relative path in a launch line onto
      a node this session is not on.
      **Yes.** All 8 renders and their sidecars under
      `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/parity/grid/`;
      the checkpoint cache under the same `/datasets` tree (checked above). Ran in-session on
      mscluster85, no cross-node launch involved.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| any comparison against SuperDiff | that at 50 steps (this project's setting) it composed on both tested pairs, and at its own 200-step default it did not — the reverse of the usual concern that cutting steps breaks a method |
| SuperDiff's sampling | that this pipeline has no DDIM mode, so its stochasticity is not controllable the way this project's other composers are |
| SuperDiff as a baseline | that the grid was two pairs at one seed each, which bounds how much it can say, and that only one pair (cat×dog) has a validated detector reading |
| the `kappa` clamp | that it made no visible or detected difference at either step count for these two pairs — the step-count effect is not an unclamped-estimator artefact, at least not one this clamp catches |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- 🟡 Why more steps makes this pipeline less compositional is unexplained (see "Asked after the
  result"). Worth a real investigation before this becomes a caption sentence, since "SuperDiff
  works better at fewer steps" is a strong enough claim that a reviewer will want a reason, not
  just an observation.
- 🟡 No validated detector exists for the butterfly×meadow pair type, so that row's eye-only read
  is the best evidence available right now.
- 🟡 The 200-step collapse holds on seeds 9–12 for both pairs (κ × λ sheets): cat×dog loses
  the second animal outright, butterfly×meadow shrinks the butterfly into the background. Both
  recover with a fraction of `r_t^SD` added back (half for cat×dog, a quarter for
  butterfly×meadow). The forced-κ sheets will say whether κ moves those switch points.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

The eight-cell grid is done and produced a real, unexpected finding (composes at 50 steps,
collapses at 200) rather than the anticipated one. Before this becomes a paper sentence, decide
with the user whether to chase the "still open" items above, or take the finding as-is to
[step 29](06-three-rules-on-one-amount-axis.md) and the `kappa`-sweep figure (task group 4 of the
design), both of which are unblocked now that `eps_M` and the clamp are verified working.
