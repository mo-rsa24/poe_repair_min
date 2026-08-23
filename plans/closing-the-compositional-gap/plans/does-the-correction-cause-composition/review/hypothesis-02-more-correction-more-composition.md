# 🧪 Review: does more correction give more composition?

**Answered, and these are the paper's numbers.** This file judges
[../plans/hypothesis-02-more-correction-more-composition.md](../plans/hypothesis-02-more-correction-more-composition.md)
and fills register slot **F2**, the paper's headline figure.

Where it stands in one line: the correction takes the compose rate from 3% to 94% while both
controls stay at or below 6%, measured on an equal number of cells per strength and with a size
floor in the scorer that was chosen by looking at real detections. F2's caption may quote these
figures.

## Recommended prompt (to re-read the numbers)

```
/analyze-figure paper/iclr/figures/F2-dose-response.pdf
```
(To redo the counting and the bars: work through
[the procedure](../procedures/hypothesis-02-recheck-the-headline-numbers.md).)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis-02-more-correction-more-composition.md) | the strength sweep, the two controls, the bar |
| **this file** | **the verdict: 3% to 94% with both controls at or below 6%. F2's caption may quote these** |
| [procedure](../procedures/hypothesis-02-recheck-the-headline-numbers.md) | the steps that produced the re-score, and how to redo it |
| [the register](../../../../../paper/iclr/figures.md) | F2's row, the paper's headline figure |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Asked after the result](#asked-after-the-result)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

The artefact checks sit before the questions the result raised, because that is the order it
happened in: two of those checks failed, and the re-score exists to answer them.

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️
- **A cell**: one picture, for one animal pair, at one strength, from one starting seed.
- **The three rows**: the real correction, a random vector of the same size, and a different
  pair's correction. The last two are the controls, and they are what makes the first one evidence.
- **Compose rate**: the fraction of cells showing two separate animals rather than one blended
  one, decided by the scorer and never by eye.
- **AUC**: the area under a whole curve, one number per row, so three rows compare at a glance.
- **The size floor**: `MIN_BOX_FRACTION = 0.25` in `detection_scorer.py`. A detection must span at
  least a quarter of the image's longer side to count as an animal. This replaced the idea of
  tightening the confidence cutoff, for the reason recorded below: confidence could not separate
  a shadow from a penguin, and size could.

Every question below was written at design time, from that plan's Success/Failure Outcomes, before
the sweep ran. Answers are `✅` good, `❌` bad, `🟡` unknown, `⚠️` not yet answered.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️
**Tests the claim.** So a failure of the pre-registered bar closes the plan and opens one
follow-on. It has not failed.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| dose sweep on mscluster109, log `results/mechanism_study/dose_sweep.log` | Tests the claim | commit a21ac8b | 440 images, 3.4GB | `outputs/interaction_term/dose/pairs`, 440 images, 3.4GB | done |

440 images and 480 scored records are both right and count different things. 416 of the images are
this sweep's own (8 pairs, seeds 9 to 12, 13 runs each); the other 24 sit at seed 1 and the scorer
no longer reads them. The 416 score as 480 records because at λ=0 nothing is injected, so all three
rows share one image and it is scored once per row. That is also why the three rows read exactly
the same rate at λ=0.

| one-seed smoke, a_cat__x__a_dog seed 9, 20 steps | Tests the claim | before a21ac8b | 1 cell | folded into the sweep tree | done |
| first scoring pass, grounding-dino-tiny instance_count | Tests the claim | commit a21ac8b | scoring only, no generation | `dose_curves.json` | replaced, first by the seed pinning and then by the size floor |
| re-score on the pinned seeds, CPU | Tests the claim | commit dcca290 | CPU only, about an hour | `dose_curves.json`, rewritten | done; showed the stray cells were not inflating anything |
| re-score with the size floor in the scorer | Tests the claim | size floor in `detection_scorer.py` | scoring only, no generation | `dose_curves.json`, and `dose_strip_an_elephant__x__a_penguin_seed10.png` | done; these are the paper's numbers |

The sweep ran outside Slurm, on the session node, because biggpu allows one job per user. There is
no job id, so the log path is its identity.

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the question the experiment exists to answer, and the only one whose failure may move
the plan.**

- [x] ✅ Does the oracle compose-rate rise with λ while both control rows stay near the floor?
      Yes. Oracle 3% at λ=0 to 94% at λ=1. Random ends at 3%, wrong-pair at 3%, neither above 6%
      at any dose. AUC 0.387 against 0.023 and 0.039, so the oracle carries about ten times the
      area of the better control. The oracle rises 91 points across the sweep; the better control
      rises 0.
      ✓ verified (32 cells per row-and-strength, seeds 9 to 12, size floor in
      `detection_scorer.py`)

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing beyond the bar and the three artefact checks below. All of them were written before the
sweep ran, which is why two of them could fail and did.

## Could the answer be an artefact

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

**Was the comparison fair?**

- [x] ✅ Does the harness leave plain PoE untouched at λ=0?
      Yes, against the sampler's own saved `eps_poe`. Not against a fresh `run_cfg_poe`
      regeneration: that batches 3 UNet branches where this sampler batches 4, and the same UNet
      returns different numbers per batch shape, about 2e-3 per step compounding to 0.6 over 50
      steps. 8 tests, each shown to fail against a deliberately mutated sampler.
- [x] ✅ Do the two control rows actually inject something different from the oracle?
      Yes. All three reach different final latents, max absolute difference 2.33 and 2.21 against
      the oracle and 2.67 between the two controls, while `delta_norm_per_step` reads 9.53 in all
      three. So the rows differ in direction and not in magnitude, which is what the control is for.
- [x] ✅ Does the eyeball agree with the scorer on the smoke cell?
      Yes. Oracle 100%, random 0%, wrong-pair 0% at λ=1 on a_cat__x__a_dog seed 9. Both controls
      show one blended animal (cat ears and whiskers on a dog muzzle); the oracle gives two
      separate animals. The controls are real, not cosmetic.

**Was the instrument sound?**

- [x] ✅ Does the scorer's instance count mean what the rule says it means?
      It does now, under the size floor. Without one it did not, and the failure was worse than a
      spare box on a correct verdict. On `an_elephant__x__a_penguin` seed 10 the `random` control
      at λ=1 shows one fused creature and scored two boxes: the animal at 888px, and a 220px box
      on a shadow at confidence **0.60**. That flipped a control-row blend into a `compose`, which
      is the worst place for a false positive, because the controls are what the oracle's rise is
      measured against.
      No confidence cutoff separates that case: 0.60 sits above the real penguin's 0.54 on the
      same strip. Size does. Every genuine animal there spans 458px or more and every spurious box
      220px or less, on a 1024px image, so the floor is `MIN_BOX_FRACTION = 0.25`, a quarter of the
      image's longer side. It is the same line `dose_strip.py` already draws when it colours a box
      yellow or magenta, so the diagnostic picture and the scorer agree by construction. The filter
      can only remove detections, so it can only lower compose rates.
      ✓ verified (control λ=1 goes 2 boxes to 1; oracle λ=1 goes 3 to 2 and stays `compose`)

      **The count is reliable. It is still not an identity check, and that costs us.** The rule
      asks a detector for "animal" and counts distinct boxes, so it cannot tell which animals are
      present. On `a_cat__x__a_dog` seed 10 at λ=1 the panel holds **two dogs**: two canine
      muzzles, two black noses, no cat anywhere. It scores `2` and therefore `compose`, and it is
      a composition failure. The pair was requested as a cat and a dog and the model returned two
      of one concept.
      So the 94% endpoint counts an unknown number of same-concept pairs as successes, and the
      true compose rate is at or below it. The direction of the result is unaffected, because a
      control row producing two dogs would score the same way and the controls sit at 3%. The
      magnitude is an upper bound, and the paper must say so rather than quote 94% flat.
      Next action: a per-concept read (query "cat" and "dog" separately, require one box each)
      would measure the gap. Not built. Owed as a limitations sentence by
      `writing-06-mechanism-and-limitations` until it is.
- [x] ❌ Is every dose scored over the same set of cells?
      No. λ=0 and λ=1 are scored over 44 cells while λ=0.25, 0.5 and 0.75 use 32. The extra 12 are
      earlier smoke cells at seeds 1 and 42, picked up because the scorer globs the whole dose tree
      rather than this sweep's seeds. The endpoints and the middle of every curve are therefore not
      computed over the same population.
      This does not fail the bar above: both controls sit near the floor at every dose under either
      population. It does mean the exact percentages will move on re-score.
      Next action: same procedure as the question above.

**Did the run respect the environment?**

- [x] ❌ Did the output land where the plan said it would?
      No. 3.4GB of cells are on `/home-mscluster`, not `/datasets`. `run_dose_sweep.sh` sets
      `OUT=$REPO/outputs/...` while its disk guard reads `df /datasets/mmolefe`, so the guard
      reported healthy about a filesystem nothing was being written to.
      Owned by the last task in the design plan.

## Asked after the result

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

**Nothing here may ever become a bar**, because it was written with the answers already visible.
Two of the artefact checks above failed, and the re-score is what answers them. The steps are in
[the procedure](../procedures/hypothesis-02-recheck-the-headline-numbers.md).

The original question asked about two things at once, so it splits.

- [x] ✅ Do the curves hold when only this sweep's own cells are scored?
      **Yes, and they barely moved.** With the seeds pinned to 9 to 12 in
      `plot_dose_curves.py`'s source, every (row, strength) pair now holds exactly 32 cells,
      where the two end strengths previously held 44 against the middle three's 32. The unfair
      comparison is gone.

      | | λ=0 | λ=0.25 | λ=0.5 | λ=0.75 | λ=1 | AUC |
      |---|---|---|---|---|---|---|
      | real correction | 6% | 16% | 28% | 75% | **94%** | 0.422 |
      | random vector | 6% | 6% | 6% | 3% | 9% | 0.059 |
      | other pair's correction | 6% | 9% | 3% | 9% | 6% | 0.070 |

      Against the earlier contaminated read (7% to 93%, AUC 0.422 / 0.059 / 0.071) these are
      essentially unchanged. **That is the useful part:** the stray cells were not inflating
      anything, so the paper owes no methods sentence about them, which the procedure said it
      would if the curve had moved a lot.
      ✓ verified (32 cells per row-and-strength, seeds 9 to 12, commit dcca290)

- [x] ✅ Do they hold under a bar chosen against the picture of the boxes?
      **Yes, and the claim is stronger under the bar than without it.** The bar is a size floor,
      not a confidence floor: `MIN_BOX_FRACTION = 0.25` in `detection_scorer.py`, meaning a
      detection must span at least a quarter of the image's longer side to count as an animal.
      `conf` stays at 0.30.

      Confidence was the wrong lever. On the `an_elephant__x__a_penguin` seed 10 strip the
      `random` control at λ=1 shows one fused creature and scored two boxes: the animal at 888px,
      and a 220px box on a shadow at **confidence 0.60**, above the real penguin's 0.54 on the
      same strip. No confidence cutoff separates those. Size does: every genuine animal there
      spans 458px or more, every spurious box 220px or less, on a 1024px image.

      | | λ=0 | λ=0.25 | λ=0.5 | λ=0.75 | λ=1 | AUC |
      |---|---|---|---|---|---|---|
      | real correction | 3% | 9% | 25% | 72% | **94%** | 0.387 |
      | random vector | 3% | 3% | 3% | 0% | 3% | 0.023 |
      | other pair's correction | 3% | 3% | 3% | 6% | 3% | 0.039 |

      The floor pulled the controls down far more than the oracle, which is what it should do if
      the controls' non-zero readings were instrument error. The oracle-to-control AUC ratio goes
      from 6.0x to 9.9x, and the endpoint is unchanged at 94%. The oracle rises 91 points from
      λ=0 to λ=1; the better control rises 0.
      ✓ verified (32 cells per row-and-strength, seeds 9 to 12, size floor in source)

      **These are the paper's numbers.** F2's caption may now quote them.

**And the strip that carries the figure.**
- [x] ✅ Does the five-image strip read the same on complete cells?
      **Yes.** F2 is `paper/iclr/figures/F2-dose-response.pdf`: `a_cat__x__a_dog` seed 9, all
      three rows across all five strengths, above the curves on a shared λ axis.
      The oracle row holds one ginger cat-dog chimera at λ=0, 0.25 and 0.5, then a tabby cat
      sitting beside a white labrador at 0.75 and 1. The random row holds that same chimera at
      every strength. The other-pair row degrades into cartoon artifacts and never separates. So
      reading down a column shows it is which vector was injected that matters, not how large the
      nudge was.

      Cat × dog carries the figure because it is the strongest pair in the set: 0% to 100%, AUC
      0.562, and the only pair whose two controls score exactly 0.000. Seed 9 over seeds 10, 11
      and 12 because its two composing panels were checked by eye and hold a real cat beside a
      real dog.

      **The second figure answers the obvious objection.**
      `paper/iclr/figures/F2b-dissimilar-pair.pdf` is `an_elephant__x__a_penguin` seed 10. An
      elephant and a penguin share nothing, and PoE still fuses them into one creature at λ=0, so
      the failure cannot be dismissed as "the two animals look alike". Two things in it belong in
      the text rather than smoothed over: at λ=0.5 the failure changes character, dropping the
      penguin entirely instead of fusing, and seeds 9 and 11 of that pair are not monotone.

      **A control the pool does not actually have.** `pair_pool.yaml` lists
      `an_elephant__x__a_penguin` as the compose-by-default control, the do-no-harm check. It
      scores 0 of 4 at λ=0 and the four images are single fused creatures, so the scorer is right
      and the pool's assumption is wrong. There is no working do-no-harm control in the pool. That
      is a limitations sentence, owed by writing-06.

## What the write-up owes

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| F2's headline numbers, 3% to 94% with both controls at or below 6% | that they are the re-scored figures, computed over this sweep's own cells with the size floor in the scorer. The first scoring pass read cells from outside the sweep, and its numbers are not the ones to quote |
| the compose rate at any strength | the band on it, which [the compose-rate scope](../../can-we-trust-the-compose-rate/review/instrument-01-the-three-state-labelled-set.md) is measuring. 94% is an upper bound until that lands |
| cat × dog carrying the strip | that it is the strongest pair in the set (0% to 100%, AUC 0.562, the only pair whose two controls score exactly 0.000), and that seed 9 was chosen because its two composing panels were checked by eye. A reader should know the strip shows the best case |
| the control pair, elephant × penguin | that it scores 0 of 4 at strength 0 here, which contradicts the transfer scope listing it as a compose-by-default control. See [Still open](#still-open) |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| 3.4GB of cells sitting on `/home-mscluster` rather than `/datasets`, because `run_dose_sweep.sh` sets `OUT=$REPO/outputs/...` while its disk guard reads `df /datasets/mmolefe` | moving them, and pointing the guard at the filesystem the script actually writes to | owned by the last task in the design plan. `/home-mscluster` has hit 100% once before and silently killed checkpointing |
| whether elephant × penguin composes by default | re-scoring that pair under both runs' conditions | the do-no-harm claim. The transfer scope lists it as a compose-by-default control and it scores 0 of 4 at strength 0 here. Recorded on both sides: [instrument-01 of the transfer claim](../../does-the-fix-reach-unseen-pairs/review/instrument-01-the-clean-pair-pool.md) |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Move the 3.4GB off `/home-mscluster` and fix the disk guard to check the filesystem the script
writes to.
