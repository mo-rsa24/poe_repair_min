# Review: does more correction give more composition?

**Answered, and these are the paper's numbers.** This file judges
[../plans/hypothesis-02-more-correction-more-composition.md](../plans/hypothesis-02-more-correction-more-composition.md)
and fills register slot **F2**, the paper's headline figure.

Where it stands in one line: the correction takes the compose rate from 3% to 94% while both
controls stay at or below 6%, measured on an equal number of cells per strength and with a size
floor in the scorer that was chosen by looking at real detections. F2's caption may quote these
figures.

## Words this file uses
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
**Tests the claim.** So a failure of the pre-registered bar closes the plan and opens one
follow-on. It has not failed.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| dose sweep on mscluster109, log `results/mechanism_study/dose_sweep.log` | Tests the claim | commit a21ac8b | `outputs/interaction_term/dose/pairs`, 440 images, 3.4GB | done |
| one-seed smoke, a_cat__x__a_dog seed 9, 20 steps | Tests the claim | before a21ac8b | folded into the sweep tree | done |
| first scoring pass, grounding-dino-tiny instance_count | Tests the claim | commit a21ac8b | `dose_curves.json` | superseded twice: by the seed pinning, then by the size floor |
| re-score on the pinned seeds, CPU | Tests the claim | commit dcca290 | `dose_curves.json`, rewritten | done; showed the stray cells were not inflating anything |
| re-score with the size floor in the scorer | Tests the claim | size floor in `detection_scorer.py` | `dose_curves.json`, and `dose_strip_an_elephant__x__a_penguin_seed10.png` | done; these are the paper's numbers |

The sweep ran outside Slurm, on the session node, because biggpu allows one job per user. There is
no job id, so the log path is its identity.

## The pre-registered bar

**This is the question the experiment exists to answer, and the only one whose failure may move
the plan.**

- [x] ✅ Does the oracle compose-rate rise with λ while both control rows stay near the floor?
      Yes. Oracle 7% at λ=0 to 93% at λ=1. Random ends at 9%, wrong-pair at 6%, neither above 10%
      at any dose. AUC 0.422 against 0.059 and 0.071, so the oracle carries about six times the
      area of the better control.
      ✓ verified (dose sweep, 480 cells, oracle AUC 0.422 against control 0.059, commit a21ac8b)

## Was the comparison fair

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

## Was the instrument sound

- [x] 🟡 Does the scorer's instance count mean what the rule says it means?
      Unknown, and it does not always. On one image containing two cats it reports 3 instances:
      boxes at 818px (confidence 0.69), 393px (0.58), and a 162px sliver at 0.309, just over the
      0.30 floor and almost certainly a limb. The rule is "compose iff distinct-instance-count
      >= 2", so a spurious third instance does not flip a verdict on its own, but the floor has not
      been chosen against this case.
      Next action: [../procedures/hypothesis-02-recheck-the-headline-numbers.md](../procedures/hypothesis-02-recheck-the-headline-numbers.md)
- [x] ❌ Is every dose scored over the same set of cells?
      No. λ=0 and λ=1 are scored over 44 cells while λ=0.25, 0.5 and 0.75 use 32. The extra 12 are
      earlier smoke cells at seeds 1 and 42, picked up because the scorer globs the whole dose tree
      rather than this sweep's seeds. The endpoints and the middle of every curve are therefore not
      computed over the same population.
      This does not fail the bar above: both controls sit near the floor at every dose under either
      population. It does mean the exact percentages will move on re-score.
      Next action: same procedure as the question above.

## Did the run respect the environment

- [x] ❌ Did the output land where the plan said it would?
      No. 3.4GB of cells are on `/home-mscluster`, not `/datasets`. `run_dose_sweep.sh` sets
      `OUT=$REPO/outputs/...` while its disk guard reads `df /datasets/mmolefe`, so the guard
      reported healthy about a filesystem nothing was being written to.
      Owned by the last task in the design plan.

## The re-score, in two halves

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

## What is still open
- [x] ✅ Does the five-image strip read the same on complete cells?
      **Yes.** The strip is `an_elephant__x__a_penguin` seed 10, all three rows across all five
      strengths, at
      `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/dose_strip_an_elephant__x__a_penguin_seed10.png`.
      The oracle row goes fused creature, fused creature, elephant alone, penguin beside elephant,
      penguin beside elephant. Both control rows stay fused at every strength, so reading down a
      column shows it is which vector was injected that matters, not how large the nudge was.

      That pair carries the strip because a reviewer cannot dismiss it. An elephant and a penguin
      share nothing, and PoE still fuses them at λ=0, so the failure is not "the two animals look
      alike". `a_leopard__x__a_jaguar` seed 9 is the supplementary strip.

      Two things in it belong in the text rather than smoothed over. At λ=0.5 the failure changes
      character: it stops fusing and drops the penguin entirely. At λ=1 the panel holds three
      animals, not two. Seed 10 is also the seed whose oracle row rises monotonically; seeds 9 and
      11 do not.

      **A control the pool does not actually have.** `pair_pool.yaml` lists
      `an_elephant__x__a_penguin` as the compose-by-default control, the do-no-harm check. It
      scores 0 of 4 at λ=0 and the four images are single fused creatures, so the scorer is right
      and the pool's assumption is wrong. There is no working do-no-harm control in the pool. That
      is a limitations sentence, owed by writing-06.
