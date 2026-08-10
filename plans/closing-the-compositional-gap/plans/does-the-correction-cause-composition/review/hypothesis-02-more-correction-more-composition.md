# 🔬 Review: the dose experiment

Verdicts for [../plans/hypothesis-02-more-correction-more-composition.md](../plans/hypothesis-02-more-correction-more-composition.md). The design lives there
and does not change because a number arrived here.

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
| scoring pass, grounding-dino-tiny instance_count | Tests the claim | commit a21ac8b | `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/dose_curves.json` | done, re-score owed |

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

## What is still open

- [ ] ⚠️ Do the curves hold after re-scoring on a pinned root and a chosen confidence floor?
      Answer this by running
      [../procedures/hypothesis-02-recheck-the-headline-numbers.md](../procedures/hypothesis-02-recheck-the-headline-numbers.md).
- [ ] ⚠️ Does the five-image strip read the same on complete cells?
      The existing strip is one pair and seed (a_leopard__x__a_jaguar seed 9) and was generated
      while the sweep was still partial.
