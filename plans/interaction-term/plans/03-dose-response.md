# 💉 The dose experiment: inject the cached correction

## Description
Inject the cached true correction into plain PoE sampling at rising strength,
λ in {0, 0.25, 0.5, 0.75, 1}, same seed, and score each output. Three rows:
the true correction, a size-matched random vector, another pair's correction.

## Purpose
The paper's headline causal claim (Goal 1): composition failure is the absence
of this term. Dose response with a flat random control proves the fix is the
term's direction, not perturbation energy. Serves DoD 3.

## Goal
The three-curve figure (compose-rate vs λ) over the held-out animal pairs with
the five-image strip, plus the scored outputs on disk for plans 05 and 06.

## Environment Facts This Plan Depends On
- Cached residuals and pinned init latents per cell (training_cache).
- In-session smoke on the 3090 (SDXL inference fits; check nvidia-smi for
  co-tenants first); full sweep as a job, biggpu first, else bigbatch.
- Disk guard: outputs to /datasets, abort at 90% full.
- W&B: log the Mono vs PoE vs corrected triptych per cell.

## Tasks
- [x] ✅ injection script + the two control rows
      The injection half was already done by plan 00
      (scripts/interaction_term_inject.py over run_teacher_residual).
      The CONTROLS were not possible: the sampler always computed its own
      delta, with no way to inject a different vector. Added
      delta_substitute={"random","wrong_pair"} to run_teacher_residual
      2026-08-05.
      Three things the hook gets right, each of which would have voided the
      control if missed:
        - both substitutes are NORM-MATCHED to the true delta per step, so a
          difference in compose rate cannot be explained by injecting more or
          less magnitude
        - the recorded delta_norm and the PMI identity keep using the TRUE
          delta: those are properties of the pair, not of what we inject
        - the lam==1.0 shortcut (eps_t = eps_j) is disabled when a substitute
          is active. That identity only holds for the real delta; leaving it
          would have made the control row silently reproduce the oracle row.
      Verified: oracle / random / wrong-pair all reach different final latents
      (max |diff| 2.33, 2.21, and 2.67 between the two controls), while
      delta_norm_per_step reads 9.53 in all three.
      The 8 canaries still pass after the sampler edit.
- [x] ✅ λ=0 canary: done by plan 00. Not "byte-identical to plain PoE
      regeneration" as worded: run_cfg_poe batches 3 UNet branches where this
      sampler batches 4, and the same UNet returns different numbers per batch
      shape (~2e-3/step, compounding to 0.6 over 50 steps). The canary compares
      against the sampler's own saved eps_poe instead. 8 tests, each shown to
      fail against a mutated sampler.
- [x] ✅ one-seed smoke in-session, all three rows at lambda=1 on
      a_cat__x__a_dog seed 9 (20 steps). Scored: oracle 100%, random 0%,
      wrong_pair 0%. Eyeball agrees with the scorer: both controls still show
      ONE blended animal (cat ears and whiskers on a dog muzzle) where the
      oracle gives two separate animals. The controls are real, not cosmetic.
- [x] ✅ full sweep: 8 held-out pairs × 4 seeds × 5 λ × 3 rows = 480 cells.
      Finished 2026-08-06T03:04:25Z on mscluster109 (A6000), 397 generated, 83
      skipped as already present, 0 failed. Not a Slurm job: biggpu allows one
      job per user. lambda=0 is sampled once and shared across rows (nothing is
      injected, so the three rows are the same image there).
      ✓ verified (440 images, oracle compose-rate 7% at λ=0 rising to 93% at
      λ=1, random 9% and wrong_pair 6% at λ=1; AUC 0.422 / 0.059 / 0.071;
      commit a21ac8b)
- [x] 🟡 score everything with the validated compose-scorer. Ran, and the
      three-curve result stands, but two things about the instrument are open
      and both are in the scorer, not the experiment:
        - the scorer globs the whole dose tree, so λ=0 and λ=1 are scored over
          44 cells while λ=0.25/0.5/0.75 use 32. The extra 12 are earlier smoke
          cells at seeds 1 and 42. The curve endpoints and the middle are not
          computed over the same set.
        - the count is not a count: on one two-cat image it reports 3 instances,
          the third being a 162px sliver at confidence 0.309, just over the 0.30
          floor. Recorded in commit a21ac8b.
      Next action: re-score with the root pinned to this sweep's seeds and a
      confidence floor chosen against the sliver case, then re-read the curves.
      The direction of the result is not in doubt: both controls sit near the
      floor at every dose.
- [x] ◑ three-curve figure plus strip. dose_curves.png written. The strip is one
      pair and seed (a_leopard__x__a_jaguar seed 9) and was generated while the
      sweep was still partial. Owed: the strip regenerated on complete cells, and
      the final figure candidates from plan 10's /design-figure pass.
- [ ] ⚠️ move the sweep outputs off /home-mscluster. run_dose_sweep.sh sets
      `OUT=$REPO/outputs/...`, so 3.4GB of cells landed in the home repo while
      the script's own disk guard checked /datasets. docs/ENVIRONMENT.md says
      large artifacts go to /datasets only, because /home-mscluster hit 100% once
      and silently killed checkpointing. Move the tree, repoint the script, and
      make the guard check the filesystem actually being written to.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| dose sweep, mscluster109, log `results/mechanism_study/dose_sweep.log` | Tests the claim | commit a21ac8b | `outputs/interaction_term/dose/pairs` (440 images, 3.4GB, wrong filesystem) | done |
| one-seed smoke, a_cat__x__a_dog seed 9, 20 steps | Tests the claim | before a21ac8b | folded into the sweep tree | done |
| scoring pass, grounding-dino-tiny instance_count | Tests the claim | commit a21ac8b | `/datasets/.../interaction_term/dose/dose_curves.json` | done, instrument re-score owed |

Every run in this plan is a claim-testing run, so a failure of the pre-registered
bar would close this plan and open a follow-on. None failed.

## Success/Failure Outcomes
- **λ=0 canary**
  - Success: max abs latent delta vs plain PoE below 1e-5.
  - Failure: nonzero delta means the harness contaminates the base path. Stop
    and fix before any sweep; this is the contamination canary, not a
    formality.
- **full sweep job**
  - Success: 480 scored images; oracle curve rises with λ; random row flat.
  - Failure: OOM on the 3090 (move to biggpu), or scorer and eyeball disagree
    (Goal 1's inconclusive arm: fix the instrument, rerun, never loosen the
    threshold).

## Illustrations
*(image not yet generated)*

**Prompt for image generation:**
> Generate an image of a flowchart showing this experiment: load cached
> correction, verify the λ=0 canary, smoke one seed, submit the full sweep,
> score images, plot three curves. Success path green with checkmark
> "Completed" pills. Failure path red on the canary stage labeled "harness
> contaminates base path" with an X icon and a dashed "Retry Stage"
> troubleshoot callout. Downstream stages muted gray with "Skipped" pills.
> Glossy, minimalistic, modern UI/UX dashboard panel, dark background, rounded
> rectangle stage cards in a horizontal row connected by directional arrows,
> clean sans-serif labels, generous spacing, no clutter.

## Recommended skill
▶ `/run-experiment` ✅ for the sweep job; `/demonstrate` ✅ for the smoke strip.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY -m poe_repair.experiments.interaction_term.inject \
  --pair a_cat__x__a_dog --seed 9 --lambda 0 --check-canary
# expect: "canary ok, delta < 1e-5"
find outputs/interaction_term/dose/pairs -name "*.png" | wc -l
# expect: 440 images (480 cells, the 40 duplicates being the shared λ=0 row)
# NOTE: this path is on /home-mscluster, which is the open task above.
$PY scripts/plot_dose_curves.py     # prints per-row AUC, writes the figure
```
