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
- [ ] ⚠️ full sweep: 8 held-out pairs × 4 seeds × 5 λ × 3 rows = 480 cells.
      RUNNING on mscluster109 GPU 0 (not a Slurm job: biggpu allows one job per
      user). scripts/mechanism_study/run_dose_sweep.sh, ~50s/cell, ~5.5h,
      resumable. lambda=0 is sampled once and shared across rows (nothing is
      injected, so the three rows are the same image there).
- [ ] ⚠️ score everything with the validated compose-scorer
- [ ] ⚠️ three-curve figure plus strip (candidates come from plan 10's
      /design-figure pass)

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
ls /datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/ | wc -l
# expect: 480 scored cells
$PY scripts/plot_dose_curves.py     # prints per-row AUC, writes the figure
```
