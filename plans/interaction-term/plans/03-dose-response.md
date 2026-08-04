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
- [ ] ⚠️ injection script: read residuals/step_*.pt, apply ε_PoE + λ·r_t per
      step, decode, save; rows for norm-matched random and wrong-pair r_t
- [ ] ⚠️ λ=0 canary: output byte-identical to plain PoE regeneration  [inferred]
- [ ] ⚠️ one-seed smoke in-session: the 5-image strip for a_cat__x__a_dog seed 9
- [ ] ⚠️ full sweep as a job: 8 held-out pairs × 4 seeds × 5 λ × 3 rows
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
