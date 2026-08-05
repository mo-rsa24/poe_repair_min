# Instrument smoke: interaction-term scope

Recorded outputs from plan `plans/interaction-term/plans/00-build-the-instruments.md`.
Every number below was produced by running the command shown, on
`a_cat__x__a_dog` seed 9 unless stated. Node mscluster85, RTX 3090, co3 env.

Date: 2026-08-05.

## Cache smoke: PASS

```
$ python scripts/cache_smoke.py --all
70/70 ok   (790 cells, 38324 step files)
pair directories: 58 heldout + 18 train = 76, but 70 distinct slugs
cached under both splits (6): a_butterfly__x__a_flower_meadow, a_cat__x__a_lion,
a_dog__x__a_horse, a_lion__x__a_dog, a_tiger__x__a_dog, a_wolf__x__a_husky
```

Zero unreadable files, zero missing keys, zero NaN or Inf across 38,324 step
files. Every file carries the four eps keys at `[1,4,128,128]` fp16.

**Correction to the plan's environment facts.** The cache holds **70 distinct
pairs**, not 76. There are 76 pair *directories* (18 train + 58 held-out), but
six slugs are cached under both splits. Anything that averages "per pair" over
the directory listing double-counts those six. Plan 01 should say 70.

## The r_t loader: PASS

```
$ python -c "from poe_repair.experiments.interaction_term import cache; ..."
pair=a_cat__x__a_dog seed=9 split=heldout steps=50 w=7.5
r_t shape: (50, 1, 4, 128, 128)
||r_t||: first=8.81  min=8.81  max=100.44  last=26.53
log-SNR: -5.15 -> 6.37
```

### Agreement with the sampler's definition of r_t

Checked against the sampler-written residuals in
`artifacts/diagnostics/residual_diagnostics/delta_structure/.../seed_49/`, which
store both the raw eps and the `delta` the sampler computed from them:

| step | sampler ‖delta‖ | loader ‖r_t‖ | relative difference |
|---|---|---|---|
| 0 | 6.839 | 6.838 | 2.47e-02 |
| 10 | 19.356 | 19.356 | 8.25e-03 |
| 25 | 51.067 | 51.068 | 3.05e-03 |
| 40 | 47.876 | 47.877 | 3.11e-03 |
| 49 | 30.054 | 30.053 | 4.24e-03 |

The formula is identical, confirmed exactly: recomputing in fp16 the way the
sampler did reproduces the stored `delta` with **zero** error. The differences
above are the loader upcasting to fp32 first, which is the more accurate path.

The gap is fp16 cancellation, and it is worth knowing about because it bounds
what any cache-derived number can claim. r_t is a small difference of large
quantities (mean |r_t| = 0.020 against eps of order 1), so fp16 rounding is
amplified. It is largest at step 0 where ‖r_t‖ is smallest.

The plan's success criterion said "matches to fp16 round-trip precision". At
2.5% on the noisiest step that criterion is not met as literally written, and
the honest reading is: same formula, different rounding, fp32 is the better
number. Recorded rather than relaxed.

## The two canaries: PASS (8 tests)

```
$ python -m pytest tests/test_interaction_term_canaries.py -q
........                                                    [100%]
8 passed in 61.22s
```

### Why these are not "bit-exact against run_cfg_poe"

The obvious test, running `run_teacher_residual` at λ=0 and demanding equality
with `run_cfg_poe`, **fails**, and not because of a bug.

`run_cfg_poe` batches three UNet branches (A, B, ∅). `run_teacher_residual`
batches four (A, B, J, ∅). The same UNet with identical inputs returns
different numbers at batch 3 and batch 4:

```
eps_a       batch-3 vs batch-4    max|diff| = 1.953e-03
eps_b                             max|diff| = 1.953e-03
eps_uncond                        max|diff| = 1.953e-03
same batch shape, run twice       bit-identical
```

cuBLAS picks different kernels by shape. Over 50 steps that compounds to 0.635
in the final latent. No amount of correct injection code makes that comparison
exact, and a tolerance loose enough to pass would be far too loose to catch a
real leak. Every canary therefore holds the batch shape fixed at four branches
and varies only λ, which is the thing under test.

### Proof the canaries can fail

Two deliberate mutations, each reverted afterwards.

**Mutation 1: window bounds `or` → `and`** (window no longer suppresses λ).

| test | result |
|---|---|
| `test_window_outside_range_contributes_nothing` | **FAILED** |
| `test_window_inside_range_does_inject` | **FAILED** |
| other 5 | passed |

Caught by exactly the two window tests, nothing spurious.

**Mutation 2: `eps_t = eps_poe + 1e-3 * delta` at λ=0** (a 0.1% dose leak).

This one is the reason there are eight tests instead of six. Six of them
compare one sampler run against another, so a leak at λ=0 moves *both* sides
together and all six still pass. `test_lambda_zero_steps_with_exactly_eps_poe`
closes that hole by comparing against the sampler's saved `eps_poe`, which is
written before the injection branch runs:

| reference | reads on mutation | catches it? |
|---|---|---|
| another λ=0 run | identical | no |
| eps_poe recomputed from saved raw eps | 0.4-1.5% (fp16 cancellation) | no, noise is larger |
| **saved `eps_poe`, normalised by ‖delta‖** | **0.09%** | **yes** |

Normalising matters as much as the reference: ‖delta‖ is ~40x smaller than
‖eps_poe‖, so dividing by eps_poe buries a 0.1% leak. Threshold set at 0.02%,
between the clean floor (~0) and the mutation (0.09%).

### The PMI identity

The sampler already checks, every step, that
`Δ_t == w·(ε_J + ε_∅ − ε_A − ε_B)`. That identity is the scope's claim about
what r_t *is*. Measured over a full 50-step run:

| step | relative residual | ‖delta‖ |
|---|---|---|
| 0 | 11.8% | 8.79 |
| 5 | 3.5% | 29.71 |
| 25 | 3.4% | 28.94 |
| 49 | 3.1% | 26.46 |
| median | **3.4%** | |

The residual is largest exactly where ‖delta‖ is smallest, which is the fp16
cancellation signature again, not a broken identity. The test asserts the
median stays under 8%. Plan 05 is where this curve gets read as a result rather
than a guard.

## Still to do in this plan

The analysis scripts (`snr_collapse`, `spectrum`, `climb`, `fork_curve`,
`plot_dose_curves`, `plot_window_curves`, `language_probes`, `quality_control`,
`manifold_slide`, `composition_scatter`) are not yet written. The two CLI
wrappers exist but their non-canary paths (actually sampling at a dose or a
window) have not been run end to end.
