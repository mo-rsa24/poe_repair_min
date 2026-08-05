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

_(Superseded: every script listed here as "still to do" was subsequently
written and smoked. See the sections below.)_

---

## snr_collapse.py: RUNS (12 pairs, 27 curves)

```
$ python scripts/snr_collapse.py --all --max-pairs 12 --max-seeds 2
collapse spread: 44.6%   (12 pairs, 27 curves)
  log-SNR range -5.15 to 6.37 in 20 bins
  peak of the median curve at log-SNR -0.30
  reading: no collapse
```

Two choices in this script that change the number, both deliberate.

Curves are scaled by their own **median**, not their peak. Peak-scaling makes
every curve hostage to one noisy point, and the first version of this figure
showed heavy step-to-step jitter purely from that. Spread is then measured
relative to the median curve's own height, so the headline number does not
depend on the scaling choice.

At 12 pairs this reads "no collapse". That is the instrument reporting, not a
verdict: the sample is small and plan 05 owns the full-cache run. The median
curve does show a clean hump peaking near log-SNR 0, and the interquartile band
is tight through the middle with the spread concentrated in the high-noise
left tail.

## spectrum.py: RUNS (18 train pairs, 901 vectors)

```
$ python scripts/spectrum.py --all --max-pairs 18 --max-seeds 3 --stride 3
train: 901 vectors x 65536 dims from 18 pairs

   k       r_t   gaussian   ratio
   1     3.1%      0.1%   22.3x
   8    15.9%      1.1%   14.4x
  64    51.9%      8.6%    6.0x

excluding 2 pair(s) cached in both splits: a_butterfly__x__a_flower_meadow, a_cat__x__a_lion

held-out energy captured by the TRAIN subspace (816 vectors, 16 pairs):
  k=  8:   0.3%   (train 15.9%)
  k= 64:   2.7%   (train 51.9%)
```

**Read the ratio, never the raw percentage.** With N vectors in D dimensions and
N far below D, the centred stack has rank N-1, so energy-at-k is largely forced
by N alone. Measured on random vectors in 65536 dims:

| N | top-16 energy |
|---|---|
| 30 | 56.0% |
| 100 | 17.1% |
| 300 | 6.0% |

A first smoke run at N=30 reported "92% energy at k=16", which is almost
entirely this artifact. The script now prints the matched Gaussian floor beside
every number and warns when N is small relative to k.

Two guards worth noting: the stack is centred before the SVD (otherwise
component 1 is just the mean, flattering the low-rank claim for free), and
slugs cached under both splits are excluded from the held-out set (otherwise
the "held-out" projection is partly measuring training pairs).

## fork_curve.py: RUNS (1 cell, generated for the smoke)

```
$ python scripts/fork_curve.py --root outputs/interaction_term/dose/pairs
a_cat__x__a_dog seed 9: elbow at step 4 of 21   d(0)=0.00  d(elbow)=19.79  d(end)=135.77
elbow at step 4 (median over 1 cells)
```

`d(0) = 0.00` is the check that matters here: both paths start from identical
noise, so any later distance is the interaction term's doing and not a
different starting point.

This script reads trajectories and does not sample. With none on disk it exits
2 and prints the commands that produce them, rather than failing obscurely.
The two used above were generated at 20 steps for this smoke.

## The phenomenon, seen

Same pair, same seed, same noise. Only the dose differs.

| λ=0 (PoE) | λ=1 (Mono) |
|---|---|
| `outputs/interaction_term/dose/pairs/a_cat__x__a_dog/seed_9/teacher_residual_const_lam000/` | `.../teacher_residual_const_lam100/` |
| one animal: cat ears and whiskers fused onto a dog muzzle and tongue | two animals, a tabby cat and a white dog, sitting side by side |

This is the chimera the scope exists to explain, and the correction closing it,
on one cell at 20 steps. Not a result (one cell, no scorer, short schedule),
but it confirms the instruments are wired to the real phenomenon.

## climb.py: RUNS (10 pairs, 23 cells)

```
$ python scripts/climb.py --all --max-pairs 10 --max-seeds 2
correction size as a fraction of ||eps_PoE||:
  median 11.5%   IQR 7.2% to 19.9%

step-to-step direction agreement (cosine, consecutive r_t):
  median 0.799   IQR 0.294 to 0.937
  random-direction floor for 65536 dims: ~0.0039
```

Two distributions, and the second is the informative one. A correction with a
stable direction is something a low-rank adapter could learn; a thrashing one
is not. At 0.799 median against a random floor of 0.0039, the direction is
strongly structured over time, not noise.

The figure shows correction size climbing from ~5% early to a ~13% plateau, and
direction agreement high throughout and rising near the end.

## plot_dose_curves.py: RUNS (1 cell, 2 doses)

```
$ python scripts/plot_dose_curves.py --root outputs/interaction_term/dose/pairs
scorer: instance_count via IDEA-Research/grounding-dino-tiny
  rule: COMPOSE iff distinct-instance-count('animal', NMS iou<0.5, conf>=0.30) >= 2

  a_cat__x__a_dog seed 9 lam 0.00: 1 instances -> blend
  a_cat__x__a_dog seed 9 lam 1.00: 2 instances -> COMPOSE
```

The scorer agrees with the eye on both images: 1 instance for the chimera, 2
for the side-by-side pair. That is the qualitative and quantitative sides
landing on the same cell.

The script refuses to run unless `scorer_validated.json` says `pass: true`, so
no dose curve can be produced with an unvetted instrument.

## plot_window_curves.py: RUNS (1 cell, 4 windows, 20 steps)

```
$ python scripts/plot_window_curves.py --root outputs/interaction_term/window/pairs
  window   0-5   (centre   2.5): 100%  (n=1)
  window   5-10  (centre   7.5):   0%  (n=1)
  window  10-15  (centre  12.5):   0%  (n=1)
  window  15-20  (centre  17.5):   0%  (n=1)

peak at window centre 2.5, band 2.5 to 2.5 (within one standard error)
```

Two independent instruments agree on the early band: the peak here sits at
window 0-5, and `fork_curve` put the path-split elbow at step 4 of 20. One cell
at 20 steps, so not a result, but the machinery is coherent.

The images show why. Correcting in the first 5 steps produces two separate
animals. Correcting only in the last 5 leaves the chimera essentially
untouched, still one animal with fused cat and dog features. Once the layout is
settled early, a late correction cannot undo it.

## quality_control.py: RUNS (1 cell)

```
$ python scripts/quality_control.py --root outputs/interaction_term/dose/pairs
  a_cat__x__a_dog seed 9: instances 1->2   confidence 0.915->0.769 (-0.146)
  reading: quality cost
```

Do not read the -0.146 as a finding. At n=1 it is one image, and the
comparison is also slightly unfair: the baseline averages one box, the
corrected averages two, and a second animal is usually detected less
confidently than a single dominant one. Plan 06 should compare like with like
(for instance, the best box on each side) before quoting a quality cost.

## language_probes.py: RUNS (20 pairs, L1 and L3)

```
$ python scripts/language_probes.py --probe l1 --probe l3 --max-pairs 20
L1 additivity gap over 20 pairs: median 1.3197, range 0.9406 to 1.5056
L3 shared binding direction over 20 pairs:
  first direction holds 27.1% of residual energy (random floor 6.5%, ratio 4.2x)
```

Both read from `embeddings.pt`, which the cache already stores, so neither
needs the UNet. The L3 number is reported against its own random floor for the
same reason the spectrum is: a small sample concentrates by chance.

## manifold_slide.py: RUNS (1 cell, 5 doses)

```
$ python scripts/manifold_slide.py --root outputs/interaction_term/dose/pairs
  lam 0.00: projection +0.000   off-axis  0.0%
  lam 0.25: projection +0.220   off-axis 86.8%
  lam 0.50: projection +0.333   off-axis 78.9%
  lam 0.75: projection +0.889   off-axis 44.3%
  lam 1.00: projection +1.000   off-axis  0.0%
  monotone in lambda: yes
```

The projection rises monotonically from 0 to 1, so the dose does move the
sample along the PoE-to-Mono axis. The endpoints are 0 and 1 by construction,
which is the check that the axis is built correctly, not a result.

The off-axis fraction is the interesting column: 79-87% in the middle of the
range. The correction moves the sample toward Mono along a curved route, not a
straight line. Worth knowing before anyone describes the dose as a simple
interpolation.

## composition_scatter.py: BLOCKED, by design

Exits with an explanation rather than a plot:

```
no normalization memo at docs/normalization_preregistration.md.
Correction size is not comparable across prompt types without a committed
measure, and choosing one after seeing the plot is how the 95% delta-field
number had to be retracted. Write the memo first: plan 01.
```

It also requires `--groups` (a pair-to-type mapping) rather than inferring
types from slugs, since a wrong grouping would be invisible in the finished
scatter.

## Final check: the plan's own Engagement Instructions

```
$ for f in cache_smoke plot_dose_curves plot_window_curves snr_collapse \
           fork_curve climb spectrum language_probes quality_control \
           manifold_slide composition_scatter interaction_term_inject \
           interaction_term_window; do
    test -f "scripts/$f.py" || echo "MISSING scripts/$f.py"; done
(no output: all 13 present)

$ python -c "...; print(cache.r_t('a_cat__x__a_dog', 9).shape)"
(50, 1, 4, 128, 128)

$ python -m pytest tests/test_interaction_term_canaries.py -q
........                                          [100%]
8 passed in 65.02s

$ python scripts/cache_smoke.py --all
70/70 ok   (790 cells, 38324 step files)
```

Plan 00 is done. Every instrument runs on real data and every headline number
above was produced by the command shown.
