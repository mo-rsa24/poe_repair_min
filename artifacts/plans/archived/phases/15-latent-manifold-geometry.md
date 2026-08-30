# Plan 15 — Latent-manifold geometry of PoE and the LoRA correction

> Companion to [04-lora-single-seed.md](04-lora-single-seed.md). Takes
> over the **load-bearing role** that plan-13's MDS panels were being
> asked (and structurally failing) to play. The MDS panels survive but
> are reframed — see [13-semantic-mds-toggle.md](13-semantic-mds-toggle.md)
> and §6 of this plan.
>
> Scientific frame: Arvanitidis et al., *Latent Space Oddity: On the
> Curvature of Deep Generative Models* — Euclidean distance in deep
> generative latent spaces does not reflect data distance, because the
> data sits on a curved sub-manifold and straight lines in z pass
> through holes (low-density regions). The meaningful distance is the
> Riemannian one under the pullback metric G(z) = J(z)ᵀJ(z), where
> J(z) = ∂x/∂z is the decoder Jacobian.

## Hypothesis (falsifiable)

**PoE-without-LoRA's terminal latents land in low-density regions of
the mono manifold — the "hole" in LSO terms. LoRA acts as a correction
that pulls the trajectory back onto the manifold — the "arc".**

The MDS figure in plan-13 was the right intuition reaching for the
wrong instrument. Raw-Euclidean MDS over noisy z_t measures
pixel-statistic geometry of latents, not concept-region geometry on the
data manifold. The known failure case proves it: at (λ=0.5, ep 900) a
single fluffy cat sits *near* mono in raw-latent MDS; at (λ=1.0,
ep 1600) a chimeric cat–dog blend sits *further* from mono. If raw L2
measured concept overlap, the ordering would be reversed.

## Scope (cross-seed pool, n_bank = 8)

Cat × dog, on the 12 seeds already cached for the cross-seed LoRA
pooling work (Plan 08): 8 train-pool seeds (1–8) and 4 held-out
seeds (9–12). LoRA artefact under test is the **cross-seed** LoRA —
the seed-42 single-seed LoRA from Plan 04 is a separate experiment
and not the object here.

**Mono bank filter.** Mono itself fails the co-occurrence semantics
on four seeds: `{1, 4, 6, 10}`. Their terminal `z_0` would pollute
the reference manifold, so they are excluded from the bank. The bank
therefore contains 8 seeds: `{2, 3, 5, 7, 8, 9, 11, 12}` — a
train-pool subsplit `{2, 3, 5, 7, 8}` (5 seeds) and a held-out
subsplit `{9, 11, 12}` (3 seeds). Mono-on-mono baseline is
leave-one-out across these 8 seeds.

**Evaluation set decision (Open Question 1, deferred).** Whether the
four mono-failed seeds belong in the *evaluation* arms (PoE-no-LoRA,
PoE+λR) is a separate choice. Two readings:
- *Headline figure*: exclude them — evaluate PoE and PoE+λR on the
  same 8 seeds the bank uses. Cleanest comparison; same noise
  samples that mono can land for cleanly.
- *Supplementary panel*: include them — these seeds let us ask "does
  LoRA help on noise samples where the mono ceiling itself failed?"
  This is a genuinely interesting probe but not the headline claim.
Default: exclude from headline, include in a marked supplementary
panel.

This is a deliberate change from the original seed-42 beachhead.
The trade is: a small bump in sample size (n = 8 per condition vs
n = 1) and a free held-out-seed split (seeds 9, 11, 12 unseen at
LoRA train time), at the cost of measuring the cross-seed LoRA
rather than the single-seed one. **No new sampling compute** is
incurred if all three conditions (mono, PoE-no-LoRA, PoE+λR at final
epoch, λ ∈ {0, 0.5, 1.0}) are already cached per-seed. If any
trajectory is missing, that gap is the only thing to backfill — see
Step 1 in the sequence.

n = 8 is small (held-out subsplit n = 3 is *very* small). Use strip
plots with bootstrap CIs and Mann–Whitney U for distribution
comparisons; do not draw smooth histograms. The held-out subsplit
remains a directional check rather than a load-bearing statistical
test.

## What this plan does *not* do

- **Delete the plan-13 MDS panels.** Both modes survive as *illustration*.
  Their captions are rewritten so they no longer claim to measure
  concept distance — see §6.
- **Touch the decoded-image rows.** Those remain the primary deliverable
  on `/` and stay untouched.
- **Train anything.** This is a measurement layer over the existing
  LoRA checkpoints, mono baselines, and cached trajectories.
- **Generalise beyond cat × dog seed 42.**

## Two claims, both load-bearing

The user wants both. They are different objects with different
instruments. The plan instruments both, and §6 explicitly notes which
trajectory-dynamics questions are *unknown unknowns* worth surfacing.

### Claim A — Endpoint claim
PoE+LoRA's terminal z_0 lies on the mono manifold (high density,
small geodesic distance to mono samples). PoE-without-LoRA's terminal
z_0 lies off it (low density, large geodesic distance). This is the
science — directly testable, single-number per sample.

### Claim B — Trajectory claim
PoE traces a high-cost straight line through low-density z-space;
PoE+LoRA traces a lower-cost arc that stays close to the data manifold.
This is the picture — supports Claim A with a time-resolved geometry
and recovers the LSO right-panel intuition for your data.

## §1 — Which manifold, exactly?

The LSO setting is a VAE's prior latent space, where the *data
manifold* is the support of the aggregate posterior — the set of
latents encoded from real images. The decoder Jacobian J(z) gives the
pullback metric G(z), and "off-manifold" means traversing low-density
regions where the decoder behaviour is ill-defined.

Your setting is **not** that directly. Your z_t is the diffusion
latent at noise level t:

- At **t = T** (pure noise): z_T ∼ N(0, I). There is no data manifold
  here — every direction is equally valid. Geometric off-manifoldness
  is undefined.
- At **t = 0** (clean): z_0 is what the VAE would have encoded. The
  LSO geometry applies directly; pullback metric well-defined.
- At **intermediate t**: the relevant density is the data manifold
  convolved with Gaussian noise of variance 1 − ᾱ_t — increasingly
  fattened until at t = T it is the standard prior.

So the **endpoint** claim is geometrically clean (t = 0, LSO regime
applies). The **trajectory** claim is harder and requires a different
formulation — pullback path length under a time-varying metric, with
the caveat that G(z_t) is only well-defined at low t.

This is not a reason to drop the trajectory claim; it is the reason
the trajectory figure is a *picture* with the endpoint figure as its
*measurement anchor*.

## §2 — Reference set: filtered mono bank, n = 8

You can't measure "off the manifold of X" without a concrete bank of
X — *and* the bank has to actually represent X. **Selected:**
terminal z_0 from the cached cat × dog mono seeds
`{2, 3, 5, 7, 8, 9, 11, 12}`, after dropping `{1, 4, 6, 10}` because
mono itself failed to produce cat-and-dog co-occurrence on those
seeds. Operationally: maintain an explicit `mono_bank_seeds` config
list; do not delete the failed cached arrays, just gate them out at
the analysis layer.

Mono-on-mono baseline is leave-one-out across the 8 retained seeds —
each kept mono sample's NN to the other 7.

Held in reserve, only if n = 8 turns out too narrow:

1. A larger natural-image VAE-latent bank containing real cat-and-dog
   co-occurrence photos, VAE-encoded. Broader manifold.
2. A learned density on the 8 mono samples (small KDE / RealNVP).
   Single-number off-manifoldness score per sample.
3. Re-sample mono out to 64 seeds (the original target). The most
   expensive fallback and the one to defer longest. The four
   already-failed seeds remain excluded.

**Subtlety to handle, not skip.** A mono failure is informative on
its own: noise samples `{1, 4, 6, 10}` lie in regions where even
the ceiling sampler cannot reach co-occurrence. Whether PoE+λR
recovers on those same seeds is a separate, valuable question
(supplementary panel — see Scope).

## §3 — Instruments

### For Claim A (endpoint, on terminal z_0)

- **k-NN distance to mono bank** in raw z_0 (Euclidean). Ordering-
  informative even though absolute distances are pixel-statistic —
  *for the comparison* PoE-no-LoRA vs PoE+λR vs mono-on-mono, the
  ordering is what the claim rests on. Cheapest first pass.
- **Pullback-metric distance under G(z_0) = J(z_0)ᵀJ(z_0)**, with
  J(z_0) the SDXL VAE decoder Jacobian. The true LSO measurement. Use
  finite differences along k principal directions or Hutchinson trace
  estimates if full Jacobian is intractable.
- **Density score under a model fit on the mono bank** (KDE or small
  flow). Single-number off-manifoldness per sample. Only built if
  k-NN is ambiguous.

### For Claim B (trajectory, along z_T → z_0)

**Cheap kinematic instruments first.** None of these need the JVP
wrapper or the pullback metric. All are direct functions of the cached
`z_t` arrays and the cached `ε̂_t` arrays. These are the load-bearing
trajectory measurements — the pullback stack below is conditional on
these leaving anything unsettled.

- **Per-step curvature.** Bending angle between consecutive
  displacements: `θ_t = ∠((z_t − z_{t-1}), (z_{t+1} − z_t))`. Curve
  vs t per condition. Does PoE go locally straight while PoE+LoRA
  arcs?
- **Euclidean arc length.** `L_euclid = Σ_t ‖z_t − z_{t-1}‖₂`. Single
  scalar per trajectory. Does PoE travel further or shorter in raw z
  than PoE+LoRA?
- **Distance to mean mono path per t.** `d_t = ‖z_t^{cond} −
  mean_i z_t^{mono,i}‖₂`, curve vs t per condition. Time-resolved,
  kinematic, no JVP. The cheapest path-level statement that compares
  PoE and PoE+LoRA to mono directly.
- **‖r_t‖ profile.** `r_t = ε̂_t^{PoE+λR} − ε̂_t^{PoE}`, then `‖r_t‖₂`
  per t. Locates the LoRA correction in time — terminal corrector vs
  trajectory-wide corrector. Single line per (epoch, λ).
- **Time-resolved k-NN to mono z_t bank.** For each t, NN distance
  from PoE `z_t` (and PoE+λR `z_t`) to the mono bank's `z_t` at the
  same t. Curve per (epoch, λ). Reveals *when* the trajectories
  diverge from mono and *where* LoRA's correction kicks in.
- **Bundle variance per t.** Spread of seeds around their per-t mean,
  for each of PoE-no-LoRA / PoE+λR / mono. Tells whether LoRA tightens
  bundles toward mono or merely shifts the centroid.

**Advanced — only if the cheap instruments leave the trajectory claim
ambiguous:**

- **Pullback path length.** Integrate `‖dz/dt‖_{G(z_t)}` along the
  trajectory. Compare PoE-no-LoRA path length vs PoE+λR path length.
  If LoRA bends *along* the manifold, its path is geometrically curved
  but *shorter* under G. Caveat: G(z_t) is only well-defined at low t;
  in practice integrate over the last ~30% of steps where the latent
  is approximately clean. Requires the JVP wrapper.

## §4 — Straight-line-through-the-hole picture, adapted

The LSO figure shows two curves with **fixed shared endpoints** — red
geodesic-of-Euclidean (straight, through the hole), green
geodesic-of-pullback (arced, along the manifold). Your PoE and PoE+LoRA
**do not share endpoints** — they end at different terminal latents.
So the literal "two routes, same start and end" picture isn't yours.

The right adaptation:

- **Endpoint claim (load-bearing).** PoE-no-LoRA's terminal z_0 lies in
  a low-density region of the mono manifold (it "fell into the hole");
  PoE+LoRA's terminal z_0 lies in a high-density region (it "stayed on
  the surface"). Measured via §3 Claim A instruments.
- **Trajectory claim (illustrative, but instrumented).** PoE traverses
  a path that, integrated under G, has *high* cost — it does not
  respect the manifold's local geometry. PoE+LoRA traverses a
  curvilinear path with *low* cost under G — it bends along the
  manifold's local geometry. Measured via §3 Claim B instruments. The
  *figure* is a 2D projection of these trajectories with their
  pullback-cost colourised on the line; the *measurement* is the
  scalar path length.

The endpoint claim is the science. The trajectory claim is the picture
with measurement teeth.

## §5 — Time-resolved trajectory dynamics: unknown unknowns

The user explicitly asked: are there unknown unknowns about
characterising trajectory dynamics over time? Yes — at least four,
each cheap to probe once §3 Claim B instruments exist:

1. **Onset profile.** Does PoE drift off-manifold *gradually* across
   all t, or in a *sudden phase transition* at low t (where the latent
   becomes clean enough for manifold structure to bite)? The
   time-resolved k-NN curve answers this.
2. **LoRA correction timing.** At which t does the LoRA correction
   become active? Plot ‖r_t‖ vs t for each (epoch, λ). If r_t is
   concentrated at low t, LoRA is a *terminal* corrector; if it is
   distributed, LoRA is a *trajectory-wide* one. These have different
   mechanistic readings.
3. **Bundle dynamics.** Do PoE seeds spread out (high bundle variance)
   while PoE+LoRA seeds tighten (low bundle variance) as t decreases?
   Or does LoRA preserve bundle spread but shift the centroid? Mode-
   finding vs mode-collapse, observable per t.
4. **Manifold-tangent alignment.** Is LoRA's correction direction
   aligned with the *tangent* of the mono manifold (an on-manifold
   correction) or *normal* to it (an off-manifold pull that happens
   to land near mono samples)? Computable as the cosine between r_t
   and the top-k singular vectors of J(z_t) at corresponding mono z_t.
   Distinguishes "LoRA learned the manifold" from "LoRA learned mono
   specifically".

These are the questions you couldn't ask of the MDS panel because the
projection collapsed them. With the instruments above they become
direct, single-number-per-t measurements.

## §6 — Where MDS is still valuable (and where it isn't)

The user's concern was whether MDS is *only* valuable as a negative
result. Answer: the raw-Euclidean variant is, but there are MDS
variants that remain legitimate.

### Negative result (worth stating as a contribution)
- **Raw-Euclidean MDS over z_t** is pixel-statistic geometry, not
  concept geometry. Known failure case: (λ=0.5, ep 900) single cat
  sits near mono; (λ=1.0, ep 1600) chimera sits further. **State this
  in the paper.** "Naïve Euclidean MDS on diffusion latents misleads
  about concept distance; the manifold structure must be accounted for
  (Arvanitidis et al.)." This is itself a small contribution.

### Variants that may still earn their keep

- **Trajectory-as-point MDS.** Each *whole* trajectory is one point in
  MDS; the pairwise distance is a trajectory-distance (e.g. dynamic
  time warping on the per-step z, or Wasserstein between trajectory
  point clouds). Visualises *which trajectories cluster together* —
  do PoE+LoRA seeds cluster with mono seeds while PoE-no-LoRA seeds
  cluster apart? Different question from latent geometry, but a clean
  one.
- **Pullback-metric MDS on endpoints.** Pairwise *geodesic* distances
  under G between terminal z_0s, then MDS. Manifold-aware version of
  the original idea. Could legitimately replace the role plan-13's
  raw-latent panel was attempting.
- **Semantic (DINO) MDS** stays valid as a *behaviour-convergence*
  panel — does the running predicted-x̂₀(t) approach mono's
  predicted-x̂₀(t) in image-semantic terms? This is a different claim
  from the latent-geometry one and the captions should say so cleanly.
  Plan-13's existing panel already supports this; the fix is caption
  discipline, not the instrument.

### What MDS cannot do here
- It cannot recover the pullback geometry from raw-Euclidean
  pairwise distances on z_t. Information about G is lost in the
  projection. So no amount of caption work salvages the raw-Euclidean
  panel as a latent-geometry measurement — it can only be reframed as
  a trajectory-shape illustration with no quantitative reading.

## Storage layout

Mirrors existing layouts, additive only:

```
<results_root>/manifold_cache/
  mono_bank/
    z0/seed_{NNN}.npy            # terminal z_0 per mono seed
    z_t/seed_{NNN}.npy           # full trajectory per mono seed
  claim_a/
    knn_distances.json           # per (epoch, λ, seed): k-NN to mono bank
    pullback_distances.json      # optional, if §3 step 2 is run
    density_scores.json          # optional, if §3 step 3 is run
  claim_b/
    curvature_per_t/<cell_id>.json      # per-t bending angle
    euclidean_arc_length.json           # per (cond, seed) scalar
    dist_to_mean_mono_path/<cell_id>.json
    r_t_norm/<cell_id>.json             # LoRA-correction time profile
    knn_per_t/<cell_id>.json            # per-t NN distance to mono
    bundle_variance/<cell_id>.json
    pullback_path_length.json           # conditional, JVP required
  unknown_unknowns/
    tangent_alignment/<cell_id>.json    # conditional, JVP required
```

Per-cell precompute so adding a new (epoch, λ) cell only re-runs the
new cell, not the bank.

## Step sequence

Ordered cheapest-first. The JVP / pullback stack is conditional on the
cheap instruments leaving anything unsettled.

1. **Inventory cached trajectories.** Tool:
   [scripts/manifold/inventory_trajectories.py](../scripts/manifold/inventory_trajectories.py).
   Walks the filesystem, reports per-(seed, condition, λ) whether
   `z_t` and `ε̂_t` are cached, writes a JSON gap manifest.
   **Verified 2026-05-26**: 0 / 84 cells complete for the cross-seed
   cat × dog pool — the cross-seed pipeline never persisted
   trajectories; only PNGs and (optional) `ε̂_t` records. The
   seed-42 single-seed LoRA cache has `z_t` but no `ε̂_t`. **Backfill
   is required**, not optional. Also: encode the mono-bank filter as
   a config (`mono_bank_seeds = {2, 3, 5, 7, 8, 9, 11, 12}`) so the
   exclusion is auditable and reversible.
   **Backfill tool:**
   [scripts/manifold/sample_with_trajectory.py](../scripts/manifold/sample_with_trajectory.py)
   — re-samples mono / PoE-no-LoRA / PoE+λR at the 12 seeds with
   `LatentTrajectoryCollector` recording both `z_t` and the guided
   `ε̂_t`. Uses pinned init latents from the existing training cache,
   so trajectories are byte-comparable with the inspector / probe
   pipelines. Compute: 12 seeds × (1 mono + 1 PoE-no-LoRA + 3
   PoE+λR) = 60 50-step SDXL DDIM runs at 1024². ~ several hours on
   one A100.
2. **Claim A — Euclidean k-NN.** Strip plot of NN-to-mono distance,
   three groups (PoE-no-LoRA, PoE+λR at λ = 1, mono-on-mono
   leave-one-out). Headline figure uses the same 8 seeds as the
   bank (`{2, 3, 5, 7, 8, 9, 11, 12}`); n = 8 per group; bootstrap
   95% CI on the median; Mann–Whitney U for pairwise tests.
   Annotate the train-pool vs held-out subsplit (`{2, 3, 5, 7, 8}`
   vs `{9, 11, 12}`) as two marker styles. **Supplementary panel**:
   same plot including the four mono-failed seeds `{1, 4, 6, 10}`
   in the PoE / PoE+λR arms (with markers tagged "mono-failed"),
   asking whether LoRA helps on noise samples where the ceiling
   itself failed.
3. **Cheap kinematic path-level panel.** Compute and plot, on the
   same 12-seed pool:
   - per-step curvature curve per condition,
   - Euclidean arc length (strip plot, scalar per trajectory),
   - distance-to-mean-mono-path per t,
   - ‖r_t‖ profile.
   All four are direct functions of the cached arrays. No JVP.
4. **Claim B — time-resolved k-NN and bundle variance.** Curves vs t.
   Same 12-seed pool. Cheap.
5. **Decision gate.** Inspect the output of steps 2–4. Two possible
   continuations:
   - **(a) Cheap instruments already settle both claims.** Stop.
     Write up. Tag the pullback / JVP stack as future work. The
     contribution is "calibrated endpoint distance + kinematic
     trajectory differences across the train / held-out split, plus
     a recaptioned MDS panel".
   - **(b) Cheap instruments are ambiguous, the mechanism claim
     needs the manifold-aware version.** Proceed to step 6.
6. **VAE Jacobian helper (conditional).** Wrap `pipeline.vae.decode`
   for cheap JVPs. Sanity-check finite-difference agreement.
7. **Claim A — pullback / density (conditional).** Pullback distance
   under G(z_0), finite-differenced along k principal mono-manifold
   directions.
8. **Claim B — pullback path length (conditional).** Integrate
   `‖dz/dt‖_{G(z_t)}` over the last 30% of steps. Compare
   distributions.
9. **Tangent alignment (conditional).** Cosine between `r_t` and the
   top singular vectors of `J(z_t^{mono})`. Distinguishes "LoRA
   learned the manifold" from "LoRA learned mono specifically".
10. **Caption rewrite for plan-13 panels.** Drop concept-distance
    language from raw-Euclidean MDS; tag semantic-MDS as a
    behaviour-convergence panel only.
11. **Sanity figure.** Single PNG: Claim A strip plot (left) and the
    most informative cheap trajectory curve from step 3–4 (right).
    Final epoch, λ ∈ {0, 0.5, 1.0}. Train vs held-out marked.

## Acceptance

Given n = 8 per condition (n = 5 train-pool, n = 3 held-out), use
bootstrap CIs and Mann–Whitney U for distribution comparisons;
significance is "CI for the difference of medians excludes zero",
not p < 0.05 in a t-test. The held-out subsplit (n = 3) is treated
as directional, not as a statistical test on its own.

- **Claim A** passes if the bootstrap CI for `median(d_PoE-no-LoRA)
  − median(d_mono-on-mono)` excludes zero from above, AND the CI
  for `median(d_PoE+λR) − median(d_mono-on-mono)` straddles zero
  (i.e. PoE+LoRA is statistically indistinguishable from a fresh
  mono seed in NN-to-mono distance), on the 8-seed headline set.
  Directional check: the same ordering visible on the 3-seed
  held-out subsplit.
- **Cheap-trajectory acceptance.** Per-step curvature or
  distance-to-mean-mono-path-per-t shows a visible separation
  between PoE-no-LoRA and PoE+λR on the 8-seed headline set, with
  the same direction on the held-out subsplit.
- **‖r_t‖ profile** identifies a clean t-window where the LoRA
  correction is active (terminal vs trajectory-wide).
- **Claim B (advanced, conditional)** passes if pullback path length
  for PoE+λR is smaller than for PoE-no-LoRA — only relevant if the
  decision gate at step 5 escalates to the manifold-aware stack.
- Plan-13's MDS captions no longer claim concept-distance.
- **Supplementary**: behaviour on the four mono-failed seeds
  (`{1, 4, 6, 10}`) is reported alongside the headline, flagged
  but not used for the acceptance test.

## Open questions

- **Cache completeness.** Are cached `(z_t, ε̂_t)` arrays available for
  all 12 seeds across mono / PoE-no-LoRA / PoE+λR at (final epoch,
  λ ∈ {0, 0.5, 1.0})? Step 1 of the sequence checks; the likely
  missing piece is per-step `ε̂_t` for PoE-no-LoRA, since the
  cross-seed pipeline may have only stored `z_t`. If `ε̂_t` is
  missing, the `‖r_t‖` profile and tangent-alignment probes need a
  small backfill run that re-emits velocities without re-sampling
  images.
- **n = 8 statistical power (held-out n = 3).** Small. The
  bootstrap-CI + Mann–Whitney acceptance criteria are honest about
  this. If the headline strip plot's CIs straddle zero where they
  shouldn't, the fallback is option 3 in §2 (re-sample mono to 64
  seeds, keeping the four current failures excluded) — but only do
  this if the cheap instruments show the right ordering visually
  and the test is the only thing missing.
- **Why mono fails on `{1, 4, 6, 10}`.** A separate, mechanistic
  question: are those four noise samples in a region of `z_T` from
  which even mono cannot reach co-occurrence, or is it a
  prompt-encoding / scheduler-luck artefact? Out of scope here but
  worth surfacing — if the answer is "those four noise samples are
  genuinely hard," it strengthens the supplementary panel reading.
- **Pullback metric cost (deferred).** Only matters if the decision
  gate at step 5 escalates. SDXL VAE decode maps 4×128×128 →
  3×1024×1024; full Jacobian intractable. Finite differences along
  k ≤ 64 principal directions of the mono bank's local covariance is
  the fallback. Worth a 50-LOC feasibility check before step 6.
- **Bundle metric.** Variance is the obvious first choice. If the
  bundle is multi-modal, switch to per-mode counts or a small GMM fit.
- **Whether to fold semantic MDS into the Claim B picture** by adding
  per-t pullback-cost annotation to the existing DINO trajectories.
  Defer until the standalone Claim B figure exists.
