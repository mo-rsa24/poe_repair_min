# Plan 13 — Semantic-embedding toggle for the latent-trajectory MDS panel

> Companion to [04-lora-single-seed.md](04-lora-single-seed.md) and the
> existing MDS pipeline in
> [scripts/build_lora_inspector_mds.py](../scripts/build_lora_inspector_mds.py).
> Frontend touch-point: the "Latent trajectories — single large MDS
> plot" panel in the LoRA Inspector.

## Problem

The current MDS is computed on flattened latents `z_t`. Distances in
that space track pixel-level appearance statistics, not the semantic
property the LoRA is trained to induce (spatial co-occurrence of both
animals). Concrete failure case on cat×dog seed 42:

- (λ=0.5, epoch 900) — single fluffy cat, no dog — sits **near** mono.
- (λ=1.0, epoch 1600) — both animals co-located but morphologically
  fused — sits **further** from mono.

The plot therefore orders configurations by appearance similarity, not
by the property we want to demonstrate (PoE+λ·r approaches mono with
training).

## Hypothesis

Latent-`z_t` geometry is inherent to the model's compressed image code
and does not encode co-occurrence. But the convergence claim itself is
recoverable if we replace the per-timestep embedding with a *semantic*
one — DINOv2 features of the decoded predicted-clean latent
`x̂₀(t)` — before running MDS.

A two-state toggle on the existing panel can switch between:

- **Mode A — Raw latent (current).** MDS on flattened `z_t`.
- **Mode B — Semantic x̂₀.** MDS on cosine distances over DINOv2
  CLS embeddings of `VAE.decode(x̂₀(t))`.

Same epochs, same λ-slider, same static A/B/A∧B paths, same terminal
thumbnails. Only the (x, y) coordinates change. Both layouts are
precomputed offline; the toggle is a file-swap at view time.

## What this plan does *not* do

- **Replace the existing raw-latent MDS.** Mode A stays; Mode B is
  additive.
- **Re-sample anything.** `LatentTrajectoryCollector`
  ([poe_repair/_sdxl/runtime.py:36-74](../poe_repair/_sdxl/runtime.py#L36-L74))
  already stores `trajectories` *and* `velocities` per step, so x̂₀(t)
  is derivable from cached `.npy` dumps. (Step 1 confirms the velocity
  convention.)
- **Touch other inspector tabs.** Only the residual MDS panel.
- **Generalise beyond cat×dog seed 42.** Same beachhead scope as
  Plan 04.
- **Add a third embedding (CLIP, OWLv2 co-occurrence) yet.** Out of
  scope; revisit if Mode B fails the gate below.

## Validation gate (do *before* building anything)

The whole toggle is wasted work if DINOv2 doesn't actually reorder the
known failure case. Smoke test on three terminal images only:

1. `case1` — terminal `x_0` from (λ=0.5, epoch 900). Single cat.
2. `case2` — terminal `x_0` from (λ=1.0, epoch 1600). Chimeric blend.
3. `mono` — terminal `x_0` from the mono ceiling.

DINOv2 ViT-S/14 CLS, L2-normalise, cosine distance to `mono`.

**Pass criterion**: `d(case1, mono) > d(case2, mono)`.

- Pass → proceed with the rest of the plan.
- Fail → DINO is also dominated by appearance here. Stop. Escalate
  to an explicit co-occurrence score (OWLv2 / Grounding DINO presence
  detection) in a follow-up plan, and demote the MDS panel to "what a
  trajectory looks like" without any convergence reading attached.

Script: `scripts/cross_seed_lora_pooling/smoke_dino_distance.py` (new,
~30 lines, single cell, no MDS).

## Pipeline (Mode B)

For each trajectory (mono, A, B, A∧B, and every (epoch, λ) PoE+λ·r
cell already enumerated by the existing MDS builder):

1. Load cached trajectory dump → `z_t` and `velocities`.
2. **Recover x̂₀(t).** Convention check first:
   - SDXL DDIM eps-prediction: `x̂₀(t) = (z_t − √(1−ᾱ_t)·ε̂_t)/√ᾱ_t`
     where ε̂_t is what `velocities[step]` holds.
   - v-prediction: `x̂₀(t) = √ᾱ_t · z_t − √(1−ᾱ_t) · v_t`.
   - Confirm during step 1 of build by cross-checking the recovered
     `x̂₀` at the final step against the saved terminal `z_0`.
3. **Subsample timesteps** every 5 (≈ 10 points per 50-step path) to
   keep VAE + DINO cost bounded. Configurable flag.
4. VAE-decode `x̂₀(t)` at 256² (re-use the inspector's existing VAE).
5. DINOv2 ViT-S/14 forward → CLS token, L2-normalise.
6. Stack all CLS tokens across all trajectories → `E ∈ ℝ^(N×384)`.
7. Pairwise dissimilarity matrix `D[i,j] = 1 − cos(E_i, E_j)`
   (cosine, **not** Euclidean — DINO similarity is angular).
8. Metric MDS to 2D → `coords_semantic.json`.
9. **Procrustes-align** to `coords.json` (Mode A) using the static
   A, B, A∧B endpoint positions as the three anchor points. This keeps
   the toggle visually coherent — same paths don't reshuffle across
   the entire canvas when the user flips the button.

## Storage layout

Mirrors the existing residual MDS cache:

```
<results_root>/mds_cache/
  coords.json                # Mode A (raw latent)  — already exists
  coords_semantic.json       # Mode B (DINO x̂₀)    — new
  semantic_cache/            # per-trajectory DINO CLS arrays
    <cell_id>/<lam>.npy      # shape (T_sub, 384)
    static/{A,B,AandB}.npy
```

Per-cell precompute (saved alongside the existing per-cell artefacts)
so adding a single new (epoch, λ) cell does not force a full re-MDS;
only the projection step (~seconds) re-runs.

## Frontend toggle

In the residual MDS panel of the LoRA Inspector:

- New control: pill toggle `Embedding: [ raw latent | semantic (DINO x̂₀) ]`.
- Default: `raw latent` (no behaviour change for existing users).
- On flip:
  - Swap the coords source from `coords.json` to `coords_semantic.json`.
  - Re-render the same paths at the new coordinates.
  - Re-draw the slider-driven PoE+λ·r path at the new coordinates for
    the current (epoch, λ).
  - Terminal thumbnails stay attached to each path's endpoint —
    unchanged.
- Tooltip on the toggle: "Mode A: MDS over flattened z_t. Mode B: MDS
  over DINOv2 features of decoded predicted-x̂₀(t) — semantic, tracks
  co-occurrence."
- Sub-caption updates per mode so the figure caption is honest about
  what the geometry encodes.

## Costs

Rough per-cell numbers on an A100 (cat×dog seed 42, ~340 paths total
across all (epoch, λ) + statics, 10 timesteps each after subsampling):

- VAE decode: ~80 ms × 3,400 ≈ 4.5 min
- DINOv2 ViT-S/14: ~10 ms × 3,400 ≈ 35 s
- Pairwise cosine + MDS: seconds

Budget **≤ 10 min per cell** offline, **zero** at view time. Frontend
toggle is a JSON swap.

## Subtleties to handle (not skip)

- **Cosine, not Euclidean.** Use `1 − cos` for the MDS dissimilarity.
- **Procrustes anchoring.** Without it, toggling will visually
  scramble. Fit a 2D similarity transform on the three static
  endpoints, apply to all Mode B coords.
- **Early-t scatter.** Predicted-x̂₀ at high noise is essentially
  random; expect a diffuse cloud at the start of each path that tightens
  toward the terminal. This is the desired reading ("paths
  indistinguishable at t=T, bend toward target near t=0"), not a bug.
- **Caption discipline.** Mode A caption must drop any
  "approaches mono" language and stay neutral about geometry. The
  convergence claim is *only* legitimate under Mode B.
- **Mono usage.** Mode B is diagnostic, comparing PoE+λ·r against the
  mono ceiling. Same rule as the rest of the project — mono is the
  ceiling reference, not a deployable sampler. No new claim about mono.

## Step sequence

1. **Confirm velocity convention.** ≤ 50 LOC sanity check: load one
   trajectory, recover `x̂₀(T_final)` from `(z_T, velocity_T)` under
   both eps and v formulas, compare against the saved terminal latent.
   Bake the right formula into a small helper:
   `poe_repair/_sdxl/predicted_x0.py`.
2. **Validation gate.** Run `smoke_dino_distance.py` on the three
   known images. Stop if it fails.
3. **Builder script.** Extend
   [scripts/build_lora_inspector_mds.py](../scripts/build_lora_inspector_mds.py)
   with a `--mode semantic` flag (or add a sibling
   `build_lora_inspector_mds_semantic.py` if cleaner). Emit
   `coords_semantic.json` and `semantic_cache/`.
4. **Procrustes step.** Small utility:
   `scripts/_procrustes_align_mds.py` — align Mode B to Mode A using
   the three static anchors; overwrite `coords_semantic.json` in place.
5. **Inspector frontend.** Add the toggle pill, route between the two
   coords sources, update caption/tooltip per mode.
6. **Sanity figure.** Single PNG, both modes side-by-side at
   (ep 900, λ=0.5) and (ep 1600, λ=1.0), to confirm the geometry flip
   reproduces the validation-gate finding inside the actual MDS view.

## Acceptance

- Validation gate passes.
- Mode B MDS, on the same paths the panel renders today, places
  (λ=1.0, ep 1600) **closer** to mono than (λ=0.5, ep 900).
- The slider sweeping λ ∈ [0, 1] at the final epoch shows the PoE+λ·r
  path bending toward mono monotonically (or near-monotonically) in
  Mode B — the demonstration the original figure was attempting.
- Toggle flip is visually coherent (Procrustes-aligned), no
  reshuffling of static anchors.
- Caption under Mode A no longer claims convergence; Mode B caption
  carries the convergence reading.

## Open questions

- DINOv2 ViT-S/14 vs ViT-B/14: S is plenty for the validation gate;
  if the production MDS looks muddled, swap to B (one-line config).
- Subsample stride (5 vs 10): start at 5, drop to 10 if cost bites.
- Should Mode B also expose a "trim to t < T/2" sub-toggle to hide
  the early-t noise cloud? Defer until we see the actual figure.
