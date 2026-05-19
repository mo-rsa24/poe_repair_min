# Phase 7 — Cross-seed Δ_t: is the residual a shared signal?

## Question

Phase 4 trained one LoRA on one seed and it worked. Before training a
*pooled* LoRA across seeds (Phase 8), we need to know: is `Δ_t` a
property of the *concept pair* (the same kind of object across seeds,
just rotated by the starting noise), or is each seed's `Δ_t` a
seed-private object that pooling will only average into something
weaker than either input?

Four narrow flavours:

| Flavour | Plain question | Probe |
|---|---|---|
| **Time** | Inside one seed, is the per-step correction just a few patterns scaled differently? | SVD effective rank of `Δ^(s) ∈ R^(T × D)`. |
| **Across seeds** | Are those few patterns the same across seeds? | Principal angles between per-seed top-k right singular subspaces, vs a permutation null. |
| **Time and space** | When and where in the image does `Δ_t` actually live? | `‖Δ_t‖` curve vs t; per-pixel energy in early / mid / late buckets. |
| **Noise floor** | Is `‖Δ_t‖` actually above what mismatched/null seed-pairings give? | Seed-shared-vs-seed-specific energy decomposition with finite-sample bias correction. |

The noise-floor probe was tightened during execution — the original
"mismatched seed Δ" comparison conflated state divergence with signal
absence. See the implementation notes at the bottom.

## Why this phase exists

Phase 4 proved a per-seed LoRA learns *something* on cat × dog seed 42.
This phase decides whether that something is a shared concept-pair
signature (pooling is well-posed) or seed-specific structure (pooling
would have to be reframed). It also produces a numerical baseline
("two different seeds' Δ_t look like *this* to each other") that
Phase 8 compares the pooled LoRA's output against.

## Code

- Cache reader: `poe_repair/training_cache.py::iter_cell_deltas`,
  `delta_t_from_raw`. Reads `Δ_t = w · (ε_J + ε_∅ − ε_A − ε_B)` per
  cached step.
- Sampler that produces Δ_t / ε_PoE / ε_Mono on the same trajectory:
  `poe_repair/methods/_sampling.py::run_teacher_residual` with
  `lambda_max=0.0` and `save_residuals_dir=<scratch>`.
- Diagnostic functions: `poe_repair/diagnostics/delta_structure.py`
  (one function per prong).
- Experiment driver:
  `poe_repair/experiments/residual_diagnostics/delta_structure/__main__.py`.
- Outputs land under
  `outputs/residual_diagnostics/delta_structure/`:
  `meta.json`, `tensors.pt`, `results.json`, `figures/fig_main.png`.

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
```

### 1. Collect Δ_t for N seeds (~1–2 h for N=4; ~2–3 h for N=8)

```bash
$PY -m poe_repair.experiments.residual_diagnostics.delta_structure \
    --seeds 42,43,44,45                       # N=4 first pass
```

Each seed: one `run_teacher_residual(lambda_max=0.0)` call with
per-step `.pt` dumps; concatenated into a `(S, T, C, H, W)` tensor
written to `tensors.pt`. Scratch files are deleted after stacking.

### 2. Run diagnostics (CPU, < 30 min)

The driver runs all four prongs automatically after collection. To re-run
just the diagnostics from existing `tensors.pt`:

```bash
$PY -m poe_repair.diagnostics.delta_structure \
    --tensors outputs/residual_diagnostics/delta_structure/tensors.pt
```

Writes `results.json` and `figures/fig_main.png` (6-panel: time
compressibility, principal-angle matrix, null histogram, `‖Δ_t‖` vs t,
spatial heatmap, matched-vs-null distribution).

### 3. Push to N=8 if cross-seed verdict is ambiguous

```bash
$PY -m poe_repair.experiments.residual_diagnostics.delta_structure \
    --seeds 42,43,44,45,46,47,48,49
```

Decision rule: if the cross-seed flavour reads `indistinguishable_from_null`
at N=4, push to N=8 before committing.

## How to read the result

The `results.json` file labels every prong with a one-word verdict.
The composite landing is the combination of those verdicts.

| Bucket | What you see | Means |
|---|---|---|
| **Poor** | Δ_t is zero, NaN, or shape-mismatched on at least one seed. | Sampler bug. Stop and debug. |
| **Bad** | Per-seed time-compressibility verdict is `not_compressible` for *all* seeds (k90 > 20 everywhere). `‖Δ_t‖` is flat across t with no peak. | Δ_t doesn't have *any* exploitable structure even within a seed. Phase 4's success has to be re-interpreted — the LoRA may be learning something other than what the cache target nominally points at. |
| **Unknown** | Per-seed compressibility passes (k90 ~ 5–15). Cross-seed alignment lands `indistinguishable_from_null` even at N=8. Noise-floor decomposition gives a small but non-zero bias-corrected SNR. | Δ_t is structured *within* a seed but not visibly shared across seeds. Phase 8 should expect "pooled LoRA averages seed-private corrections" — and the cheap inference-time mono-average pre-screen (Phase 8 Step 0) should already be close to mono on held-outs *only* if the shared component is large enough to matter. |
| **Good (the result on disk)** | Per-seed `Δ_t` is low-rank in time (k90 ~ 12–20). Spatial concentration tightens through late steps. **Cross-seed mean is bias-corrected SNR ≈ 0** (the result was `landing_6`: at the cross-seed mean, Δ_t is consistent with seed-private noise). Principal-angle gap is statistically detectable but practically negligible. | Pooling will recover at best the seed-average direction. Phase 8 should be framed as "test whether the seed-average is enough, since cross-seed alignment of the per-seed Δ_t directions is essentially null." This is the result on disk as of 2026-05-19. |

The `Good` row reflects the executed outcome. It changes Phase 8's
framing without making it dead: a pooled LoRA can still beat per-seed
LoRAs through epoch parity, capacity sharing, or convergence
properties, even if the raw Δ_t cross-seed signal is weak.

## Implementation notes that survived execution

1. **The plan's original Prong E (matched vs mismatched-seed Δ
   norms) was wrong.** At step `t`, two seeds occupy different latents
   `x_t`, so `‖ε̃_mono(i) − ε̃_poe(j)‖` is dominated by *state
   divergence*, not by absence of structured correction. The replaced
   probe is a seed-shared-vs-seed-specific energy decomposition with
   *mandatory* finite-sample bias correction (naïve shared-fraction
   scales as 1/N under a zero-signal null and gives spurious "signal"
   at small N).
2. **CFG sanity is trivial by algebraic identity.**
   `Δ_guided = guidance_scale × Δ_unguided` (PMI identity). All
   scale-invariant metrics (angles, energy ratios, concentration
   ratios, bias-corrected SNR) are identical between guided and
   unguided ε-space by construction. The empirical check is for
   confidence, not necessity.

## What this phase does *not* do

- Cross-pair sweep. Cat × dog only.
- Direct comparison of LoRA's output to Δ_t (that comparison lives in
  Phase 8 Task D as the residual-of-residual analysis).
- Mono deployment.
- Any claim past N=8 seeds.

## Status — 2026-05-19

| Item | Done | To do |
|---|:---:|:---:|
| Sampler patch — `run_teacher_residual` saves `eps_poe` and `eps_mono` per step | ✅ | |
| N=4 collection (seeds 42, 43, 44, 45) | ✅ | |
| N=8 push (seeds 42–49) after ambiguous N=4 verdict | ✅ | |
| Prong A — time compressibility | ✅ | |
| Prong B — cross-seed alignment with permutation null | ✅ | |
| Prong C — energy vs time | ✅ | |
| Prong D — spatial locus by timestep bucket | ✅ | |
| Prong E — replaced shared-vs-specific energy decomposition with bias correction | ✅ | |
| `results.json` + 6-panel `fig_main.png` | ✅ | |
| `SYNTHESIS.md` documenting `landing_6` outcome | ✅ | |
| Frame Phase 8 around the `landing_6` prior (pooling expects weak shared signal) | | ⬜ |
