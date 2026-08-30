# Phase 1 — Veracity

## Question

If we knew the perfect per-step correction `r_t = ε̃_Mono − ε̃_PoE`,
could we walk a failing PoE trajectory to a working Mono-like image
just by adding `r_t` at each step? In other words, *is the gap
reachable from where PoE currently is*?

This phase does not learn anything. It uses the oracle (Mono itself)
to compute `r_t`, injects it during DDIM sampling at varying strengths
`λ ∈ [0, 1]`, and checks whether the output sweeps from chimera (λ=0)
to clean co-occurrence (λ=1).

## Why this phase exists

If injecting the oracle residual doesn't recover Mono, no learner can
either — the loss landscape we'd be training on doesn't lead anywhere
useful. Conversely, a clean λ-sweep proves the repair problem is
well-posed *before* we spend training compute.

Three claims this phase has to establish:

- **Existence.** `ε̃_PoE + r_t = ε̃_J` holds in code, to fp16 tolerance.
  Trivial by definition; the check is that the codepath is correct.
- **Reachability.** `‖r_t‖` is non-trivial, larger on contested pairs
  than on cooperative pairs, and concentrated in a basin-commit time
  window — i.e. `r_t` is real signal, not numerical jitter.
- **Sufficiency.** Walking λ from 0 to 1 produces a visible chimera →
  co-occurrence transition, monotonic in latent / perceptual distance,
  with GroundingDINO actually detecting two distinct boxes at λ=1.

## Code

- Sampler: `poe_repair/methods/_sampling.py::run_teacher_residual`
  takes `lambda_max` and (optionally) `save_residuals_dir`. At λ=0 it
  reproduces vanilla PoE bit-identically; at λ=1 it reproduces literal
  Mono.
- Experiment package: `poe_repair/experiments/veracity/`
  (figures + metrics drivers).
- Detection helper: `poe_repair/experiments/veracity/metrics.py::detect_boxes`
  (GroundingDINO-Tiny via HuggingFace).
- VQA helper: same file, `vqascore_yesno` (LLaVA-1.5).
- Box-overlay rendering: `poe_repair/figures/_common.py::overlay_boxes`.
- Style module: `poe_repair/figures/_veracity_style.py`.

The on-disk artefacts live under `outputs/veracity/`. The figure set is
fixed at four main figures plus three appendix figures — see
[veracity-figure-plan.md](../claude/veracity-figure-plan.md)
for the per-figure spec; this plan only enumerates the run order.

## Commands

Environment.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
```

### 1. λ-sweep cache (oracle injection at 11 strengths)

```bash
$PY -m poe_repair.experiments.veracity --sweep-only \
    --pair "a cat|a dog" --seed 42
```

Produces `outputs/veracity/pairs/a_cat__x__a_dog/seed_42/teacher_residual_const_lam{000,010,...,100}/`.

### 2. PMI-identity check + per-step metrics

```bash
$PY -m poe_repair.experiments.veracity --metrics-only \
    --pair "a cat|a dog" --seed 42
```

Writes `outputs/veracity/metrics/pmi_identity.json` and `distances.json`.

### 3. Control sweeps (for the anti-corroboration figure)

```bash
$PY -m poe_repair.experiments.veracity --sweep-only \
    --pair "a cat|a cat" --seed 42        # self-pair control
$PY -m poe_repair.experiments.veracity --sweep-only \
    --pair "a butterfly|a meadow" --seed 42   # cooperative control
```

### 4. Final figure render (after caches are on disk)

```bash
$PY -m poe_repair.experiments.veracity --skip-sweep --skip-metrics
```

Renders all 7 figures under `outputs/veracity/figures/`.

## How to read the result

| Bucket | What you see | Means |
|---|---|---|
| **Poor** | At λ=1, image is identical to PoE chimera. PMI identity fails (>1e-3 relative). | Sampler or definition bug. The codepath does not implement `ε̃_PoE + r_t`. Nothing else in this plan is interpretable until fixed. |
| **Bad** | PMI identity passes but λ-sweep is non-monotonic; GroundingDINO finds zero or one box at λ=1; VQAScore < 0.3 on the joint question. | `r_t` exists but doesn't move the image. The PoE trajectory has wandered somewhere `r_t` can't undo, or the residual is overwhelmed by sampler dynamics. The whole project should pause for diagnosis before Phase 2. |
| **Unknown** | λ-sweep moves the image qualitatively but GroundingDINO confidence is in `(0.35, 0.55)` for one concept; chimera vs co-occurrence is ambiguous on the headline strip. | Detection threshold is borderline. Lower threshold to 0.20 with caption caveat and re-run figures. Treat as a soft pass for moving on to Phase 2. |
| **Good (the result on disk)** | At λ=0: chimera; at λ=1: two distinct boxes detected with confidence > 0.6; VQAScore on `"is the cat clearly separate from the dog?"` ≥ 0.7; latent-L2 and CLIP distances monotone in λ; `‖r_t‖` peaks inside the measured basin-commit window. | The repair problem is well-posed. Phase 2 can proceed: a learner that approximates `r_t` would, in principle, drive PoE → Mono. |

## What this phase does *not* prove

- That `r_t` is *easy* to learn (Phase 4 tests that).
- That `r_t` is *the same* across seeds (Phase 7 tests that).
- That a Mono-free corrector can match this performance (Phase 4, 5, 6).
- Anything about pairs other than `cat × dog` or the controls listed.

## Status — 2026-05-19

| Item | Done | To do |
|---|:---:|:---:|
| λ-sweep cache on cat × dog seed 42 (11 strengths) | ✅ | |
| PMI-identity numerical check | ✅ | |
| Self-pair control sweep (`a cat \| a cat`) | ✅ | |
| Cooperative-pair control sweep (`a butterfly \| a meadow`) | ✅ | |
| Fig 1 — existence / PMI self-consistency | ✅ | |
| Fig 2 — reachability anti-corroboration + basin-commit | ✅ | |
| Fig 4 — sufficiency λ-sweep with GroundingDINO boxes | ✅ | |
| App-A — trajectory dependence (PoE vs Mono anchor) at seeds {4, 42, 123} | ✅ | |
| App-B′ — detection-based failure modes on seeds {4, 42, 123} | ✅ | |
| App-C — CFG × timestep qualitative grid | ✅ | |
| GroundingDINO + LLaVA helpers wired | ✅ | |
| App-E — window-localised injection sweep + figure | | ⬜ (10 GPU runs at `outputs/veracity_window_injection/`) |
| Final end-to-end figure render after App-E lands | | ⬜ |
