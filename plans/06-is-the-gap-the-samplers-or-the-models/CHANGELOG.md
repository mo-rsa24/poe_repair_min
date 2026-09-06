## 2026-09-02
Rewrote plan `baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md`. The plan previously said to
run SuperDiff "at DDIM, 50 steps." Reading the actual SDXL pipeline
(`superdiff/superdiff-sdxl-v1-0`, `pipeline.py`, 531 lines, pulled and read in full) showed this
is wrong: the pipeline is a hand-written Euler-Maruyama stochastic integrator with no scheduler
object and no `eta`/deterministic-sampling option anywhere in the code, so "DDIM" was never an
available setting, at any step count. Every DDIM mention is removed from the plan.

The same read confirmed `num_inference_steps=200` and `guidance_scale=7.5` are the pipeline's own
real defaults, and that its blending weight `kappa` is computed with no clamp anywhere in the
source, a genuine risk at low step counts since `dsigma` is larger per step and a spike has less
room to recover from.

Widened the check from one pair and one seed to an 8-cell grid: `a_cat__x__a_dog` and
`a_butterfly__x__a_flower_meadow` (seed 9, this scope's standard pair of pairs), crossed on step
count (200, 50) and on a new `kappa` clamp (on, off). This separates a real step-count effect from
an unclamped-estimator artefact, rather than conflating them, and it means the plan does not have
to assume the clamp matters, it measures whether it does.

Renamed the plan and its paired review file from `05-superdiff-at-this-repos-fifty-steps` to
`05-what-changes-when-superdiff-leaves-its-own-defaults`, since the old name asserted a fixed step
count the plan no longer runs at exclusively. Repaired every cross-reference: the root and scope
`MASTER_PLAN.md` running-order tables, `06-three-rules-on-one-amount-axis.md`,
`04-does-the-corrector-compose-in-the-same-window.md`, and `03-what-is-left-once-the-chain-settles.md`.

Reading `06-three-rules-on-one-amount-axis.md` in full showed its two existing grids already put
SuperDiff's row on this project's own shared `λ` axis across seeds 9 to 12, using the `eps_M` hook
this plan exposes — so the seeds-by-`λ` figure this plan's earlier close-out task planned to hand
off there was redundant with work that plan already does. Added a new task group instead (group 4)
building the one figure that was genuinely missing: `r_t^SD` against SuperDiff's own `kappa`,
swept directly to five forced values rather than through the shared `λ` axis, at seeds 9 to 12 and
both step counts. This is SuperDiff's own internal-parameter characterization, not a cross-rule
comparison, so it stays in this plan rather than moving to the "three rules" plan.

## 2026-09-03
Redesigned task group 4 of `baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md`
after the eight-cell grid came back reversed (composes at 50 steps, collapses to one concept at
200). The κ sweep as first written (rows seeds, columns forced κ, one panel per step count) was
replaced by a κ × λ set: one sheet per (pair, κ setting), rows seeds 9–12, columns
λ ∈ {0, 0.25, 0.5, 0.75, 1} where λ is the fraction of `r_t^SD = eps_J − eps_M` added back,
i.e. step 29's generalised injection applied inside SuperDiff with its blending weight held
fixed per sheet. `balanced` means the pipeline's own unclamped κ, its published behaviour.

Scoped to 200 steps only, cat×dog at six κ settings and butterfly×meadow at κ=0.5, to keep the
run near five hours rather than twelve; the 50-step half was dropped and its 30 finished renders
kept on disk. The composer gained `kappa_override` and `lam` arguments for this, so the λ
injection now lives in `poe_repair/composers/superdiff.py` rather than only in step 29's driver.

Added objective 7 and Definition-of-Done items 13 and 14 to this scope, with two new plans:
`baselines/07-does-the-poe-trained-correction-reach-superdiff.md` (step 49) and, gated on it,
`baselines/08-adapters-that-learn-superdiffs-own-residual.md` (step 50). Plan 05's κ × λ sheets
showed SuperDiff at its own settings is missing a residual of the same shape PoE is (one blended
animal at λ=0, two once half of `r_t^SD` is added back), which made "does the PoE-trained adapter
carry over to a different composition rule" a live question with no home in the scope's six
objectives. Step 49 injects the rank 8/16/32 adapters into SuperDiff on every one of its 200
steps at `kappa` 0.5 and reads the sheets against plan 05's no-adapter κ=0.5 sheets. Step 50
trains adapters against SuperDiff's own residual at that `kappa` only if step 49 says the
PoE-trained one does not transfer, since it costs three ~30-hour trainings.

Later the same day, step 49 came back "hurts" on the first three of six sheets (rank 8 both
pairs, rank 16 cat×dog: no seed separates at any λ, and the adapter's correction is mostly
orthogonal to SuperDiff's missing residual, median cosine peaking at +0.34), and the user stopped
it at 49 of 96 cells. Step 50 opened. Its task 1.3 was rewritten from "add a SuperDiff loader and
target to `training_cache.py`" to "keep the PoE key names and put the blend in the trainer's
step": the cache builder writes `x_t` as the UNet input SuperDiff fed and `eps_a_raw`/`eps_b_raw`
as prompts 1/2, so the existing loader reads it unchanged, and the only new code is a `compose`
switch in `one_pair_one_seed/trainer.py` selected by `--compose superdiff --kappa 0.5`. One
formula in one place beats a second loader that could drift from it.

Plan 08's three trainings changed one bar after a number was seen, and this is the record of
it. The trainer's stall guard requires the commit-bucket loss to halve by optimizer step 5000;
all three SuperDiff-residual runs fell 44–48% by then (0.0102 → 0.0053–0.0058) and were killed
just short. The guard was calibrated on the PoE residual, where the same step budget makes ~4×
more passes over the data. The deadline is scaled by that factor, to 20,000 steps, the threshold
rule and the halving rule are kept, and the runs resume from their step-5000 checkpoints under
the same W&B ids. The commit window itself had already been rescaled from steps 5–24 to 20–99
before launch, for the same 50-to-200-step reason.

## 2026-09-04
Plan 08's three SuperDiff-residual trainings were stopped by the user at optimizer steps 61k
(rank 8), 62k (rank 16) and 36k (rank 32) of 100k, before the final λ sweep in task 3.1. The
decision rests on the checkpoint strips rendered every 10k steps through SuperDiff itself
(mono target | SuperDiff κ 0.5 | adapter at λ 1): every rank separates cat×dog into two bodies
by step 10k–20k, then smears with further training at fixed λ, rank 32 first (no separation by
30k), rank 16 next (none by 50k), rank 8 last (still three of four seeds at 60k but softer each
checkpoint), while the training loss falls and plateaus without any matching rise. The λ sweep of
rank 16's step-30k checkpoint showed λ 0.5 usable on two seeds, so part of the smear is dose, but
the worsening at fixed λ across checkpoints is learned. Tasks 3.1 and 3.2 (the 96-cell final set
and its six sheets) are not run; the six timelines under `corrector/superdiff/timelines/` stand in
as the plan's figure output, and the checkpoints stay on disk. Plan 08 closes on that evidence
rather than on a full-budget verdict.

## 2026-09-05
Added a fifteenth Definition-of-Done item and a third task group to
`hypothesis/04-does-the-corrector-compose-in-the-same-window.md`: the eight-seed sheets every
parallel session reports on (joint prompt, plain PoE, the corrector on all 50 steps, seeds 9 to
16, both pairs) and the fidelity read, where the corrector runs only on the last fifteen steps on
top of the rank-32 λ 1.2 adapter run at `k ∈ {0, 5, 20}`, scored for compose and Laplacian-variance
sharpness. The read asks whether corrector steps at low noise sharpen the corrected run without
costing composition, which no plan in the scope asked; the sampler-share read at step 26 is
untouched. Its thresholds went into `scripts/corrector_window_sweep.py` as named constants and into
plan 04's review file before any of it ran. Plan 04's environment facts gained the adapter
checkpoint and the note that the control pair's seeds 13 to 16 start from the from-seed draw the
cache equals.

## 2026-09-06
Added task group 5 to `hypothesis/04-does-the-corrector-compose-in-the-same-window.md` and a
matching pre-registered question in its review file: the clean tail, where the rank-32 adapter
runs only on steps 0 to 19 or 0 to 29, the frozen model's plain PoE step takes over after, and
the corrector, when on, settles the latent on the frozen score at steps 35 to 49. Asked because
the tail condition in group 3 came back null (sharpness +8%, inside the band), so the corrector on
the corrected score does not remove the adapter's softness, and the question the person actually
has is how to get the adapter's renders crisp. The bar is session B's: sharpness back inside the
plain-PoE seed band with composition within one seed. It overlaps scope 01's plan 14 on purpose;
that plan keeps the λ schedules and the re-noise cell, this group carries the corrector variant.
