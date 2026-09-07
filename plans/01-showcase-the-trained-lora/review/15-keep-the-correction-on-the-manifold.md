# 🧪 Review: is the adapter's softness an off-manifold push, and does dropping that part restore fidelity?

**Both questions are answered: question 1 inconclusive by its own linearity rule and a null on its statistic, question 2 null.** The reading that survives is in the first figure: at the steps that decide the composition, both corrections lie in directions the frozen denoiser responds to *less* than to a random direction, and from step 10 on both lie in the directions it responds to most. The adapter's correction is not more off-manifold than the true one anywhere. This file judges [the design](../plans/experiments/15-keep-the-correction-on-the-manifold.md). Its answers fill the showcase wall's "projected correction" row if a condition is supported, and otherwise turn the paper's fidelity caveat into a sentence about direction.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/15-keep-the-correction-on-the-manifold.md) | the probe, the three conditions, the code, the constants |
| **this file** | **the verdict: how normal each correction is, and whether projecting it moves the render nearer the joint image** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The normal share** of a direction `v` at a state: `‖σ_t · J_ε v‖ / ‖v‖` with `J_ε` the Jacobian of the frozen base model's unconditional denoiser, read by one finite-difference forward in float32; 0 means along the manifold, 1 means fully off it where the manifold picture holds.
- **The true correction** `r_t`: guided joint-prompt prediction minus guided PoE prediction at the cached state. **The adapter's correction**: the PoE prediction with the rank-32 step-30050 adapter on minus with it off, same state, λ 1.
- **The ratio**: mean over seeds 9 to 16 of the per-seed mean over steps 0 to 10 of the adapter's normal share divided by the true correction's.
- **Adapter alone, projected, projected late**: λ 1.2 on all 50 steps unprojected; the same with the normal part of the correction dropped on every step; the same dropped only from step 10.
- **d(joint)**: DINOv2 ViT-S/14 cosine distance to the seed's joint-prompt render, lower is nearer. **Compose count**: seeds of 8 the validated detector counts two or more animals on.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (group 1: a read on cached states, then an intervention on the direction of the correction, no training). A missed bar on question 1 closes the mechanism half; a missed bar on question 2 closes the fix and the fidelity caveat is written with the cache read's number in it.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| smoke on a shared biggpu device | proves the process | 2026-09-06 02:58, mscluster109 device 0 over SSH; log `.../showcase/logs/tangent_projection_smoke.log` | one second | none | refused by the launcher's guard: the device had read 2.9 GB at 0% minutes earlier and 27.7 GB at 100% at launch (another user's job ramped); nothing ran, which is the guard doing its job |
| job 50336, `bigbatch` | proves the process | 2026-09-06 03:03, Slurm placed it on mscluster65 | one second | none | failed in the guard: the node's GPU is hardware-faulted (`Unable to determine the device handle for GPU0`), now recorded in [nodes](../../../environment/hpc/nodes.md) with the four other faulted idle nodes a probe found |
| job 50341, `bigbatch`, `STAGE=smoke_then_all`: the smoke, then the cache read, renders, score, figures, W&B | tests the claim | 2026-09-06 03:07, mscluster75 (RTX 3090, 24 GB), `co3`, pid 118730; log `.../showcase/logs/tangent_projection_50341.log` | about 3 to 4 GPU-hours on this card | none | both UNets loaded on the 24 GB card (the memory fit is proven), then the smoke's first row failed on a device mismatch in the cosine helper (a cached tensor on the CPU against a live one on the GPU); fixed in source, job resubmitted |
| job 50344 then 50346, `bigbatch`, `STAGE=hsweep`: the instrument against its step size | proves the instrument | 2026-09-06 03:15 and 03:25, mscluster55 (RTX 3090), `co3` | 6 and 10 minutes | `h_sweep.json`, `normal-share-against-the-finite-difference-step.png` | done; verdict under [Asked after the result](#asked-after-the-result) |
| job 50342, `bigbatch`, `STAGE=smoke_then_all`: smoke, cache read, 48 renders, score, figures, W&B | tests the claim | 2026-09-06 03:11 to 05:54, mscluster53 (RTX 3090, 24 GB), `co3`, pid 240739; log `.../showcase/logs/tangent_projection_50342.log` | 2h 43m wall | `cache_read.json` (400 rows), `renders/` (48), `results.json`, six figures, W&B run `prime_lab/poe-repair-animals-compose/sgvdb9gl` | done. Slurm records the job as FAILED on exit code 2: the launcher's `case` block was edited while bash was still reading the file, so the shell hit a syntax error after the final stage had finished and printed its run id. Every stage ran and every output is on disk. The pattern is already in the [cluster shell pitfalls](../../../environment/known-failures.md) memory: never edit a running bash launcher |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Question 1. Over steps 0 to 10 of the cached plain-PoE run, is the adapter's correction more normal to the base model's manifold than the true correction?**
      The bar, in source as `NORMAL_RATIO_SUPPORT = 1.25` and `NORMAL_RATIO_NULL = 1.05` in `scripts/showcase/tangent_projection.py`, on the ratio defined above:
      **support** if the ratio is at or above 1.25 (the adapter pushes off the manifold at least a quarter harder per unit norm);
      **null** if at or under 1.05;
      **inconclusive** between, or whenever the linearity check fails (`FD_MAX_LINEARITY_GAP = 0.15` on the mean relative gap between the share at `FD_REL_STEP = 0.01` and at 0.02).
      Answer: **⚪ inconclusive by the rule, and a null on the statistic.** The linearity gap is 0.182 against a 0.15 bar, so the rule fires. The statistic it would have judged is 0.985, under the 1.05 null bar and nowhere near the 1.25 support bar, and it is 0.981 read on steps 0 to 10 and 0.978 on steps 11 to 49, so no window of the run has the adapter pushing off the manifold harder than the true correction. Both readings say the same thing and the hypothesis behind question 1 is dead: whatever makes the corrected render soft, it is not that the adapter's direction is more off-manifold than the target's. From `cache_read.json`, fields `statistic.ratio_adapter_over_true_early_mean` and `statistic.linearity_gap_mean`, 400 rows over 8 seeds and 50 steps.

- [ ] ⚠️ **Question 2. Does a projected condition bring the cat × dog render nearer the joint-prompt image while holding composition?**
      The bar, in source as `MIN_MONO_GAIN = 0.05` and `MAX_COMPOSE_LOSS_SEEDS = 1`, against the adapter-alone condition rendered in the same process:
      **support** if a projected condition's 8-seed mean d(joint) is at least 0.05 below the adapter's, its compose count within one seed, and the control pair's compose count within one seed;
      **composition breaks** if it is nearer by 0.05 but loses two or more composed seeds;
      **inconclusive** if the control pair loses two or more;
      **null** otherwise.
      Answer: **❌ null, and one condition damages the picture.** Neither projected condition gets nearer the joint render; both move away from it. Mean DINOv2 distance to the seed's joint render over 8 seeds: adapter alone 0.472, subtracted every step 0.666, subtracted from step 10 0.482 (gains −0.194 and −0.010 against a bar of +0.05). Composition holds or improves by the detector (7 of 8, then 8 of 8 and 7 of 8) and the control pair keeps 8 of 8 in every condition, so the null is on fidelity alone. What the every-step column actually renders is worse than the number suggests: on seeds 9, 10, 11 and 15 it leaves the photograph and draws a grey engraving, a sepia painting, or in seed 10's case three grey blobs that are not animals at all. Its higher mean Laplacian sharpness (79.3 against the adapter's 57.1) is that engraved texture being counted as edges, the failure mode already recorded for this measure. From `results.json`, `summary.a_cat__x__a_dog`.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ **Where does the random direction sit?** Flat near 1 for most of the run and slowly falling: 0.994 over steps 0 to 10, 0.902 over steps 11 to 49. The PoE prediction itself tracks it (1.034, then 0.941). That flat reference is what makes the two corrections readable, and it splits the run in two. Over steps 0 to 10 the true correction reads 0.780 and the adapter's 0.766, both *under* the random direction, so at the steps that decide the composition both corrections lie in directions the frozen denoiser reacts to about a fifth less than to a random direction. Over steps 11 to 49 they read 1.665 and 1.628, so from step 10 on both sit in the directions it reacts to nearly twice as strongly as to a random one. The adapter's error (adapter minus true) reads 0.730 early and 1.090 late, which is the random direction's own profile: the part the adapter gets wrong has no geometry of its own.
- [x] ✅ **Does projecting the adapter's correction bring it closer to the true one?** No, it moves it slightly away. Mean over the run, cos(adapter, true) 0.867 against cos(tangent parts) 0.844; over steps 0 to 10, 0.818 against 0.764. The adapter's error is not concentrated in the part the probe calls normal, so removing that part cannot recover fit. This is the cache-side twin of question 2's null and it was readable before a single render was spent.
- [x] ❌ **Is the projection idempotent?** No, and where it fails is exactly where the picture breaks. Normal share of the adapter's correction before and after one subtraction, mean of 8 seeds: step 0 0.70 → 0.66, step 5 0.72 → 0.53, step 10 0.98 → 0.81, then it inverts: step 20 2.08 → 2.09, step 30 2.17 → 2.53, step 35 1.42 → 2.06, step 45 1.13 → 1.72. A projector would drive the second reading toward zero. Instead, once the share passes 1 the subtraction overshoots and leaves a vector the denoiser responds to *more* than before. So `v − σ_t J v` is a projection only over roughly steps 0 to 10, and the every-step condition spends the other forty steps adding an overshoot, which is what the grey engravings in its column are.
- [x] ✅ **Does the joint-prompt probe tell the same story as the unconditional one?** Yes. Run means: true correction 1.471 under the unconditional probe against 1.425 under the joint-prompt one; adapter's correction 1.439 against 1.373. The two agree to about 5%, so the read is about the base model's geometry at that state and not about which prompt the probe was conditioned on.
- [x] ✅ **Did the identity check pass?** Yes, and by a wide margin. Largest mean absolute pixel difference over the 16 compared renders (both pairs, 8 seeds) between the adapter-alone render here and the corrector tail stage's k=0 render of the same configuration: 0.025 grey levels, against a 6.0 bar and a cross-device fp16 drift band of about 2. The unprojected baseline is the shipped path op for op. From `renders.json`, `identity_check`.
- [x] ⚪ **Does the projected-late condition keep more seeds than the every-step one?** No, it keeps one fewer: 7 of 8 against 8 of 8, both against the adapter's 7. Composition is untouched by either projection, so nothing here says the early normal part carries the second animal. The two conditions separate on fidelity instead, and in the direction the idempotence read predicts: leaving steps 0 to 10 alone, where the subtraction really is a projection, costs almost nothing (d(joint) 0.482 against the adapter's 0.472, 4 of 8 seeds nearer the joint render), while subtracting on all fifty steps costs a lot (0.666, 2 of 8 nearer).

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was
written with the answer already visible.

- [ ] ⚠️ **Is the finite difference in its linear regime at all, and is the split a projector?** (raised by the smoke's three rows, before any full-run number existed.) On seed 9, steps 0 to 2, the true correction's normal share reads 0.52, 0.51, 0.80 at the run's step and 0.43, 0.51, 0.62 at twice it, so the read falls as the step grows and the pre-registered linearity gap (0.23 on those rows) is over its 0.15 bar. And one projection lowers the adapter's share only from 0.55 to 0.40 at step 0 and from 0.85 to 0.80 at step 2, where a projector would take it near zero. So `σ_t J` is not the 0-or-1 operator the ideal manifold picture gives; it has a spectrum. The random direction reads 1.00 and the PoE prediction itself 1.02 at every step, so the corrections at 0.5 to 0.8 are still far more tangent than a random direction. The step-size sweep (`--h-sweep`, job 50344 on mscluster55, seed 9 at the twelve frame steps, six steps from 0.0025 to 0.08 of ‖x_t‖) answers it: **the read is not linear in the step at any step index, and the split is not a projector.** From step 10 on the "share" exceeds 1 (true correction at the run's step: 1.02 at step 10, 1.73 at 15, 2.80 at 20, 2.46 at 30, 2.12 at 40, 0.86 at 49), while the random direction reads 1.00 falling to 0.74. A share over 1 means `σ_t J` amplifies the correction, so its eigenvalues along these directions are above 1 and `v − σ_t J v` overshoots: one subtraction lowers the share at steps 0 to 10 only partly (0.55 to 0.46 at step 0) and raises it from step 15 on (0.67 to 0.99 to 1.49 at step 35). The reading: the early corrections lie in directions the base denoiser barely reacts to (share 0.5 to 0.8 against 1.0 for a random direction), and the mid-run corrections lie in the directions it reacts to *most*, on both the true and the adapter's correction alike. Numbers in `h_sweep.json`; picture `normal-share-against-the-finite-difference-step.png`. Consequences, decided before any render was scored: question 1 is read as designed (its linearity flag will fire and it will read inconclusive by its own rule); question 2's two subtraction conditions are still scored as the interventions they are, with this caveat beside them; and a follow-on condition `projected_ls` is added that does what Proposition 1 asks in the paper's own form, minimise the normal energy `‖J w‖`, as a one-dimensional line search along `σ_t J v` with `α = ⟨Jv, Jn⟩ / ‖Jn‖²` (two probe forwards a step, exact on any eigen-direction whatever its eigenvalue). It is judged by question 2's bar and labelled a follow-on, never as the pre-registered condition. The sweep was rerun with the line-search read beside the subtraction's (job 50346 on mscluster55, `h_sweep.json`), and the line search does not rescue the read: at steps 0 to 10 it behaves as the linear picture predicts (α 0.9 to 1.1 at the run's step, normal energy down to 0.45 to 0.8 of itself), but from step 15 on the α that minimises the energy is negative on most steps (−0.05 at 20, −0.31 at 30, −0.43 at 35, −0.39 at 40, −0.45 at 45, adapter's correction, step 0.01) and the energy falls by at most a third. A negative α means the denoiser's response to the "normal" direction is not the linear image of it, so no one-forward operator built on this difference is a projection there. Decision, taken before any render was scored: the follow-on condition is not rendered, since a render of an operator that does not do what its name says would be a picture with no reading; the code path stays in `scripts/showcase/tangent_projection.py` (`--conditions projected_ls`) for a version of the probe that is linear, which on this evidence means a proper least-squares removal with vector-Jacobian products rather than a finite difference. What the sweep does establish stands on its own: at steps 0 to 10 both corrections sit in directions the base denoiser responds to about half as much as to a random direction, and they are alike in this (question 1's ratio, read on the smoke's rows, 0.99).

- [ ] ⚠️ **Why is the corrected render soft, if not this?** Question 1 rules out the direction being more off-manifold than the target's, and question 2 rules out removing that part as a fix. What the cache read does show is that from step 10 on, both the true correction and the adapter's lie in the directions the frozen denoiser amplifies most, roughly twice a random direction. The adapter spends fifty steps pushing along the model's most responsive directions, and the plan-14 result that the softness is already in the running estimate by step 20 sits in the same window. A next read that would separate these: the same normal-share curve computed on the *corrected* trajectory rather than the cached plain-PoE one, since the states diverge and the amplification is a property of the state.

- [ ] ⚠️ **The detector counted two animals on three grey blobs.** Seed 10 of the every-step column is an abstract grey shape on a pale ground with no animal in it, and the validated instance-count scorer returned 2. Seed 11 returned 3 on a sepia painting of a cat and a kitten. The compose count is load-bearing in six findings, so this belongs beside the existing note that the count cannot tell a cat and a dog from two dogs. It does not change any verdict here, because question 2 is a null on fidelity and composition was never the deciding read.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** The three conditions run through the same sampler, from the same cached noise, in one process, differing only in whether and from which step the normal part is dropped. Mono and plain PoE are reference columns from the corrector sheet stage, not compared cells. The counts confirm 8 seeds per condition per pair, 48 renders in all, and the identity check above confirms the baseline is the shipped path.
- [x] ⚠️ **Was the measuring tool sound?** Partly, and the parts are separable. The state the probe was fed is right: the live frozen PoE prediction agrees with the cached one at cosine 0.9996 or better on all 400 rows. The probe itself is a float32 copy of the frozen UNet with TF32 off. What is not sound is reading `σ_t J` as the projector `I − P`: it fires the linearity check, the step-size sweep shows no flat regime in `h`, and the idempotence read shows the split is a projection only over steps 0 to 10. Every claim above is written on what the quantity is (how much the frozen denoiser's output moves per unit of input, against a random direction as the yardstick) rather than on the manifold interpretation, and that quantity is measured soundly. Sharpness is Laplacian variance and counts edges, which is why the every-step column scores high while looking worse; the eye read and the DINOv2 distance both say the opposite and are quoted instead.
- [x] ✅ **Did the run respect the environment?** Output under `/datasets`, the launcher's guards passed, node, device and PID in the log header and here, `torch.cuda.is_available()` true on the pinned device. Three launches were refused before this one: a shared biggpu device whose foreign job ramped between the check and the launch, and a `bigbatch` node whose GPU is hardware-faulted. Both refusals came from the guards rather than from a crash mid-run.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a "projected correction" row on the wall, if supported | the probe (unconditional denoiser, fp32 finite difference), the ratio from question 1, and that d(joint) is DINOv2 distance on one pair |
| the fidelity caveat, if null | the ratio and the random-direction reference, so the caveat says whether the softness is an off-manifold push or not |
| the instrument's source | Saito and Matsubara, arXiv 2510.05509, Proposition 1, cited as the origin of the tangent read and nothing else |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| `σ_t J` is a projector only over about steps 0 to 10; past that it amplifies, and the subtraction overshoots | a proper least-squares removal built on vector-Jacobian products rather than a finite difference, which would hold whatever the eigenvalue | nothing in this plan; both its questions are answered without it. The line-search code path stays in the script behind `--conditions projected_ls` and was deliberately not rendered, since a render of an operator that does not do what its name says is a picture with no reading |
| the same normal-share curve on the corrected trajectory rather than the cached plain-PoE one | one more cache read with the states taken from a corrected run | the "why is it soft" question above |
| the training-time version (a penalty on the normal part of the adapter's output) | it needed a supported question 2, and question 2 is null, so this is closed rather than open | nothing |
| the instance-count scorer returning 2 on a picture with no animal in it | a pass over the every-step column's eight renders against the scorer, filed beside the existing scorer caveats | six findings lean on the compose count |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Both questions are answered, so nothing in this plan is owed a run. What the result feeds: the paper's fidelity caveat is now written with a number behind it (the adapter's correction is no more off-manifold than the target's, ratio 0.985), and the showcase wall gets no new row. The mechanism question the result opened, why the corrected render is soft when the direction is not the reason, is the "why is it soft" item under [Still open](#still-open) and belongs to whichever plan next reads the corrected trajectory.
