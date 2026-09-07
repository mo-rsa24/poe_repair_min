# 🧪 Keep the correction on the manifold: is the adapter's softness an off-manifold push, and does dropping that part restore fidelity?

**Both questions are answered (W&B run `sgvdb9gl`, verdicts in the review file): question 1 inconclusive by its linearity rule and a null on its statistic, question 2 null. This plan asks two questions about one object. First, read on the cached plain-PoE states whether the rank-32 adapter's correction points off the base model's image manifold more than the true correction does. Second, render cat × dog with that off-manifold part removed at every step and ask whether the picture lands nearer the joint-prompt image while still showing two animals.**

**Step 57 in the root running order. Waits on nothing: the cache, the checkpoint and the reference renders all exist. Sits beside [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md), which asks *when* the softness enters; this plan asks *what direction* it is.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/01-showcase-the-trained-lora/plans/experiments/15-keep-the-correction-on-the-manifold.md — the cache read, the three conditions rendered and scored, the figures filed, the W&B run id and both verdicts in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 53 (sibling) | [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md) | when the softness enters: committed by step 20 on the saved frames; the schedules keep five seeds and sharpen |
| 53 (feeds this) | [07-what-the-correction-is-made-of](../../../05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md) | the same cached states, the correction split against the experts' own span; this plan splits it against the manifold instead |
| **57 (current)** | **15: keep the correction on the manifold** | the normal share of each correction per step, then the projected renders against the adapter alone |
| 35 (next in scope) | [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) | the wall; a supported condition here becomes a row on it |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Words this plan uses](#words-this-plan-uses)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Words this plan uses

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **The correction**: the rank-32 adapter at training step 30050, held out on cat × dog. At each denoising step it changes the product-of-experts (PoE) noise prediction by a vector; **λ** is the multiplier on that vector, 1.2 the shipped setting.
- **The true correction, `r_t`**: the guided joint-prompt prediction minus the guided PoE prediction at the same cached state, the thing the adapter was trained to reproduce ([what the correction is made of](../../../../report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md) uses the same definition).
- **The manifold**: the set of noisy latents the frozen base model treats as natural at noise level `t`. It is read through the base model's unconditional denoiser `ε_∅`: for data on a manifold with tangent projector `P`, the denoiser's Jacobian satisfies `σ_t ∂ε_∅/∂x = I − P`, so `σ_t · J v` is the part of a direction `v` that points off the manifold. This is Saito and Matsubara's Proposition 1 (arXiv 2510.05509, "the null directions of the score Jacobian are the tangent space") written on the denoiser.
- **The normal part** of a direction `v`: `σ_t · (ε_∅(x_t + h v̂) − ε_∅(x_t)) / h · ‖v‖`, one extra forward of the frozen UNet in float32, with `h = 0.01 · ‖x_t‖`. **The tangent part** is `v` minus that. **The normal share** is `‖normal‖ / ‖v‖`, the sine of the angle between `v` and the tangent space where the manifold picture holds, 0 when `v` lies along the manifold.
- **The probe**: the float32 copy of the frozen UNet that computes the normal part. No adapter is ever attached to it. Its primary prompt is the empty one (the base model's own manifold); its secondary prompt is the joint prompt "a cat and a dog", reported beside it.
- **Projected**: the adapter run at λ 1.2 on all 50 steps where, at every step, the adapter's correction has its normal part removed before it is scaled and added. **Projected late**: the same, but only from step 10 on, so the early steps that decide the composition run as the adapter alone.
- **The adapter alone**: λ 1.2 on all 50 steps, unprojected, rendered in the same process as the projected conditions so the comparison differs on one axis.
- **Mono**: the joint-prompt render, the reference the fidelity read measures distance to. **Plain PoE**: λ 0.
- **d(joint)**: DINOv2 ViT-S/14 cosine distance between a render and the seed's Mono render, the compose scorer's embedder; lower is nearer the clean image. The same read the clean-tail cells of plan 04 in scope 06 use.
- **Compose count**: seeds of 8 where the validated detector counts two or more animal instances ([compose rate](../../../../context/world/compose-rate.md)). On the control pair butterfly × flower meadow the validated rule does not apply, so a butterfly box and a flower box both present is reported as unvalidated.
- **Both-ness**: projection of a render's DINOv2 embedding onto the axis from the solo-cloud midpoint toward the joint-prompt cloud, on the axes fitted for [where each condition lands](../../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md), refit here from the saved features and checked against the sidecar.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** The corrected render is softer than plain PoE and than the cached true correction's render ([the best-case panel](../figures/12-close-f8a-and-the-best-case-panel.md)). Plan 14 found the softness is already in the running estimate by step 20, so a clean tail cannot reach it. Nothing yet says which *component* of the adapter's push causes it. A denoiser averages whatever is pushed off its manifold into grey, low-contrast texture, which is what the soft renders look like. So the question is whether the adapter's correction carries a larger off-manifold part than the true correction, and whether removing that part, step by step, gives a crisper render without losing the second animal.

**The hypothesis.** The true correction lies mostly along the base model's manifold; the adapter's correction carries a larger normal part; dropping the normal part brings the render nearer the joint-prompt image while composition holds.

**If true.** The cache read's ratio clears its bar, the projected condition's mean d(joint) falls by at least 0.05 against the adapter alone with the compose count within one seed, and the showcase wall gets a "projected correction" row. The paper's fidelity caveat becomes a sentence about direction, with a fix that costs one forward pass per step and no training.

**If false.** Either the two corrections are equally normal (the softness is not an off-manifold push and the read closes that door), or projection kills the composition (the normal part carries the second animal, which is itself a finding about what the correction is), or it does neither and the render is unchanged (the normal part is too small to matter at λ 1.2).

**Dataset.** Cat × dog, held-out seeds 9 to 16, the cached initial noise per seed and the cached plain-PoE trajectories under `artifacts/caches/training_cache/heldout/`; the composing control pair butterfly × flower meadow on the same seeds for the renders, so a projection that breaks what already works is caught.

**Associated materials.**
- Review questions: [the review file](../../review/15-keep-the-correction-on-the-manifold.md)
- The paper the instrument comes from: Saito and Matsubara, "Be Tangential to Manifold: Discovering Riemannian Metric for Diffusion Models", arXiv 2510.05509. It is an interpolation paper and says nothing about composition; only its Proposition 1 is used.
- The finding this extends: [what the correction is made of](../../../../report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md), which split the same vectors against the experts' span rather than the manifold.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** The cache read: 8 seeds × 50 steps × about 12 float32 UNet forwards, about one GPU-hour on an RTX A6000. The renders: 3 conditions × 8 seeds × 2 pairs = 48 renders; the projected ones cost two fp16 three-branch forwards plus two fp32 forwards per step, about two minutes each, so about 1.5 GPU-hours. Scoring about 15 minutes. About 3 GPU-hours in all.

**Buys.** Both review questions, one row on the showcase wall if supported, and a per-step manifold decomposition of the correction that no other plan has.

**Prerequisites.** The rank-32 checkpoint at step 30050; the held-out cache; the Mono and plain-PoE references under the corrector output root (`sheet/references/` and `sheet/pairs/`); the corrector tail stage's k=0 renders for the identity check; the landing finding's saved DINOv2 features for the cloud axes.

**W&B project:** `prime_lab/poe-repair-animals-compose`. Output root: `/datasets/mmolefe/poe_repair_min/outputs/showcase/tangent_projection/`.

**Memory.** The process holds the fp16 UNet with the adapter (5 GB) and a float32 copy of the frozen UNet (10 GB) at once. With the text encoders moved off the card after encoding, the VAE decoding in tiles and slices, and the allocator set to expandable segments, the peak is about 21 GB, so it fits one 24 GB card that carries nothing else. The smoke stage runs first inside the job and proves the fit before the long stages start. It does not fit beside another SDXL job on the same card.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The shared-device path: `biggpu` allows one Slurm job per user, so the run goes over SSH with `nohup` on a pinned device, invisible to `squeue`, harvested with `pgrep` ([execution protocol](../../../../environment/hpc/execution-protocol.md)). The device guard is `launch_sdlora.sh`'s: under 4 GB of idle foreign residency at 0% utilisation is shared, anything active is refused.
- `co3` python on the RTX 8000 and A6000 nodes (106, 108, 109); `co3_bw` on 110 to 112 ([nodes](../../../../environment/hpc/nodes.md)).
- The launch script asserts `torch.cuda.is_available()` under the pinned device before real work ([poe-launch-002](../../../../environment/known-failures.md)).
- Outputs on `/datasets` only, with the disk guard reading the filesystem the script writes to ([storage](../../../../environment/storage.md)).
- The probe runs in float32 with TF32 off, because a float16 difference of two forwards at this step size is the size of the rounding noise; the sampler's own forwards stay fp16 so the adapter-alone condition matches the shipped path within the cross-device drift band.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The adapter's correction pushes off the base model's manifold harder than the true correction does, and removing that off-manifold part at every step gives a render nearer the joint-prompt image with the second animal kept.**

**Why this matters right now.** The paper's fidelity caveat is a sentence about softness with no mechanism behind it. If the softness is a normal component, the caveat becomes a statement about direction, and the fix is a one-forward-pass projection that needs no retraining and no schedule.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Plan 07 varied how much correction, plan 14 varied when, and the span finding split the correction against the experts' own predictions. None of them split it against the geometry of the base model. The softness could be an off-manifold push, and that is testable on the cache with one extra forward per step.

**The solution.** Read the normal share of five directions at every cached state: the true correction, the adapter's, a random direction, the PoE prediction itself and the adapter's error. Then render the projected conditions beside the adapter alone and score them the way the clean-tail cells were scored.

**Key insights.**
1. The same finite-difference object is both the read and the fix, so a supported read comes with the intervention already built.
2. The random-direction line is the reference for how large the normal space reads at each step; without it a share of 0.3 has no meaning.
3. The projection is done on the frozen unconditional denoiser, so it can only remove what the base model calls unnatural; it cannot add composition. Any gain in composition is therefore a side effect to be reported, never a claim.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
at one cached state x_t:

   true correction  r_t  ──┐                         ┌─ tangent part  (along the manifold)
   adapter's        r̂_t  ──┼─ σ_t · J_ε · v  ──────►─┤
   random           z    ──┤   (one fp32 forward)    └─ normal part   (off the manifold)   share = ‖normal‖/‖v‖
   PoE prediction   ε_PoE ─┘

the projected run, step k:
   ε_frozen, ε_lora  (fp16, adapter off / on)  →  Δ = ε_lora − ε_frozen
   Δ_tan = Δ − σ_t J_ε Δ                        (fp32 probe on the frozen ε_∅)
   ε_t = ε_frozen + 1.2 · Δ_tan                  →  DDIM step

conditions (8 seeds, both pairs):   adapter alone │ projected every step │ projected from step 10
references (already on disk):       Mono (joint prompt) │ plain PoE
```

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The cache read** (`--cache-read`): for every cached step of seeds 9 to 16, the adapter's correction at the cached state (adapter on minus adapter off, same process), the true correction from the cache, and the normal share of each direction under the unconditional probe (at `FD_REL_STEP` and, for the two corrections, at twice that as a linearity check) and under the joint-prompt probe; the cosine between the adapter's and the true correction whole, tangent parts and normal parts; the idempotence read at the frame steps. Verdict on question 1 by the constants in source. Output: `cache_read.json`.
2. **The renders** (`--render`): the three conditions on both pairs, 8 seeds, from the cached initial noise; Tweedie frames at twelve steps on cat × dog; the identity check of the adapter-alone render against the corrector tail's k=0 render. Output: `renders/<pair>/<condition>/seed_<n>.png` with a per-step sidecar each, `renders.json`.
3. **The scores** (`--score`): compose count, d(joint), sharpness, both-ness and the frame tracks on the cloud axes; verdict on question 2 by the constants in source. Output: `results.json`.
4. **The figures** (`--figures`): the six files in the Figure Catalog, each with a sidecar.
5. **The W&B run** (`--wandb`): per-step mean curves, the per-seed table, the per-render table, every image, the sidecars as an artifact, both branches in the summary.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objective 3 (the softness question measured rather than argued) and goal 4's reading of experiment C. Checkable outcomes:

1. `cache_read.json` with the question-1 branch and the linearity check passing.
2. Every render on disk, the identity check passing, `results.json` with the question-2 branch.
3. The six figures filed under `artifacts/results/does-keeping-the-correction-on-the-manifold-restore-fidelity/` with their card.
4. The review file's two questions answered support, null or inconclusive, by the constants in source, with the W&B run id.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/01-showcase-the-trained-lora/plans/experiments/15-keep-the-correction-on-the-manifold.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--prove-the-process-runs-on-the-claimed-device)**.

### 1. 🔬 Prove the process runs on the claimed device

◀ **Needs:** a free device claimed per [the launch recipe](../../../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#2-launch-on-a-shared-device), with `pgrep -af 'sweep|train|corrector|tangent'` checked on the node first.

- [x] **1.1 Run the smoke stage first, on the device the long stages will use.**
  - On a shared biggpu device: `ssh <node> 'GPU=<idx> STAGE=smoke bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/tangent_projection_shared_device.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/tangent_projection_smoke.log 2>&1'`, then task 2.1 with `STAGE=all`.
  - When every biggpu device is busy: `sbatch --exclude=<the faulted nodes in environment/hpc/nodes.md> scripts/showcase/tangent_projection.sbatch`, which runs `STAGE=smoke_then_all` on a fresh `bigbatch` card, the smoke first and the long stages only if it ends clean; task 2.1 is then the same job.
  - One seed, three cached steps, one three-step render under the output root's `smoke/`. Done when: the log shows the smoke's two stages ending without a traceback and `smoke/cache_read.json` has three rows whose `sanity_cos_live_frozen_vs_cached_poe` is above 0.99 (the live frozen prediction agrees with the cache).

▶ **Next: [task 2.1](#2--run-the-cache-read-and-the-renders)**.

### 2. 🧪 Run the cache read and the renders

◀ **Needs: [task 1.1](#1--prove-the-process-runs-on-the-claimed-device)** passed.

- [x] **2.1 Launch every stage in order** (`STAGE=all` over SSH with `nohup`, log `tangent_projection_all.log`; or the `sbatch` job from task 1.1, whose log is `tangent_projection_<jobid>.log`). Record node, device, PID or job id, and the W&B run id in the review file's Runs table.
- [x] **2.2 Harvest.** `pgrep -af tangent` on the node, then the counts: 400 rows in `cache_read.json`, 48 files under `renders/`, `results.json` present, six PNGs under the results folder. Read the log on the launch node, not through the session node's NFS view.

▶ **Next: [instruction 3.1](#3--judge-the-pictures)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 3. 👁️ Judge the pictures

◀ **Needs: [task 2.2](#2--run-the-cache-read-and-the-renders)** done.

3.1 **Open the two sheets** under `artifacts/results/does-keeping-the-correction-on-the-manifold-restore-fidelity/` (or the W&B run's Media tab). Rows are seeds 9 to 16; columns Mono, plain PoE, adapter alone, projected every step, projected from step 10. ✅ a projected column shows two animals on the same rows the adapter column does and looks nearer the Mono column in texture and colour; ❌ animals vanish, or the column looks like the adapter column.

3.2 **Check the numbers agree with the eye**: `results.json` → `summary.a_cat__x__a_dog.<condition>` has `composed_of_8`, `mean_dino_dist_to_mono`, `mono_gain_vs_adapter_alone`; `branch` and `reasons` at the top. Then write both verdicts into the [review file](../../review/15-keep-the-correction-on-the-manifold.md).

3.3 **Read the normal-share graph** once: the true, adapter and random lines against step. The pre-registered window is the grey band. If the random line sits at or below the two corrections, the normal space is large at that step and the share means little there; say so in the review.

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

**Pass criteria:**
- The smoke stage ends clean and the live frozen prediction agrees with the cache (cosine above 0.99 on the smoke rows).
- The linearity check: the mean relative gap between the normal share at `FD_REL_STEP` and at twice it is at or under `FD_MAX_LINEARITY_GAP = 0.15`. Above it, question 1 reads inconclusive whatever the ratio.
- The identity check: the adapter-alone render agrees with the corrector tail's k=0 render within `IDENTITY_MAX_MEAN_ABS_DIFF = 6.0` grey levels (cross-device fp16 drift is about 2). A failure means the sampler is not the shipped path, and the projected conditions are not read.

**Fail criteria:**
- The control pair loses more than one composed seed under a projected condition: the projection breaks what already works and the condition reads inconclusive.

**When you get results, answer the questions in the [review file](../../review/15-keep-the-correction-on-the-manifold.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

Every figure below lands in `artifacts/results/does-keeping-the-correction-on-the-manifold-restore-fidelity/`, with its card entry in that folder's `README.md`.

| Figure | What is plotted | What it argues | What it may not claim | File |
|---|---|---|---|---|
| normal share over denoising steps | left: y the normal share `‖σ_t J v‖/‖v‖`, x step 0 to 49, five directions (true, adapter, random, PoE prediction, adapter's error), thin per seed, thick mean, grey band steps 0 to 10; right: cosine between the adapter's and the true correction, whole and by part | whether the adapter's push is more normal than the true one, and where in the run | that the share is an angle, except where the manifold picture holds; the random line says how large the normal space reads | `normal-share-over-denoising-steps.png`, `.json` |
| corrections split along and off the manifold | one panel per step (0, 2, 5, 10, 20, 30, 49): arrows from the origin, x the tangent norm, y the normal norm, green true, orange adapter, thin per seed, thick mean; the manifold is the x axis | the same split as a picture: which correction leans off the manifold and by how much, in latent units | that the tangent spaces at different steps or seeds are the same subspace; the picture aligns them | `corrections-split-along-and-off-the-manifold.png`, `.json` |
| normal share during the projected runs | y the normal share of the adapter's correction at the live state before it is dropped, x step; both projected conditions | whether the live trajectory reads like the cached one | anything about the unprojected run's states | `normal-share-during-the-projected-runs.png`, `.json` |
| cat × dog in DINOv2 space with the projected correction | left: the landing finding's 48 points as faint clouds plus this run's endpoints (◆) on the which-animal and both-ness axes; right: the Tweedie-frame tracks of each condition at twelve steps | where the projected renders land relative to the joint-prompt cloud, and the path they take | distances in the full space (two axes of 384; off-plane norm about 0.8) | `cat-x-dog-in-dino-space-with-the-projected-correction.png`, `.json` |
| the two eight-seed sheets | rows seeds 9 to 16; columns Mono, plain PoE, adapter alone, projected every step, projected from step 10; the count, d(joint) and sharpness under each tile; green frame = the detector counts two | what the numbers are counting, beside the pictures they count | fidelity beyond what the eye sees at sheet size | `mono-vs-poe-vs-adapter-vs-projected-cat-dog-eight-seed-sheet.png`, `...-butterfly-meadow-...`, `.json` each |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 4.1 | new patterns into the catalogs |
| Close out | `/sync-plan-tree` | task 4.2 | statuses aggregated up |

### 4. 🧹 Close out

◀ **Needs: [instruction 3.2](#3--judge-the-pictures)** done.

- [ ] **4.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **4.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md)**, which takes a supported condition as a wall row.

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/showcase/tangent_projection.py`: the float32 probe (`Probe`, `split`), the cache read and its verdict (`cache_read`, `cache_verdict`), the projected sampler (`run_projected`, op for op the corrected sampler's k=0 path plus the projection), the identity check, the scoring and its verdict (`score`, `fix_verdict`), the figures, the W&B logging, and every constant. `scripts/showcase/tangent_projection_shared_device.sh`: the shared-device launcher with the disk, python, foreign-process, fault and CUDA guards. `scripts/showcase/tangent_projection.sbatch`: the Slurm wrapper for a `bigbatch` card, which calls the same launcher.

**Reused:** the cache reader and the guided rule from `scripts/showcase/correction_span_common.py`; `_attach_and_load_lora` from `scripts/showcase/lambda_boundary_probe.py` (rank overridden to 32); `_adapter_disable` and `_adapter_enable` from `poe_repair/methods/_poe_langevin.py`; `_laplacian_var` from `scripts/showcase/lambda_window_grid.py`; the instance-count scorer and the DINOv2 embedder from the compose-scorer validation; the landing finding's saved features for the cloud axes.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

```
/run-experiment plans/01-showcase-the-trained-lora/plans/experiments/15-keep-the-correction-on-the-manifold.md — the smoke in the foreground on one claimed device, then STAGE=all with nohup on the same device
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) takes a supported condition as a row; a null goes to the paper's fidelity caveat as a sentence about direction, bounded by the cache read.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From project catalog

From [environment/known-failures.md](../../../../environment/known-failures.md):

| Symptom | Cause | Fix |
|---|---|---|
| the SSH launch returns exit 2 with no log | a relative path on the launch line, or a script under `/tmp` (`poe-launch-001`) | every path absolute; the script lives under the repo on `/home-mscluster` |
| the run crawls with zero GPU utilisation | a faulted device passed a memory-only check (`poe-launch-002`) | the launcher refuses `[N/A]` utilisation and asserts `torch.cuda.is_available()` |
| the plain reference differs from the cached one by tens of grey levels | a windowed sampler left the adapter enabled (2026-09-05) | the sampler disables the adapter on every frozen forward and on exit; the identity check reads it |

### Specific to this plan

| Symptom | Cause | Fix |
|---|---|---|
| CUDA out of memory at the first probe | two UNets (fp16 with adapter, fp32 frozen) plus 1024 px activations do not fit the card | run on a 48 GB card; the session node's 24 GB is not enough |
| the normal share at twice the step differs by more than 15% | the finite difference is outside the linear range, or fp32 was not used | `FD_REL_STEP` down by half and rerun the cache read; check TF32 is off |
| the sanity cosine between the live frozen prediction and the cached one is under 0.99 | the wrong pair, seed or timestep was fed to the three-branch forward, or the adapter was left on | stop; the cache read is measuring the wrong state |
