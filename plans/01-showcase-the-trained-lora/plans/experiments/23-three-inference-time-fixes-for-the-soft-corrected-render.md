# 🧪 Three inference-time fixes for the soft corrected render: does CFG++ re-noising, the APG projection or keeping PoE's norm make the two-animal render crisp?

**This plan asks one question: with the rank-32 step-30050 correction at its shipped setting, does any of three published guidance fixes return the plain run's sharpness on the seeds where the animals are kept?**

**Step 65 in the root running order. Ran 2026-09-06; verdict null for all three fixes, in [the review file](../../review/23-three-inference-time-fixes-for-the-soft-corrected-render.md). Waits on nothing: the reference renders exist from [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md). Sits beside plan 14, which varied when the correction acts, and plan 19, which varied the sampler's noise; this one varies how the correction enters the DDIM step.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/01-showcase-the-trained-lora/plans/experiments/23-three-inference-time-fixes-for-the-soft-corrected-render.md — the three fix cells rendered, scored and stripped; verdict in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 53 (feeds this) | [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md) | the Mono, plain PoE and full-window λ 1.2 renders this plan reuses as its reference columns; found the corrected running estimate is smoother than every plain seed by step 20 |
| 61 (sibling) | [19-does-a-stochastic-sampler-sharpen-the-corrected-render](19-does-a-stochastic-sampler-sharpen-the-corrected-render.md) | the same question with fresh noise per step |
| **65 (current)** | **23: three inference-time fixes** | CFG++ re-noising, the APG projection and the norm-keeping rescale, each on the full window, against the full window itself |
| 35 (next in scope) | [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) | the wall; a supported cell here becomes a row on it |

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

- **The correction**: the rank-32 adapter at training step 30050, held out on cat × dog. At each of the 50 DDIM steps it changes the product-of-experts (PoE) noise prediction; **λ** multiplies that change. λ 1.2 on every step is the shipped setting and plan 14's **full window**.
- **The same-loop full window**: the full window rendered through this plan's loop with the hook off, condition `full_1.2_sameloop`. It is the column every fix is judged against, because it differs from a fix by the hook alone. Plan 14's full-window render is the identity check's reference and a strip column is not drawn from it.
- **The PoE prediction**: the guided noise prediction the three frozen branches give, `eps_A + eps_B − eps_∅` at guidance 7.5. **The corrected prediction** is the PoE prediction plus λ times the correction.
- **CFG++ re-noising** (Chung et al. 2024, arXiv 2406.08070): the corrected prediction forms the running estimate of the finished image (the Tweedie mean), and the DDIM step that moves to the next noise level re-adds noise using the PoE prediction instead of the corrected one. In CFG++ the analogous move is to re-noise with the unconditional prediction rather than the guided one.
- **The APG projection** (Sadat et al. 2025, arXiv 2410.02416): the correction is split into the part parallel to the PoE prediction and the part orthogonal to it. The parallel part is multiplied by `APG_ETA` (0 here, APG's recommended default), the orthogonal part is kept whole. APG found the parallel part of a guidance update drives oversaturation and the orthogonal part carries the quality.
- **Keeping PoE's norm**: the corrected prediction is rescaled to the PoE prediction's norm at every step, direction unchanged. This isolates the norm argument from the direction change.
- **Parallel share**: `‖parallel part‖ / ‖correction‖` at one step. **Parallel coefficient**: `⟨correction, PoE⟩ / ‖PoE‖²`; negative means the correction shortens the PoE prediction, which is the damping [what the correction is made of](../../../../report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md) measured (the reachable part weights each expert at 1 to 3 where PoE weights it 7.5).
- **Sharpness**: Laplacian variance of the greyscale 1024 px render, `_laplacian_var` in `scripts/showcase/lambda_window_grid.py`. Higher is crisper. It counts edges, so a sketch-style seed scores above a photograph; that is why every read here is paired per seed against the same seed's full-window render and never a mean against the plain band (plan 14's premise gate fired on that band, seeds 11, 14 and 15 being line drawings).
- **d(Mono)**: the DINOv2 ViT-S/14 cosine distance from a render to the same seed's joint-prompt render. Lower is nearer the joint-prompt image.
- **Both-ness**: projection of the DINOv2 embedding toward the "a cat and a dog" cloud on the axes fitted for the landing finding; cat × dog only.
- **Compose count**: seeds of 8 where the validated detector counts at least two animal instances ([compose rate](../../../../context/world/compose-rate.md)).
- **A strip**: one row per seed, tiles Mono | plain PoE | same-loop full window λ 1.2 | CFG++ | APG | norm kept, with the detector's count, the sharpness, d(Mono) and both-ness under each tile.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** The corrected render composes and comes out soft. Plan 14 asked when the softness enters and found the corrected running estimate is already smoother than every plain seed by step 20. Plan 19 asks whether fresh noise per step brings the texture back. This plan asks a third question about the same render: the correction is a guidance-shaped additive term on a DDIM chain, and the classifier-free-guidance literature has three fixes for exactly the blur and wash such a term produces. Each is one line in the sampler and costs one render sweep.

**The hypothesis.** The softness is the corrected prediction leaving the model's own manifold, through one of two routes the fixes separate: the re-noising step carrying the correction (CFG++), or the correction's parallel part shrinking the prediction (APG, norm kept).

**If true.** At least one fix keeps the compose count within one seed of the same-loop full window and is sharper than it on 6 or more of the 8 seeds. The showcase wall gets a "fix" row, the paper's fidelity caveat becomes a sampler choice, and the per-step diagnostics say which route carried the blur.

**If false.** Every fix either loses two or more seeds (the composition needs the part the fix removes) or gains sharpness on 4 or fewer seeds (the softness is not in how the correction enters the step). Then the blur is in the correction's direction itself, and the training-side plans (15 weight decay, 20 EMA, 16 clean-estimate loss) are the remaining routes.

**Dataset.** Cat × dog, held-out seeds 9 to 16, the cached initial noise per seed. No control pair: the fixes are judged against the full-window render of the same seed, and a fix that breaks the composition shows up in the compose count directly.

**Associated materials.**
- Review questions: [the review file](../../review/23-three-inference-time-fixes-for-the-soft-corrected-render.md)
- The paper-scout shortlist that ranked these three fixes: [the selection record](/home-mscluster/mmolefe/goal-setting/learning/t2i-compositional-failure/paper-scout/selection-2026-09-06-manifold-fixes.md)
- The finding the parallel-share diagnostic tests: [what the correction is made of](../../../../report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** One identity render, then 3 fixes × 8 seeds plus the same-loop full window × 8 seeds at 50 steps, each step two UNet passes: 33 renders at about 62 s each on a Quadro RTX 8000, about 40 minutes. Scoring about 5 minutes. The three reference columns are plan 14's renders, copied in.

**Buys.** The one review question, the per-step parallel-share and norm-ratio curves (a free reading of the correction finding on the live trajectory), and one wall row if a fix is supported.

**Prerequisites.** The rank-32 checkpoint at step 30050; plan 14's `render_manifest.json` with its Mono, PoE and full-window renders for cat × dog; the cloud-axes features file; the training cache's step-0 latents.

**W&B project:** `prime_lab/poe-repair-animals-compose`, run name `crisp_fix_strip_r32_030050`. Output root: `/datasets/mmolefe/poe_repair_min/outputs/showcase/crisp_fix_strip/`.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The shared-device path: `biggpu` allows one Slurm job per user, so the render runs with `nohup` over SSH on a pinned free device, invisible to `squeue`, harvested with `pgrep` ([execution protocol](../../../../environment/hpc/execution-protocol.md)).
- `co3` python on the RTX 8000 and A6000 nodes (106, 108, 109) and on the session node; `co3_bw` on 110 to 112 ([nodes](../../../../environment/hpc/nodes.md)).
- The launch script checks `torch.cuda.is_available()` under the pinned device before real work ([poe-launch-002](../../../../environment/known-failures.md)).
- Outputs on `/datasets` only, and the disk guard reads the filesystem the script writes to ([storage](../../../../environment/storage.md)).
- fp16 end to end. The identity check against plan 14's render is a mean absolute pixel difference bound of 6.0 grey levels; combining the two predictions in fp32 rather than fp16 alone moves a 50-step render by about 3 grey levels on the same card, which is why the bar's reference is rendered through this loop and plan 14's render is the check, not the reference.
- DINOv2 embeds on the GPU: on the CPU it routes through a CUDA-only xformers kernel ([poe-mem-002](../../../../environment/known-failures.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Changing how the correction enters the DDIM step, with the correction itself unchanged, returns the plain run's sharpness while keeping the two animals.**

**Why this matters right now.** The paper's fidelity caveat says the corrected run is softer. If a one-line sampler change removes the softness, the caveat becomes a sampler setting and the shipped render on the wall is the fixed one.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Plans 07, 14 and 19 varied how much correction, when it acts, and how much noise the sampler injects. None varied what the sampler does with the corrected prediction once it has it. The three published fixes each change exactly that and nothing else.

**The solution.** One sampler with a hook between the two predictions and the DDIM step. The hook has four settings: off (the same-loop full window, the bar's reference, and on seed 9 the identity check against plan 14), CFG++, APG, norm kept. Render, score with the same instruments as plan 14, strip per seed.

**Key insights.**
1. The correction finding measured that the reachable part of the true correction damps both experts. If the adapter learned that damping, the corrected prediction is shorter than PoE's, and a shorter prediction at the same noise level is a smoother running estimate. The norm-kept cell tests this alone.
2. CFG++ separates the two roles of the prediction in a DDIM step: pointing at the image, and putting the noise back. Only the first needs the correction.
3. The per-step parallel share and norm ratio are recorded on every render, so whichever way the verdict falls, the diagnostic that explains it is already on disk.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
at every step k of 50:   eps_poe  = frozen branches         eps_lora = branches with the adapter
                          delta    = eps_lora − eps_poe       par = <delta, eps_poe> / |eps_poe|^2 · eps_poe

full window (hook off)   eps_t = eps_poe + 1.2 delta          x0 ← eps_t     re-noise ← eps_t
cfgpp                    eps_t = eps_poe + 1.2 delta          x0 ← eps_t     re-noise ← eps_poe
apg (eta 0)              eps_t = eps_poe + 1.2 (delta − par)  x0 ← eps_t     re-noise ← eps_t
normkeep                 eps_t = (eps_poe + 1.2 delta) · |eps_poe| / |eps_poe + 1.2 delta|
```

Every cell: the seed's cached initial noise, guidance 7.5, 1024 square, DDIM eta 0, the adapter disabled on the sampler's exit.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The reference copy**: plan 14's Mono, plain PoE and full-window renders for cat × dog seeds 9 to 16 copied under the output root with their source path in `render_manifest.json`.
2. **The identity check** (`--identity`): the hook set to off on seed 9 must reproduce plan 14's full-window render within 6.0 grey levels (`IDENTITY_MAX_MEAN_ABS_DIFF_CROSS_DEVICE`); the 1.0 same-device figure is recorded beside it and is informative only, since the two loops combine the predictions in different precisions. Output: `identity_check.json`.
3. **The render** (`--render`, `--render-baseline`): the three fixes and the same-loop full window on 8 seeds, each with a per-step diagnostics file (`‖delta‖`, parallel share, parallel coefficient, norm ratio of the prediction used).
4. **The scoring** (`--score`): compose (instance count), sharpness at 1024 px, d(Mono), both-ness; per fix the paired counts against the same-loop full window; the fp16 floor between plan 14's full window and the same-loop one, per seed; the verdict by `verdict_for_fix`; `results.json`, `cell-table.md`, the diagnostics figure, the W&B run with its cells table and per-step curves.
5. **The strips** (`--strips`): one strip per seed, logged to the W&B run as `strips/seed_<n>` and as the `strips_by_seed` table.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objective 3 (the softness question measured rather than argued). Checkable outcomes:

1. The identity check passing before any fix cell is trusted.
2. Every fix cell rendered, scored, on a strip, in W&B, with `results.json` beside the strips.
3. The review file's one question answered support, null or inconclusive by the constants in source.
4. The per-step diagnostics figure filed in `artifacts/results/`.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/01-showcase-the-trained-lora/plans/experiments/23-three-inference-time-fixes-for-the-soft-corrected-render.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--render-score-and-strip-on-a-shared-device)**.

### 1. 🧪 Render, score and strip on a shared device

◀ **Needs:** a free device claimed per [the launch recipe](../../../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#2-launch-on-a-shared-device), with `pgrep -af 'crisp_fix|correct_early'` checked on the node first.

- [x] **1.1 Launch the full chain** (identity, render, score, strips) on the claimed device.
  - Command: `ssh <node> 'GPU=<idx> STAGE=all nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/crisp_fix_strip_shared_device.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/crisp_fix_all.log 2>&1 &'`
  - Done when: the log ends in `DONE stage=all`, `identity_check.json` reads `pass_cross_device: true`, `results.json` carries a verdict per fix, and the W&B run's Media tab shows `strips/seed_<n>` for 8 seeds. Record node, device, PID and the W&B run id in the review file's Runs table.
  - Launched 2026-09-06 03:26 on mscluster108 device 1 over SSH, bash PID 321031.
- [x] **1.2 Render the same-loop full window, then re-score and re-strip against it** (`STAGE=baseline`, same launch shape, log `crisp_fix_baseline.log`). Queued behind task 1.1 by a waiter on the node. Done when `results.json` reads `reference_full_window: full_1.2_sameloop`, `fp16_floor_grey_levels` carries 8 seeds, and the strips' third tile is the same-loop full window.

▶ **Next: [instruction 2.1](#2--judge-the-cells)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 2. 👁️ Judge the cells

◀ **Needs: [task 1.1](#1--render-score-and-strip-on-a-shared-device)** done.

2.1 **Open the eight strips** under `/datasets/mmolefe/poe_repair_min/outputs/showcase/crisp_fix_strip/strips/` or the W&B run's Media tab (`strips/seed_9` to `strips/seed_16`). Tiles left to right: Mono, plain PoE, full window λ 1.2, CFG++, APG, norm kept. A green frame is a tile the detector counts as two animals, red is one. ✅ a fix column keeps green where the full-window column is green and looks as crisp as the plain PoE tile; ❌ the fix column turns red or stays as soft as the full-window tile.

2.2 **Check the numbers agree with the eye**: `results.json` → `summary.<fix>` has `compose_n`, `kept_full_window_seeds_n`, `sharper_than_full_n`, `nearer_mono_than_full_n`, `verdict`. Open `correction-parallel-share-and-norm-per-step.png` beside them: the middle panel's sign says whether the correction shortens PoE's prediction, and the right panel says which fix restored the norm. Then write the verdict into the [review file](../../review/23-three-inference-time-fixes-for-the-soft-corrected-render.md).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

**Pass criteria:**
- The identity check passes: the hook set to off reproduces plan 14's full-window render of seed 9 within 6.0 grey levels. If it fails the chain stops and nothing else is read. It read 2.9 on mscluster108 device 1, the card plan 14 also used.

**Fail criteria:**
- Plan 14's manifest lacks a Mono, PoE or full-window render for any seed: the reference copy raises and the chain stops.

**When you get results, answer the questions in the [review file](../../review/23-three-inference-time-fixes-for-the-soft-corrected-render.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

Every figure below lands in `artifacts/results/does-an-inference-time-fix-make-the-corrected-render-crisp/`, with its card entry in that folder's `README.md`.

| Figure | What is plotted | What it argues | What it may not claim | File |
|---|---|---|---|---|
| one strip per seed | tiles Mono, plain PoE, same-loop full window λ 1.2, CFG++, APG, norm kept for one seed; under each tile the detector's animal count, Laplacian sharpness at 1024 px, d(Mono) and both-ness; frame green for two animals, red for one | what each fix does to one seed, so the eye can check the table's numbers | anything about the mean; a strip is one seed | `strips/strip-seed_<n>.png`, W&B `strips/` and `strips_by_seed` |
| the cell table | one row per condition: compose count of 8, seeds of the same-loop full window kept, mean sharpness, seeds sharper than it, mean d(Mono), seeds nearer Mono than it, both-ness mean, verdict; below it the fp16 floor between plan 14's full window and the same-loop one | which fix, if any, is supported | anything about pairs other than cat × dog | `cell-table.md`, `results.json` |
| correction parallel share and norm per step | three panels against denoising step 0 to 49: the parallel share of the correction (thin lines one seed, thick mean); the parallel coefficient with its sign; the norm ratio of the prediction used to PoE's, one curve per fix | whether the correction shortens PoE's prediction and which fix undoes that | anything about the direction of the orthogonal part | `correction-parallel-share-and-norm-per-step.png` |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.2](#2--judge-the-cells)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md)**, which takes a supported cell as a wall row.

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/showcase/crisp_fix_strip.py`: `apply_fix` (the hook and its diagnostics), `FixSampler.fixed` (plan 14's loop with the hook), `identity_check`, `verdict_for_fix`, the stages, the constants. `scripts/showcase/crisp_fix_strip_shared_device.sh`: the shared-device launcher with the disk, python, memory, fault and CUDA guards.

**Reused from plan 14** (`scripts/showcase/correct_early_then_clean_up.py`): `Sampler` (context, prompts, pinned noise, adapter attach), `_cloud_axes`, `_laplacian_var`, `mean_abs_diff`, `_write`; the instance-count scorer; `DinoEmbedder`.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

`/analyze-run` on the W&B run once the chain lands, then `/analyze-figure` on the eight strips before the verdict is written.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Task 3.2: `/sync-plan-tree plans/01-showcase-the-trained-lora/`. The strips are judged and the verdict written.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

| Symptom | Cause | Fix |
|---|---|---|
| identity check reads 2 to 6 grey levels | fp16 drift: a different card, or the two loops combining the predictions in different precisions (fp32 here, fp16 in plan 14) | the 6.0 bar applies; the bar's reference is the same-loop full window, never plan 14's render |
| identity check reads above 6 | the hook is not a no-op when off, or the adapter state differs | stop; diff `FixSampler.fixed` against `Sampler.scheduled` |
| scoring dies in DINOv2 on the CPU | the xformers kernel is CUDA-only ([poe-mem-002](../../../../environment/known-failures.md)) | run `--score` on a device |
| launcher exits 4 | the device has more than 1 GiB in use | pick another device |
