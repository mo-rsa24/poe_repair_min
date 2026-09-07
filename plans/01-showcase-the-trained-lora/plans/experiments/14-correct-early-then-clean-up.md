# 🧪 Correct early, then clean up: does a clean plain-PoE tail give two animals at plain-PoE sharpness?

**This plan asks one question: if the correction is switched off after the early steps and plain product-of-experts finishes the run, do we keep the two animals and get the plain run's sharpness back?**

**Step 53 in the root running order. Waits on nothing: task 1 needs no GPU. Sits beside [07-experiment-c-lambda-window](07-experiment-c-lambda-window.md), whose grid found that sharpness does not fall steadily with λ.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/01-showcase-the-trained-lora/plans/experiments/14-correct-early-then-clean-up.md — the schedule grid, the tail-length cell and the re-noise cell rendered and scored; verdict in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 37 (sibling) | [07-experiment-c-lambda-window](07-experiment-c-lambda-window.md) | λ swept at a fixed window: sharpness 75.6 at λ 0, 55.8 at 0.5, 69.0 at 1.0, no steady fall |
| 52 (feeds this) | [06-where-each-condition-lands](../../../05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md) | the per-step frames this plan's first task reads; commit step median 15 for the corrected run, range 8 to 35 |
| **53 (current)** | **14: correct early, then clean up** | when the softness enters, then a λ schedule, a longer tail and a re-noise cell against it |
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

- **The correction**: the rank-32 adapter at training step 30050, held out on cat × dog. At each denoising step it changes the product-of-experts noise prediction; **λ** is the multiplier on that change, so λ 0 is plain product-of-experts (PoE) and λ 1.2 is the shipped setting.
- **The full-window run**: λ 1.2 at every one of the 50 steps. This is "adapter alone" on the sheets.
- **A schedule**: λ as a function of the denoising timestep `t` (1000 is pure noise, 0 is the finished image). At 50 DDIM steps the sampler visits `t = 981 − 20k` at step index `k`, so step 10 is `t 781` and step 20 is `t 581`. Defining the schedule in `t` is what lets the 200-step run switch off at the same noise level.
- **The hard cut**: λ 1.2 while `t ≥ 781` (steps 0 to 10 inclusive), 0 after.
- **The decay**: λ 1.2 while `t ≥ 781`, then falling in a straight line to 0 at `t 581` (step 20), 0 after. The decay exists because the corrected run's commit step reaches 35 on some seeds, and a cut at 10 may let those seeds fall back.
- **The tail**: the steps after λ reaches 0, run as plain PoE with the adapter disabled.
- **Sharpness**: the Laplacian variance of the greyscale image, the same function plan 07 used (`_laplacian_var` in `scripts/showcase/lambda_window_grid.py`). Higher is crisper. It is computed on the 1024 px final render for the cells, and on the 256 px saved frames for the per-step read, so the two sets of numbers are not on one scale.
- **The plain-PoE band**: the lowest and highest sharpness over the 8 plain-PoE seeds of a pair. "Back in the band" means a cell's mean sharpness is at or above the band's lower edge.
- **Both-ness**: the projection of a render's DINOv2 embedding onto the axis from the midpoint of the "a cat" and "a dog" clouds toward the "a cat and a dog" cloud, in cosine units, from the axes already fitted in `artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space-dino-feats.npy`. Defined for cat × dog only, because only that pair has reference clouds.
- **The running estimate**: the model's guess of the finished image at a step (the Tweedie mean), decoded. The saved frames under `where_each_condition_lands/frames/` are these.
- **The re-noise cell**: the schedule run to step 20, its running estimate taken as a clean image, noise added back to the level of step 35 (`t 281`), then plain PoE from step 35 to the end.
- **The re-noise level sweep**: the same move with the level varied, `RENOISE_LEVELS_T = (481, 381, 281, 181)` (steps 25, 30, 35, 40), and the committed image taken from two sources: the running estimate at step 20 (`x0s20`, the re-noise cell's source) and the finished decay render (`final`). This is the knob that Consistency Trajectory Models' γ-sampling and Restart sampling turn: how much of the committed image is re-synthesised by the tail. Cells are named `decay_10_20_renoise_<source>_t<level>`.
- **A strip**: one row per seed and pair, tiles Mono | plain PoE | full window | best schedule | best re-noise cell, the detector's count, the sharpness and the both-ness printed under each tile.
- **Compose rate**: the fraction of the 8 seeds where the validated detector counts at least two animal instances ([compose rate](../../../../context/world/compose-rate.md)).

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** Plan 07 asked whether the blur grows with λ and got no steady trend. This plan asks the other question about the same blur: does it come from the early steps, where the correction decides the composition, or from the late steps, where the correction keeps acting on an already-committed image? The corrected run commits by step 15 on the median seed ([where each condition lands](../../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md)), so a correction that stops around there has done its work.

**The hypothesis.** The softness enters late. Switching the correction off after the early window, and letting plain PoE finish, keeps the two animals and returns the plain run's sharpness.

**If true.** A schedule cell keeps compose rate within one seed of the full-window run and its mean sharpness is back in the plain-PoE band. The showcase wall gets a "correct early" row and the paper's fidelity caveat shrinks to the early window.

**If false.** Either the animals go with the correction (the schedule composes two or more seeds fewer), or the softness is already in the committed structure at step 20 and no tail fixes it; then only the re-noise cell can reach it, and if that fails too the softness is the correction's own property.

**Dataset.** Cat × dog, held-out seeds 9 to 16, the cached initial noise per seed; and the composing control pair butterfly × flower meadow on the same seeds, so a schedule that breaks what already works is caught. The training cache stores one initial noise per seed, shared across pairs (the step-0 latents of the two pairs are identical on seeds 9 to 12), and the control pair's cache stops at seed 12, so its seeds 13 to 16 take the cat × dog cache's noise for those seeds.

**Associated materials.**
- Review questions: [the review file](../../review/14-correct-early-then-clean-up.md)
- Ledger entry: [the longer-training question runs as three experiments](../../decisions-taken-here.md#the-longer-training-question-runs-as-three-experiments-re-scoped-by-evidence), which this plan extends with a fourth read on the same blur
- The per-step frames this plan reads first: [where each condition lands](../../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** Task 1 is CPU only, minutes. The render stages: 5 conditions × 8 seeds × 2 pairs at 50 steps (80 runs), 2 conditions × 16 cells at 200 steps (32 runs at four times the length), and 16 re-noise runs of 15 steps each. Corrected steps run two UNet passes. About 2 GPU-hours on a Quadro RTX 8000. Scoring adds about 15 minutes.

**Buys.** The one review question, and one row on the showcase wall if any cell is supported.

**Prerequisites.** The rank-32 checkpoint at step 30050; the saved frames; the cloud-axes features file; the training cache's step-0 latents for both pairs.

**W&B project:** `prime_lab/poe-repair-animals-compose`. Output root: `/datasets/mmolefe/poe_repair_min/outputs/showcase/correct_early_then_clean_up/`.

**The refiner cell.** The SDXL refiner is not in the Hugging Face cache on this cluster (only `stabilityai/stable-diffusion-xl-base-1.0` is). The cell that would replace the base tail with the refiner is not run and not downloaded.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The shared-device path: `biggpu` allows one Slurm job per user, so the render runs with `nohup` over SSH on a pinned free device, invisible to `squeue`, harvested with `pgrep` ([execution protocol](../../../../environment/hpc/execution-protocol.md)).
- `co3` python on the RTX 8000 and A6000 nodes (106, 108, 109) and on the session node; `co3_bw` on 110 to 112 ([nodes](../../../../environment/hpc/nodes.md)).
- The launch script checks `torch.cuda.is_available()` under the pinned device before real work, because a card can list in `nvidia-smi` and still run on the CPU ([poe-launch-002](../../../../environment/known-failures.md)).
- Outputs on `/datasets` only, and the disk guard reads the filesystem the script writes to ([storage](../../../../environment/storage.md)).
- fp16 end to end; frames and renders are compared within the fp16 drift band, so a "byte-close" check is a mean absolute pixel difference bound, never equality.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A correction confined to the steps before the run commits, followed by a plain-PoE tail, gives two animals at plain-PoE sharpness.**

**Why this matters right now.** The paper's fidelity caveat is currently "the corrected run is softer". If the softness is a late-window effect, the caveat becomes a schedule choice rather than a property of the fix.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Plan 07 varied how much correction; nothing has varied when. The two are different questions, and the second is the cheaper fix if it works, since it needs no training.

**The solution.** Read when the sharpness of the running estimate separates between the plain and corrected runs, using frames already on disk. Then render the schedules, the longer tail and the re-noise cell, and score each the same way.

**Key insights.**
1. The corrected run's commit step (median 15, range 8 to 35) says the composition is decided early; the correction after that is free to be switched off.
2. If sharpness is already below the plain band at step 20, the blur is in the structure, and a tail cannot reach it; a re-noise can, because it re-synthesises texture from the committed layout.
3. The 200-step tail tests whether the tail was simply too short, with the switch-off pinned to the same noise level so only the tail's length changes.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
denoising step k (50 DDIM):   0 ..... 10 ..... 20 ..... 35 ..... 50
timestep t:                 981      781      581      281       1

full window   λ: 1.2 ─────────────────────────────────────────── 1.2
hard cut      λ: 1.2 ────────┐0 ──────────────────────────────── 0
decay         λ: 1.2 ────────╲ (straight line) ╲0 ────────────── 0
tail 200      same λ(t) as the best schedule, 200 steps, t_k = 996 − 5k
re-noise      best schedule to k 20 → x̂0 → + noise to t 281 → plain PoE k 35..50
plain PoE     λ: 0 ───────────────────────────────────────────── 0
```

Every cell: the seed's cached initial noise, guidance 7.5, 1024 square, the adapter disabled on every off-window step and on the sampler's exit.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The per-step sharpness read** (`--sharpness-over-steps`): Laplacian variance of every saved frame for all six conditions; one thin line per seed, a thick mean per condition; the falsification read at steps 20 and 50 written to a sidecar. Output: `artifacts/results/does-correcting-early-then-cleaning-up-restore-sharpness/sharpness-over-denoising-steps.{png,json}`.
2. **The detachment proof** (`--detach-check`): in one process, plain PoE on seed 9 before the adapter is attached, then a hard-cut run, then plain PoE again through a sampler that never touches the adapter. The two plain renders must agree within `DETACH_MAX_MEAN_ABS_DIFF` grey levels, and the second is also compared to the cached `poe.png`. Output: `detach_check.json` under the output root.
3. **The schedule grid** (`--render --stage schedule`): mono, plain PoE, full window, hard cut, decay; 8 seeds; both pairs. Mono is rendered before the adapter is attached.
4. **The scoring** (`--score`): compose (instance count), sharpness at 1024 px, both-ness (cat × dog), butterfly presence (control pair), `results.json`, one sheet per condition with the four columns Mono | PoE | this cell | full window, logged to W&B.
5. **The tail and re-noise cells** (`--render --stage tail`, `--render --stage renoise`): the best schedule by the rule in source (`pick_best_schedule`: highest compose count, ties to higher mean sharpness) at 200 steps beside plain PoE at 200 steps; and the re-noise cell. Scored the same way.
6. **The re-noise level sweep** (`--render --stage renoise-sweep`): one full decay run per seed supplies both committed images; each is noised to each level and finished by plain PoE (`Sampler.renoise_heads`, `Sampler.renoise_tail`). Scored the same way, plus the post-hoc per-seed paired read (`sharper_than_full_n`, `sharper_than_poe_n`, `kept_full_window_seeds_n`) beside the bar.
7. **The strips** (`--strips`, no GPU): one strip per seed and pair, saved under `strips/` and logged to the W&B run as images and as the `strips_by_seed` table (one row per seed, one tile per column).

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objective 3 (the softness question measured rather than argued) and goal 4's reading of experiment C. Checkable outcomes:

1. The per-step sharpness figure and its sidecar, with the step-20 and step-50 read answered.
2. The detachment proof passing before any windowed cell is trusted.
3. Every cell rendered, scored, on a sheet, in W&B, with `results.json` beside the sheets.
4. The review file's one question answered support, null or inconclusive, by the constants in source.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/01-showcase-the-trained-lora/plans/experiments/14-correct-early-then-clean-up.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--read-when-the-softness-enters-no-gpu)**.

### 1. 📊 Read when the softness enters (no GPU)

◀ **Needs:** the frames under `/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames/<cond>/seed_<n>/step_<kk>.png` (they exist, 6 conditions × 8 seeds × 14 saved steps).

- [ ] **1.1 Run the per-step sharpness read.**
  - Command: `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python scripts/showcase/correct_early_then_clean_up.py --sharpness-over-steps`
  - Writes the figure and sidecar named in the Figure Catalog. The falsification constants `EARLY_STEP = 20` and `LATE_STEP = 50` and the band rule live in the script.
  - Done when: the sidecar's `verdict` field reads `late`, `committed` or `inconclusive`, and the review file's first "written before" question carries it.

▶ **Next: [task 2.1](#2--prove-the-adapter-is-detached-after-a-windowed-run)**.

### 2. 🔬 Prove the adapter is detached after a windowed run

◀ **Needs: [task 1.1](#1--read-when-the-softness-enters-no-gpu)** done, and a free device claimed per [the launch recipe](../../../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#2-launch-on-a-shared-device) with `pgrep -af 'sweep|train|corrector'` checked on the node first.

- [ ] **2.1 Run the detachment proof on the claimed device.**
  - Command: `ssh <node> 'GPU=<idx> STAGE=detach nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/correct_early_then_clean_up_shared_device.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/correct_early_detach.log 2>&1 &'`
  - Done when: `detach_check.json` reports `pass: true` (mean absolute pixel difference between the plain render before attaching and the plain render after a windowed run at or under `DETACH_MAX_MEAN_ABS_DIFF`). If it fails, nothing after this task runs; fix the sampler's exit first.

▶ **Next: [task 3.1](#3--render-and-score-the-schedule-grid)**.

### 3. 🧪 Render and score the schedule grid

◀ **Needs: [task 2.1](#2--prove-the-adapter-is-detached-after-a-windowed-run)** passed.

- [ ] **3.1 Render the schedule grid** (`STAGE=schedule`), same launch shape, log `correct_early_schedule.log`. 80 renders under the output root's `renders/<pair>/<condition>/seed_<n>.png`.
- [ ] **3.2 Score it** (`STAGE=score`): `results.json`, the per-condition sheets, the W&B run. Record node, device, PID and the W&B run id in the review file's Runs table.

▶ **Next: [task 4.1](#4--the-tail-length-and-the-re-noise-cell)**.

### 4. 🧪 The tail length and the re-noise cell

◀ **Needs: [task 3.2](#3--render-and-score-the-schedule-grid)** done, because `pick_best_schedule` reads `results.json`.

- [ ] **4.1 Render the 200-step tail** (`STAGE=tail`): the best schedule at 200 DDIM steps with λ(t) unchanged, and plain PoE at 200 steps, both pairs, 8 seeds.
- [ ] **4.2 Render the re-noise cell** (`STAGE=renoise`): best schedule to step 20, running estimate re-noised to `t 281`, plain PoE for steps 35 to 50.
- [ ] **4.3 Re-score** (`STAGE=score`); the new cells join `results.json` and get their own sheets and W&B images.
- [ ] **4.4 Render, score and strip the re-noise level sweep** (`STAGE=sweep`), same launch shape, log `correct_early_sweep.log`. 128 tails behind 16 head renders, about 45 minutes on a Quadro RTX 8000. Done when the eight `decay_10_20_renoise_<source>_t<level>` cells sit in `results.json` with verdicts, `renoise_duplicate_check.max` is at or under `DETACH_MAX_MEAN_ABS_DIFF`, and the review file's sweep question is answered.
- [ ] **4.5 Log the per-seed strips** (`STAGE=strips`, or on the session node: `CUDA_VISIBLE_DEVICES="" /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python scripts/showcase/correct_early_then_clean_up.py --strips`). Re-run after every score so the strip's last two columns pick up the best schedule and the best re-noise cell. Done when the W&B run's Media tab shows `strips/<pair>/seed_<n>` for 16 seeds and the `strips_by_seed` table.

▶ **Next: [instruction 5.1](#5--judge-the-cells)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 5. 👁️ Judge the cells

◀ **Needs: [task 4.3](#4--the-tail-length-and-the-re-noise-cell)** done.

5.1 **Open each sheet** under `/datasets/mmolefe/poe_repair_min/outputs/showcase/correct_early_then_clean_up/sheets/` (or the W&B run's Media tab). Rows are seeds 9 to 16; columns Mono, plain PoE, the cell, full window. ✅ the cell's column shows two animals on the same rows the full-window column does, and looks as crisp as the PoE column; ❌ animals vanish or the column is as soft as the full-window one.

5.2 **Check the numbers agree with the eye**: `results.json` → `summary.<pair>.<condition>` has `compose_n`, `sharpness_mean`, `both_ness_mean`, and `verdict`. Then write the verdict into the [review file](../../review/14-correct-early-then-clean-up.md).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

**Pass criteria:**
- The detachment proof passes (task 2.1). A windowed sampler that leaves the adapter attached contaminated the reference renders once already (2026-09-05).
- The premise holds at 1024 px: the full-window run's mean sharpness sits below the plain-PoE band's lower edge on cat × dog. If it does not, there is no softness to fix at this checkpoint and the plan closes 🟡 on that fact rather than running the tail cells.

**Fail criteria:**
- Plain PoE at λ 0 through the new sampler differs from the cached `poe.png` by more than `IDENTITY_MAX_MEAN_ABS_DIFF` grey levels: the sampler is wrong, and nothing else is read.

**When you get results, answer the questions in the [review file](../../review/14-correct-early-then-clean-up.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

Every figure below lands in `artifacts/results/does-correcting-early-then-cleaning-up-restore-sharpness/`, with its card entry in that folder's `README.md`.

| Figure | What is plotted | What it argues | What it may not claim | File |
|---|---|---|---|---|
| sharpness over denoising steps | y: Laplacian variance of the 256 px running estimate; x: denoising step 0 to 50; one thin line per seed, thick mean per condition, six conditions; grey band: plain-PoE seed min to max | whether the corrected run's sharpness leaves the plain band before or after step 20 | anything about the 1024 px render; the frames are 256 px thumbnails, so absolute values are not those of plan 07 | `sharpness-over-denoising-steps.png`, `.json` |
| one sheet per cell | rows seeds 9 to 16; columns Mono, plain PoE, the cell, full window; the scorer's count and the sharpness printed under each tile | what the numbers are counting | fidelity beyond what the eye sees at sheet size | `sheet-<pair>-<condition>.png` |
| the cell table | one row per cell: compose count of 8, mean sharpness, plain band edges, both-ness mean, verdict | which cell, if any, is supported | anything about pairs other than cat × dog and the control | `cell-table.md`, `results.json` |
| one strip per seed | tiles Mono, plain PoE, full window λ 1.2, best schedule, best re-noise cell for one seed of one pair; under each tile the detector's animal count, the Laplacian sharpness at 1024 px and the both-ness | what the fix does to one seed, so the eye can check the table's numbers | anything about the mean; a strip is one seed | `strips/strip-<pair>-seed_<n>.png`, W&B `strips/` and `strips_by_seed` |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 6.1 | new patterns into the catalogs |
| Close out | `/sync-plan-tree` | task 6.2 | statuses aggregated up |

### 6. 🧹 Close out

◀ **Needs: [instruction 5.2](#5--judge-the-cells)** done.

- [ ] **6.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **6.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md)**, which takes a supported cell as a wall row.

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/showcase/correct_early_then_clean_up.py`: the schedule-aware sampler (`Sampler.scheduled`, adapter disabled on every off-window step and on exit), the re-noise sampler (`Sampler.renoise`), the sweep's heads and tails (`Sampler.renoise_heads`, `Sampler.renoise_tail`), the strips (`_strips`, `_wandb_strips`), the stages, the constants. `scripts/showcase/correct_early_then_clean_up_shared_device.sh`: the shared-device launcher with the disk, python, memory, fault and CUDA guards.

**Reused:** `_laplacian_var` and `_attach_and_load_lora` from `scripts/showcase/lambda_window_grid.py` (rank overridden to 32), `run_cfg` and `run_cfg_poe` from `poe_repair/methods/_sampling.py`, the instance-count scorer, `DinoEmbedder`.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

```
/run-experiment plans/01-showcase-the-trained-lora/plans/experiments/14-correct-early-then-clean-up.md — task 1 on the session node, then the detachment proof before any windowed cell, then the stages in order on one claimed device
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) takes a supported cell as a row; a null goes to the paper's fidelity caveat as a bounded sentence.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From project catalog

From [environment/known-failures.md](../../../../environment/known-failures.md):

- **poe-lora-003**: every condition in a comparison runs through the identical sampler, differing only in λ(t). Mono is the one exception (single-prompt CFG) and is a reference column, never a compared cell.
- **poe-lora-004**: the new samplers carry `@torch.no_grad()`.
- **poe-mem-002**: `_Embedders` is not used here; the DINOv2 embedder is `DinoEmbedder` on CPU and the detector takes an explicit device.
- **poe-launch-002**: the launcher refuses a device whose utilisation reads `[N/A]` and probes `torch.cuda.is_available()` before starting.
- **The windowed sampler leaves the adapter attached** (memory note, 2026-09-05): task 2.1 exists to prove the new sampler does not.

## Cross-references

- The next lever this plan owns is handed to it by [the finding "can a corrector or a clean tail sharpen the adapter's renders"](../../../../report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md), with its bar (a 0.05 fall in DINOv2 distance to the joint render) and its baseline (the adapter alone at 0.472) already measured.
