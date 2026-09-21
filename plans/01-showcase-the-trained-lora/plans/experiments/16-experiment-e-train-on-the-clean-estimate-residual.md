# 🧪 Experiment E: train the rank-32 adapter on the clean-estimate residual, and read whether the held-out fix gets crisper

**This plan asks one question: if the adapter's loss is the error between clean estimates instead of the error between noise predictions, so the early steps carry the weight they carry in the picture, does the rank-32 adapter at 30k render cat × dog nearer the joint-prompt image and with its contrast back?**

**Step 58 in the root running order. Waits on nothing: the code is a per-timestep weight on the existing loss, the run needs one free device for about 13 hours, and the readout reuses the 8-seed grid, the per-step frames and the DINOv2 axes that plans 14 and 15 already use.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/01-showcase-the-trained-lora/plans/experiments/16-experiment-e-train-on-the-clean-estimate-residual.md — the run reached 40k, the 30k and 40k grids are scored, the readout figures and strips are filed, verdict in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 53 (sibling) | [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md) | when the softness enters, read on the saved frames: committed by step 20 |
| 57 (sibling) | [15-experiment-d-weight-decay](15-experiment-d-weight-decay.md) | the same baseline run with weight decay 0.1; the launcher and verdict shape this plan copies |
| **58 (current)** | **16: experiment E, the clean-estimate loss** | one training axis moved, the 30k and 40k checkpoints read against the baseline's 30,050 |
| 35 (next in scope) | [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) | the wall; a supported checkpoint replaces the baseline adapter in its rows |

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

- **The baseline**: `phase1_r32_100k`, the rank-32 pooled adapter trained on the noise-space error, whose step-30,050 checkpoint is the one every showcase figure uses. Held out on cat × dog it composes 7 of 8 seeds at λ 1 with an 8-seed DINOv2 drift of −0.091.
- **The noise-space error**: at a cached state `x_t`, the mean squared difference between the adapter's product-of-experts noise prediction and the cached joint-prompt noise prediction. This is the baseline's loss, unweighted over the 50 cached DDIM steps.
- **The clean estimate**: the model's guess of the finished image at a step, `x̂0 = (x_t − √(1 − ᾱ_t) ε) / √ᾱ_t` (the Tweedie mean). Two clean estimates from the same `x_t` differ by `√((1 − ᾱ_t)/ᾱ_t)` times the difference of their noise predictions.
- **The clean-estimate error**: the noise-space error multiplied by `(1 − ᾱ_t)/ᾱ_t`. It is the mean squared difference between the two clean estimates. The multiplier is 172 at the first DDIM step (t 981), 22 at step 10 (t 781), 1.2 at step 30 and 0.002 at the last step.
- **The cap**: the multiplier is clipped at 22, its value at step 10, so steps 0 to 10 share one weight and the later steps fall off as the multiplier does. After clipping, the weights are divided by their mean over the 50-step grid, so the loss scale and the learning rate stay comparable with the baseline. Steps 0 to 10 then carry 65% of the total weight (the baseline gives them 22%, one fiftieth each), steps 20 to 49 carry 8%.
- **λ**: the multiplier on the adapter's change to the product-of-experts noise prediction at inference; 0 is plain PoE, 1.0 the setting every read here uses.
- **Drift**: DINOv2 drift, the corrected render's cosine distance to the joint-prompt render minus its distance to the plain PoE render, mean over the 8 held-out cat × dog seeds at λ 1 over all 50 steps. Negative means nearer the joint-prompt image. From `summary.full.1.0.mean_dino_drift` in a probe folder's `results.json`.
- **Compose count**: seeds of 8 where the validated detector counts two or more animal instances ([compose rate](../../../../context/world/compose-rate.md)).
- **Contrast**: the standard deviation of the grey-level pixels of a render, 0 to 255 scale. Higher is punchier; the haze the baseline's corrected renders show is a low value. On the 1024 px final renders the baseline adapter's 8-seed mean is 37.5, plain PoE's 40.6, the joint prompt's 46.4.
- **Both-ness**: the projection of a render's DINOv2 embedding onto the axis from the midpoint of the "a cat" and "a dog" clouds toward the "a cat and a dog" cloud, cosine units, the second view of [where each condition lands](../../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md).
- **The running estimate**: the decoded clean estimate at a step. The saved frames under `where_each_condition_lands/frames/` are these, at 256 px.
- **A strip**: one row per seed, tiles Mono | plain PoE | baseline adapter at λ 1 | new adapter at λ 1, the detector's count, the drift and the contrast printed under each tile.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** The baseline adapter reproduces the length-changing, re-weightable part of the correction almost exactly and about half of its new direction ([what the correction is made of](../../../../report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md)). Its corrected renders are the flattest condition from step 0 and never recover ([the correct-early review](../../review/14-correct-early-then-clean-up.md): committed by step 20). The composition is decided over steps 0 to 10, and the correction measured in clean-estimate space is largest there (0.56 RMS at step 0 against 0.30 at step 30), while the noise-space error the baseline trains on is smallest there (0.04 RMS at step 0 against 0.28 at step 30). So the baseline spends most of its gradient on the steps that decide the least. This plan moves the loss to where the picture is decided and nothing else.

**The hypothesis.** Trained on the clean-estimate error, the rank-32 adapter at 30k renders the held-out pair nearer the joint-prompt image than the baseline does at 30,050, still composes, and gets part of the contrast back.

**If true.** The new checkpoint replaces the baseline in the showcase rows, the fidelity caveat shrinks, and the paper gains a one-line method change with a stated reason (Li and He 2025 on predicting the clean image).

**If false.** Either the animals go (compose count below 6, so the re-weighting traded composition for nothing), or the drift and the contrast stay where the baseline's are, in which case the softness is not a matter of which steps the loss weights and the on-policy arm in the pressure-test route moves up.

**Dataset.** The same 11 training pairs × 8 seeds (88 cells, 4,400 cached steps) the baseline trained on; cat × dog held-out seeds 9 to 16 for every read.

**Associated materials.**
- Review questions: [the review file](../../review/16-experiment-e-train-on-the-clean-estimate-residual.md)
- The read that motivated it: [what the correction is made of](../../../../report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md), and the contrast and clean-estimate-length numbers in the review file's **Words** block
- The baseline's own decay finding, which fixed the drift margin this plan reuses: [does training longer keep improving the held-out fix](../../../../report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** Training: 40k steps at about 1.15 s per step on a Quadro RTX 8000 or an A6000, about 13 hours, plus four tracking evaluations of about 4 minutes each. Probing: two 8-seed grids (λ 0, 1.0, 1.2, 24 renders each) at about 30 minutes each. The frames: 8 runs at about 1 minute each. The readout: minutes. About 15 GPU-hours in all, on one device, in one chain.

**Buys.** The two review questions, and, if supported, the adapter that replaces the baseline on the showcase wall.

**Prerequisites.** The training cache; the baseline's probe folder `figure_r32_030050`; the baseline's per-step frames; the joint-prompt, cat-alone and dog-alone reference renders under `where_each_condition_lands/`. All exist.

**Why one arm.** A cap sweep would triple the cost. The cap is fixed at 22 because that is the multiplier's own value at the last step of the window the outcome is decided in; a second arm (no cap, the exact clean-estimate error, which puts 86% of the weight on steps 0 to 10) is named in the review file's **Still open** and runs only if this arm reads inconclusive.

**Why 40k and not 100k.** The baseline's best checkpoint is 30k and it decays from 70k for a reason plan 15 is testing separately. The question here is asked at 30k on equal terms, with 40k as the one later look.

**W&B project:** `prime_lab/poe-repair-animals-compose`. Training run: `phase1_r32_x0loss_40k`. Readout run: `experiment_e_x0_loss_readout`. Output root: `/datasets/mmolefe/poe_repair_min/outputs/showcase/`, folders `phase1_r32_x0loss_40k/`, `figure_r32_x0loss_030000/`, `figure_r32_x0loss_040000/`, `where_each_condition_lands/frames_x0loss/`, `experiment_e_x0_loss/`.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The shared-device path: `biggpu` allows one Slurm job per user, so the run goes over SSH with `nohup` on a pinned free device of a biggpu node, invisible to `squeue`, harvested with `pgrep` ([execution protocol](../../../../environment/hpc/execution-protocol.md)). When no device is free, a poller on the session node claims the first one that frees ([the wait-and-launch script](../../../../scripts/showcase/experiment_e_wait_and_launch.sh)).
- `co3` python on the RTX 8000 and A6000 nodes (106, 108, 109) and on the session node; `co3_bw` on 110 to 112 ([nodes](../../../../environment/hpc/nodes.md)). The launcher picks by hostname.
- The launcher's guards: at most 4096 MiB in use and 5% utilisation on the device, `/datasets` under 90%, `torch.cuda.is_available()` true under the pinned device ([poe-launch-002](../../../../environment/known-failures.md)).
- Outputs on `/datasets` only, and the disk guard reads the filesystem the script writes to ([storage](../../../../environment/storage.md)).
- Rank 32 training holds about 25 GB on the device; the two RTX 3090s on the session node hold 24 GB and are not candidates.
- The DINOv2 embedder on CPU needs `XFORMERS_DISABLED=1` ([poe-mem-002](../../../../environment/known-failures.md)); in the chain it runs on the claimed device.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Weighting the adapter's loss so that the error is measured between clean estimates, and so falls mostly on the steps that decide the composition, gives a held-out fix that is nearer the joint-prompt image and crisper than the baseline's at the same training step.**

**Why this matters right now.** The showcase wall and the paper's fidelity caveat are both written around the baseline's 30,050 checkpoint, whose renders are two animals under a haze. A one-line loss change that lifts the haze changes what the wall shows and what the caveat has to say.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Plans 07 and 14 varied how much correction and when, at inference, and neither returned the plain run's sharpness. Plan 15 varies the weights' growth. Nothing has varied what the adapter is asked to learn.

**The solution.** Keep every part of the training run and change the loss's per-timestep weight, so the target is the clean-estimate residual the thread's second paper argues for. Read the 30k checkpoint the way the baseline's 30,050 was read, on the same seeds, with the same scorer and the same DINOv2 axes.

**Key insights.**
1. The noise-space error is one fiftieth per step by construction. The clean-estimate error puts 65% of its weight on steps 0 to 10 with the cap at 22, which is where the outcome locks in.
2. The adapter already has the re-weightable, length-changing part of the correction; what it half-learns is the direction, and the direction is decided early. Weighting the early steps is the cheapest way to spend rank 32 there.
3. The read is two numbers with bars fixed in source, one picture in DINOv2 space, and one strip per seed, so the eye can check the table.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
DDIM step k:              0 ...... 10 ...... 20 ...... 30 ...... 40 ...... 49
timestep t:             981      781      581      381      181        1
(1 − ᾱ_t)/ᾱ_t:          172       22      4.6      1.2     0.28    0.002

baseline weight:          1        1        1        1        1        1     (noise-space MSE)
this run's weight:       2.96     2.96     0.61     0.16     0.04     0.00   (clipped at 22, mean 1)

train 40k steps  →  probe 30k and 40k (8 seeds × λ 0, 1.0, 1.2)  →  frames of the 30k adapter at λ 1
                 →  figures, strips, verdict  →  one W&B readout run
```

Every render: the seed's cached initial noise, guidance 7.5, 1024 square, 50 DDIM steps, the adapter on all 50 steps.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The weight** (`_loss_weights` in `poe_repair/experiments/one_pair_one_seed/trainer.py`): per-sample multiplier on the noise-space squared error, `1` under `--loss-space eps`, `min((1 − ᾱ_t)/ᾱ_t, cap) / mean over the 50-step grid` under `--loss-space x0`. The loss is the weighted mean; the unweighted noise-space MSE is returned beside it as `loss_eps`.
2. **The flags** (`--loss-space`, `--loss-weight-cap` on the pooled trainer), written into `config.json`. The bucket curves and the kill criteria read `loss_eps`, so `train/loss_bucket/*` stays comparable with the baseline; `train/loss_eps` and `train/loss_weight_mean` are logged beside `train/loss`.
3. **The launcher** `scripts/showcase/experiment_e_x0_loss.sh <device>`: the guards, a one-epoch dry run with W&B off, the 40k run, then `STAGE=probe` (the two 8-seed grids by the `figure_r32_030050` recipe) and `STAGE=readout`.
4. **The poller** `scripts/showcase/experiment_e_wait_and_launch.sh`: polls the biggpu devices every 5 minutes, confirms a free one twice 60 s apart, launches the chain on it over SSH, and records the claim in `logs/experiment_e_claimed_device.txt`.
5. **The readout** `scripts/showcase/experiment_e_x0_loss.py`: `--frames` (the new adapter's per-step running estimates at λ 1, same renderer and steps as the baseline's frames), `--figures` (five figures below), `--strips` (one per seed, one sheet), `--verdict` (the bars in source, `verdict.json`, `cell-table.md`), `--wandb` (everything into one run). Every figure and sidecar is copied into the results folder named in the Figure Catalog.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 4 (the softness question measured rather than argued) and objective 3. Checkable outcomes:

1. The run reaches 40k with 8 checkpoints and the tracking renders on `/datasets`, `config.json` carrying `loss_space: x0` and `loss_weight_cap: 22.0`.
2. The two probe folders exist with `results.json`; the frames folder holds 8 seeds × 14 steps.
3. The five figures, the strips and `verdict.json` sit under the output root and in the results folder; the readout W&B run shows them.
4. The review file's two questions answered support, null or inconclusive by the constants in source.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/01-showcase-the-trained-lora/plans/experiments/16-experiment-e-train-on-the-clean-estimate-residual.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--the-weight-the-flags-and-the-scripts)**.

### 1. 🔧 The weight, the flags and the scripts

- [x] **1.1 The per-timestep weight in the trainer, and the two flags on the pooled trainer.** `python -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled --help` lists `--loss-space {eps,x0}` and `--loss-weight-cap`; `_loss_weights` on the 50-step grid has mean 1.0 and gives 2.96 at steps 0 to 10, 0.61 at 20, 0.16 at 30.
- [x] **1.2 The launcher, the poller and the readout script**, syntax-checked; the readout run end to end in smoke mode (`EXPERIMENT_E_SMOKE=<dir>`, the baseline's files standing in for the new adapter's) on the session node's CPU.

▶ **Next: [task 2.1](#2--launch-the-run)**.

### 2. 🚀 Launch the run

◀ **Needs: [task 1.2](#1--the-weight-the-flags-and-the-scripts)** done, and every biggpu device checked per [the launch recipe](../../../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#1-decide-where-a-run-goes).

- [ ] **2.1 Start the poller on the session node** (or, when a device is already free, the launcher directly on that node).
  - Command: `nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/experiment_e_wait_and_launch.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/experiment_e_wait.log 2>&1 &`
  - Direct form: `ssh <node> 'nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/experiment_e_x0_loss.sh <device> > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/experiment_e_x0loss.log 2>&1 &'`
  - Done when: `logs/experiment_e_claimed_device.txt` names the node and device, the chain's log shows `guards passed`, `dry run passed` and a W&B run URL, and the review file's Runs table carries node, device, PID and run id.
- [ ] **2.2 Harvest while it runs**, per [the harvest recipe](../../../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#3-harvest): `ssh <node> "pgrep -af 'train_pooled'"`, the checkpoint count under `phase1_r32_x0loss_40k/checkpoints/` (8 at the end), and the W&B run's `train/loss_eps` beside the baseline's `train/loss`.

▶ **Next: [task 3.1](#3--probe-the-checkpoints-and-run-the-readout)**.

### 3. 🧪 Probe the checkpoints and run the readout

◀ **Needs: [task 2.1](#2--launch-the-run)**; the chain runs these on its own after training, this task exists for the case where the chain died and the stage has to be rerun by hand.

- [ ] **3.1 The two 8-seed grids** (`STAGE=probe`): `figure_r32_x0loss_030000/results.json` and `figure_r32_x0loss_040000/results.json`, each with `summary.full.1.0.compose_rate` and `mean_dino_drift`.
- [ ] **3.2 The readout** (`STAGE=readout`, or the five flags of `scripts/showcase/experiment_e_x0_loss.py` one by one on the claimed device): the frames folder, the five figures, the strips, `verdict.json`, the W&B run. Record the readout run id in the review file.

▶ **Next: [instruction 4.1](#4--judge-the-checkpoints)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 4. 👁️ Judge the checkpoints

◀ **Needs: [task 3.2](#3--probe-the-checkpoints-and-run-the-readout)** done.

4.1 **Open the sheet** `/datasets/mmolefe/poe_repair_min/outputs/showcase/experiment_e_x0_loss/strips/sheet-all-seeds.png` (or the readout run's Media tab, `strips/sheet_all_seeds`). Rows are seeds 9 to 16; columns Mono, plain PoE, the baseline adapter, the new adapter. ✅ the fourth column shows two animals on the rows the third does, and reads as punchy as the Mono column; ❌ animals vanish, or the fourth column is under the same haze as the third.

4.2 **Open the two W&B runs side by side**: `phase1_r32_x0loss_40k` and the baseline `6xc2l8ix`. On `train/loss_eps` against `train/loss` of the baseline, the new run should sit above the baseline late (it spends less gradient there) and at or below it in `train/loss_bucket/early`. On `eval/tracking/embedding_drift/dino/out_out/*` the new run's points should sit below the baseline's at 10k, 20k, 30k, 40k. Write what you see, then the numbers from `verdict.json`, into the [review file](../../review/16-experiment-e-train-on-the-clean-estimate-residual.md).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

**Pass criteria:**
- The dry run (one epoch, W&B off) completes and prints a finite loss; the launcher removes its folder and starts the real run.
- `config.json` of the real run differs from the baseline's in exactly three fields: `loss_space`, `loss_weight_cap`, and `kill.commit_bucket_halve_after_steps` (plus `schedule.total_epochs`, 800 against 2000, which is the run's length and not a training setting).
- The smoke-mode readout ran end to end on the baseline's files before launch.

**Fail criteria:**
- The compose count at 30k is below 6 of 8: the re-weighting cost the composition, question 1 is invalid on its own terms, and the arm closes as a null on the fidelity question.
- The probe's λ 0 column differs from the baseline probe's λ 0 column by more than the fp16 drift band (mean absolute pixel difference above 6 of 255 on any seed): the sampler or the cache changed under the run and nothing is read.

**When you get results, answer the questions in the [review file](../../review/16-experiment-e-train-on-the-clean-estimate-residual.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

Every figure below lands in `artifacts/results/does-training-on-the-clean-estimate-residual-sharpen-the-fix/`, with its card entry in that folder's `README.md`, copied there by the readout from `/datasets/mmolefe/poe_repair_min/outputs/showcase/experiment_e_x0_loss/`.

| Figure | What is plotted | What it argues | What it may not claim | File |
|---|---|---|---|---|
| training curves | left: y the unweighted noise-space MSE (EMA 0.02), x optimizer step, one line per run; right: y the DINOv2 drift of the 2 held-out tracking cells, x optimizer step, one point per 10k per run | whether the new loss changes what the adapter fits, and whether the held-out fix improves sooner | anything about the 8-seed grid; the tracking set is 2 cells | `training-curves.png` |
| contrast over steps | y the grey-level standard deviation of the 256 px running estimate, x DDIM step 0 to 50, thin line per seed, thick mean per condition, four conditions | when the haze enters for the new adapter against the baseline, on the same axis as the correct-early read | the 1024 px contrast; the frames are thumbnails | `contrast-over-steps.png`, `.json` |
| where each condition lands | x the cat-to-dog axis, y both-ness, in DINOv2 cosine units; the three reference clouds as hulls; per seed, plain PoE as a square, the baseline adapter as a circle, the new adapter as a diamond, arrows from PoE to each | whether the new adapter lands nearer the joint-prompt cloud than the baseline | which cloud is nearest (the clouds overlap; the landing finding's bar on that failed) | `where-each-condition-lands.png`, `.json`, `final-render-dino-feats.npz` |
| both-ness over steps | y the both-ness of the running estimate on axes fitted on the step-50 reference frames, x DDIM step, thin per seed, thick mean, four conditions, dashed line at the joint-prompt centroid | when the new adapter commits to two animals against the baseline | anything about the final 1024 px render | `both-ness-over-steps.png`, `.json` |
| checkpoint bars | three bars per panel (baseline 30,050, new 30,000, new 40,000): seeds composing of 8; 8-seed DINOv2 drift at λ 1 with the two bars drawn; mean contrast at λ 1 with the joint prompt's as a line | the two pre-registered reads at a glance | anything beyond cat × dog | `checkpoint-bars.png`, `.json` |
| one strip per seed, and the sheet | tiles Mono, plain PoE, baseline adapter λ 1, new adapter λ 1; under each the detector's count, the drift and the contrast | what the loss change does to one seed, so the eye can check the table | anything about the mean; a strip is one seed | `strips/strip-seed_<n>.png`, `mono-vs-poe-vs-adapters-all-seeds.png`, W&B `strips/` and `strips_by_seed` |
| the cell table | one row per checkpoint: compose count, drift, contrast, gap closed, verdict | which reads are supported | anything about the training pairs | `cell-table.md`, `verdict.json` |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 5.1 | new patterns into the catalogs |
| Close out | `/sync-plan-tree` | task 5.2 | statuses aggregated up |

### 5. 🧹 Close out

◀ **Needs: [instruction 4.2](#4--judge-the-checkpoints)** done.

- [ ] **5.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **5.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md)**, which takes a supported checkpoint as the adapter on the wall.

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**Changed:** `poe_repair/experiments/one_pair_one_seed/trainer.py` (`_loss_weights`, the weighted loss and `loss_eps` in `_train_one_step`); `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py` (the two flags, `cfg.loss_space`, `cfg.loss_weight_cap`, `config.json`); `poe_repair/experiments/cross_pair_lora_pooling/multi_pair_trainer.py` (buckets and kill criteria on `loss_eps`; `train/loss_eps`, `train/loss_weight_mean` logged).

**New:** `scripts/showcase/experiment_e_x0_loss.sh` (the chain), `scripts/showcase/experiment_e_wait_and_launch.sh` (the poller), `scripts/showcase/experiment_e_x0_loss.py` (the readout, the bars).

**Reused:** `scripts/showcase/lambda_boundary_probe.py` (the 8-seed grid, the detector, the drift), `scripts/showcase/where_each_condition_lands_trajectories.py` (the frames, with its checkpoint constant overridden), `DinoEmbedder` from `scripts/build_lora_inspector_mds_semantic.py`.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

```
/run-experiment plans/01-showcase-the-trained-lora/plans/experiments/16-experiment-e-train-on-the-clean-estimate-residual.md — the poller claims a device, the chain trains to 40k, probes 30k and 40k, and runs the readout; harvest with pgrep on the claimed node
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) takes a supported checkpoint as the wall's adapter; a null goes to the paper's fidelity caveat, and the review file's **Still open** names the uncapped arm.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From project catalog

From [environment/known-failures.md](../../../../environment/known-failures.md):

- **poe-launch-001**: every path in the SSH launch line is absolute; `cd <repo> &&` is not relied on.
- **poe-launch-002**: the launcher probes `torch.cuda.is_available()` under the pinned device before real work.
- **poe-mem-002**: the DINOv2 embedder runs on the claimed device in the chain; on the session node's CPU only with `XFORMERS_DISABLED=1`.
- **The windowed sampler leaves the adapter attached** (memory note, 2026-09-05): the probe renders λ 0 first, before the adapter is attached, and the frames script renders no references (they exist already), so no reference is rendered with the adapter active.
- **The kill criterion** aborted the fresh baseline at 6,500 steps; this run disables it like experiment D, and the buckets it reads are on the unweighted loss so the numbers mean what they meant.
