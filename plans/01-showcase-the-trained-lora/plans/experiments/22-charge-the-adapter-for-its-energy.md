# 🧪 Charge the adapter for its energy: does a running cost on the correction give two animals with the haze gone?

**This plan asks one question: if the rank-32 adapter keeps training from its best checkpoint with a price on the size of the correction it adds at each step, does the held-out cat × dog render land nearer the joint-prompt image while still showing two animals?**

**Step 64 in the root running order. Waits on nothing: the free read is done, the code is written, the arms run on idle bigbatch cards. Sits beside [15-experiment-d-weight-decay](15-experiment-d-weight-decay.md), which prices the weights; this plan prices the output.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/01-showcase-the-trained-lora/plans/experiments/22-charge-the-adapter-for-its-energy.md — three arms at 40,050 probed, the readout run on W&B, verdicts in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 53 (sibling) | [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md) | switches the correction off after the early steps at inference; the softness is committed by step 20 |
| 57 (sibling) | [15-experiment-d-weight-decay](15-experiment-d-weight-decay.md) | weight decay 0.1 on the LoRA factors from scratch; prices the parameters |
| 58 (sibling) | [16-experiment-e-train-on-the-clean-estimate-residual](16-experiment-e-train-on-the-clean-estimate-residual.md) | re-weights the fit loss toward the early steps |
| **64 (current)** | **22: charge the adapter for its energy** | a running cost on the correction's own size, resumed from the shipped checkpoint, three arms, one readout |
| 35 (next in scope) | [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) | the wall; a supported arm replaces the shipped checkpoint in its rows |

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

- **The shipped checkpoint**: the rank-32 pooled adapter `phase1_r32_100k` at training step 30,050, the one every showcase figure uses. Held out on cat × dog it composes 7 of 8 seeds at λ 1 with a DINOv2 drift of −0.091.
- **The correction**: at a denoising step, the change the adapter makes to the product-of-experts noise prediction, `r̂ = ε_PoE^lora − ε_PoE`. **λ** multiplies it at inference; every read here uses λ 1.0.
- **The true correction**: `r_t = ε_joint − ε_PoE`, the joint-prompt prediction minus the product prediction at the same state, both guided at 7.5. The adapter is trained to match it.
- **Control energy**: read the reverse run as a stochastic differential equation with the same marginals. A change `r` to the noise prediction at timestep `t` is a change of `−r/σ_t` to the score, and Girsanov's theorem prices the path-space divergence between the corrected run and the plain run at `Σ_k ½ γ_k ‖r_k‖²` nats, with `γ_k = β_step,k / (1 − ᾱ_k)` and `β_step,k = 1 − ᾱ_k / ᾱ_prev` the diffusion integrated over one DDIM step. `γ_k` sits between 0.12 and 0.21 for steps 0 to 48 and is 0.50 at the last step. The derivation is chapter 3 of Tang's Schrödinger-bridge monograph (arXiv 2603.18992), definition 3.1 and corollary 2.21.
- **The running cost, or the penalty**: the trainer's loss becomes `fit + β · mean_k[ w_k · mean(r̂_k²) ]`, where `fit` is the existing noise-space mean squared error against the cached joint prediction, `w_k` is `γ_k` divided by its mean over the 50-step DDIM grid (so `w_k` has mean 1, runs 0.65 to 1.1 over steps 0 to 40 and is 2.6 at the last step), and **β** is the price. β 0 is the trainer as it was. `control_energy_weights_normalised` in `poe_repair/experiments/one_pair_one_seed/trainer.py` is the function.
- **An arm**: one resumed training. Three arms, β 0 (the control), 0.01 and 0.05, each from the shipped checkpoint for 10,000 steps to step 40,050, identical in every other flag, seed and state sequence.
- **The baseline lineage**: `phase1_r32_100k` itself, whose own 40,050 checkpoint is the reference for whether the launch shape (a 24 GB card with gradient checkpointing) changed the training at all.
- **On-policy**: measured along the trajectory the adapter itself produces at λ 1, with the true correction recomputed by one joint-prompt forward at every state the run actually visits. The cache holds states on the plain-PoE trajectory, which the corrected run leaves.
- **Mismatch**: `Σ_k ½ γ_k ‖r̂_k − r_true,k‖²` on-policy, in nats. How far the adapter's correction is from the true one, in the same unit as the energy.
- **Drift**: DINOv2 drift, the corrected render's cosine distance to the joint-prompt render minus its distance to the plain-PoE render, mean over the 8 held-out seeds at λ 1. Negative means nearer the joint-prompt image. From `summary.full.1.0.mean_dino_drift` in a probe folder's `results.json`.
- **Compose count**: seeds of 8 where the validated detector counts two or more animal instances ([compose rate](../../../../context/world/compose-rate.md)).
- **Contrast**: the standard deviation of the grey-level pixels of the 1024 px render, 0 to 255. The haze reads as a low value; the shipped adapter's 8-seed mean is 37.5 against the joint prompt's 46.4.
- **Both-ness and the DINOv2 plane**: a render embedded with the scorer's DINOv2 encoder and projected onto two axes fitted on the "a cat", "a dog" and "a cat and a dog" reference clouds: x from the cat centroid to the dog centroid, y from their midpoint toward the joint-prompt centroid. Both-ness is y. The same construction as [where each condition lands](../../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md).
- **A track**: the running estimate (the model's guess of the finished image, decoded at 256 px) at 14 steps of one run, embedded and projected onto that plane, so one run is one path.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** The shipped adapter composes but its renders are under a haze, and training it longer makes the haze worse while the weights grow without bound ([the decay finding](../../../../report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md)). Read as a control problem, the adapter is a drift added to the plain run, and the objective a bridge would minimise charges for that drift's squared size at every step. The trainer has no such charge. The free read below says where the true correction spends: 82% of its control energy falls on steps 20 to 49, after the composition is decided (median commit step 15 for the corrected run), and 3.5% on steps 0 to 10. A correction spent after commitment cannot change which animals appear; it can only change how they look. This plan adds the charge and reads what the adapter gives up.

**The hypothesis.** With the running cost on, the adapter keeps the early correction that decides the composition and sheds late energy it does not need, so the 40,050 checkpoint renders nearer the joint-prompt image than the control arm at the same step, with the same seeds composing.

**If true.** The β 0.05 checkpoint replaces the shipped one on the wall, the fidelity caveat shrinks to a sentence about the price, and the paper gains a one-term method change with a stated reason.

**If false.** Either the animals go with the energy (compose count under 6, the penalty priced away the part that composes), or the render does not move (drift within 0.03 of the control), in which case the haze is not excess energy and the on-policy arm named in [the pressure-test route](../../../../artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md) moves up. The on-policy read separates the two: if mismatch holds while energy falls and the render does not improve, the haze was never the energy.

**Dataset.** The same 11 training pairs × 8 seeds (88 cells, 4,400 cached steps) the shipped adapter trained on; cat × dog held-out seeds 9 to 16 for every read.

**Associated materials.**
- Review questions: [the review file](../../review/22-charge-the-adapter-for-its-energy.md)
- The free read, already filed: [the control energy of the true correction](../../../../artifacts/results/does-charging-the-adapter-for-its-energy-sharpen-the-fix/README.md)
- The reading that motivated it: chapter 3 (stochastic optimal control) and section 6.5 (adjoint matching) of arXiv 2603.18992; the plain-words map from that monograph to this repository is in the review file's **Words** block

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** Per arm: 10,000 training steps with gradient checkpointing on an RTX 3090, about 1.5 s per step, so about 4.5 hours, plus one tracking-set pass, the 8-seed grid at λ 0 and 1.0 (16 renders, about 25 minutes), the 8 frame runs (about 8 minutes) and the on-policy read (8 rollouts with a joint forward at each step, about 15 minutes). Three arms in parallel on three idle bigbatch nodes: about 5.5 hours wall. The readout job: the baseline's own 40,050 grid, two on-policy reads and the figures, about 1.5 hours. About 18 GPU-hours in all.

**Buys.** The three review questions, the on-policy mismatch-against-energy figure that decides which fix runs next, and, if supported, the adapter that replaces the shipped one on the wall.

**Prerequisites.** The training cache; the shipped checkpoint; the baseline's probe folder `figure_r32_030050`; the baseline's per-step frames and the three reference clouds under `where_each_condition_lands/`. All exist.

**Why resume rather than retrain.** The question is about the checkpoint the paper ships. Resuming from it holds the first 30,050 steps fixed across arms and answers in 10,000 steps what a fresh run would answer in 40,000, at a quarter of the cost. The price is that the arms start from an adapter trained without the charge; a fresh run is the follow-on if the resumed arm supports.

**Why three arms and not one.** β 0 is the control: same launch shape, same card, same gradient checkpointing, same state sequence, so the only difference to the other two is the price. Without it, the comparison to the baseline's own 40,050 would mix the price with the launch shape. β 0.01 is the dose check: if energy and drift move monotonically in β, the effect is the price and not a seed of the resume.

**Why β 0.05.** At the shipped checkpoint the correction's per-element mean square is 0.002 at step 0 and 0.08 at step 30, while the fit loss after a 0.97 cosine on training cells is about 6% of that. So β 1 would make the price fifteen times the fit and crush the correction; β 0.05 puts the price at roughly the size of the fit, and β 0.01 at a fifth of it. The smoke run prints the measured ratio before any arm launches, and the plan says what to do if it is far off (see **What has to pass**).

**W&B project:** `prime_lab/poe-repair-animals-compose`. Training runs `phase1_r32_energy{0,0.01,0.05}_from30050_40k`; readout run `energy_penalty_readout`. Output root: `/datasets/mmolefe/poe_repair_min/outputs/showcase/`, folders `phase1_r32_energy<β>_from30050_40k/`, `figure_r32_energy<β>_040050/`, `figure_r32_040050/`, `where_each_condition_lands/frames_energy<β>/`, `energy_penalty/`.

**Known issues:** see the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- Every `biggpu` device was busy on 2026-09-06 and `biggpu` allows one Slurm job per user, so the arms go to `bigbatch` through Slurm, one 24 GB RTX 3090 per node, excluding the nodes whose cards are faulted (44, 45, 50, 51, 65, 74, 83) and the session node 85 ([nodes](../../../../environment/hpc/nodes.md), [execution protocol](../../../../environment/hpc/execution-protocol.md)).
- Rank-32 training holds about 23.5 GB without gradient checkpointing and went out of memory on a 3090 (Slurm job 50339); with checkpointing it fits, at about a third more time per step. The numbers are the same; only the activation memory changes.
- `co3` python at `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python` on every bigbatch node.
- The job script's guards: `/datasets` under 90%, the card not faulted (`utilization.gpu` without `[N/A]`), `torch.cuda.is_available()` true ([poe-launch-002](../../../../environment/known-failures.md)).
- Outputs on `/datasets` only; the disk guard reads the filesystem the script writes to ([storage](../../../../environment/storage.md)).
- fp16 end to end; the cache read upcasts every tensor to float32 before summing squares.
- The DINOv2 embedder runs on the job's card; on the session node's CPU only with `XFORMERS_DISABLED=1` ([poe-mem-002](../../../../environment/known-failures.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A running cost on the correction's control energy, added to the adapter's loss, gives a held-out fix that is nearer the joint-prompt image than the same training without the cost, at the same step, with the same seeds composing.**

**Why this matters right now.** The wall and the fidelity caveat are written around a checkpoint whose renders are two animals under a haze. The Schrödinger-bridge reading says the haze is the price of unpaid energy; this run is the test of that reading, and either answer changes what the caveat has to say.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Plans 07 and 14 varied how much correction and when, at inference, and neither returned the plain run's sharpness. Plan 15 prices the weights. Plan 16 re-weights the fit. Nothing prices what the adapter adds to the run.

**The solution.** The stochastic-optimal-control objective a Schrödinger bridge minimises is fit plus half the squared control, so the trainer gets that second term, weighted per step the way Girsanov weights it. Three arms resumed from the shipped checkpoint, one axis moved, and a readout that reads energy, mismatch, drift, contrast and the compose count on the same eight seeds.

**Key insights.**
1. The true correction spends 82% of its control energy after step 20, where the composition is already decided. That energy can only be changing how the animals look, and the haze is the candidate.
2. The adapter fits the true correction at cosine 0.97 on training cells and its late correction has no cross-seed structure, so it is the late, high-energy, hard-to-predict part the price should remove first.
3. The on-policy read gives two numbers per step in one unit, energy and mismatch. If energy falls and mismatch holds, the price removed only what the fit did not need.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
the cache (plain-PoE states, 88 cells)        the shipped checkpoint, step 30,050
          │                                                │
          ▼                                                ▼  resume, optimizer moments included
  free read: ½ γ_k ||r_k||² per step   ┌── β 0    ──┐  ┌── β 0.01 ──┐  ┌── β 0.05 ──┐   10,000 steps each
  → 82% after step 20 (done)           │ loss = fit │  │ + 0.01·E  │  │ + 0.05·E  │   E = mean_k w_k·mean(r̂_k²)
                                       └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
                                             ▼ 40,050        ▼ 40,050        ▼ 40,050
                       per arm: 8-seed grid at λ 0 and 1 · frames at λ 1 · on-policy energy and mismatch
                                             │
                       readout: baseline 30,050 and 40,050 read the same way, then
                       curves · mismatch-vs-energy · DINOv2 plane and tracks · bars · strips · verdict · W&B
```

Every render: the seed's cached initial noise, guidance 7.5, 1024 square, 50 DDIM steps, the adapter on all 50 steps at λ 1.0.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The weights and the penalty** in `poe_repair/experiments/one_pair_one_seed/trainer.py`: `control_energy_weights` (γ_k) and `control_energy_weights_normalised` (w_k); `_train_one_step` returns `loss = loss_fit + β · loss_energy` and reports `loss_fit`, `loss_energy` and β in its info. `loss_eps` is unchanged, so the bucket curves and the kill criterion read what they read before.
2. **The flags** `--energy-penalty` and `--gradient-checkpointing` on the pooled trainer, written into `config.json`; `train/loss_fit`, `train/loss_energy`, `train/energy_penalty_beta` logged beside `train/loss`.
3. **The free read** `scripts/showcase/control_energy_from_cache.py`: `½ γ_k ‖r_k‖²` per step and seed from the cache, its figure and sidecar, copied to the results folder.
4. **The on-policy read** `scripts/showcase/on_policy_energy.py`: per step, `‖r̂‖`, `‖r_true‖`, their cosine, the mismatch and the energies in nats, plus the corrected final render, for one checkpoint.
5. **The arm job** `scripts/showcase/energy_penalty.sbatch`: guards, `STAGE=smoke` (one epoch, W&B off, prints the price-to-fit ratio and peak memory), `STAGE=arm` (train, then the grid, the frames and the on-policy read of the 40,050 checkpoint), `STAGE=probe` (the post-training stages alone).
6. **The readout** `scripts/showcase/energy_penalty_readout.py`, the bars as constants: `--baseline` (the baseline's 40,050 grid and the two baseline on-policy reads, GPU), `--figures`, `--strips`, `--verdict`, `--wandb`; and `scripts/showcase/energy_penalty_readout.sbatch` that runs it after the arms.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 4 (the softness question measured rather than argued) and objective 3. Checkable outcomes:

1. Three run folders each with `lora_step_040050.pt`, `config.json` carrying `energy_penalty` at its arm's value and `gradient_checkpointing: true`.
2. Four probe folders (three arms and the baseline's 40,050) with `results.json`; three frame folders with 8 seeds × 14 steps; five on-policy sidecars.
3. The figures, the strips, `verdict.json` and `cell-table.md` under `energy_penalty/` and in the results folder; the readout W&B run shows them.
4. The review file's three questions answered support, null or inconclusive by the constants in source.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/01-showcase-the-trained-lora/plans/experiments/22-charge-the-adapter-for-its-energy.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--the-code-and-the-free-read)**.

### 1. 🔧 The code and the free read

- [x] **1.1 The energy weights, the penalty and the two flags.** `python -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled --help` lists `--energy-penalty` and `--gradient-checkpointing`; `control_energy_weights_normalised` on the 50-step grid has mean 1.0, reads 1.09 at step 0, 0.65 at step 30, 2.64 at step 49.
- [x] **1.2 The free read.** `control_energy_from_cache.py` run on the session node's CPU; the figure and sidecar sit in the results folder. Mean total 11,044 nats over the 8 seeds (2,499 to 19,127); steps 0 to 10 carry 3.5% of it, steps 20 to 49 carry 82%; the mean curve peaks at step 28.
- [x] **1.3 The on-policy read and the arm job**, syntax-checked.

▶ **Next: [task 2.1](#2--smoke-then-launch-the-three-arms)**.

### 2. 🚀 Smoke, then launch the three arms

◀ **Needs: [task 1.3](#1--the-code-and-the-free-read)**.

- [x] **2.1 The smoke job**: one epoch from the shipped checkpoint on a bigbatch card, W&B off.
  - Command: `STAGE=smoke BETA=0.05 sbatch --job-name=energy_smoke --time=00:45:00 scripts/showcase/energy_penalty.sbatch`
  - Done when: the log prints `[smoke] 50 steps: mean loss_fit …, mean loss_energy …, beta*energy / fit = …` and `[smoke] peak VRAM GB: …` under 23, and the ratio is inside the band in **What has to pass**.
- [x] **2.2 The three arms**, one job each.
  - Command: `for b in 0 0.01 0.05; do BETA=$b sbatch --job-name=energy_b$b scripts/showcase/energy_penalty.sbatch; done`
  - Done when: `squeue -u mmolefe` shows the three running and each run folder's `config.json` reads its `energy_penalty`; the review file's Runs table carries job ids, nodes and W&B ids.
- [x] **2.3 The readout job**, queued behind the arms.
  - Command: `sbatch --dependency=afterany:<j0>:<j1>:<j2> scripts/showcase/energy_penalty_readout.sbatch`
  - Done when: the readout W&B run exists with the figures in its Media tab and `verdict.json` sits under `energy_penalty/`.

▶ **Next: [task 3.1](#3--harvest)**.

### 3. 🌾 Harvest

◀ **Needs: [task 2.2](#2--smoke-then-launch-the-three-arms)**.

- [x] **3.1 While the arms run**: `squeue -u mmolefe`, then `ls <run>/checkpoints` (two files at the end, 35,050 and 40,050), then the W&B runs' `train/loss_energy` against `train/loss_fit` and `train/lora_weight_norm`. Record the step time from `train/timing/step_time_s`.
- [x] **3.2 When the readout lands**: paste the three verdict lines of `verdict.json` into the review file, with the run id and the commit.

▶ **Next: [instruction 4.1](#4--judge-the-arms)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 4. 👁️ Judge the arms

◀ **Needs: [task 3.2](#3--harvest)** done.

4.1 **Open the sheet** `/datasets/mmolefe/poe_repair_min/outputs/showcase/energy_penalty/strips/sheet-all-seeds.png` (or the readout run's Media tab, `strips/sheet_all_seeds`). Rows are seeds 9 to 16; columns Mono, plain PoE, shipped 30,050, control 40,050, β 0.01, β 0.05. ✅ the last column shows two animals on the rows the control does and reads as punchy as Mono; ❌ animals vanish, or it sits under the same haze as the control.

4.2 **Open the mismatch-against-energy figure** `energy_penalty/figures/mismatch-vs-energy.png`. Five curves, one per checkpoint. ✅ the β 0.05 curve sits below the control on energy and on top of it on mismatch; ❌ mismatch rises with the price, the adapter is losing the fit rather than the excess.

4.3 **Open the three training runs side by side in W&B** with the baseline `6xc2l8ix`. `train/loss_fit` should sit on the baseline's `train/loss` for the control and slightly above it for β 0.05; `train/loss_energy` should fall in β; `train/lora_weight_norm` should flatten for β 0.05 where the control's climbs. Write what you see, then the numbers from `verdict.json`, into the [review file](../../review/22-charge-the-adapter-for-its-energy.md).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

**Pass criteria:**
- The smoke job completes one epoch with peak memory under 23 GB and a finite `loss_fit` and `loss_energy`.
- The price-to-fit ratio the smoke prints, `β · loss_energy / loss_fit` at β 0.05 over the first 50 steps, lies between 0.2 and 5. If it does not, the arms are not launched; β is rescaled so the ratio is 1.0, the new value is written here with the measured ratio, and only then do the arms launch. This is a scale calibration made before any outcome exists, and it is disclosed as one.
- The control arm's `config.json` differs from the baseline's in exactly `energy_penalty` (0.0 written explicitly), `gradient_checkpointing`, `kill.commit_bucket_halve_after_steps`, `schedule.total_epochs` (800 against 2000, the run's length) and `resume_from`.

**Fail criteria:**
- The control arm at 40,050 composes fewer than 6 of 8, or its drift differs from the baseline's own 40,050 by more than `CONTROL_VALIDITY_DRIFT_TOL = 0.03`: the launch shape changed the training and no arm is read against the shipped lineage. The within-experiment comparison (arms against the control) still stands and is said to stand alone.
- Any arm's λ 0 column differs from the baseline probe's λ 0 column by more than the fp16 drift band (mean absolute pixel difference above 6 of 255 on any seed): the sampler or the cache changed under the run and nothing is read.

**When you get results, answer the questions in the [review file](../../review/22-charge-the-adapter-for-its-energy.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

Every figure below lands in `artifacts/results/does-charging-the-adapter-for-its-energy-sharpen-the-fix/`, with its card entry in that folder's `README.md`, copied there from `/datasets/mmolefe/poe_repair_min/outputs/showcase/energy_penalty/`.

| Figure | What is plotted | What it argues | What it may not claim | File |
|---|---|---|---|---|
| control energy over steps | three panels against DDIM step 0 to 49: `‖r_t‖` of the true correction; `½ γ_k ‖r_k‖²` in nats with `γ_k` dotted; the cumulative sum with steps 0 to 10 shaded. Thin per seed, thick mean of 8 | where the true correction spends its energy: after the composition is decided | anything about the adapter; this is the cached true correction on the plain-PoE trajectory, and the nats are a Girsanov price under an SDE reading of a run that was sampled as an ODE | `control-energy-over-steps.png`, `.json` |
| training curves | left: `train/loss_fit` (EMA 0.02) against optimizer step for the three arms and the baseline's `train/loss`; middle: `train/loss_energy`; right: `train/lora_weight_norm`; one line per run | whether the price changes what the adapter fits and stops the weights growing | anything about the render | `training-curves.png` |
| mismatch against energy | per DDIM step, on-policy: top, `½ γ_k ‖r̂_k‖²` (the adapter's spend); bottom, `½ γ_k ‖r̂_k − r_true,k‖²` (the mismatch); one curve per checkpoint: baseline 30,050, baseline 40,050, control 40,050, β 0.01, β 0.05; mean of 8 seeds | whether the price removed excess energy or the fit | anything about the render; energy in nats is the SDE reading | `mismatch-vs-energy.png`, `.json` |
| where each condition lands | the DINOv2 plane: x cat-to-dog, y both-ness; the three reference clouds as hulls; per seed plain PoE as a square, shipped 30,050 as a circle, control as a triangle, β 0.05 as a diamond, arrows from PoE | whether the priced adapter lands nearer the joint-prompt cloud than the control | which cloud is nearest (the clouds overlap; the landing finding's bar on that failed) | `where-each-condition-lands.png`, `.json`, `final-render-dino-feats.npz` |
| tracks in the plane | the same plane; one path per run over the 14 saved steps for the joint prompt, plain PoE, the shipped adapter, the control and β 0.05; seed 15 large, the other seeds faint; a marker at the commit step | how each run travels toward "a cat and a dog", and whether the priced adapter commits where the shipped one does | anything about the latent itself; the plane is the image encoder's, fitted on 256 px running estimates | `tracks-in-the-dino-plane.png`, `.json` |
| both-ness over steps | y both-ness of the running estimate, x DDIM step, thin per seed, thick mean, five runs, dashed line at the joint-prompt centroid | when each run commits to two animals | the final 1024 px render | `both-ness-over-steps.png`, `.json` |
| checkpoint bars | five bars per panel (baseline 30,050, baseline 40,050, control, β 0.01, β 0.05): seeds composing of 8; DINOv2 drift at λ 1 with the margin drawn; mean contrast with the joint prompt's as a line; on-policy energy in nats with the late-step share hatched | the three pre-registered reads at a glance | anything beyond cat × dog | `checkpoint-bars.png`, `.json` |
| one strip per seed, and the sheet | tiles Mono, plain PoE, shipped 30,050 λ 1, control 40,050 λ 1, β 0.01 λ 1, β 0.05 λ 1; under each the detector's count, the drift and the contrast | what the price does to one seed, so the eye can check the table | anything about the mean; a strip is one seed | `strips/strip-seed_<n>.png`, `sheet-all-seeds.png`, W&B `strips/` and `strips_by_seed` |
| the cell table | one row per checkpoint: compose count, drift, contrast, on-policy energy, mismatch, verdicts | which reads are supported | anything about the training pairs | `cell-table.md`, `verdict.json` |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 5.1 | new patterns into the catalogs |
| Close out | `/sync-plan-tree` | task 5.2 | statuses aggregated up |

### 5. 🧹 Close out

◀ **Needs: [instruction 4.3](#4--judge-the-arms)** done.

- [ ] **5.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **5.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md)**, which takes a supported checkpoint as the wall's adapter.

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**Changed:** `poe_repair/experiments/one_pair_one_seed/trainer.py` (`control_energy_weights`, `control_energy_weights_normalised`, the penalty and the three info fields in `_train_one_step`); `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py` (the two flags, `cfg.energy_penalty`, `cfg.gradient_checkpointing`, `config.json`, `enable_gradient_checkpointing` after the LoRA attach); `poe_repair/experiments/cross_pair_lora_pooling/multi_pair_trainer.py` (`train/loss_fit`, `train/loss_energy`, `train/energy_penalty_beta`).

**New:** `scripts/showcase/control_energy_from_cache.py` (the free read), `scripts/showcase/on_policy_energy.py` (the on-policy read), `scripts/showcase/energy_penalty.sbatch` (smoke, arm, probe), `scripts/showcase/energy_penalty_readout.py` (the figures, the bars), `scripts/showcase/energy_penalty_readout.sbatch`.

**Reused:** `scripts/showcase/lambda_boundary_probe.py` (the 8-seed grid, the detector, the drift), `scripts/showcase/where_each_condition_lands_trajectories.py` (the frames, with its checkpoint constant overridden), `scripts/showcase/correction_span_common.py` (the cache loader), `DinoEmbedder` from `scripts/build_lora_inspector_mds_semantic.py`, the axis and strip helpers of `scripts/showcase/experiment_e_x0_loss.py`.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

```
/run-experiment plans/01-showcase-the-trained-lora/plans/experiments/22-charge-the-adapter-for-its-energy.md — smoke on bigbatch, then the three arms as parallel Slurm jobs, the readout job queued behind them; harvest with squeue and the run folders
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) takes a supported checkpoint as the wall's adapter; a null goes to the paper's fidelity caveat, and the review file's **Still open** names the fresh-run and on-policy-training follow-ons.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From project catalog

From [environment/known-failures.md](../../../../environment/known-failures.md):

- **poe-launch-002**: the job script rejects a card whose utilisation reads `[N/A]` and asserts `torch.cuda.is_available()` before any work; bigbatch nodes 44, 45, 65 read idle with a dead card on 2026-09-06 and are excluded.
- **poe-mem-002**: the DINOv2 embedder runs on the job's card; the readout never embeds on a CPU without `XFORMERS_DISABLED=1`.
- **poe-lora-003**: every compared cell runs through the identical windowed sampler at λ 1.0 from the same cached noise; Mono is a reference column, never a compared cell.
- **The windowed sampler leaves the adapter attached** (memory note, 2026-09-05): the probe renders λ 0 first, and the frames script renders no references (they exist already), so no reference is rendered with an adapter active.
- **Rank 32 out of memory on a 24 GB card** (Slurm job 50339, 2026-09-06): every arm passes `--gradient-checkpointing`; the smoke prints the peak memory before the arms launch.
- **The kill criterion**: a resume never latches it, and the arms pass `--kill-halve-after-steps 1e9` so the fact is explicit in `config.json`.
