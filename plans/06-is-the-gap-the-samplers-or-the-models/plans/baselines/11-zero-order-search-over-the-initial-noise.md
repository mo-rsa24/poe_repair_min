# 🎯 Zero-order search over the initial noise

Take each held-out seed's cached starting noise, nudge it eight ways, render plain
product-of-experts from every nudge, ask the validated compose scorer which finished image has two
animals, and keep that one. No correction is added and nothing is learned; the only thing that
changes is which noise the run starts from. If a small search over the noise is enough to compose,
the early-window correction has a training-free rival that forks at step 0.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md — <support / null / inconclusive, the kept compose rates per sigma against the unperturbed seed>
```

## Recommended skill

▶ `/run-experiment` ✅ for the launch; `/analyze-run` ✅ for the W&B sheets and the sidecar.

## Position in the plan tree

**Unnumbered until the next sync; proposed step 56 of 56.** Waits on nothing: the pinned initial
noise it reads exists in the training cache, and the detector is validated. The one order is the
`## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md); the row
for this plan is queued in
[the pending-sync list](../../../../artifacts/ideas/parallel-sessions-on-poe-dynamics-and-fixes/PENDING_SYNC.md).

| Step | Plan | What it does |
|------|------|-------------|
| 51 | [baseline-05: twisted-smc-on-a-learned-joint-vs-poe-twist](09-twisted-smc-on-a-learned-joint-vs-poe-twist.md) 〰️ | selection among particles with a learned weight; inconclusive because the weight memorised its training set |
| 55 | [baseline-06: feynman-kac-steering-on-a-detector-reward](10-feynman-kac-steering-on-a-detector-reward.md) ◑ | selection among particles mid-run with the compose scorer as the weight |
| **56 (proposed, current)** | **baseline-07: zero-order-search-over-the-initial-noise** ◑ | **selection among starting noises, before the run, with the compose scorer as the verifier** |

Design only. Verdicts and run state live in
[the paired review file](../../review/11-zero-order-search-over-the-initial-noise.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Words this plan uses](#words-this-plan-uses)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [The check before moving on](#the-check-before-moving-on)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#words-this-plan-uses) ➡️

Does the best of eight small perturbations of a seed's starting noise, rendered with plain
product-of-experts and picked by the compose scorer, show two animals more often than the
unperturbed seed does?

## Words this plan uses

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **The pivot**: the seed's cached starting noise, the same tensor every other render of that seed
  in this repo starts from. Rendered with plain product-of-experts it is the unperturbed baseline.
- **A candidate**: the pivot mixed with a fresh standard-normal draw, `z' = (z + σ·u) / sqrt(1 + σ²)`.
  The division keeps the candidate a unit-variance Gaussian, so the sampler sees a legitimate
  starting noise; its cosine to the pivot is `1 / sqrt(1 + σ²)`, 0.995 at σ 0.1 and 0.958 at σ 0.3.
- **σ (sigma)**: how far a candidate moves from the pivot. Two values, 0.1 and 0.3.
- **The verifier**: the validated instance-count compose scorer (two or more detected animal
  instances means compose). Read two ways on the same candidates: on the finished image, and on
  x0-hat at step 10.
- **x0-hat at step 10**: the model's guess of the finished image at denoising step index 10 of 50,
  by the Tweedie formula, decoded through the VAE. Step 10 is the end of the window where the
  compose rate is decided, so a verifier that works there could steer a run instead of only
  picking one.
- **The kept image**: the candidate with the highest verifier count, ties broken by the summed
  confidence of the detector's kept boxes, then the lowest candidate index. One kept image per
  seed, per σ, per verifier.
- **Random search at N 8**: the paper's simplest algorithm, best of N independent noises. The
  eight held-out seeds are eight independent noises, so the existing renders already answer it.
- **Both-ness**: the projection of a render's DINOv2 embedding onto the axis from the midpoint of
  the "a cat" and "a dog" clouds toward the "a cat and a dog" cloud, from
  [where each condition lands](../../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md).
  A fused face scores as one animal on the count and can still sit high on this axis, so the two
  reads together show whether a "compose" is two bodies.

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** `poe_repair/experiments/noise_search/`: the perturbation, the batched plain-PoE
sampler that also captures x0-hat at the early step, the pick rule and the bars (`search.py`); the
runner that renders references, candidates and sheets, scores everything, writes the sidecar and
the both-ness read, and logs to W&B (`run.py`). Launched by
`scripts/noise_search/run_noise_search.sh`.

**What it does.** Ma et al., "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising
Steps" ([arXiv 2501.09732](https://arxiv.org/abs/2501.09732), unpacked in
[the note](../../../../artifacts/notes/inference-time-scaling-as-a-search-over-noise/note.md)),
frame extra inference compute as a search over the starting noise with a verifier. Random search
is best of N draws; zero-order search draws N candidates near a pivot and keeps the best. Here the
sampler is plain product-of-experts, the verifier is the project's compose scorer, and one round of
zero-order search at N 8 is run around every held-out seed.

**Key components.** The pivot, the perturbation, the batched sampler, the two verifier reads, the
sheets, the both-ness read.

**Testing approach.** One sheet per pair, rows the held-out seeds 9 to 16, columns Mono (the
joint prompt), plain PoE from the pivot, the existing rank-32 λ 1.2 render where it exists, then
best of 8 at σ 0.1 and 0.3 by the final-image verifier, then best of 8 at σ 0.1 and 0.3 by the
step-10 verifier. Every tile is a plain-PoE render from some starting noise; only the noise differs.

**Associated materials.** [The review questions](../../review/11-zero-order-search-over-the-initial-noise.md),
[plan 10](10-feynman-kac-steering-on-a-detector-reward.md) for the shared sampler pieces and the
sheet drawer, and [the register row](../../../standing/literature/reading-register.md) for the
paper.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**The paper never writes the perturbation.** Its zero-order neighbourhood is "distance λ from the
pivot" with no metric and no formula. The variance-preserving mixture above is this project's
reading, chosen so a candidate is still a standard normal draw; the one public reimplementation
checked filters candidates at cosine 0.95, which σ 0.3 sits inside. Every caption says the formula
is ours.

**One round, not a climb.** The paper iterates: the best candidate becomes the next pivot. One
round at N 8 is the cheapest version that can show whether the neighbourhood holds composing
noises at all. If it does, iterating is the follow-up; if it does not, iterating cannot help.

**The verifier is the judge.** The kept image is chosen by the same scorer that scores it, so the
kept column can only win ties in its own favour. The both-ness read and the eye read in the
Instructions are what stop a detector error from counting as a compose.

**The step-10 verifier reads a blur.** x0-hat at step 10 is soft, and the detector count on it can
disagree with the count on the finished image. The runner records, per candidate, whether the
two agree, and the kept-by-step-10 column is scored on its finished image, so the read is "did
picking at step 10 find a composing finish", not "did step 10 look composed".

**Every render shares the seed's initial noise family.** The pivot is the training cache's pinned
noise for that seed (`embeddings.pt`, key `init_latents`). Butterfly × meadow's cache stops at seed
12; seeds 13 to 16 are drawn by the same formula, checked byte-identical against cat × dog's cached
seeds 9 and 13 on 2026-09-05 by plan 10.

**Identity check.** The pivot's plain-PoE render is compared pixel-wise against the existing
rank-32 λ 0 render of the same seed (`figure_r32_030050/renders/full/seed_<n>_lambda_0.0.png`).
The two come from different samplers (this plan's batched DDIM against the LoRA sampler at λ 0) and
different cards, so a few grey levels of difference is the expectation; a large difference means a
different starting noise, and the run is wrong.

**Deterministic sampler.** DDIM `eta` 0 throughout, so the starting noise is the only randomness
and the paper's framing applies: a candidate is an image.

**Cost.** Per (pair, seed, σ) one batched run of 8 candidates at 50 steps, 1024², three UNet
branches per candidate: about 5 to 6 minutes on a Quadro RTX 8000 (scaled from plan 10's K 16 at
9 minutes on an RTX 3090), plus 16 VAE decodes and detector reads. 32 such runs for two pairs, two
σ and eight seeds, plus 16 Mono references and 16 pivots at about 25 s each: about 3.5 hours.
Buys: the pre-registered question, the two sheets, the both-ness read.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**One round of zero-order search over the starting noise, with the compose scorer as verifier,
raises plain product-of-experts' compose rate over the unperturbed seed.**

**Independent variables.** σ in {0.1, 0.3}; which verifier picks (final image, x0-hat at step 10).
Fixed: N 8, one round, DDIM `eta` 0, 50 steps, guidance 7.5, 1024², plain PoE, two pairs (cat ×
dog, the failing pair; butterfly × meadow, the composing control pair), seeds 9 to 16.

**Dependent variable.** Compose rate over the 8 seeds of the kept image by the validated detector
on the finished image. Secondary, recorded and not judged: the fraction of all 8 candidates that
compose, whether any candidate composes, the step-10-against-final agreement per candidate, the
mean cosine of candidates to the pivot, and both-ness of every cat × dog tile.

**Falsify condition.** The bars sit in `poe_repair/experiments/noise_search/search.py` as
`PASS_MARGIN` and `NULL_MARGIN`, and the runner writes `verdict.json` from them. Judged on cat ×
dog, final-image verifier; the step-10 verifier gets the same bars applied and reported beside it.

- **Support.** The kept image composes at least 0.25 more often than the unperturbed seed at either
  σ. Composing noises sit within a small angle of the failing ones, and a verifier can find them.
- **Null.** The kept rate is within 0.10 of the unperturbed rate at both σ. Eight nudges of this
  size do not reach a composing noise.
- **Inconclusive.** Anything between: one σ above 0.10 and under 0.25, or the two σ disagreeing
  across the null margin.
- **Butterfly × meadow** is a control: the kept image must not compose less often than the pivot.
  A search guided by the scorer cannot lower it unless the detector prefers something other than
  a butterfly over a meadow, in which case that sentence goes in the caption.

**Why this matters right now.** The correction forks the run at step 1 (fork step median 1 in the
landing finding). A search over the noise forks at step 0 with no correction at all. If it works,
the early-window correction has a training-free rival and the paper must say why the adapter is
still worth it; if it does not, the composing state is not a small nudge away, which is a sentence
the mechanism section can use.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Plans 09 and 10 select among particles mid-run. Nothing selects among starting
noises, which is the cheapest inference-time lever there is and the one the paper says scales.

**The approach.** Read random search off the existing renders for free, then run one round of
zero-order search at two step sizes with the same scorer as verifier.

**Key insights.**

1. With a deterministic sampler a noise is an image, so the search never touches the score and
   whatever it recovers is the sampler's share by construction, as in plans 09 and 10.
2. Reading the verifier at step 10 as well as at the end asks the same blindness question plan 10
   asks, on a method where the read is a pick rather than a resample.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  seed s, cached noise z (the pivot)

  z ──plain PoE, eta 0, 50 steps──► pivot image ──detector──► count (the baseline column)

  for sigma in {0.1, 0.3}:
     u_1..u_8 ~ N(0, I);  z'_k = (z + sigma·u_k) / sqrt(1 + sigma²)
     z'_1..z'_8 ──plain PoE, batched──► at step 10: x0hat_k ──VAE──► png ──detector──► early count_k
                                       at step 49: image_k  ──VAE──► png ──detector──► final count_k
     kept_final = argmax_k final count_k          (paper's setting)
     kept_early = argmax_k early count_k, scored on image_k   (does the verifier work at step 10)

  sheet per pair, rows seeds 9..16:
  [ Mono | PoE pivot | PoE + λ1.2 (existing) | best-8 σ0.1 final | best-8 σ0.3 final | best-8 σ0.1 step10 | best-8 σ0.3 step10 ]
  both-ness of every cat x dog tile on the landing finding's axes
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The search.** `search.py`: `perturb` (the mixture above), `cosine_to_pivot`,
   `run_poe_candidates` (plain PoE at `eta` 0 over K candidates at once, capturing x0-hat at the
   early step), `pick`, `verdict`, and the constants `PASS_MARGIN`, `NULL_MARGIN`, `SIGMAS`,
   `N_CANDIDATES`, `VERIFIER_EARLY_STEP`, `ETA`.
2. **The runner.** `run.py`: per pair and seed, the Mono reference, the pivot with its step-10
   x0-hat, the identity check against the existing λ 0 render, then per σ the 8 candidates with
   both reads scored; `cell.json` per seed; the sheet per pair with counts drawn on every tile;
   `summary.json`, `verdict.json`; both-ness of every cat × dog tile on the landing finding's axes
   (`bothness.json`), with a `--bothness-only` mode for a node without network; W&B run in
   `prime_lab/poe-repair-animals-compose`, group `noise-search`.
3. **The launcher.** `scripts/noise_search/run_noise_search.sh {smoke|full}` with the disk guard on
   the output root, the python-by-node switch, the faulted-device guard (`poe-launch-002`) and a
   `torch.cuda.is_available()` check on the pinned device before any model loads.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves the scope's mission from the cheapest selection side: if the composing image is a small
nudge of the noise away, the sampler's share is large and reachable without any model change.

**Goals**

1. The random-search row at N 8 written from the existing renders, source named.
2. A smoke run proves the whole path: references, candidates, both reads, sheet, sidecar,
   verdict.
3. Two sheets (cat × dog, butterfly × meadow), seeds 9 to 16, with `.json` sidecars, logged to
   W&B; `verdict.json` quoted in the review file with the run id, node, device and PID; both-ness
   of the kept images beside the counts.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- Python by node: `/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python` on mscluster110 to
  112 (Blackwell), `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python` elsewhere, per
  [environment/hpc/nodes.md](../../../../environment/hpc/nodes.md). Never a bare `python`.
- `torch.cuda.is_available()` checked on the pinned device before real work; a faulted card lists
  in `nvidia-smi` and runs on the CPU, per
  [poe-launch-002](../../../../environment/known-failures.md).
- Outputs under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/`, disk
  guard on that path, per [environment/storage.md](../../../../environment/storage.md). Nothing
  under `/home-mscluster`.
- The training cache at `/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/` with
  `embeddings.pt` per cell: cat × dog seeds 9 to 16, butterfly × meadow seeds 9 to 12.
- The landing finding's saved features at
  `artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space-dino-feats.npy` (48
  rows, six conditions by seeds 9 to 16), which fix the both-ness axes.
- DINOv2 loads through `torch.hub` from the shared home cache; a node without network may fail
  there, which is what `--bothness-only` on the session node is for.
- biggpu allows one Slurm job per user; long runs start with `nohup` on a free device and
  `squeue` is blind to them; check `pgrep -af 'sweep|train|corrector|noise_search'` before
  claiming a device, per
  [environment/hpc/execution-protocol.md](../../../../environment/hpc/execution-protocol.md).
- The tracker is W&B, project `prime_lab/poe-repair-animals-compose`; the review file carries the
  run id and verdict, never curves.
- fp16 SDXL; the VAE decodes in fp32.

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against context/, environment/, runbook/, report/,
      and any learning journey that names this project.
  - Paste: `/xref-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--read-the-paper-and-write-the-free-row)**.

### 1. 📄 Read the paper and write the free row

- [x] **1.1** Read arXiv 2501.09732 in full and record what this plan copies and what it had to
      invent: random search as best of N, zero-order search as N candidates around a pivot, the
      verifier on the finished image, and the missing perturbation formula.
  - Paste: `/unpack-paper https://arxiv.org/abs/2501.09732`
  - Done when: the register row for 2501.09732 exists in
    [the reading register](../../../standing/literature/reading-register.md) and the note is
    filed under `artifacts/notes/inference-time-scaling-as-a-search-over-noise/`.
- [x] **1.2** Write the random-search row at N 8 into the review file from the existing renders:
      plain PoE and the rank-32 λ 1.2 run on cat × dog seeds 9 to 16, from
      `/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_030050/results.json`, and their
      both-ness from
      `artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space.json`.
  - Done when: the row names the source file and the field each number came from.
- [x] **1.3** Write the falsification criterion with its numbers into the review file and the
      constants into `search.py` before anything runs.
  - Done when: `PASS_MARGIN`, `NULL_MARGIN`, `SIGMAS`, `N_CANDIDATES`, `VERIFIER_EARLY_STEP` and
    `ETA` are in source and the review file's pre-registered question quotes them.

▶ **Next: [task 2.1](#2--build-and-smoke-the-path)**.

### 2. 🔧 Build and smoke the path

◀ **Needs: [task 1.3](#1--read-the-paper-and-write-the-free-row)**, the bars in source.

- [x] **2.1** Write `poe_repair/experiments/noise_search/` (search, runner) and the launcher
      `scripts/noise_search/run_noise_search.sh`.
  - **Done when:** the package imports, `perturb` returns unit-variance candidates at the expected
    cosine to the pivot, `pick` breaks ties by confidence then index, and `verdict` returns
    support, null and inconclusive on hand-made rates.
- [x] **2.2** Run the smoke mode on a free device: cat × dog seed 9 only, N 2, σ 0.3, 10 steps,
      512², x0-hat captured at step 2, detector on, W&B off.
  - Command: `ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_search/run_noise_search.sh smoke > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/logs/smoke.log 2>&1 &'`
  - **Done when:** the run dir holds `mono_eta0.png`, `poe_pivot.png`, its step-2 x0-hat, a
    `sigma_0.3/` folder with two finals, two x0-hats and `run.json`, a one-row sheet with its
    sidecar, `summary.json` and `verdict.json`, and the log shows the device name and `cuda`
    from torch. The images are noise at 10 steps and 512²; this proves the plumbing and reads
    nothing.
  - Ran 2026-09-05 18:28 on mscluster107 device 1, python PID 44055, 51 s for the seed after
    the model load; every listed output present (the review file's Runs table).

▶ **Next: [task 3.1](#3--the-full-run)**.

### 3. 🏋️ The full run

◀ **Needs: [task 2.2](#2--build-and-smoke-the-path)**, the smoke clean.

- [x] **3.1** Launch the full mode with `nohup` on a free device: both pairs, seeds 9 to 16, σ 0.1
      and 0.3, N 8, 50 steps, 1024², W&B online.
  - Command: `ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_search/run_noise_search.sh full > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/logs/full.log 2>&1 &'`
  - 💡 `/run-experiment` ✅ for the launch and the harvest.
  - **Done when:** the W&B run id, the run dir, the node, the device and the PID are in the review
    file's Runs table.
  - Launched 2026-09-05 18:33 on mscluster107 device 1, python PID 46316, W&B `3yxrkwgx`, run
    dir `zo_N8_s50_20260905-183320`; finished 20:27.
- [x] **3.2** Harvest: if `bothness.json` is missing, run
      `--bothness-only --run-id <run>` on the session node; copy the two sheets, `summary.json`
      and `bothness.json` into `artifacts/results/is-the-gap-the-samplers-or-the-models/` with
      README entries; quote `verdict.json` in the review file; answer every pre-registered question
      there; write the finding in `report/is-the-gap-the-samplers-or-the-models/`.
  - **Done when:** the review file's verdict is written and the finding exists.

▶ **Next: [instruction 4.1](#4--read-the-sheets)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [the check before moving on](#the-check-before-moving-on).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 4. 👁️ Read the sheets

◀ **Needs: [task 3.2](#3--the-full-run)**, the sheets filed.

- [ ] **4.1** Open `artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-cat-dog-sheet.png`.
      Row by row, write down whether a kept tile that counts 2 shows two bodies or one fused
      animal with two detected parts. A "compose" that is a fused face is a detector error, and
      its both-ness in `noise-search-bothness.json` should sit near the PoE band (about 0.2) rather
      than the joint band (about 0.5).
- [ ] **4.2** Open the butterfly × meadow sheet the same way. Expected result: every column shows
      a clear butterfly over a meadow. ❌ A kept column that lost the butterfly means the verifier
      prefers something else; record it in the review file's caveats.
- [ ] **4.3** In W&B, project `prime_lab/poe-repair-animals-compose`, group `noise-search`, open
      the run, Charts tab, and read `secondary/a_cat__x__a_dog/sigma_0.3_early_agrees_with_final`
      (the share of candidates whose step-10 count gives the same compose verdict as their final
      count). Near 0.5 or below means the step-10 verifier is guessing.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** this is the cheapest inference-time baseline in the scope, and
> the one that decides whether "search the noise" belongs in the paper's related-work sentence or
> in its results table.

```bash
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search
ls "$OUT"                                          # run dirs and logs/
ls "$OUT"/<run>/a_cat__x__a_dog | wc -l            # expect 8 seed folders
ls "$OUT"/<run>/a_cat__x__a_dog/seed_9/sigma_0.3   # expect c0..c7.png, their x0-hats, run.json
ls "$OUT"/<run>/*.png                              # two sheets
cat "$OUT"/<run>/verdict.json
```

**Pass criteria**

- The smoke run's outputs listed in task 2.2, then the full run's id, node, device and PID in the
  review file.
- Two sheets with no red "missing" boxes, two sidecars, `bothness.json`.
- `verdict.json` quoted in the review file.

**Fail criteria (STOP)**

- The pivot's render is a different scene from the existing λ 0 render of the same seed (a
  different starting noise). The mean absolute pixel difference is recorded per seed but cannot
  decide this alone: the two samplers part late on some seeds (up to 14.5 grey levels on seed 12
  with the same scene and a different face), so the layout match is the read.
- `torch.cuda.is_available()` printed `False` in the log, or a batch of 8 took more than 20
  minutes.

**Partial pass guidance**

- A run stopped early still has its per-seed `cell.json`; draw the sheets with red boxes for the
  missing tiles and read what exists, marking the verdict as read at that seed count.

**When you get results, answer**
[the review file](../../review/11-zero-order-search-over-the-initial-noise.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `<run>/noise_search_<pair>_sheet.png` | — | rows seeds 9 to 16; columns Mono, plain PoE from the pivot, the existing rank-32 λ 1.2 render (cat × dog only), best of 8 at σ 0.1 and 0.3 by the final-image verifier, best of 8 at σ 0.1 and 0.3 by the step-10 verifier; the detector's count drawn on every tile; the varied setting (the starting noise) named in the column label | task 3.1 | ✅ | one per pair, logged to W&B as `sheets/<pair>` |
| `<run>/noise_search_<pair>_sheet.json` | — | per tile: source path, detector count, compose verdict; per column: compose rate over 8 seeds; secondary: any-candidate-composes, candidate compose fraction, step-10 agreement, mean cosine to pivot | task 3.1 | ✅ | sidecar, bundled in the W&B artifact |
| `<run>/summary.json`, `<run>/verdict.json` | — | the compose rates by column, the secondary reads, the verdict string with its bars, for both verifiers | task 3.1 | ✅ | quoted in the review file |
| `<run>/bothness.json` | — | which-animal and both-ness of every cat × dog tile on the landing finding's axes, per seed and mean per column | task 3.1 or 3.2 | ✅ | read beside the counts |
| `<run>/<pair>/seed_N/sigma_S/c{k}_xhat_step10.png` | — | every x0-hat the step-10 verifier read, for the eye to check what the detector saw | task 3.1 | ✅ | **Supplementary** |

### Organization workflow

1. Everything under the run dir on `/datasets`.
2. The two sheets, `summary.json` and `bothness.json` promoted to
   `artifacts/results/is-the-gap-the-samplers-or-the-models/` as `noise-search-cat-dog-sheet.png`,
   `noise-search-butterfly-meadow-sheet.png`, `noise-search-summary.json` and
   `noise-search-bothness.json`, with README entries.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| a run launches or finishes | the review file's Runs table with its W&B id, node, device and PID |
| `verdict.json` is written | the review file's pre-registered question, and the finding in `report/` |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, by `sync-plan-tree` from the pending-sync list |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md` | **task 0.1** | Conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md` | **the close out** | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto | Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/11-zero-order-search-over-the-initial-noise.md` | **the close out** | Statuses and running order match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [poe_repair/experiments/noise_search/search.py](../../../../poe_repair/experiments/noise_search/search.py) | the perturbation, the batched sampler with the early capture, the pick rule, the constants, the verdict |
| [poe_repair/experiments/noise_search/run.py](../../../../poe_repair/experiments/noise_search/run.py) | the runner, the sheets, the sidecar, the both-ness read, W&B |
| [poe_repair/experiments/fk_steering/run.py](../../../../poe_repair/experiments/fk_steering/run.py) | the pinned-noise loader, the detector wrapper and the sheet drawer this plan reuses |
| [poe_repair/experiments/twisted_smc/sampler.py](../../../../poe_repair/experiments/twisted_smc/sampler.py) | the DDIM step and the Mono reference sampler |
| [poe_repair/experiments/compose_scorer_validation/detection_scorer.py](../../../../poe_repair/experiments/compose_scorer_validation/detection_scorer.py) | the validated instance count, used as the verifier and the judge |
| [scripts/showcase/where_each_condition_lands_plot.py](../../../../scripts/showcase/where_each_condition_lands_plot.py) | the cloud axes the both-ness read reproduces |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

A support makes noise search a row in [step 29's](06-three-rules-on-one-amount-axis.md) comparison
and the sampler-side sentence in the paper's corrector section, and the follow-up is the paper's
iterated climb. A null is the model-side reading and goes to the same section as the reason the
adapter adds a direction. Either is read against the in-span share once
[what the correction is made of](../../../05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md)
lands: a search over noises can only reach what plain PoE renders from some noise.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>Catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🟡 a listed card that runs on the CPU

**When it happens:** the pinned device is in the fault state of `poe-launch-002`.
**Fix:** the launcher refuses a device whose utilisation reads `[N/A]` or `[GPU requires reset]`,
and the runner exits if `torch.cuda.is_available()` is `False`.

#### 🟡 an SSH launch with a relative path finds nothing

**When it happens:** a non-interactive SSH command starts in `$HOME` (`poe-launch-001`).
**Fix:** every path on the launch line is absolute, and the launcher does its own `cd`.

#### 🟡 the both-ness read fails on a node without network

**When it happens:** `torch.hub` cannot reach GitHub to validate the DINOv2 repo.
**Fix:** the runner catches it and names the `--bothness-only` rerun on the session node.

</details>
