# 🎯 Noise search on top of the adapter

The rank-32 correction already gives two animals on seven of eight held-out cat × dog seeds, and
its renders are softer than plain product-of-experts. This plan runs the same eight-candidate
search over each seed's starting noise as plan 11, with the adapter attached at λ 1.2, and lets
the verifier pick by compose count first and sharpness second. The goal is a render that snaps
into two clean animals at full sharpness, from a seed whose corrected render is soft, without
changing the adapter or the sampler.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/12-noise-search-on-top-of-the-adapter.md — <support / null / breaks / inconclusive, the sharpness ratio and the compose rates>
```

## Recommended skill

▶ `/run-experiment` ✅ for the launch; `/analyze-run` ✅ for the W&B sheets and the sidecar.

## Position in the plan tree

**Unnumbered until the next sync; proposed step 57 of 57.** Waits on plan 11's runner, which
exists. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md); the row for this plan is queued in
[the pending-sync list](../../../../artifacts/ideas/parallel-sessions-on-poe-dynamics-and-fixes/PENDING_SYNC.md).

| Step | Plan | What it does |
|------|------|-------------|
| 56 | [baseline-07: zero-order-search-over-the-initial-noise](11-zero-order-search-over-the-initial-noise.md) ❓ | the same search on plain PoE: one counted success in 128 candidates, a two-headed body |
| **57 (proposed, current)** | **baseline-08: noise-search-on-top-of-the-adapter** ◑ | **the same search with the correction attached, picking for sharpness once the count is satisfied** |

Design only. Verdicts and run state live in
[the paired review file](../../review/12-noise-search-on-top-of-the-adapter.md).

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

Among eight small perturbations of a seed's starting noise, rendered with the correction
attached, is there a two-animal render sharper than the one the cached noise gives, and how close
does it get to plain PoE's and the joint prompt's sharpness?

## Words this plan uses

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **The adapter**: the rank-32 LoRA at step 30050, applied on every step at λ 1.2 as
  `eps = eps_PoE(adapter off) + 1.2 · (eps_PoE(adapter on) − eps_PoE(adapter off))`, the on-step
  of the repo's LoRA sampler, here batched over candidates.
- **The adapter pivot**: the seed's cached noise rendered with the adapter. The unperturbed
  corrected render, and the baseline every kept image is judged against.
- **The plain pivot**: the same noise rendered with plain PoE by the same code; its sharpness is
  the band the corrected render fell out of.
- **A candidate**: `z' = (z + σ·u) / sqrt(1 + σ²)`, the same draws as plan 11 for the same seed
  and σ (same generator seed), rendered with the adapter.
- **Sharpness**: the variance of the second differences of the greyscale 1024 px render, plan 07's
  Laplacian-variance function. It counts edges, and sketch-style seeds score high whatever their
  blur, so every comparison is per seed against that seed's own pivots, and the statistic is a
  median of per-seed ratios, never a band across seeds.
- **The verifier**: the compose count clipped at 2 first, then sharpness, then the lowest index.
  A one-animal candidate never beats a two-animal one however sharp it is.
- **The kept image**: the candidate the verifier picks, one per seed per σ.
- **The sharpness ratio**: the kept image's sharpness over the adapter pivot's, per seed.

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** `poe_repair/experiments/noise_search/adapter.py`: the attach recipe, the batched
adapter sampler, the sharpness function, the pick rule, the bars. `run_adapter.py`: the runner in
three phases (references before the adapter is attached, the attach with its inert-when-disabled
check, then the search), the sheets, the sidecar, both-ness, W&B. Launched by
`scripts/noise_search/run_noise_search.sh adapter-smoke|adapter-full`.

**What it does.** Plan 11 asked whether a nudge of the noise composes plain PoE; it does not. The
adapter composes already, so here the count is the constraint and sharpness is the objective: the
search asks whether a nearby noise gives the adapter a crisper two-animal render.

**Key components.** The three phases, the verifier, the per-seed ratios against three references
(adapter pivot, plain pivot, Mono).

**Testing approach.** One sheet per pair, rows the held-out seeds 9 to 16, columns Mono, the plain
pivot, the adapter pivot, best of 8 at σ 0.1, best of 8 at σ 0.3. Sharpness per tile in the
sidecar. Only the starting noise differs between the adapter pivot and a kept tile.

**Associated materials.** [The review questions](../../review/12-noise-search-on-top-of-the-adapter.md),
[plan 11](11-zero-order-search-over-the-initial-noise.md) for the search and its result, session
B's [correct early, then clean up](../../../01-showcase-the-trained-lora/plans/experiments/14-correct-early-then-clean-up.md)
for the sharpness read this plan shares.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**References before the adapter.** The repo's LoRA samplers leave the adapter enabled when they
return (the re-render of 2026-09-05). Phase 1 renders Mono and the plain pivot for every seed
before the adapter is attached; phase 2 attaches it, disables it, renders plain PoE seed 9 again
and records the pixel difference against phase 1, which must be zero or a few grey levels.

**Sharpness counts edges.** Sketch seeds (11, 14, 15) sit high on Laplacian variance at any blur.
Every number is a per-seed ratio against that seed's own adapter pivot, plain pivot and Mono
render, and the verdict reads the median ratio over seeds.

**The count is the constraint.** The adapter fails on seed 14 (count 1 at λ 1.2). There the search
may find a composing candidate; it may also find none, in which case the verifier picks the
sharpest one-animal render, and the sheet shows that.

**The identity check.** The adapter pivot is compared pixel-wise against the existing λ 1.2 render
of the same seed. Plan 11 showed the two DDIM code paths part late on some seeds (up to 14.5 grey
levels with the same scene), so the layout match is the read and the number is recorded.

**Cost.** Two UNet passes per step with the adapter (off, then on), so twice plan 11: about 14 min
per (seed, σ) batch of 8 on a Quadro RTX 8000, 16 batches per pair, about 3.7 h per pair plus the
references. The two pairs run on two devices at once, one run dir each. Buys: the pre-registered
question, the two sheets, the both-ness read.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**With the correction attached, the best of eight nearby starting noises gives a two-animal render
sharper than the cached noise's, on most seeds, without losing the compose rate.**

**Independent variables.** σ in {0.1, 0.3}. Fixed: N 8, one round, the rank-32 step-30050 adapter
at λ 1.2 on every step, DDIM `eta` 0, 50 steps, guidance 7.5, 1024², two pairs, seeds 9 to 16.

**Dependent variables.** Compose rate of the kept image over 8 seeds; the median over seeds of
the kept image's sharpness over the adapter pivot's. Secondary, recorded and not judged: the
same ratio against the plain pivot and against Mono, the share of seeds where the kept image
reaches each of those bands, the kept rate under plan 11's count-then-confidence rule, the
candidate compose fraction, both-ness of every cat × dog tile.

**Falsify condition.** The bars sit in `poe_repair/experiments/noise_search/adapter.py` as
`MAX_COMPOSE_DROP`, `MIN_SHARPNESS_GAIN` and `NULL_SHARPNESS_BAND`, and the runner writes
`verdict.json` from them. Judged on cat × dog.

- **Support.** At either σ, the kept compose rate is within one seed of the adapter pivot's
  (`MAX_COMPOSE_DROP` 0.125) and the median sharpness ratio is at least 1.10
  (`MIN_SHARPNESS_GAIN` 0.10). A nearby noise gives a crisper two-animal render.
- **Null.** Compose holds at both σ and the median ratio is within 0.05 of 1 at both
  (`NULL_SHARPNESS_BAND`). The softness is not a property of the starting noise.
- **Breaks.** The kept compose rate falls more than one seed below the adapter pivot's at a σ.
  The verifier's taste for sharpness costs animals, and that sentence goes in the caption.
- **Inconclusive.** Anything else.
- **Butterfly × meadow** is a control and is read by eye (a meadow is not an animal, so the count
  is not that pair's read): every kept tile keeps a butterfly over flowers, or the caption says
  which seeds lost it.

**Why this matters right now.** Session B is testing schedules and tails to return the corrected
render's sharpness; this is the zero-cost version of the same goal, and if it works the paper's
showcase renders can be picked this way.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The corrected renders compose and are soft. Every fix on the table changes the
sampler or the schedule.

**The approach.** Change nothing but the starting noise, and let the judge pick.

**Key insights.**

1. Plan 11 showed a count verifier selects for second heads on plain PoE. With the adapter, the
   count is mostly saturated and the second key, sharpness, does the picking; the count still
   vetoes any one-animal candidate.
2. The same candidate draws as plan 11 mean each seed's neighbourhood is seen twice, once
   without and once with the correction.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  phase 1  (no adapter)   z ──plain PoE──► plain pivot      z ──Mono──► Mono reference     for every seed
  phase 2  attach rank-32 @30050; disable; plain PoE seed 9 again == phase 1 (detach check)
  phase 3  (adapter on, lambda 1.2)
           z ──────────────────────────────────► adapter pivot ──detector, sharpness──► the baseline
           z'_1..z'_8 = (z + sigma u_k)/sqrt(1+sigma²) ──► 8 renders ──detector, sharpness──► kept = argmax (count, sharpness)

  per seed: ratio = sharpness(kept) / sharpness(adapter pivot); ceilings: plain pivot, Mono
  verdict on the median ratio over 8 seeds, with compose within one seed of the adapter pivot

  sheet per pair, rows seeds 9..16:
  [ Mono | plain pivot | adapter pivot λ1.2 | best-8 σ0.1 | best-8 σ0.3 ]
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The adapter search.** `adapter.py`: `attach_and_load_lora` (rank 32, alpha 32,
   cross-attention q, k, v, the probe helper's recipe), `run_adapter_candidates` (batched, two
   forwards per step), `laplacian_var` (plan 07's function), `pick_count_then_sharpness`,
   `verdict`, and the constants.
2. **The runner.** `run_adapter.py`: the three phases, `detach_check.json`, per-seed `cell.json`
   with sharpness on every tile and the per-seed ratios, the sheet per pair, `summary.json`,
   `verdict.json`, `bothness.json` (with `--bothness-only`), W&B group `noise-search-adapter`.
3. **The launcher.** Two new modes on `scripts/noise_search/run_noise_search.sh`,
   `adapter-smoke` and `adapter-full`, behind the same guards.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

A corrected render that is two clean animals at full sharpness, picked rather than re-sampled.

**Goals**

1. A smoke run proves the three phases, the detach check, the sheet, the sidecar and the verdict.
2. Two sheets, seeds 9 to 16, with sidecars carrying sharpness per tile, logged to W&B;
   `verdict.json` quoted in the review file with the run ids, nodes, devices and PIDs.
3. The per-seed ratios against the plain pivot and Mono, so the write-up can say how much of the
   softness a picked noise recovers.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- Everything plan 11 depends on, unchanged: python by node, the CPU-fallback guard, outputs under
  `/datasets/.../noise_search/`, the training cache, the landing finding's saved features, the
  `nohup` launch and `pgrep` harvest, W&B, fp16 SDXL with fp32 VAE decode.
- The checkpoint at
  `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`,
  key `lora_state`, rank 32; alpha equals rank in the showcase trainings.
- `XFORMERS_DISABLED=1` for any CPU DINOv2 embed on a launch node (plan 11's both-ness failure).
- Two free devices for the two pairs at once, each checked with `nvidia-smi` and `pgrep` first.

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/12-noise-search-on-top-of-the-adapter.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against context/, environment/, runbook/, report/.
  - Paste: `/xref-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/12-noise-search-on-top-of-the-adapter.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--pin-the-bars-and-build-the-path)**.

### 1. 🔧 Pin the bars and build the path

- [x] **1.1** Write the falsification criterion with its numbers into the review file and the
      constants into `adapter.py` before anything runs.
  - Done when: `MAX_COMPOSE_DROP`, `MIN_SHARPNESS_GAIN`, `NULL_SHARPNESS_BAND`, `LAMBDA`,
    `LORA_RANK`, `LORA_ALPHA`, `SIGMAS`, `N_CANDIDATES` and `ETA` are in source and the review
    file's pre-registered question quotes them.
- [x] **1.2** Write `adapter.py` and `run_adapter.py` and the two launcher modes.
  - **Done when:** the package imports, the pick rule prefers a count-2 candidate over a sharper
    count-1 one, and `verdict` returns support, null, breaks and inconclusive on hand-made rates.
- [x] **1.3** Run the smoke mode on a free device: cat × dog seed 9, N 2, σ 0.3, 10 steps, 512².
  - Command: `ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_search/run_noise_search.sh adapter-smoke > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/logs/adapter_smoke.log 2>&1 &'`
  - **Done when:** the run dir holds `mono_eta0.png`, `poe_pivot.png`, `detach_check.json` with a
    difference near zero, `adapter_pivot_lambda1.2.png`, `sigma_0.3/` with two finals and
    `run.json`, a one-row sheet with its sidecar, `summary.json` and `verdict.json`, and the log
    shows the attach line and the device.
  - Ran 2026-09-06 02:38 on mscluster107 device 1, python PID 56767: 210 modules matched, 420
    tensors loaded at step 30050, detach check 0.0 grey levels, every listed output present.

▶ **Next: [task 2.1](#2--the-full-run)**.

### 2. 🏋️ The full run

◀ **Needs: [task 1.3](#1--pin-the-bars-and-build-the-path)**, the smoke clean.

- [x] **2.1** Launch the full mode with `nohup`, one pair per free device: cat × dog on one,
      butterfly × meadow on the other, seeds 9 to 16, σ 0.1 and 0.3, N 8, 50 steps, 1024², W&B
      online.
  - Command: `ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_search/run_noise_search.sh adapter-full --pairs <pair> > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/logs/adapter_full_<pair>.log 2>&1 &'`
  - **Done when:** both W&B run ids, run dirs, nodes, devices and PIDs are in the review file's
    Runs table.
  - Launched 2026-09-06 02:42 (cat × dog, mscluster107 device 1, PID 57966, W&B `kqj26oiz`) and
    02:43 (butterfly × meadow, mscluster109 device 1, PID 1860430, W&B `6oai0d99`).
- [x] **2.2** Harvest: if `bothness.json` is missing on the cat × dog run, run
      `--bothness-only --run-id <run>` on the session node with `XFORMERS_DISABLED=1`; copy the
      two sheets, both `summary.json` files, `bothness.json` and `verdict.json` into
      `artifacts/results/is-the-gap-the-samplers-or-the-models/` as `noise-search-adapter-*` with
      README entries; quote `verdict.json` in the review file; answer every pre-registered
      question; write the finding in `report/is-the-gap-the-samplers-or-the-models/`.
  - **Done when:** the review file's verdict is written and the finding exists.

▶ **Next: [instruction 3.1](#3--read-the-sheets)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit.**
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/12-noise-search-on-top-of-the-adapter.md`
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/12-noise-search-on-top-of-the-adapter.md — <one line>`

▶ **Next: [the check before moving on](#the-check-before-moving-on).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.**

### 3. 👁️ Read the sheets

◀ **Needs: [task 2.2](#2--the-full-run)**, the sheets filed.

- [ ] **3.1** Open `artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-adapter-cat-dog-sheet.png`.
      Row by row, compare the kept tiles against the adapter pivot: are they two separate
      animals, and are they visibly crisper (fur texture, edges, eyes)? Write down the seeds
      where the kept tile looks like the Mono tile's sharpness, and the seeds where a kept tile is
      one animal or a two-headed body with a count of 2 or more.
- [ ] **3.2** Open the butterfly × meadow sheet. Expected: a butterfly over flowers in every kept
      tile. ❌ A kept tile that is a print, a pattern or a meadow with no butterfly is the
      verifier's taste showing; record the seeds in the review file's caveats.
- [ ] **3.3** In W&B, group `noise-search-adapter`, Charts tab, read
      `secondary/a_cat__x__a_dog/sigma_0.3_median_ratio_kept_over_adapter_pivot` and
      `..._kept_reaches_mono_band`. A ratio near 1 with the eye read agreeing means the softness
      is not in the noise.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** if a picked noise returns the corrected render's sharpness, the
> showcase figures can be picked this way and session B's schedules become the fallback.

```bash
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search
ls -d "$OUT"/zo_adapter_*                          # two run dirs, one per pair
cat "$OUT"/<cat-dog run>/detach_check.json
ls "$OUT"/<cat-dog run>/a_cat__x__a_dog/seed_9/sigma_0.3   # c0..c7.png, run.json
cat "$OUT"/<cat-dog run>/verdict.json
```

**Pass criteria**

- The detach check under a few grey levels; the smoke's outputs; both runs' ids, nodes, devices
  and PIDs in the review file.
- Two sheets with no red "missing" boxes, sidecars with sharpness per tile, `bothness.json`.
- `verdict.json` quoted in the review file.

**Fail criteria (STOP)**

- The detach check above a few grey levels: the adapter is not inert when disabled, and every
  phase-1 reference rendered after it would be contaminated (they are rendered before, so this
  would say the attach itself changed the UNet).
- The adapter pivot is a different scene from the existing λ 1.2 render of the same seed.
- `torch.cuda.is_available()` printed `False`, or a batch of 8 took more than 40 minutes.

**When you get results, answer**
[the review file](../../review/12-noise-search-on-top-of-the-adapter.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `<run>/noise_search_adapter_<pair>_sheet.png` | — | rows seeds 9 to 16; columns Mono, the plain pivot, the adapter pivot at λ 1.2, best of 8 at σ 0.1 and 0.3 with the adapter on, picked by count then sharpness; the count on every tile | task 2.1 | ✅ | one per pair, W&B `sheets/<pair>` |
| `<run>/noise_search_adapter_<pair>_sheet.json` | — | per tile: path, count, compose, sharpness; per column: compose rate and mean sharpness; secondary: the per-seed sharpness ratios (median), the share of seeds reaching the plain and Mono bands, the count-then-confidence kept rate, the candidate compose fraction, the cosine to the pivot | task 2.1 | ✅ | sidecar |
| `<run>/detach_check.json`, `<run>/summary.json`, `<run>/verdict.json`, `<run>/bothness.json` | — | the inert-when-disabled check, the numbers, the verdict with its bars, the both-ness of every cat × dog tile | tasks 2.1 and 2.2 | ✅ | quoted in the review file |

### Organization workflow

1. Everything under the two run dirs on `/datasets`.
2. The two sheets and the sidecars promoted to
   `artifacts/results/is-the-gap-the-samplers-or-the-models/` as `noise-search-adapter-*`.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| a run launches or finishes | the review file's Runs table with its W&B id, node, device and PID |
| `verdict.json` is written | the review file's pre-registered question, and the finding in `report/` |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, by `sync-plan-tree` from the pending-sync list |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/12-noise-search-on-top-of-the-adapter.md` | **task 0.1** | Conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/12-noise-search-on-top-of-the-adapter.md` | **the close out** | Errors added to catalogs |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/12-noise-search-on-top-of-the-adapter.md` | **the close out** | Statuses and running order match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [poe_repair/experiments/noise_search/adapter.py](../../../../poe_repair/experiments/noise_search/adapter.py) | the attach recipe, the batched adapter sampler, sharpness, the pick rule, the bars |
| [poe_repair/experiments/noise_search/run_adapter.py](../../../../poe_repair/experiments/noise_search/run_adapter.py) | the three phases, the sheets, the sidecar, both-ness, W&B |
| [poe_repair/experiments/noise_search/search.py](../../../../poe_repair/experiments/noise_search/search.py) | the perturbation and the plain-PoE sampler the references use |
| [poe_repair/methods/_sampling.py](../../../../poe_repair/methods/_sampling.py) | `run_lora_residual_inject_masked`, whose on-step this plan batches |
| [scripts/showcase/lambda_window_grid.py](../../../../scripts/showcase/lambda_window_grid.py) | `_laplacian_var`, the sharpness function copied here |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

A support means the showcase renders are picked this way and the paper says so in the fidelity
paragraph; the follow-up is the paper's iterated climb with sharpness as the objective. A null
says the softness is not in the noise, which hands the question to session B's schedules and the
corrector tail in plan 04. A break says the verifier's taste costs animals and a fidelity term
needs a compose veto stronger than the count.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>Catalogued failures and their fixes</summary>

### From project catalog

#### 🟡 the adapter stays attached after a windowed run

**When it happens:** any render after a LoRA sampler returns, in the same process.
**Fix:** phase 1 renders every reference before the attach; phase 2 proves the adapter inert when
disabled.

#### 🟡 a listed card that runs on the CPU

**Fix:** the launcher's faulted-device guard and the runner's `torch.cuda.is_available()` exit.

#### 🟡 CPU DINOv2 fails on a node where xformers imports

**Fix:** `XFORMERS_DISABLED=1` and `--bothness-only` on the session node.

</details>
