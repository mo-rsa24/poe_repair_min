# 🎲 Twisted SMC on a learned joint-versus-PoE twist

Run K particles of the plain product-of-experts sampler, weight each by a learned estimate of how
much more likely its state is under the joint prompt than under the product, and resample. No
correction is ever added to the score. If choosing among what the product already proposes is
enough to compose, the gap is the sampler's to close.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md — <pass / null / inconclusive, the compose fractions>
```

## Recommended skill

▶ `/run-experiment` ✅ for the launch; `/analyze-run` ✅ for the W&B curves and the strips.

## Position in the plan tree

**Step 51 of 51.** Waits on nothing: the training cache it reads exists. The one order is the
`## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 26 | [hypothesis-02: what-is-left-once-the-chain-settles](../hypothesis/03-what-is-left-once-the-chain-settles.md) ⚠️ | the Langevin read of the same question; this plan is its particle-filter cousin and does not wait on it |
| **51 (current)** | **baseline-05: twisted-smc-on-a-learned-joint-vs-poe-twist** 〰️ | **a contrastive twist head trained on the cache, a K-particle PoE sampler that resamples on it, a strip every 10k steps** |

Design only. Verdicts and run state live in
[the paired review file](../../review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
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

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

Train a classifier that tells a re-noised joint-prompt latent from a re-noised PoE latent at the
same timestep, use its logit as the weight in a K-particle resampling sampler over the plain PoE
score, and read whether the resampled particle composes more often than the same particles left
unweighted.

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** `poe_repair/experiments/twisted_smc/`: the twist head (`twist.py`, a 4.7M
parameter conv classifier on the 4×128×128 latent, conditioned on the timestep and the joint
prompt's pooled text embedding), the contrastive bank (`data.py`: `mono.png` and `poe.png` of every
training-cache cell, VAE-encoded once), the particle sampler (`sampler.py`), and the trainer that
renders the strips (`train.py`). Launched by `scripts/twisted_smc/train_twist.sbatch`.

**What it does.** The optimal twist for a sampler whose proposal is PoE and whose target is the
joint prompt is `psi_t(x_t) = p_J,t(x_t) / p_PoE,t(x_t)`. A classifier trained with binary
cross-entropy on samples of the two distributions at time `t` has that ratio as its optimal logit.
Positives are Mono finals and negatives PoE finals, both re-noised to one shared random timestep
by the forward kernel, which is the amortisation trick of Contrastive Distribution Matching
([arXiv 2605.23346](https://arxiv.org/abs/2605.23346)) carried to continuous latents. At sampling
time each particle's log-weight gains `log psi_{t'}(x_{t'}) − log psi_t(x_t)` per step and the
population is systematically resampled when the effective sample size drops below half of K.

**Key components.** The bank, the head, the sampler, the strip.

**Testing approach.** Three panels from the same K noise draws at the same step stochasticity:
Mono (joint prompt, target), PoE (product, unweighted, the control), twisted SMC (product,
weighted and resampled). Only the weighting differs between the last two.

**Associated materials.** [The review questions](../../review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md),
[the cache format](../../../../poe_repair/training_cache.py), the Feynman-Kac read at
[step 30](../ideas/07-feynman-kac-correctors.md), and the amortised-twist paper above.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**The proposal has to be stochastic.** Deterministic DDIM makes duplicated particles identical
forever, so resampling can only lose diversity. The sampler steps with DDIM at `eta` 1.0 (fresh
noise each step). Mono and the PoE control use the same `eta`, K and noise draws, so the
comparison stays on one axis; the repo's usual `eta` 0 renders are not the control here.

**The trained corrector is the same object seen from the other side.** `r_t = ε̃_J − ε̃_PoE` is,
up to the noise schedule, the gradient of `log psi_t`. The LoRA learned that gradient by
regression and adds it to the score; this plan learns the potential itself contrastively and
never adds anything. If both compose, the paper can say the correction is one object with two
delivery routes. If only the LoRA composes, choosing among the product's own proposals is not
enough, which is the model-side reading.

**The positive set is small.** 120 Mono finals across 18 training pairs. The head is regularised
by dropout, weight decay, horizontal flips and the re-noising itself (every clean latent yields
a fresh `x_t` each draw), and validated on 16 held-out cells. The validation accuracy curve is
the first thing to read; a head that memorises the training pairs will show it there.

**The strip's third panel is one of K particles.** The image shown is the particle with the
largest final weight; the other K−1 are saved beside it and the detector reads all of them. The
compose fraction in the verdict is over every particle of every render cell, not the shown one.

**Cost.** The head trains at about 37 steps per second on a Blackwell node; a 100k-step run took 45 minutes
including the renders (job 49853). Each render cell costs two particle runs at start (the Mono and
PoE references) and one per strip: at K 4, 20 steps, 1024², about 12 seconds on a Blackwell
node per run, measured on job 49850.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Reweighting and resampling the plain PoE sampler on a learned joint-versus-PoE twist raises
the compose fraction over the same particles left unweighted.**

**Independent variables.** The twist on or off, at fixed K 4, `eta` 1.0, 20 render steps,
guidance 7.5, three render cells (two training pairs, one held-out pair), one seed each.

**Dependent variable.** Compose fraction by the validated detector (two or more animal instances)
over all K particles of all render cells, per panel.

**Falsify condition.** The bars sit in `train.py` as `PASS_MARGIN`, `NULL_MARGIN` and
`MIN_VAL_ACC`, and the run writes `verdict.json` from the final render.

- **Pass.** SMC compose fraction exceeds the PoE control's by 0.25 or more. Choosing among the
  product's proposals is enough on these cells; the sampler holds a real share of the gap.
- **Null.** The two fractions are within 0.10 of each other while validation accuracy is above
  0.60. The twist learned something and it changed nothing: the composing states are not among
  the product's proposals.
- **Inconclusive.** Validation accuracy never reaches 0.60. The head did not learn the ratio and
  the sampler was never tested.

**Why this matters right now.** It is the sampler-side baseline the scope's question needs and
the only one in this repo that adds no direction the model did not propose.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Every intervention in this repo so far adds a vector to the score. A reviewer
can ask whether the product already contained composing samples that a better sampler would have
found.

**The approach.** A particle filter that only selects. Its twist is learned, so the selection is
informed by the joint prompt without the joint prompt's score ever entering the chain.

**Key insights.**

1. The classifier logit between two diffused distributions at the same `t` is the twist a
   twisted-SMC sampler needs, and the training cache already holds both distributions' finals.
2. Re-noising clean latents with the closed-form forward kernel gives unlimited training states
   from a fixed set of images, so no UNet call happens during training.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  training cache cell           bank                       twist head
  mono.png ──VAE──► z_J  ─┐    re-noise to t ──► x_t, label 1 ─┐
  poe.png  ──VAE──► z_P  ─┘    re-noise to t ──► x_t, label 0 ─┴─► BCE ──► log psi_t(x_t | pool_J)

  sampling, K particles on the PoE score, eta 1.0
  step t ──► x_t' for each particle ──► log w += log psi_t'(x_t') − log psi_t(x_t)
         ──► ESS < K/2 ? systematic resample : continue

  every 10k steps, per render cell:   [ Mono | PoE control | twisted SMC ]  ──► W&B image + artifact
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The bank.** `data.py`: discover complete cache cells, VAE-encode `mono.png` and `poe.png`
   with the scaling factor applied, store with each cell's `pool_j`, cache under
   `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc/latent_bank/`.
2. **The head.** `twist.py`: conv stem, four FiLM-conditioned stride-2 blocks, global pool, MLP
   to one logit. Conditioning is a sinusoidal timestep embedding plus a projection of `pool_j`.
3. **The sampler.** `sampler.py`: `run_particles(composition="poe"|"mono", twist=None|head, eta,
   ess_threshold)`, chunked UNet forwards over particles, DDIM step with `eta`, per-step weight
   increments, systematic resampling, ESS and ancestry recorded.
4. **The trainer.** `train.py`: AdamW on the BCE, validation on held-out cells, checkpoints under
   `checkpoints/`, references rendered once, strips every `--render-every` steps logged as
   `samples/<split>/<pair>/seed_NN` images and a `strips-step<N>` artifact, detector reads under
   `eval/compose/{mono,poe,smc}/...`, `verdict.json` at the end.
5. **The launcher.** `scripts/twisted_smc/train_twist.sbatch {smoke|full}` with the disk guard on
   the output root, python chosen by node (`co3_bw` on the Blackwell nodes).

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves the scope's mission from the selection side: a corrector that never adds a direction, so
whatever it recovers is the sampler's share by construction.

**Goals**

1. A smoke run proves the whole path: bank, head, references, strip, artifact, checkpoint.
2. A 100k-step run in W&B with validation accuracy, ESS and compose curves and ten strips.
3. `verdict.json` written by the code's own bars, quoted in the review file with the run id.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- Python by node: `/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python` on mscluster110 to
  112 (Blackwell), `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python` elsewhere, per
  [environment/hpc/nodes.md](../../../../environment/hpc/nodes.md). Never a bare `python`.
- Bank, checkpoints and renders under
  `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc/`, disk guard on that
  path, per [environment/storage.md](../../../../environment/storage.md).
- The training cache at `/datasets/mmolefe/poe_repair_min/outputs/training_cache/` with
  `mono.png`, `poe.png`, `embeddings.pt` and `meta.json` per cell; 120 train cells over 18 pairs
  as of 2026-09-05.
- biggpu allows one job per user; the launch follows
  [environment/hpc/execution-protocol.md](../../../../environment/hpc/execution-protocol.md).
- The tracker is W&B, project `prime_lab/poe-repair-animals-compose`; the review file carries run
  ids and verdicts, never curves.
- fp16 SDXL, fp32 for the head and for every norm and weight.
- Renders at 20 DDIM steps, `eta` 1.0, guidance 7.5, 1024², K 4.

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against context/, environment/, runbook/, report/,
      and any learning journey that names this project.
  - Paste: `/xref-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--build-and-smoke-the-path)**.

### 1. 🔧 Build and smoke the path

- [x] **1.1** Write the package `poe_repair/experiments/twisted_smc/` (bank, head, sampler,
      trainer) and the launcher `scripts/twisted_smc/train_twist.sbatch`.
  - **Done when:** the package compiles, the head returns one logit per latent at 1024² and 512²,
    and the resampler returns K indices.
- [x] **1.2** Run the smoke mode on an idle biggpu node: 2 train pairs × 2 seeds, 1 held-out
      cell, 40 steps, K 2, 4 render steps, 512², renders at steps 0, 20 and 40.
  - Command: `sbatch --nodelist=<idle node> scripts/twisted_smc/train_twist.sbatch smoke`
  - **Done when:** the run dir holds `references/`, three `samples/step_*/` folders with strips,
    `checkpoints/twist_step_000040.pt`, and the W&B run shows the three strip images and the
    `strips-step000040` artifact.
  - Job 49849, W&B run `5isfz35a`; every listed output present, in
    [the review file](../../review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md).

▶ **Next: [task 2.1](#2--the-full-run)**.

### 2. 🏋️ The full run

◀ **Needs: [task 1.2](#1--build-and-smoke-the-path)**, the smoke clean.

- [x] **2.1** Launch the full mode: 100k steps, batch 16 positives + 16 negatives, lr 1e-4,
      renders every 10k steps at K 4, 20 steps, 1024², two training pairs and one held-out pair.
  - Command: `sbatch --nodelist=<idle node> scripts/twisted_smc/train_twist.sbatch full`
  - 💡 `/run-experiment` ✅ for the launch and the harvest.
  - **Done when:** the W&B run id and the run dir are in the review file's Runs table.
  - Slurm job 49853 on mscluster110, W&B run `3cwrxlw0`, finished 2026-09-05 08:10; verdict in the review
    file at harvest.

▶ **Next: [instruction 3.1](#3--read-the-curves-and-the-strips)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [the check before moving on](#the-check-before-moving-on).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 3. 👁️ Read the curves and the strips

◀ **Needs: [task 2.1](#2--the-full-run)**, the run launched.

- [ ] **3.1** In W&B, project `prime_lab/poe-repair-animals-compose`, group `twisted-smc`, open
      the run, Charts tab, and read `val/acc`, `val/loss`, `train/loss_bucket/*`,
      `eval/ess_mean` and `eval/compose_*_mean`.
  - Expected result: `val/acc` climbs above 0.6 and flattens; the near-clean bucket's loss falls
    fastest and the high-noise bucket's least (the two distributions coincide at high noise);
    `eval/ess_mean` sits below K, so resampling fires.
  - ❌ `val/acc` stuck near 0.5: inconclusive, the twist did not learn; record and stop.
- [ ] **3.2** Media tab, `samples/train/...` and `samples/heldout/...`: step through the strips
      by step. Write down, per cell and checkpoint, whether the third panel shows two animals
      where the second shows one, and whether it looks like a different draw or a re-selected one
      (the caption lists the steps at which resampling fired).
  - ✅ Pass / ❌ null / 〰️ inconclusive per the claim's definitions, quoting `verdict.json`.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** this is the one baseline in the scope that adds nothing to
> the score, so its verdict is the cleanest read of the sampler's share.

```bash
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc
ls "$OUT"                                         # run dirs and latent_bank/
ls "$OUT"/<run>/samples | wc -l                   # expect 11 render folders (step 0 + 10 × 10k)
ls "$OUT"/<run>/checkpoints | tail -2             # twist_step_100000.pt, latest.json
cat "$OUT"/<run>/verdict.json
```

**Pass criteria**

- The smoke run's outputs listed in task 1.2, then the full run's run id in the review file.
- Eleven strips per render cell in W&B and on disk.
- `verdict.json` quoted in the review file.

**Fail criteria (STOP)**

- The PoE control panel and the SMC panel come from different noise draws (the `*__p0.png`
  of the reference and of the SMC render at step 0 should be near-identical before any resample).

**Partial pass guidance**

- A run stopped early still has its strips and checkpoints; read what exists and mark the verdict
  as read at that step.

**When you get results, answer**
[the review file](../../review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `<run>/samples/step_NNNNNN/<split>__<pair>__seed_NN__strip.png` | — | three panels, Mono, PoE control, twisted SMC, from the same K noise draws; the SMC panel is the largest-weight particle at that checkpoint | task 2.1 | ⏳ | logged to W&B under `samples/<split>/<pair>/seed_NN` and bundled in the `strips-step<N>` artifact |
| `<run>/samples/step_NNNNNN/*__smc__p{k}.png` | — | every particle, for the detector's compose fraction | task 2.1; scored by `scripts/twisted_smc/score_all_particles.py` into `<run>/all_particles_scores.json` | ✅ | **Supplementary** |
| `artifacts/results/is-the-gap-the-samplers-or-the-models/twisted-smc-checkpoint-timeline.png` | — | three cells by Mono, PoE and eleven SMC checkpoints, composed from the run's PNGs | `scripts/twisted_smc/report_figures.py` | ✅ | 📊 Drawn in [Figure 1 of the figure explainer](../../../../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer.md#figure-1-the-checkpoint-timeline-three-cells-by-eleven-checkpoints) |
| `artifacts/results/is-the-gap-the-samplers-or-the-models/twist-training-curves.png` | — | validation and training accuracy against step with the 0.60 bar; training loss per noise bucket on a log axis | the same script | ✅ | 📊 Drawn in [Figure 3 of the figure explainer](../../../../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer.md#figure-3-what-the-twist-head-learned-accuracy-and-loss-by-noise-level) |

### Organization workflow

1. Everything under the run dir on `/datasets`.
2. A strip promoted to the paper goes to `artifacts/results/is-the-gap-the-samplers-or-the-models/` with a README entry.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| a run launches or finishes | the review file's Runs table with its W&B id |
| `verdict.json` is written | the review file's pre-registered question |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, by `sync-plan-tree` |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md` | **task 0.1** | Conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md` | **the close out** | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto | Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md` | **the close out** | Statuses and running order match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [poe_repair/experiments/twisted_smc/twist.py](../../../../poe_repair/experiments/twisted_smc/twist.py) | the head |
| [poe_repair/experiments/twisted_smc/data.py](../../../../poe_repair/experiments/twisted_smc/data.py) | the bank and the re-noising sampler |
| [poe_repair/experiments/twisted_smc/sampler.py](../../../../poe_repair/experiments/twisted_smc/sampler.py) | the particle sampler and the weights |
| [poe_repair/experiments/twisted_smc/train.py](../../../../poe_repair/experiments/twisted_smc/train.py) | the loop, the strips, the bars |
| [poe_repair/_sdxl/metrics.py](../../../../poe_repair/_sdxl/metrics.py) | guided and PoE eps, the Tweedie estimate |
| [poe_repair/experiments/compose_scorer_validation/detection_scorer.py](../../../../poe_repair/experiments/compose_scorer_validation/detection_scorer.py) | the validated instance count |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

A pass makes twisted SMC a row in [step 29's](06-three-rules-on-one-amount-axis.md) comparison
and a sentence in the paper's corrector section. A null is the model-side reading and goes to
the same section as the reason the adapter adds a direction.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>3 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🔴 the twist separates the classes by an artefact

**When it happens:** positives and negatives differ in something other than content: a
different timestep distribution, a different encoding path, a different resolution.
**What you see:** validation accuracy near 1.0 from the first thousand steps.
**Why:** the classifier finds the easiest separating feature.
**How to fix:** one timestep draw shared by both classes per position, both classes through the
same VAE encode, and the near-clean bucket's loss read against the high-noise bucket's (they must
differ; at high noise the two distributions coincide).

#### 🟡 resampling collapses the population

**When it happens:** `eta` 0, or a twist with very large logits.
**What you see:** every particle identical after the first resample; ESS at 1 every step.
**Why:** duplicated particles under a deterministic step never separate again.
**How to fix:** `eta` 1.0 (the default), `--twist-temperature` below 1 if ESS still sits at 1.

#### 🟡 the co3 environment on a Blackwell node

**When it happens:** the launcher run on mscluster110 to 112 with `co3`.
**What you see:** no CUDA output at all rather than an error.
**Why:** `co3`'s torch has no `sm_120` kernels.
**How to fix:** the sbatch picks `co3_bw` by hostname; keep that switch.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
