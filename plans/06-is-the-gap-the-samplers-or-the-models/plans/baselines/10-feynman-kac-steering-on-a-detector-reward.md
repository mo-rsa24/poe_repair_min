# 🎲 Feynman-Kac steering on a detector reward

Run K particles of the plain product-of-experts sampler, read the validated compose scorer on each
particle's current best guess of the finished image at five points in the run, and resample toward
the particles the scorer already likes. No correction is ever added to the score, and nothing is
learned. If choosing among what the product proposes, with the final judge itself as the guide,
is enough to compose, the gap is the sampler's to close.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md — <support / null / inconclusive, the compose rates>
```

## Recommended skill

▶ `/run-experiment` ✅ for the launch; `/analyze-run` ✅ for the W&B sheets and the sidecar.

## Position in the plan tree

**Step 55 of 65.** Waits on nothing: the pinned initial noise it reads exists in the training cache,
and the detector is validated. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 30 | [idea-01: feynman-kac-correctors-gated](../ideas/07-feynman-kac-correctors.md) ⚠️ | the other Feynman-Kac paper, Skreta et al. arXiv 2503.02819, which changes the score; this plan does not wait on it and must not be confused with it |
| 51 | [baseline-05: twisted-smc-on-a-learned-joint-vs-poe-twist](09-twisted-smc-on-a-learned-joint-vs-poe-twist.md) 〰️ | the same particle sampler with a learned weight; inconclusive because the weight memorised its training set |
| **55 (current)** | **baseline-06: feynman-kac-steering-on-a-detector-reward** ⚪ | **the same particle sampler with the compose scorer as the weight, so nothing can memorise** |

Design only. Verdicts and run state live in
[the paired review file](../../review/10-feynman-kac-steering-on-a-detector-reward.md).

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

Does resampling K plain product-of-experts particles toward the ones the compose scorer already
counts two animals in, at five points in the run, produce a finished image with two animals more
often than the same particles left alone?

## Words this plan uses

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **x0-hat**: the model's current best guess of the finished image, computed from the noisy latent
  and the noise prediction at that step by the Tweedie formula, then decoded through the VAE so
  the detector can look at it. Blurry early, sharp late.
- **The reward**: the validated instance-count scorer's count of distinct animals in x0-hat,
  clipped at 2. So each particle scores 0, 1 or 2 at each read.
- **The max potential**: a particle's weight is `exp(λ · m)` where `m` is the highest reward that
  particle's lineage has ever scored. The paper's recommended potential; λ is 10, the paper's
  value for text-to-image.
- **Resampling**: at five evenly spaced steps (indices 0, 10, 20, 30 and 40 of 50), the K
  particles are redrawn in proportion to their weights, so high-weight particles are copied and
  low-weight ones dropped. Between those steps every particle just runs the plain sampler.
- **The unweighted control**: the same K particles, same initial noise and same fresh noise
  draws, never weighted and never resampled. Particle 0 of the control starts from the seed's
  cached initial noise, so it is the plain product-of-experts render at `eta` 1.0.
- **The picked particle**: the one particle per seed the FK column shows and scores: the highest
  final reward, ties broken by the highest lineage max, then the lowest index.
- **Best of K**: the paper's own baseline, the unweighted particle with the highest final reward.
  Separates what resampling adds from what merely picking the best of K draws adds.

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** `poe_repair/experiments/fk_steering/`: the sampler with its constants
(`steer.py`), the runner that renders the references, the controls and the steered runs, scores
everything, draws the sheets and writes the sidecar (`run.py`). Launched by
`scripts/fk_steering/run_fk_steering.sh`.

**What it does.** Singhal et al., "A General Framework for Inference-time Scaling and Steering of
Diffusion Models" ([arXiv 2501.06848](https://arxiv.org/abs/2501.06848)), run a system of K
particles through a diffusion sampler and resample them on potentials built from a reward read on
x0-hat. With the diffusion model itself as the proposal, nothing about the score changes; only
which particles survive does. Here the reward is the project's own compose scorer, so the guide and
the judge are one instrument. The plan reads whether that selection reaches two-animal images
the plain sampler does not.

**Key components.** The proposal (plain PoE at `eta` 1.0), the reward on x0-hat, the max
potential, systematic resampling at five steps, the sheets.

**Testing approach.** One sheet per pair, rows the held-out seeds 9 to 16, columns Mono (the joint
prompt, `eta` 0), plain PoE (`eta` 0), the unweighted control's particle 0 at K 16 (`eta` 1.0),
best of 16 unweighted, FK at K 4, FK at K 16. Only the weighting differs between a control column
and an FK column at the same K.

**Associated materials.** [The review questions](../../review/10-feynman-kac-steering-on-a-detector-reward.md),
[plan 09](09-twisted-smc-on-a-learned-joint-vs-poe-twist.md) for the shared particle sampler, and
[the register row](../../../standing/literature/reading-register.md) for the paper.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**The reward may be blind where the decision is made.** The compose rate is decided in steps 0 to
10 (the timing result in scope 03), and x0-hat is a blur there. A detector count on a blur can
disagree with the count on the finished image. That is why the third bar exists: if the step-10
reward disagrees with the final verdict on more than half the particles, the method was never
tested and the answer is inconclusive.

**The guide is the judge.** The picked particle is chosen by the same scorer that scores it, so
the FK column can only win ties in its own favour. The best-of-K control has the same advantage
without any resampling, which is why it is on the sheet: the resampling's own contribution is the
FK column against best of K, and the method's contribution as a user sees it is the FK column
against the control's particle 0.

**Every render shares the seed's initial noise.** Particle 0 of every K-particle run starts from
the training cache's pinned noise for that seed (`embeddings.pt`, key `init_latents`), which is
`torch.randn` from a CPU generator seeded with the seed number, in float16; particles 1 to K−1
are fresh draws from a generator seeded from the seed. Butterfly × meadow's cache stops at seed 12,
so seeds 13 to 16 use the same formula, checked byte-identical against cat × dog's cached seeds
9 and 13 on 2026-09-05.

**`eta` 1.0 for the particle runs, `eta` 0 for the two reference columns.** Resampling needs a
stochastic proposal, or duplicated particles never separate. Mono and plain PoE are the repo's
deterministic renders so the sheet's left two columns match every other sheet in the paper; the
control column at `eta` 1.0 is the fair comparison for the FK columns, and the sheet names both.

**Systematic, not multinomial, resampling.** The paper resamples multinomially. With integer
rewards most weight vectors tie, and multinomial resampling of equal weights duplicates and drops
particles for no reason. Systematic resampling of equal weights is the identity. Stated in the
sampler's docstring.

**No aesthetic term.** The paper's SDXL runs steer on ImageReward alone and report aesthetic and
HPS scores only as metrics, so no second reward is added here.

**Cost.** Per cell on the session node's RTX 3090: two `eta` 0 references (about 20 s each), a
K 16 control and a K 16 steered run (about 9 min each at 50 steps, 1024², 3 UNet branches per
particle), a K 4 control and a K 4 steered run (about 2 min each), and 100 x0-hat decodes with
detector reads (about 1 min). About 25 min per cell, 16 cells, about 7 hours. A Blackwell node
would halve it; on 2026-09-05 both healthy Blackwell cards were fully loaded by other users and
the third was in the fault state of `poe-launch-002`.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Resampling plain product-of-experts particles on the compose scorer's read of x0-hat raises the
compose rate over the same particles left unweighted.**

**Independent variables.** Steering on or off; K in {4, 16}. Fixed: `eta` 1.0, 50 DDIM steps,
guidance 7.5, 1024², λ 10, max potential, resampling at step indices 0, 10, 20, 30, 40; two pairs
(cat × dog, the failing pair; butterfly × meadow, the composing control pair); seeds 9 to 16.

**Dependent variable.** Compose rate over the 8 seeds by the validated detector (two or more
animal instances in the finished image). For an FK column, the picked particle per seed; for the
control, particle 0 per seed. Secondary reads, recorded but not judged: the fraction over all K
particles of each run, best of K unweighted, and the reward-versus-final agreement per read step.

**Falsify condition.** The bars sit in `poe_repair/experiments/fk_steering/steer.py` as
`PASS_MARGIN`, `NULL_MARGIN`, `MAX_REWARD_DISAGREEMENT` and `REWARD_BLIND_STEP`, and the runner
writes `verdict.json` from them. Judged on cat × dog.

- **Support.** FK at K 16 composes at least 0.25 more often than the unweighted control at K 16.
  Choosing among the product's proposals, guided by the judge, is enough on these seeds.
- **Null.** FK is within 0.10 of the control at both K 4 and K 16. The composing states are not
  among the product's proposals at these K, or the reward cannot find them.
- **Inconclusive.** The reward read on x0-hat at step 10 disagrees with the final detector verdict
  on more than half of the final particles (each traced back through the resampling ancestry to
  its step-10 state). The reward is blind where the decision is made and the method was never
  tested.
- **Butterfly × meadow** is a control: FK must not lower its compose rate below plain PoE's. If it
  does, the method breaks what works and that sentence goes in the caption.

**Why this matters right now.** Plan 09's twist memorised its training set, so the sampler-side
question stayed open from that side. A reward that cannot memorise closes that gap in the
argument, whichever way it lands.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The scope needs a selection-only sampler whose weight is trusted. Plan 09's was
learned and did not generalise, so its null-leaning result cannot be read as a null.

**The approach.** Use the validated scorer itself as the weight, following a published
inference-time steering method at its published settings.

**Key insights.**

1. Feynman-Kac steering with the base model as proposal changes nothing about the score, so
   whatever it recovers is the sampler's share by construction, as in plan 09.
2. The reward needs no training, so the memorisation failure of plan 09 cannot recur; what can
   fail instead is the reward's blindness on early, blurry x0-hat, and the third bar catches that.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  K particles, plain PoE score, DDIM eta 1.0, 50 steps

  step 0   ──► x0hat_k ──VAE──► png ──detector──► count_k ──clip 2──► r_k ──► m_k = max(m_k, r_k)
           ──► weights exp(10 · m_k) ──► systematic resample ──► continue
  step 10  ──► same read and resample
  step 20, 30, 40 ──► same
  step 49  ──► finished images, all K scored; picked = highest final reward

  sheet per pair, rows seeds 9..16:
  [ Mono eta0 | PoE eta0 | control p0 (K16, eta1) | best of 16 unweighted | FK K4 | FK K16 ]
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The sampler.** `steer.py`: `run_fk_steering(...)` over the plain PoE score with `eta`, K,
   the reward callback, the resample step set and λ; the constants above; `trace_ancestor` for
   the reward-versus-final check; `verdict` from the four compose rates and the disagreement.
2. **The runner.** `run.py`: per pair and seed, the two `eta` 0 references, the K 16 and K 4
   controls, the K 16 and K 4 steered runs; every final particle and every x0-hat read scored by
   the detector; per-cell JSON; the sheet per pair with labels drawn on it; `summary.json` with
   the compose rates and the agreement per read step; `verdict.json`; W&B run in project
   `prime_lab/poe-repair-animals-compose`, group `fk-steering`, with the sheets as images and
   the sidecar as an artifact.
3. **The launcher.** `scripts/fk_steering/run_fk_steering.sh {smoke|full}` with the disk guard on
   the output root, the python-by-node switch, the GPU fault guard (`poe-launch-002`) and a
   `torch.cuda.is_available()` check on the pinned device before any model loads.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves the scope's mission from the selection side with a weight that cannot memorise, so the
sampler-side answer plan 09 could not give is given.

**Goals**

1. A smoke run proves the whole path: references, control, steered run, x0-hat reads, sheet,
   sidecar, verdict.
2. Two sheets (cat × dog, butterfly × meadow), seeds 9 to 16, six columns each, with `.json`
   sidecars, logged to W&B.
3. `verdict.json` written by the code's own bars, quoted in the review file with the run id, node,
   device and PID.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- Python by node: `/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python` on mscluster110 to
  112 (Blackwell), `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python` elsewhere, per
  [environment/hpc/nodes.md](../../../../environment/hpc/nodes.md). Never a bare `python`.
- `torch.cuda.is_available()` checked on the pinned device before real work; mscluster111's card
  lists in `nvidia-smi` and runs on the CPU, per
  [poe-launch-002](../../../../environment/known-failures.md).
- Outputs under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering/`, disk
  guard on that path, per [environment/storage.md](../../../../environment/storage.md). Nothing
  under `/home-mscluster`.
- The training cache at `/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/`
  with `embeddings.pt` per cell: cat × dog seeds 9 to 16, butterfly × meadow seeds 9 to 12
  (13 to 16 drawn by the same formula, see Considerations).
- biggpu allows one Slurm job per user; long runs start with `nohup` on a free device and
  `squeue` is blind to them; check `pgrep -af 'sweep|train|corrector|fk_steering'` before
  claiming a device, per
  [environment/hpc/execution-protocol.md](../../../../environment/hpc/execution-protocol.md).
- The tracker is W&B, project `prime_lab/poe-repair-animals-compose`; the review file carries the
  run id and verdict, never curves.
- fp16 SDXL; the VAE decodes in fp32 (diffusers' `force_upcast`).

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against context/, environment/, runbook/, report/,
      and any learning journey that names this project.
  - Paste: `/xref-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--read-the-paper-and-pin-the-bars)**.

### 1. 📄 Read the paper and pin the bars

- [x] **1.1** Read arXiv 2501.06848 in full and record the settings this plan copies: the max
      potential, λ 10, the five-step resampling schedule, DDIM `eta` 1.0, the base model as
      proposal, K 4 in the paper, ImageReward alone as the SDXL reward.
  - Paste: `/unpack-paper https://arxiv.org/abs/2501.06848`
  - Done when: the register row for 2501.06848 exists in
    [the reading register](../../../standing/literature/reading-register.md) and names it apart
    from 2503.02819.
- [x] **1.2** Write the falsification criterion with its numbers into the review file and the
      constants into `steer.py` before anything runs.
  - Done when: `PASS_MARGIN`, `NULL_MARGIN`, `MAX_REWARD_DISAGREEMENT`, `REWARD_BLIND_STEP`,
    `LAMBDA`, `REWARD_CLIP`, `RESAMPLE_STEP_INDICES` and `K_VALUES` are in source and the review
    file's pre-registered question quotes them.

▶ **Next: [task 2.1](#2--build-and-smoke-the-path)**.

### 2. 🔧 Build and smoke the path

◀ **Needs: [task 1.2](#1--read-the-paper-and-pin-the-bars)**, the bars in source.

- [x] **2.1** Write `poe_repair/experiments/fk_steering/` (sampler, runner) and the launcher
      `scripts/fk_steering/run_fk_steering.sh`.
  - **Done when:** the package imports, `trace_ancestor` returns the right index on a hand-made
    ancestry, and the runner's `--help` lists every setting above.
- [x] **2.2** Run the smoke mode on a free device: cat × dog seed 9 only, K 2, 10 steps, 512²,
      resampling at step indices 0, 2, 4, 6, 8, detector on.
  - Command: `bash scripts/fk_steering/run_fk_steering.sh smoke`
  - **Done when:** the run dir holds the two references, a control and a steered run with their
    x0-hat reads, a one-row sheet, `summary.json` and `verdict.json`, and the log shows the
    device name and `cuda` from torch. The images are noise at 10 steps and 512²; this proves the
    plumbing and reads nothing.
  - Ran 2026-09-05 16:57 on mscluster109 device 1, PID 1856060, 29 s; every listed output present
    (the review file's Runs table).

▶ **Next: [task 3.1](#3--the-full-run)**.

### 3. 🏋️ The full run

◀ **Needs: [task 2.2](#2--build-and-smoke-the-path)**, the smoke clean.

- [x] **3.1** Launch the full mode with `nohup` on a free device: both pairs, seeds 9 to 16, K 4
      and 16, 50 steps, 1024², W&B online.
  - Command: `nohup bash scripts/fk_steering/run_fk_steering.sh full > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering/logs/full.log 2>&1 &`
  - 💡 `/run-experiment` ✅ for the launch and the harvest.
  - **Done when:** the W&B run id, the run dir, the node, the device and the PID are in the review
    file's Runs table.
  - Launched 2026-09-05 16:59 on mscluster109 device 1, PID 1856701, W&B `czim1n0w`, run dir
    `fk_K4-16_s50_20260905-165958`; finished 20:04, null.
- [x] **3.2** Harvest: copy the two sheets and the sidecar into
      `artifacts/results/is-the-gap-the-samplers-or-the-models/` with README entries, quote
      `verdict.json` in the review file, and answer every pre-registered question there.
  - **Done when:** the review file's verdict is written and the finding in `report/` is updated.
  - Filed 2026-09-06 as `fk-steering-cat-dog-sheet.png`, `fk-steering-butterfly-meadow-sheet.png`,
    their `.json` sidecars and `fk-steering-summary.json`; the finding is
    [does steering on the scorer find a composing proposal](../../../../report/is-the-gap-the-samplers-or-the-models/does-steering-on-the-scorer-find-a-composing-proposal.md).

▶ **Next: [instruction 4.1](#4--read-the-sheets)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [the check before moving on](#the-check-before-moving-on).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 4. 👁️ Read the sheets

◀ **Needs: [task 3.2](#3--the-full-run)**, the sheets filed.

- [ ] **4.1** Open `artifacts/results/is-the-gap-the-samplers-or-the-models/fk-steering-cat-dog-sheet.png`.
      Row by row, write down whether the FK K 16 tile shows two bodies where the control tile
      shows one, and whether the best-of-16 tile already does. A composing FK tile that the
      best-of-16 tile matches means picking did the work, not resampling.
- [ ] **4.2** Open the butterfly × meadow sheet the same way. Expected result: every column shows
      a clear butterfly over a meadow. ❌ An FK column that lost the butterfly means the method
      breaks what works; record it in the review file's caveats.
- [ ] **4.3** In W&B, project `prime_lab/poe-repair-animals-compose`, group `fk-steering`, open the
      run, Charts tab, and read `agreement/step_10` (the share of final particles whose step-10
      reward agrees with their final verdict). Under 0.5 is the inconclusive branch, whatever the
      compose rates say.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** this is the scope's cleanest selection-only baseline, because
> its weight is the judge itself and cannot have memorised anything.

```bash
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering
ls "$OUT"                                          # run dirs and logs/
ls "$OUT"/<run>/a_cat__x__a_dog | wc -l            # expect 8 seed folders
ls "$OUT"/<run>/*.png                              # two sheets
cat "$OUT"/<run>/verdict.json
```

**Pass criteria**

- The smoke run's outputs listed in task 2.2, then the full run's id, node, device and PID in the
  review file.
- Two sheets with no red "missing" boxes, two sidecars.
- `verdict.json` quoted in the review file.

**Fail criteria (STOP)**

- The control's particle 0 and the FK run's particle 0 differ at step 0 before any resample (they
  share the initial noise; a difference there is a bug).
- `torch.cuda.is_available()` printed `False` in the log, or a run took minutes per step.

**Partial pass guidance**

- A run stopped early still has its per-cell JSON; draw the sheets with red boxes for the missing
  tiles and read what exists, marking the verdict as read at that seed count.

**When you get results, answer**
[the review file](../../review/10-feynman-kac-steering-on-a-detector-reward.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `<run>/fk_steering_<pair>_sheet.png` | — | rows seeds 9 to 16; columns Mono (`eta` 0), plain PoE (`eta` 0), control particle 0 at K 16 (`eta` 1), best of 16 unweighted, FK K 4, FK K 16; the detector's count drawn on every tile; the varied setting (`eta`) named in the column label | task 3.1 | ✅ filed under `artifacts/results/is-the-gap-the-samplers-or-the-models/` | one per pair, logged to W&B as `sheets/<pair>` |
| `<run>/fk_steering_<pair>_sheet.json` | — | per tile: source path, detector count, compose verdict; per column: compose rate over 8 seeds | task 3.1 | ⏳ | sidecar, bundled in the W&B artifact |
| `<run>/summary.json`, `<run>/verdict.json` | — | the four compose rates, the all-particle fractions, best of K, agreement per read step, the verdict string with its bars | task 3.1 | ⏳ | quoted in the review file |
| `<run>/<pair>/seed_N/fk_K16/xhat/step_SS/p{k}.png` | — | every x0-hat the reward read, for the eye to check what the detector saw at each resample | task 3.1 | ⏳ | **Supplementary** |

### Organization workflow

1. Everything under the run dir on `/datasets`.
2. The two sheets and `summary.json` promoted to
   `artifacts/results/is-the-gap-the-samplers-or-the-models/` as `fk-steering-cat-dog-sheet.png`,
   `fk-steering-butterfly-meadow-sheet.png` and `fk-steering-summary.json`, with README entries.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| a run launches or finishes | the review file's Runs table with its W&B id, node, device and PID |
| `verdict.json` is written | the review file's pre-registered question, and the finding in `report/` |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, by `sync-plan-tree` |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md` | **task 0.1** | Conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md` | **the close out** | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto | Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md` | **the close out** | Statuses and running order match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [poe_repair/experiments/fk_steering/steer.py](../../../../poe_repair/experiments/fk_steering/steer.py) | the sampler, the potential, the constants, the verdict |
| [poe_repair/experiments/fk_steering/run.py](../../../../poe_repair/experiments/fk_steering/run.py) | the runner, the sheets, the sidecar, W&B |
| [poe_repair/experiments/twisted_smc/sampler.py](../../../../poe_repair/experiments/twisted_smc/sampler.py) | the DDIM step with `eta` and the systematic resampler this plan reuses |
| [poe_repair/experiments/compose_scorer_validation/detection_scorer.py](../../../../poe_repair/experiments/compose_scorer_validation/detection_scorer.py) | the validated instance count, used as the reward and the judge |
| [poe_repair/_sdxl/metrics.py](../../../../poe_repair/_sdxl/metrics.py) | guided and PoE eps, the Tweedie estimate |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

A support makes FK steering a row in [step 29's](06-three-rules-on-one-amount-axis.md) comparison
and the sampler-side sentence in the paper's corrector section. A null is the model-side reading
and goes to the same section as the reason the adapter adds a direction. Either is then read
against the in-span share once rung 2 of scope 05's [what the correction is made of](../../../05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md) lands: selection can only reach what the
proposal spans.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>3 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🟡 a listed card that runs on the CPU

**When it happens:** the pinned device is in the fault state of `poe-launch-002`.
**What you see:** `torch.cuda.is_available()` is `False` while `nvidia-smi` lists the card; a
50-step run takes many minutes.
**Why:** the driver reports the card but torch cannot open it.
**How to fix:** the launcher checks torch before loading models and aborts; pick another device.

#### 🟡 resampling collapses the population

**When it happens:** `eta` 0, or a reward that gives one particle 2 and every other 0 at step 0.
**What you see:** every particle identical after the first resample.
**Why:** duplicated particles under a deterministic step never separate again.
**How to fix:** `eta` 1.0 (the default); the fresh noise separates duplicates by the next read.

#### 🟡 the co3 environment on a Blackwell node

**When it happens:** the launcher run on mscluster110 to 112 with `co3`.
**What you see:** no CUDA output at all rather than an error.
**Why:** `co3`'s torch has no `sm_120` kernels.
**How to fix:** the launcher picks `co3_bw` by hostname; keep that switch.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
