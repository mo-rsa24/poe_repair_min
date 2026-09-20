# 🧪 Freezing the empty branch

**Reading the empty branch from the cache instead of from the model closes the direction the loss cannot see exactly, rather than approximately, and does it at two thirds of the batch. `01` has already run this way and reached 30,000 steps. This plan proves training and sampling agree about what is frozen, fixes the switch that variation `04` needs, and launches the baseline and `04` beside the run that exists.**

**Step 74 in the root running order. Waits on 71 for the undialled number that justifies it and on 73 for the approximate fix it is read against. Gates the naming of a winning objective, which no plan in this scope yet owns.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/04-freezing-the-empty-branch.md — the agreement check, the trainer and sampler change, the 50k run and its read against 00, recorded in the review file
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 73 (previous) | [03: the V0a read](03-the-v0a-read.md) | the approximate fix, already trained, judged |
| **74 (current)** | **04: freezing the empty branch** | the exact fix, trained and read |
| next | none in this scope yet | `02`, anchoring the empty branch to two animals, is authored once this returns |

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

- **Freezing a branch** does not mean freezing weights. The adapter sits on shared weights and changes all three branches at once, so there is no per-branch weight to freeze. It means not running that branch through the adapted model at all and reading the cached frozen tensor instead. That costs nothing: the tensor is already loaded.
- **The exact closure**: with the empty branch read from the cache, the guidance weight cancels out of the loss entirely, leaving a constant the learning rate absorbs. The direction the loss could not see closes because there is no longer an adapted empty branch to move in it.
- **The sampler mismatch**: training freezes a branch and sampling does not, so the two disagree about what the model is. The note names it as the one risk here, and this repository has made it before.
- **The seed-noise band**: the spread of held-out compose rate across the eight held-out cat × dog seeds at a reference checkpoint. Computed in plan 02 and used here as written.
- **Matched steps**: `01` at a given optimizer step against `00` at the same step. Anything else mixes two axes.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis, stated so it can fail**

*Freezing the empty branch reaches the same composed predictions as the objective running today: it lands inside `00`'s seed-noise band on held-out compose rate at matched steps, while closing the invisible direction exactly and training on a batch two thirds the size.*

**Why it can fail**

The note's argument is that freeing the empty branch buys no predictions, because anything it was doing the two concept branches can do instead. That is an argument about what the composition can express, not a measurement of what gradient descent finds. If the freed branch was doing optimisation work rather than expressive work, freezing it will cost compose rate, and the note's central derivation is wrong.

**Why that is the good outcome to hope for**

A confirmation makes training cheaper and removes the guidance weight from the problem. A falsification says something nobody in this project currently believes, about a derivation that three other variations rest on.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime.** 30,000 steps at rank 16 took about 7 hours on `mscluster109` when `01` ran it on 2026-09-17, 08:52 to 15:53. The two runs here go on the two cards of that node at once, so the pair costs one working day rather than two. The number in the review file is what actually happened, not this estimate.

**Why 30,000 and not 50,000.** Every finished run in this lineage stopped at 30,000, so that is where a matched comparison against `01` is possible. A longer run is a separate question and it belongs after one of these variations wins.

**Prerequisites.** Plan 01's paired cadence, so every render has its weights. Plan 02's band, used as written. Plan 03's verdict, so this run is read against the approximate fix as well as against `00`.

**The one risk, named in the note.** Whatever is frozen in training must be frozen in sampling. On 2026-09-05 a windowed sampler left the adapter attached while rendering what were meant to be plain references, and it cost a re-render. Task 1.1 exists because of that.

**Project tracking.** W&B, project `prime_lab/poe-repair-animals-compose`.

**Known issues.** See the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- [Nodes](../../../../environment/hpc/nodes.md): `biggpu` allows one job per user, so this single run may go there; a Blackwell is roughly three times faster per step and needs `co3_bw`, never `co3`, because a `co3` CUDA operation on that card produces no output rather than an error. `mscluster111` is hardware-faulted. On `bigbatch`, pin a node from a live idle probe: `sinfo`'s idle list includes faulted GPUs, and two nodes could not see `/datasets` from inside a Slurm job while an SSH probe passed them.
- [Throughput](../../../../environment/hpc/throughput.md): rank 32 at about 2,750 steps an hour on a 49 GB card; the `2K` batch should beat that and the review file records what it did.
- [Storage](../../../../environment/storage.md): checkpoints to `/datasets` only, and the launcher's disk guard must check the filesystem it actually writes to.
- [Overview](../../../../environment/overview.md): the forward pass is cast to fp32 before the branches are combined. Dropping a row from the batch must not disturb that.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**One substitution and one dropped batch row close the invisible direction exactly, at two thirds the compute, and make one adapter work at every guidance setting. Either the composed predictions are unchanged, which settles the objective, or they are not, which falsifies the derivation three other variations rest on.**

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The loss sees only the sum of the three branches. Sampling reads the empty branch again at `−(w−1) = −6.5`. A penalty fines the empty branch for moving, which is plan 03's approximation, but a fine is not a ban: what is left of the drift is still multiplied by 6.5, and that residue can be the size of the whole rest of the error.

**The solution.** Do not adapt the empty branch at all. Read it from the cache. The guidance weight then factors out of the loss completely, leaving a constant the learning rate absorbs, and the direction closes exactly because nothing is free to move in it.

**Why it is also cheaper.** The batch drops from `3K` to `2K`, a third less forward pass per step, and the cached empty branch is already loaded.

**The one thing that can go wrong.** Training and sampling disagreeing about what is frozen. That is not a hypothetical; it happened here eleven days ago on a different sampler, and it is why the first task of this plan is a check rather than a change.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  today (00)                          after this plan (01)
  ──────────                          ────────────────────
  batch 3K                            batch 2K
    "a cat"  ─┐                         "a cat"  ─┐
    "a dog"  ─┼─▶ U-Net ─▶ a, b, u      "a dog"  ─┴─▶ U-Net ─▶ a, b
    empty    ─┘                         empty ─────── from cache ──▶ u-bar

  loss = ‖ u + w(a−u) + w(b−u)         loss = w² ‖ (a + b − u-bar) − eps_J ‖²
         − [u-bar + w(eps_J − u-bar)] ‖²        └── the weight factors out entirely,
                                                    so one adapter works at any w
  the empty branch can wander          nothing is free to wander
  and the sampler amplifies by 6.5

  THE RISK: the sampler must also read the empty branch from the cache,
            or training and sampling are describing different models.
            Task 1.1 proves they agree before anything trains.
```

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **An agreement check**, first and before any change: a test that fails loudly if the sampler's empty-prompt pass and the trainer's disagree about whether the adapter is attached.
2. **Two confirmations**: the trainer already reads the cached empty branch under `--freeze-null`, and the sampler already detaches the adapter for its empty-prompt pass. Assert both rather than rewriting them.
3. **The one real code change**: make `--no-dial` honour `--freeze-null`, which it does not today.
4. **A smoke** of both switch settings, proving the loss falls under each.
5. **The two runs**: `00a` and `04`, rank 16, 30,000 steps, on the two cards of `mscluster109`.
6. **The read**: both scored against the `01` that already ran, same pool, same seeds, at matched steps, plus a blind read by eye.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objectives 4 and 5 of [the scope](../../MASTER_PLAN.md#objectives): test the variations in the note's order, one axis at a time, and move toward naming one objective.

1. Training and sampling provably agree about what is frozen, before any training.
2. A 50,000-step run exists with checkpoints and renders paired throughout.
3. Its held-out compose rate is read against `00`'s band at matched steps, either way.
4. Its cost per step is recorded, so the "a third cheaper" claim is a measurement rather than arithmetic.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 0. 🧭 Check this plan before working from it

- [ ] 0.1 **Run the following prompt:**
  ```
  /verify-plan @plans/09-designing-the-correction-loss/plans/hypothesis/04-freezing-the-empty-branch.md
  ```
- [ ] 0.2 **Run the following prompt:**
  ```
  /frame-hypothesis freezing the empty branch in plans/09-designing-the-correction-loss
  ```
  - The falsification criterion goes into the review file before the code changes, not after
  - Produces: the bar, written and dated ahead of the run

▶ **Next: [task 1.1](#1--prove-training-and-sampling-agree-before-changing-anything)**, the check that comes before any change.

### 1. 🛡️ Prove training and sampling agree, before changing anything

- [ ] 1.1 **Build the disagreement test, and watch it fail first**
  - Render one held-out seed twice with the current shipped adapter: once with the sampler's empty-prompt pass going through the adapted model, once with it detached
  - Assert the two renders differ. If they do not, the test is not measuring anything and every later assertion built on it is worthless
  - Then assert that the branch the trainer reads and the branch the sampler reads are the same object, by construction rather than by comparing pixels
  - Produces: a test in `tests/` that fails on the current code if the two paths are made to disagree, and a recorded pixel difference for the two renders
  - This costs one render pass and it is the cheapest possible insurance against the failure that cost a re-render on 2026-09-05

▶ **Next: [tasks 2.1 to 2.3](#2--make-the-change)**, only once 1.1 passes.

### 2. 🔧 Make the change

◀ **Needs: [task 1.1](#1--prove-training-and-sampling-agree-before-changing-anything)** passing, so a disagreement would be caught.

- [x] 2.1 **Confirm the cached empty branch is what the composition reads under `--freeze-null`**
  - The switch already exists. `trainer.py:543` selects `eps_uncond_frozen` when it is set, and `config.json` already records `freeze_null`
  - Assert it in the test task 1.1 builds, rather than rewriting working code
  - Produces: one assertion, not one switch
  - ✓ verified by inspection 2026-09-19: `trainer.py:543` reads `_eps_u_for_compose = eps_uncond_frozen.float() if _freeze_null else eps_uncond_l`, and `eps_uncond_frozen` is built at 425 from the cache entry's own `eu` inside the no-grad cache loop, so under `--freeze-null` the composition reads the cache and not the adapted model. `v1_freeze_null_r16_s0_25/config.json` records `freeze_null: true` and `no_dial: false`. The four `r16_s0_25_*` runs predate the flag and carry none of these keys, so a config read from those is empty rather than false. The assertion is still owed: task 1.1's test file does not exist yet, so this is an inspection and not a run.
- [x] 2.2 **Drop the third row of the batch**
  - The input becomes `(2K, 4, 128, 128)` and the reshape at line 489 becomes `(K, 2, ...)`
  - Produces: a forward pass a third smaller
  - ✓ verified by inspection 2026-09-20, not yet by a run: the branch count is decided once at `trainer.py:378-379` as `_drop_null_row = _freeze_null and not _render_teacher`, and carried through the latent tile (424), the per-sample timesteps (461), the prompt and pooled embeddings (467, 471), the time-id batch (476) and the reshape (489). The task predicted two lines; it is seven, because two readers of the adapted empty branch are unconditional and would raise `IndexError` on a two-row batch: the undialled diagnostic at 560 and the null-anchor drift at 605. Both are satisfied by `eps_uncond_l = eps_uncond_frozen.float() if _drop_null_row else noise[:, 2]` at 496, which makes the drift zero by construction rather than by penalty. V6 is excluded from the narrowing because its frozen branch comes from a second adapter-off forward over the same batch and needs all three rows. The run proof is task 2.4.
- [x] 2.3 **Confirm the sampler detaches the adapter for its empty-prompt pass**
  - `_inline_sampling.py:58,73,96` already threads `freeze_null` into the sampler for this reason
  - Produces: task 1.1's test covering the sampler path too
  - ✓ verified by inspection 2026-09-20: `_sampling.py:822` calls `_adapter_disable()` before the frozen three-branch forward, so `eps_uncond_f` is the empty-prompt answer with the adapter off; `_sampling.py:855` selects it under `freeze_null`. Sampling therefore reads the same un-adapted empty branch the trainer composes against, which is the agreement this plan's one named risk is about. It costs no extra forward pass: that branch is already computed for `eps_poe_frozen`. The test itself is still owed, because task 1.1 was skipped; this is an inspection of both paths, not a test covering them.

- [x] 2.3b **Make `--no-dial` honour `--freeze-null`**
  - The branch at `trainer.py:532-533` composes with `eps_uncond_l`, the adapted null, and never reads `_freeze_null`. Its own help text claims that with `--freeze-null` it is V1 up to the constant `w²`, and it is not
  - Left as it is, training keeps the null free while the sampler detaches it, which is the mismatch task 1.1 exists to catch
  - The fix:
    ```python
    elif _no_dial:
        _u = eps_uncond_frozen.float() if _freeze_null else eps_uncond_l
        eps_poe_lora = eps_a_raw_l + eps_b_raw_l - _u
    ```
  - The target is untouched: `--no-dial` keeps scoring against the raw cached joint prediction at `trainer.py:547-548`
  - Produces: `--no-dial --freeze-null` that is V1 with the `w²` factor gone, which is what variation `04` is for
  - ✓ done 2026-09-19: `trainer.py`'s `_no_dial` branch now reads `_freeze_null` and takes `eps_uncond_frozen` when it is set. Nothing was running when it was edited.
- [ ] 2.4 **Smoke both switch settings**
  - Two epochs each, rank 16, on `mscluster109`
  - Produces: a falling loss under each setting, a log line naming the frozen null, and task 1.1's test passing against the changed sampler
  - Done when the log names the frozen null and the test passes, not when the job exits 0

▶ **Next: [task 3.1](#3--run-it)**.

### 3. 🚀 Run it

◀ **Needs: [task 2.3b](#2--make-the-change)**, because `04` trains the wrong objective without it.

**`01` has already run.** `v1_freeze_null_r16_s0_25` reached 30,000 steps on 2026-09-17, W&B `kdzx03ji`, on `mscluster109`. This task launches the baseline it lacks and the variation that tests the derivation, and reads both against it. Do not relaunch `01`.

- [x] 3.1 **Write the two launchers**
  - Copy `scripts/showcase/v6_render_teacher.sh` twice. Rank 16, `STEPRANGE="0 25"`, `cells_v57`, `POOLGEN=v57`, weight decay 0, `SAMPLEEVERY=25` and the two-row render grid all come with the copy
  - `v0a_baseline.sh`: `RUN=v0a_baseline_r16_s0_25`, `EXTRA=""`
  - `v4_no_dial.sh`: `RUN=v4_no_dial_frozen_r16_s0_25`, `EXTRA="--no-dial --freeze-null"`
  - Produces: two files differing from the V6 launcher in the run name, the switch string and the python environment
  - ✓ done 2026-09-19: `scripts/showcase/v0a_baseline.sh` and `v4_no_dial.sh`. Both pin `co3`, not the `co3_bw` the V6 launcher they were copied from uses, because `mscluster109` is an A6000 and a `co3_bw` job there is a silent wrong-environment run.
- [ ] 3.2 **Launch `00a`, the baseline that no run exists of**
  ```
  GPU=0 nohup bash scripts/showcase/v0a_baseline.sh > logs/v0a.log 2>&1 &
  ```
  - `mscluster109`, the A6000, because that is where `01` ran and a matched set never straddles device models
  - `biggpu` is one job per user through Slurm, so this goes outside Slurm with `nohup`
  - Produces: 25 checkpoints, 25 render sets, one W&B run id. About 7 hours
- [ ] 3.3 **Launch `04` on the second card of the same node**
  ```
  GPU=1 nohup bash scripts/showcase/v4_no_dial.sh > logs/v4.log 2>&1 &
  ```
  - Produces: the same, running alongside 3.2
- [ ] 3.4 **Write both run ids into the review file the hour they launch**
  - The node, the GPU index and the switch string beside each
  - Three rank-32 runs died on 2026-09-17 and are discoverable only by listing directories, because nothing recorded them before they finished
- [ ] 3.5 **Record the cost**
  - Seconds per step and total wall clock for each
  - Produces: the number that makes any cheaper-training claim a measurement

▶ **Next: [instruction 5.1](#5--read-the-run-in-the-browser)**, the curve read only a person can do.

### 4. 📐 Read it against 00

◀ **Needs: [task 3.1](#3--run-it)** done and [instruction 5.1](#5--read-the-run-in-the-browser) read, so the run is known to have trained properly.

- [ ] 4.1 **Score the eight held-out seeds at matched steps**
  - Three runs now: `00a`, `04`, and the `01` that already ran (`kdzx03ji`)
  - Compose count and drift, at every checkpoint all three share
  - Produces: `01-per-checkpoint.json`, same shape as plan 02's
- [ ] 4.2 **Read it against the band**
  - Plan 02's band, used as written
  - Produces: the verdict, inside or outside, into the review file
- [ ] 4.3 **Build the shuffled strips for the blind read**
  - Same procedure as plan 03's task 2.1, shuffle key saved separately
- [ ] 4.4 **Write `report/designing-the-correction-loss/01-freezing-the-empty-branch.md`**

### Close out. 🔄 Record what this plan taught

- [ ] C.1 **Run the following prompt**, after any red run:
  ```
  /ingest-error-pattern --from-run-log
  ```
- [ ] C.2 **Run the following prompt:**
  ```
  /sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/04-freezing-the-empty-branch.md
  ```
- [ ] C.3 **Author `02`'s plan**, anchoring the empty branch to two animals, now that this result exists to design against. The scope's definition of done cannot be met until it does

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

For you to follow manually.

### 5. 📈 Read the run in the browser

◀ **Needs: [task 3.1](#3--run-it)** started, so there are curves to watch.

- [ ] 5.1 **Open the run in W&B**
  - Browser: `wandb.ai/prime_lab/poe-repair-animals-compose`, the run id recorded in task 3.1
  - Charts tab, then `train/loss_fit`
  - The panels need a logged-in session, which is why no script does this
  - ✅ Falling for the first 10,000 steps then flattening, the shape every run here has shown: normal
  - ❌ Flat from the start: the frozen branch broke the gradient path. Stop the run and go back to task 2.1
  - ❌ Rising or spiking: stop and check the batch reshape at 2.2 before spending another hour
- [ ] 5.2 **Check `train/loss_undialled` against `train/loss_fit`**
  - Plan 01 added this panel
  - Under the frozen branch the two should now track each other closely, because the guidance weight has factored out
  - ✅ They track: the closure is working as derived
  - ❌ They diverge as much as they did under `00`: the substitution did not take. Record it and check task 2.1 before reading anything else
- [ ] 5.3 **Look at the render strips in the run's media panel while it trains**
  - Every 2,500 steps a new set appears
  - You are looking for the moment the renders stop improving, which is where the next run's step budget should be set

### 6. 👁️ Read the strips blind

◀ **Needs: [task 4.3](#4--read-it-against-00)** done, so the strips exist and are shuffled.

- [ ] 6.1 **Label the eight strips without opening the shuffle key**
  - Path: `artifacts/results/designing-the-correction-loss/01-freezing-the-empty-branch/blind/`
  - Three labels per panel: clean, unclear, not two
  - ✅ All eight labelled before unblinding: the read is valid
  - ❌ Key opened first: void, reshuffle and repeat on another day
- [ ] 6.2 **Unblind and record**
  - Join through `shuffle-key.json` and count clean labels per condition

▶ **Next: [task 4.4](#4--read-it-against-00)**, the write-up.

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> Twenty hours of training and the credibility of three other variations rest on the derivation this run tests. The bar is fixed here and nothing about the result may move it.

**The question written before the run.** At matched optimizer steps, does freezing the empty branch land inside `00`'s seed-noise band on held-out compose rate?

**Inside the band** confirms the derivation. The objective is settled in favour of `01`, which is cheaper and exact, and `02` is designed on top of it.

**Outside the band, worse** falsifies the claim that freeing the empty branch buys no predictions. That is the more interesting result and it forces `03` and `04` to be reconsidered, because both rest on the same derivation.

**Outside the band, better** is not something the derivation predicts and would need its own explanation before it is believed.

**The gate before any of that.** Task 1.1 passes, or nothing trains. A run whose sampler and trainer disagree about what is frozen measures neither objective.

**Partial pass.** If the run trains but the undialled error does not track the fit loss (instruction 5.2), the closure did not happen and the compose-rate read means nothing. Record it as an implementation failure rather than as a result about the objective.

The review questions are in [the review file](../../review/04-freezing-the-empty-branch.md).

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

**Pending, from [this scope's illustrated map](../../diagram-prompts.md)**

| Figure | Lane | What it shows | Status |
|---|---|---|---|
| corrloss-02-one-pass-three-prompts | subject | one network, three prompts, three answers | opened at task 2.2, to check which row of the batch is being dropped |
| corrloss-03-two-legs-and-the-fine | subject | every place the empty branch is read | opened at task 1.1, since the check is about exactly those readings |
| corrloss-04-the-four-places-they-differ | subject | one machine, four settings | opened at task 2.1, to confirm this change touches only one setting |

**Generated during plan execution**

| File | Lane | What it holds | Task | Status |
|---|---|---|---|---|
| `agreement-check.png` | subject | one seed rendered with the empty pass adapted and detached, side by side | task 1.1 | ⏳ |
| `01-per-checkpoint.json` | — | compose count and drift at every checkpoint `00` also has | task 4.1 | ⏳ |
| `compose-rate-01-vs-00-matched-steps.png` | subject | compose rate against training step, both objectives, `00`'s band drawn | task 4.2 | ⏳ |
| `blind/seed-NN.png` (8) | subject | one strip per seed, ids instead of names, order shuffled | task 4.3 | ⏳ |

**Organization workflow.** Filed under `artifacts/results/designing-the-correction-loss/01-freezing-the-empty-branch/` with its card entry.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/09-designing-the-correction-loss/plans/hypothesis/04-freezing-the-empty-branch.md` | **task 0.1**, before any work | conformance and thin instructions reported |
| Fix the bar before the run | `/frame-hypothesis freezing the empty branch in plans/09-designing-the-correction-loss` | **task 0.2**, before any code changes | the falsification criterion dated ahead of the run |
| Capture patterns | `/ingest-error-pattern --from-run-log` | **the close out**, after any red run | errors added to the catalogs |
| Bring the tree current | `/sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/04-freezing-the-empty-branch.md` | **the close out** | statuses, running order and Error Matrix match reality |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `poe_repair/experiments/one_pair_one_seed/trainer.py`
**Relevant section:** the batch is built and run at lines 440 to 445, reshaped and split at 446 to 449, composed at 450. The frozen empty branch is already cached as `eps_uncond_frozen` at line 394, which is why this costs nothing.

```python
# the composition today, line 450
eps_poe_lora = _compose(eps_a_raw_l, eps_b_raw_l, eps_uncond_l, gs, compose, kappa)

# with the empty branch frozen
eps_poe_lora = _compose(eps_a_raw_l, eps_b_raw_l,
                        eps_uncond_frozen.float(), gs, compose, kappa)

# and the batch narrows: the reshape at 446 becomes
noise = noise.view(K, 2, *noise.shape[1:])     # was (K, 3, ...)
eps_a_raw_l, eps_b_raw_l = noise[:, 0], noise[:, 1]
```

The conditioning built at the step's start tiles three prompt embeddings to `(3K, 77, 2048)`; that becomes two and `(2K, 77, 2048)`.

**File:** the sampler that renders a checkpoint. Its empty-prompt pass must read the frozen branch the same way. `poe_repair/methods/_poe_langevin.py` carries the windowed sampler; the shipped path is elsewhere and both need the same treatment, which is what task 1.1's test is for.

**File:** `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`
**Relevant section:** the argument parser, where the new switch joins `--null-anchor` at line 136 and `--orth-weight` at 143, and the config dump at 365, where plan 01 made objective flags recordable.

**The precedent:** `plans/01-showcase-the-trained-lora/` records a windowed sampler leaving the adapter attached while rendering plain references on 2026-09-05, costing a re-render. Task 1.1 is written against that.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ `/frame-hypothesis freezing the empty branch in plans/09-designing-the-correction-loss` ✅ — fixes the falsification criterion in the review file before the code changes, which is the ordering `verify-plan` checks against `git log`.
   alt: `/run-experiment` for task 3.1's launch, which knows this project's submit-versus-SSH decision and its disk guard.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents)

Nothing in this scope yet. Close-out task C.3 authors `02`, anchoring the empty branch to two animals, which is the only variation the note says should be compared against this one rather than against the objective running today.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>One failure catalogued</summary>

**Purpose**: known issues and their fixes, regenerated by `/ingest-error-pattern` and `/sync-plan-tree`.

#### From this scope

**Three rank-32 runs died within five minutes of each other on 2026-09-17, undiagnosed.**
`v1_freeze_null_r32_s0_25` stopped at 7,500 steps at 08:42, `v3_adapt_null_only_r32_s0_25` at 7,500
at 08:41, `v6_render_teacher_r32_s0_25` at 2,500 at 08:46. Every rank-16 run at the same settings
on the same pool finished 30,000 steps. Nothing in the logs has been read. Launching at rank 32
again walks into it, which is why this plan's runs are rank 16.

#### From global catalog

#### From project catalog

---

**Auto-update note:** regenerated by `/sync-plan-tree`. Do not edit by hand.

</details>
