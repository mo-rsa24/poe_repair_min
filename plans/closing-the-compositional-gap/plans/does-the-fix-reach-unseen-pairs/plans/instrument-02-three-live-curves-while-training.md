# 🔌 Three live curves while training

## Recommended prompt (after run completes)

After you finish running this plan and want to ingest error patterns into the catalogs, use this prompt:

```
/ingest-error-pattern --from-run-log
```

This extracts error patterns from the run transcript, deduplicates against global and project catalogs, and adds new entries. New errors are automatically propagated to all affected plan files.

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 8 (previous) | [hypothesis-01: what-the-fix-changes-inside-the-model](../../does-the-correction-cause-composition/plans/hypothesis-01-what-the-fix-changes-inside-the-model.md) ✅ | Measured what $r_t$ does inside training |
| **9 (current)** | **instrument-02: three-live-curves** ◑ run in flight | **Wire live logging for three diagnostic axes** |
| 10 (next) | [hypothesis-01: does-one-pooled-fix-transfer-at-all](hypothesis-01-does-one-pooled-fix-transfer-at-all.md) | Use those live curves to run the 15-run sweep |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What the fix actually does (visual)](#what-the-fix-actually-does-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment:**  
We train a LoRA correction (denoted $r_t$, the interaction term) that fixes a bug in image generation. The fix works on the training pairs, but we don't know if it transfers to unseen pairs.

**The hypothesis:**

*The corrective interaction term $r_t$ is small in magnitude, shared across pairs, and encodes a generic fix to the underlying bug. Therefore, it should transfer to unseen pairs and generalize across domains.*

**If true:** The fix is small and shared across pairs, so it should transfer to unseen evaluation pairs. We'd see high compose-rate and aligned direction-cosine across the sweep.

**If false:** The fix is pair-specific or encodes brittle workarounds, so it won't transfer or transfers poorly. We'd see compose-rate stuck at zero or direction-cosine diverged across many runs.

**Rationale for the hypothesis:**  
The bug appears consistently across all training pairs (cat×dog, eagle×hawk, etc.), suggesting it's a systematic issue in the base model. If $r_t$ corrects this systematic issue, it should be both small (a minor adjustment) and portable (since the issue is fundamental, not pair-specific). This is testable by checking whether the same $r_t$ improves unseen pairs.

**This scope's job:**  
Diagnose why a fix doesn't transfer (or confirm that it does). To do that, we run a 15-run sweep where we test the fix on new pairs in a controlled way.

**What this plan does:**  
Before running that 15-run sweep, we need to know *while it's training* whether the fix is even arriving at the eval set. Post-run analysis is too late; you'd waste GPU hours. So this plan wires three diagnostic metrics into the live training loop, so you can kill bad runs early.

**Dataset details:**  
- **Training pairs:** 11 blend-prone animal pairs (wolf×husky, lion×tiger, and so on; the pool is `artifacts/results/does-the-fix-reach-unseen-pairs/pair_pool.yaml`), each over 8 train seeds.
- **Held-out pairs:** 8, each over 8 held-out seeds: 4 unseen blend pairs (the transfer test), cat×dog (the known-failure reference), and 3 compose-by-default controls (the do-no-harm check).
- **Unseen pairs:** Tested later in step 10 (the 15-run sweep tests transfer)
- **Known phenomenon:** The $\sim 40\%$ plateau in correction magnitude is well-established; we measure it live in this plan.

**Associated materials:**
- **Review questions:** [../review/instrument-02-three-live-curves-while-training.md](../review/instrument-02-three-live-curves-while-training.md)
- **Procedures:** (if any; add link here)
- **Assets/outputs:** Saved to `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/live_curves_smoke_run/` (large artifacts live on `/datasets`, never under the repo)
  - **Figure organization:** Use [figure-coverage-prompt.md](diagrams/figure-coverage-prompt.md) to scan the repo, rename all related figures to the step-09 naming convention, and consolidate them into `outputs/interaction_term/live_curves_smoke_run/figures/`. This prompt will generate a FIGURE_CATALOG.md that maps each figure to axes, meaning, and original location.
  - **Locations scanned:** `artifacts/results/ (per-question) and report/paper-evidence-index.md`, `outputs/interaction_term/`, `paper/iclr/figures/`, ``artifacts/notes/` items (formerly `/show-me`), results/ folders

For the full picture, see the [repo MASTER_PLAN.md](../../../../../MASTER_PLAN.md) and [hypothesis-01](../../does-the-correction-cause-composition/plans/hypothesis-01-what-the-fix-changes-inside-the-model.md).

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:**  
On one biggpu device (49GB Quadro RTX 8000 on mscluster106, or the Blackwell-class GPUs on mscluster110; see [environment/hpc/nodes.md](../../../../../environment/hpc/nodes.md#partitions)), a 1-epoch smoke run is roughly **1 to 3 hours**: the training epoch itself is minutes, and nearly all the wall time is the inline eval pass (every train and held-out cell rendered at 25 DDIM steps). The measured wall time of each run is recorded in the [review file](../review/instrument-02-three-live-curves-while-training.md). The full 15-run sweep in step 10 takes approximately 6 hours per run.

**Prerequisites:**  
The launcher `scripts/animals_compose_transfer/smoke_live_curves.sh` carries the setup: the `co3_bw` python, `POE_REPAIR_TRAINING_CACHE`, the device and disk guards, and the W&B flags. Nothing needs exporting by hand beyond `GPU=<free device index>` at launch. Cluster facts and absolute python paths are in [environment/00-INDEX.md](../../../../../environment/00-INDEX.md).

**GPU allocation:**  
Two launch paths, decided by [environment/hpc/execution-protocol.md](../../../../../environment/hpc/execution-protocol.md): an idle biggpu node gets a Slurm submit; when none is idle but an allocated node has a free device (Slurm here cannot see GPUs, so this is common), the run goes over SSH onto that device with `CUDA_VISIBLE_DEVICES` pinned and `nohup`, and `squeue` is blind to it, so monitoring is the nohup log plus `pgrep` on the node. Output goes to `/datasets` only; the launcher's disk guard checks that filesystem.

**W&B project:**  
All runs log to `prime_lab/poe-repair-animals-compose`. This is where you'll inspect the three live curves after the run finishes. Credentials are loaded from environment; verify with `echo $WANDB_API_KEY`.

**Known issues:**  
This plan may encounter common errors when running. See the [Error Matrix](#error-matrix) section at the bottom of this file for a full catalog of known issues and solutions. The Error Matrix is automatically updated after each run; check it before running again.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- biggpu nodes carry two GPU devices each; Slurm registers no GPUs (no GRES), so a free device on an allocated node is found only by SSH + `nvidia-smi`, and the shared-device launch path in [environment/hpc/execution-protocol.md](../../../../../environment/hpc/execution-protocol.md) is how this plan's run launches when no node is idle.
- Python is the `co3_bw` env at `/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python` (the launcher hard-codes it); never a bare `python`.
- Large outputs go to `/datasets/mmolefe/poe_repair_min/` only; the launcher's disk guard checks `/datasets`, the filesystem it writes to.
- W&B credentials load from `~/.netrc` / `WANDB_API_KEY` with no human step; the project is `prime_lab/poe-repair-animals-compose`.
- The training cache the run reads is `/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/` (train and heldout splits, mono/poe references and residuals per cell).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Wire the composer-scorer and direction metrics into the eval hook during training, so every run reports three separate live W&B curves: compose-rate, direction-cosine, and fraction-of-distance-reached.**

**Why this matters right now:**  
[Step 10 (hypothesis-01)](hypothesis-01-does-one-pooled-fix-transfer-at-all.md) runs a 15-run sweep, testing the fix on unseen pairs with different hyperparameters. Those 15 runs take hours each. You need live metrics so you can kill a run that's not working halfway through, instead of waiting 6 hours for garbage numbers that look plausible but are wrong.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-the-fix-actually-does-visual) ➡️

**The dilemma:**  
A 15-run unattended sweep takes 90+ GPU hours. When a run finishes, you can compute whether the fix arrived and transferred. But by then it's too late. If 10 runs failed silently, you've already wasted 60 hours. You need to know *while it runs* whether to kill it or let it complete.

See [why-this-plan-exists.png](diagrams/figures/why-this-plan-exists.png) for a visual of the problem: 90 wasted hours without live logging versus 2 hours to early decision with it.

**The solution:**  
This plan puts two diagnostic measurements inside the training loop, so you can read and react to them in real time.

**1. Did the fix deliver?**

Track `compose-rate` live (the scorer's output: is the generated image PoE-blend or Mono?). If this stays at zero, the correction never reached the eval set during training.

**2. Did it transfer?**

Track `direction-cosine` live (how aligned is the current correction to the pool-mean direction?). If the direction diverges, the fix isn't moving in the ensemble direction.

Together, these two split a floor result into actionable answers:

- **Compose-rate stuck at zero:** the fix never arrived at eval time (debug the delivery mechanism)
- **Compose-rate climbs, direction-cosine high:** it transferred (good news, move on)
- **Compose-rate climbs, direction-cosine low:** fix arrived but direction is wrong (debug the fix itself, not the delivery)

Without live logging, you won't know which case you're in until the sweep finishes. With it, you know within an hour.

---

## What the fix actually does (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

This diagram shows the difference:

```
BEFORE: Post-run logging only
┌────────────────────────────────────────────────┐
│ Run 1 → training for 6+ hours → finished       │
│         (no metrics visible yet)               │
│ Run 2 → training for 6+ hours → finished       │
│ Run 3 → training for 6+ hours → finished       │
│                                                 │
│ Now you can inspect the curves.                │
│ But it's too late: you've wasted 18+ GPU hours │
│ on a sweep that isn't working.                 │
└────────────────────────────────────────────────┘

AFTER: Live logging during training
┌────────────────────────────────────────────────┐
│ Run 1: epoch 1 → compose_rate: 0.23 ✅ signal  │
│                  direction_cos: 0.87 ✅ aligned│
│ Run 2: epoch 1 → compose_rate: 0.0  ⚠️ dead   │
│                  Kill run 3, 4, 5. Debug now. │
│                                                 │
│ You know within 1 hour, not 18.                │
└────────────────────────────────────────────────┘
```

The three curves appear on W&B as each epoch finishes, not after the run is done. You stop bad runs early.

---

## Description: what to build

⬅️ [Previous](#what-the-fix-actually-does-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

Import the compose-scorer module and run it inside the training eval hook (the same place where you currently compute loss). Three separate metrics, logged independently:

**1. Compose-rate**

- **What it measures:** Fraction of images the scorer classifies as PoE-blend (vs Mono).
- **Where it's logged:** `eval/compose_rate/{quadrant}/{pair}/seed_{NN}` plus an `eval/compose_rate/mean` aggregate.
- **Code path:** Reuse the inline-sampling + eval-crossbar already in `cross_pair_lora_pooling`.
- **Imports:** `_Embedders`, `score_output` from `compose_scorer.scorer`.
- **Implementation:** [train_pooled.py::_run_inline_sample](#code-reference-compose-rate)

**2. Direction-cosine**

- **What it measures:** Cosine similarity between the current run's correction ($r_t$) and the pool-mean correction (the Task D metric).
- **Where it's logged:** `eval/direction_cosine/{quadrant}/{pair}/seed_{NN}` plus mean.
- **Code path:** `_inline_sampling.py::direction_metrics` + `build_pool_mean_cache`.
- **What it tells you:** Are we moving in the same direction as the ensemble, or diverging?

**3. Fraction-of-distance-reached**

- **What it measures:** How far the correction has moved toward the PoE→Mono target.
- **Where it's logged:** `eval/frac_distance_reached/{quadrant}/{pair}/seed_{NN}` plus mean.
- **Why it matters:** Captures the $\sim 40\%$ plateau we see in correction magnitude, visible live.
- **Code path:** Same `_run_inline_sample` hook.

All three as separate live curves on W&B, not merged or post-processed. They appear independently so you can read each one.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose:**  
Serves [Objective 4 (Diagnose) and Definition-of-Done item 2](../../../../../MASTER_PLAN.md). Unblocks the whole unattended sweep ([step 10: does-one-pooled-fix-transfer-at-all](hypothesis-01-does-one-pooled-fix-transfer-at-all.md) and [step 11: hypothesis-02-more-correction-more-composition](../../../does-the-correction-cause-composition/plans/hypothesis-02-more-correction-more-composition.md)).

**Goals:**

1. A 1-epoch smoke run that completes without error and logs all three metrics as separate, non-empty W&B curves.
2. Confirmed that live logging works correctly before the 15-run sweep begins (this is the gate).
3. Evidence that the three diagnostic axes are reliable and actionable (compose-rate, direction-cosine, fraction-of-distance-reached).

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 1. ✅ High-level code wiring

- [x] **Wire the compose-scorer module into the eval hook.**
  - Reuse the eval-crossbar / inline-sampling path in `cross_pair_lora_pooling`.
  - Compute a compose/blend label per held-out eval output.
  - Imports: `_Embedders`, `score_output` from `compose_scorer.scorer`.
  - Implementation: [train_pooled.py:_run_inline_sample](#code-reference-compose-rate).

- [x] **Add the direction-cosine (Task D) computation.**
  - Cosine of the current correction to the pool-mean correction.
  - Verified: `_inline_sampling.py:direction_metrics` and `build_pool_mean_cache` in place.
  - Logged per-cell as `eval/direction_cosine/{quadrant}/{pair}/seed_{NN}`.
  - Aggregate: `eval/direction_cosine/mean` from `train_pooled.py:_run_inline_sample`.

- [x] **Add the fraction-of-distance-reached metric.**
  - Toward the PoE→Mono target, logged per eval so the ~40% plateau is visible live.
  - Logged as `eval/frac_distance_reached/{quadrant}/{pair}/seed_{NN}` plus mean.

▶ **Next: task 2.1**, which launches the run this wiring makes readable.

◀ **Needs: tasks 1.1 to 1.3**, the wiring this run exercises.

### 2. ⚙️ Run the 1-epoch smoke test

- [x] **Launch the smoke run in the background.**
  - Launcher: `scripts/animals_compose_transfer/smoke_live_curves.sh` (1 epoch on the 11 training pairs, then one inline-sampling eval pass over every train and held-out cell), started over SSH on a biggpu device per the shared-device path in [environment/hpc/execution-protocol.md](../../../../../environment/hpc/execution-protocol.md).
  - Cost: roughly 1 to 3 hours wall time on one 49GB biggpu device.
  - W&B project: `prime_lab/poe-repair-animals-compose` (live logging).
  - Output root: `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/live_curves_smoke_run/`
  - The run's identity, launch history, and verdict live in the [review file](../review/instrument-02-three-live-curves-while-training.md); run state never sits in this plan.
  - **Ask Claude:** "Run the step 9 smoke test for three live curves while training. Start it in the background and give me the W&B link when it begins logging."

- [x] ~~**Teach the environment the shared-device launch path.**~~
  - Done while launching: [environment/hpc/execution-protocol.md](../../../../../environment/hpc/execution-protocol.md) gained the shared-device step (SSH into an allocated biggpu node, read per-device nvidia-smi, run on a free device with `CUDA_VISIBLE_DEVICES` pinned), the `/run-experiment` node picker gained `--probe-shared`, and the launch-failure pattern landed as `poe-launch-001` in [environment/known-failures.md](../../../../../environment/known-failures.md).

▶ **Next: instruction 3.1**, reading the curves in W&B while the run cooks.

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself while Claude runs the experiment.

◀ **Needs: task 2.1** launched and logging.

### 3. 📊 Monitor the live run in W&B

3.1 🌐 **While the smoke run is training (or after it completes), open W&B:**

- Open browser: [prime_lab/poe-repair-animals-compose](https://wandb.ai/prime_lab/poe-repair-animals-compose/overview)
- Find the most recent run named `smoke_YYYYMMDD_HHMMSS` (the current run's id and URL are in the [review file's Runs table](../review/instrument-02-three-live-curves-while-training.md))
- Click into the run

3.2 ✔️ **Verify compose-rate curve exists:**
- Click **Charts** tab
- Search for `eval/compose_rate/mean` in the chart list
- You should see a line graph (flat if fix didn't arrive, climbing if it did)
- ✅ If present and non-null, mark as verified
- ❌ If missing or all-null, note the failure

3.3 ✔️ **Verify direction-cosine curve exists:**
- Search for `eval/direction_cosine/mean`
- Should show a scalar line, ideally high (0.7–1.0 means aligned with pool)
- ✅ If present and non-null, mark as verified
- ❌ If missing, note the failure

3.4 ✔️ **Verify fraction-of-distance-reached curve exists:**
- Search for `eval/frac_distance_reached/mean`
- Should show a line capped around 0.4 (the known plateau)
- ✅ If present and non-null, mark as verified
- ❌ If missing, note the failure

3.5 🔍 **If any metric is missing or all-null:**
- Click **Logs** tab and search for the metric key (e.g., "compose_rate"). If it doesn't appear, the logging hook failed
- Go to **System** tab and check stderr for import errors (`ImportError: compose_scorer.scorer`) or scoring crashes
- Screenshot the error and save for debugging
- **Do not proceed to step 10 until all three curves exist and have data**

3.6 📝 **Record observations:**
- Copy the run ID from the URL (e.g., `abc123def456`)
- Screenshot the three curves together
- Note what each curve shows (e.g., "compose_rate climbed to 0.3 by end of epoch", "direction_cosine stayed at 0.85")
- Save notes and screenshots for the review file
- Coverage caveat when reading compose-rate: anchors exist for 2 of the 19 sampled pairs (`a_wolf__x__a_husky` in-train, `a_cat__x__a_dog` held-out), so `eval/compose_rate/mean` averages over those cells only; the direction and distance curves cover every cell

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **This is the single safety check for the entire sweep.** If live logging is wrong here, the 15-run fan-out finishes producing silent garbage. The metrics look plausible, but they are measuring the wrong thing or not appearing at all. Do not proceed past this gate without a green smoke run.

**Pass criteria:**
- Smoke run completes with exit code 0 (no errors).
- All three metric keys appear in W&B as separate logged series: `eval/compose_rate`, `eval/direction_cosine`, `eval/frac_distance_reached`.
- Each series contains at least one non-null value per eval step.

**Fail criteria (STOP before step 10):**
- Eval hook errors or stalls on the smoke (import failure, scorer crash, out-of-memory).
- Any of the three metrics don't log or log with all-null values.
- The three curves are merged or post-processed instead of separate.

**When you get results, answer the open questions in the [review file](../review/instrument-02-three-live-curves-while-training.md).**

**If one or two metrics log but not all three:**

You can still move forward to step 10 if compose-rate (the most critical signal) logs successfully with non-null values. This tells you the fix is arriving at eval time. Direction-cosine and fraction-of-distance-reached are diagnostic aids; missing them doesn't block the sweep, but they make it harder to understand why a run failed. Document which metrics failed in the review file and note that future runs should investigate the logging for the missing metric(s).

**If all three metrics are missing or all-null:**

Do not proceed to step 10. The logging hook failed entirely. Check stderr for import errors (e.g., `ImportError: compose_scorer.scorer`) or crashes in the eval hook. Fix the issue before running the sweep.

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

**Purpose:** Every figure related to this plan is tracked in a single catalog. Figures fall into two categories: (1) ones to be generated from diagram prompts (status: Pending), and (2) ones expected to be generated during plan execution (status: Generated during run).

### Pending: to be generated from diagram prompts

Run each `.prompt.md` file through Claude (or /prompt-storyboard) and save outputs to `diagrams/figures/` with the filenames below:

| Figure | Prompt file | What it shows | Save to |
|--------|-------------|---------------|---------|
| Why this plan exists | [why-this-plan-exists.prompt.md](diagrams/why-this-plan-exists.prompt.md) | Two bars drawn to scale: 90 GPU hours spent blind against ~2 hours to the same decision | `diagrams/figures/why-this-plan-exists.png` |
| Three metrics explained | [three-metrics-explained.prompt.md](diagrams/three-metrics-explained.prompt.md) | Three panels explaining compose-rate, direction-cosine, fraction-of-distance-reached | `diagrams/figures/three-metrics-explained.png` |
| Where this plan sits | [context-diagram.prompt.md](diagrams/context-diagram.prompt.md) | This instrument's place between the wiring it needs and the 15-run sweep it gates | `diagrams/figures/context-diagram.png` |

### Built

| Figure | Prompt file | What it shows | File |
|--------|-------------|---------------|------|
| Before/after logging | [before-after-logging.prompt.md](diagrams/before-after-logging.prompt.md) | Side-by-side: post-run analysis versus live logging during training | [diagrams/figures/before-after-logging.png](diagrams/figures/before-after-logging.png) |

### Generated during plan execution

These figures will be created when the smoke run completes. Use [figure-coverage-prompt.md](diagrams/figure-coverage-prompt.md) to scan the repo, rename, and consolidate them:

| Figure | Description | Generated by | Status | Axes |
|--------|-------------|--------------|--------|------|
| step-09_metric-compose-rate_climb.png | Compose-rate curve (smoke run output) | `/run-experiment` smoke run, W&B | ⏳ Generated during run | X: epoch step; Y: compose-rate (0.0 to 1.0) |
| step-09_metric-direction-cosine_alignment.png | Direction-cosine curve (smoke run output) | `/run-experiment` smoke run, W&B | ⏳ Generated during run | X: epoch step; Y: cosine similarity (-1.0 to +1.0) |
| step-09_metric-frac-distance_plateau.png | Fraction-of-distance-reached curve (smoke run output) | `/run-experiment` smoke run, W&B | ⏳ Generated during run | X: epoch step; Y: fraction (0.0 to 1.0, caps at 0.4) |

### Organization workflow

1. Generate pending figures: Run each prompt through Claude and save to `diagrams/figures/`.
2. Execute the plan: Run `/run-experiment` smoke. This produces W&B curves.
3. Organize all figures: Run [figure-coverage-prompt.md](diagrams/figure-coverage-prompt.md) to scan the repo, rename figures to step-09 naming convention, move to `outputs/interaction_term/live_curves_smoke_run/figures/`, and generate [FIGURE_CATALOG.md](outputs/interaction_term/live_curves_smoke_run/FIGURE_CATALOG.md).

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

When this plan runs and produces output, three things need to stay in sync: the error catalogs, the plan file's Error Matrix section, and the figure catalog.

**After the run completes:**

1. **Ingest errors (automatic propagation):**
   - Run `/ingest-error-pattern --from-run-log` to extract patterns from the run transcript.
   - The skill deduplicates against `~/.claude/GLOBAL_ERROR_CATALOG.md` and `environment/known-failures.md`.
   - New errors are appended to the appropriate catalog.
   - The skill automatically triggers `/sync-plan-tree --update-error-matrices` when done.

2. **Update the Error Matrix section (automatic):**
   - `/sync-plan-tree` reads both catalogs and regenerates the Error Matrix section of this plan file.
   - New errors from step 9's run are now visible in the "Considerations" section.
   - This happens without manual intervention.

3. **Organize figures (manual, but guided):**
   - After the smoke run finishes, W&B outputs three metric curves to your project.
   - Use the [figure-coverage-prompt.md](diagrams/figure-coverage-prompt.md) to scan the repo for all related figures (existing artifacts/results/ (per-question) and report/paper-evidence-index.md, outputs/, paper/ figures plus new W&B screenshots).
   - The prompt renames them to the step-09 naming convention and consolidates them into `outputs/interaction_term/live_curves_smoke_run/figures/`.
   - It generates a `FIGURE_CATALOG.md` that maps each figure to its axes, meaning, and original location.

**Why this matters:**

Without orchestration, the Error Matrix section becomes stale after a run completes, and figures scatter across the repo. With it, you run two commands post-run (`/ingest-error-pattern` and the figure-coverage prompt) and everything stays current. The next time you visit this plan file, you see what actually happened, not what was planned.

**Quick reference:**

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | Manual (after run) | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Organize figures | Run [figure-coverage-prompt.md](diagrams/figure-coverage-prompt.md) | Manual (after run) | Figures renamed, consolidated, cataloged |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

### Code reference: Compose-rate

**File:** `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`  
**Function:** `_run_inline_sample`  
**Relevant section:** The eval hook where `compose_rate` is computed and logged.

```python
# Pseudocode (actual implementation in train_pooled.py)
def _run_inline_sample(model, eval_outputs, compose_scorer):
    """
    Compute three diagnostic metrics live during training.
    """
    scores = compose_scorer(eval_outputs)  # PoE-blend vs Mono
    wandb.log({"eval/compose_rate/mean": scores.mean()})
    
    direction = compute_direction_cosine(model.lora_correction, pool_mean)
    wandb.log({"eval/direction_cosine/mean": direction})
    
    distance = compute_fraction_reached(model.lora_correction, target)
    wandb.log({"eval/frac_distance_reached/mean": distance})
```

### Code reference: The launcher

**Script:** `scripts/animals_compose_transfer/smoke_live_curves.sh`  
**What it does:** Runs the 1-epoch smoke directly on a biggpu device (never sbatch: biggpu allows one job per user and Slurm cannot see GPUs here). Guards its own device (`GPU=<index>`, aborts if the device has >1GB in use) and the `/datasets` disk, stamps the launch commit into the log header, then calls `train_pooled` with 1 epoch, inline sampling every epoch, and W&B online.  
**Node choice:** `/run-experiment`'s `pick_node.py --gpu --probe-shared` prints the decision (RUN HERE / SUBMIT / SSH SHARED / QUEUE).

---

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next step: 10](hypothesis-01-does-one-pooled-fix-transfer-at-all.md) ➡️

Once this gate passes (verdict: Green), proceed to [step 10: hypothesis-01-does-one-pooled-fix-transfer-at-all](hypothesis-01-does-one-pooled-fix-transfer-at-all.md).

👉 That step runs the actual 15-run sweep, using these live curves to diagnose which runs to trust.

---

## Error Matrix

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

Known issues and solutions for this plan. This section is automatically updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`. New errors from failed runs are added to the catalogs and propagated here.

### From global catalog

Global patterns applicable across all projects. See [~/.claude/GLOBAL_ERROR_CATALOG.md](~/.claude/GLOBAL_ERROR_CATALOG.md) for the full catalog.

#### 🔴 py-001: Python 3.8 incompatible with DINOv2 imports

**When it happens:** During import of compose_scorer.scorer

**What you see:** `ImportError: cannot import name '_Embedders'`

**Why:** Your venv is on Python 3.8, which doesn't support the DINOv2 package.

**How to fix:** Activate Python 3.9 or later.

**Reference:** [~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-py-001](~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-py-001)

---

#### 🔴 dist-001: Pool-mean cache race condition on rank >0

**When it happens:** During eval logging on multi-GPU runs

**What you see:** `eval/direction_cosine/mean` logs as 0.0 for all steps despite the fix being active

**Why:** The cache wasn't synchronized across GPU ranks. Rank 0 builds the cache while rank 1+ tries to use it.

**How to fix:** Wrap `build_pool_mean_cache()` in `torch.distributed.barrier()` to synchronize before eval.

**Reference:** [~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-dist-001](~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-dist-001)

---

#### 🟡 dist-002: Variable scope inaccessible to eval hook on rank >0

**When it happens:** During eval logging on multi-GPU runs

**What you see:** Metrics log normally on rank 0 but show NaN on rank 1 and higher

**Why:** Reference tensors are defined inside local scopes in the training loop and aren't accessible to the eval hook on other ranks.

**How to fix:** Move reference tensors to module level in `_inline_sampling.py` so all ranks can access them.

**Reference:** [~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-dist-002](~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-dist-002)

---

#### 🟡 mem-001: Batch size too large for eval hook on single GPU

**When it happens:** During eval hook execution

**What you see:** `RuntimeError: CUDA out of memory` on V100 or A100

**Why:** Full eval batch (11 training pairs × 4 seeds = 44 images) plus DINOv2 and CLIP embeddings exceeds VRAM.

**How to fix:** Reduce batch size to 16-32 on A100, 8-16 on V100. Or run on A100 nodes which have more memory.

**Reference:** [~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-mem-001](~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-mem-001)

---

#### 🔵 train-001: Loss oscillates 10-20x with gradient checkpointing

**When it happens:** During training with gradient checkpointing enabled

**What you see:** Training loss jumps by an order of magnitude between steps

**Why:** Expected behavior with checkpointing. The loss is correct; the oscillation is normal.

**How to verify:** Check eval metrics (compose-rate, direction-cosine). They should be smooth. If eval metrics are also jumping, investigate further.

**Reference:** [~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-train-001](~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-train-001)

---

#### 🔴 train-002: Silent checkpoint loading failure

**When it happens:** When resuming from a saved checkpoint

**What you see:** Model loads without error but weights are stale or uninitialized; training doesn't converge

**Why:** Checkpoint file may be corrupted, incomplete, or from a different architecture.

**How to fix:** Before loading, verify the checkpoint with `torch.load(path)` and inspect weight shapes. Ensure the checkpoint matches the current model.

**Reference:** [~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-train-002](~/.claude/GLOBAL_ERROR_CATALOG.md#entry-id-train-002)

---

### From project catalog

Patterns specific to poe_repair_min. See [environment/known-failures.md](../../../../../environment/known-failures.md) for the full catalog.

#### 🔴 poe-score-001: Scorer returns all zeros or NaNs despite valid inputs

**When it happens:** During eval hook scoring

**What you see:** `eval/compose_rate/mean = 0.0` or `NaN` for all steps, even though images look visually correct

**Why:** Input tensors have wrong dtype (int instead of float), wrong device (CPU vs GPU), or invalid range (0-255 instead of 0-1).

**How to fix:** Validate eval outputs before calling the scorer:
```python
assert eval_output.dtype == torch.float32, f"Expected float32, got {eval_output.dtype}"
assert eval_output.device.type == 'cuda', f"Expected GPU, got {eval_output.device}"
assert eval_output.min() >= -0.1 and eval_output.max() <= 1.1, f"Range error: [{eval_output.min()}, {eval_output.max()}]"
```

**Reference:** [environment/known-failures.md#entry-id-poe-score-001](../../../../../environment/known-failures.md#entry-id-poe-score-001)

---

#### 🟡 poe-score-002: Compose-rate stuck at 0.0 even though fix is active

**When it happens:** Throughout training

**What you see:** Correction is being applied (loss decreases) but `eval/compose_rate/mean = 0.0` throughout

**Why:** Scorer was trained on a specific PoE style or value range. Your fix is working but produces images the scorer doesn't recognize as blended.

**How to fix:**
1. Manually inspect a batch of generated images (save to disk, visualize in Jupyter).
2. Compare to reference PoE images used to train the scorer.
3. If they look different (color space, brightness), recalibrate the scorer or adjust the correction target.
4. If images look correct, the scorer may need retraining on this data distribution.

**Reference:** [environment/known-failures.md#entry-id-poe-score-002](../../../../../environment/known-failures.md#entry-id-poe-score-002)

---

#### 🟡 poe-mem-001: Eval hook OOM on biggpu with full-size quadrants

**When it happens:** During eval on V100

**What you see:** `RuntimeError: CUDA out of memory` on biggpu partition (V100 32GB)

**Why:** Full eval batch for cat×dog (11 training pairs × 4 seeds = 44 images) plus DINOv2 and CLIP embeddings exceeds V100 VRAM. A100 nodes have enough memory.

**How to fix:**
- Use A100 nodes (mscluster110) instead of V100 (mscluster85).
- OR: Shard eval batch to 8-16 images per forward pass and loop to aggregate.
- OR: Reduce eval frequency (eval every 5 steps instead of every step).

**Reference:** [environment/known-failures.md#entry-id-poe-mem-001](../../../../../environment/known-failures.md#entry-id-poe-mem-001)

---

#### 🟡 poe-lora-001: Fraction-of-distance-reached plateau at 20% instead of 40%

**When it happens:** By mid-training

**What you see:** `eval/frac_distance_reached/mean` climbs to ~0.2 by epoch 1, then stalls instead of reaching ~0.4

**Why:** LoRA rank too low (r=4 or r=8) to capture full correction magnitude. The correction saturates the LoRA space before reaching the target.

**How to fix:**
1. Check your LoRA rank in the config (default is r=16).
2. If r < 16, increase to r=32 or r=64 and re-run.
3. If r >= 16, the plateau is correct for this correction. Document in run notes.

**Reference:** [environment/known-failures.md#entry-id-poe-lora-001](../../../../../environment/known-failures.md#entry-id-poe-lora-001)

---

#### 🟡 poe-lora-002: Direction-cosine diverges despite low training loss

**When it happens:** During training

**What you see:** Training loss decreases smoothly but `eval/direction_cosine/mean < 0.3` or becomes negative (anti-aligned)

**Why:** LoRA correction is optimizing for a different objective than the pool-mean direction. Possible causes: stale pool-mean cache, mismatched loss function, or the correction is legitimately pair-specific.

**How to fix:**
1. Check pool-mean cache is built before training starts (see dist-001 for synchronization).
2. Verify loss function in `train_pooled.py` matches the loss used to compute pool-mean.
3. If both are correct, the divergence is data, not an error. Document in run notes.

**Reference:** [environment/known-failures.md#entry-id-poe-lora-002](../../../../../environment/known-failures.md#entry-id-poe-lora-002)

---

#### 🟡 poe-launch-001: SSH-direct launch dies instantly on relative paths

**When it happens:** Launching a run over SSH on a node the session is not on (the shared-device path on biggpu)

**What you see:** `bash: line 1: logs/<name>.log: No such file or directory` immediately; no process on the node, no log file ever created

**Why:** A non-interactive SSH command starts in `$HOME`, so relative paths in the launch line resolve there, and the nohup log redirect fails before the script starts. Prefixing `cd <repo> &&` is unreliable: Claude Code's permission layer can strip `cd` from compound commands (three identical failures in a row, 2026-08-19).

**How to fix:** Make every path on the SSH launch line absolute (script and log redirect); the launch script does its own `cd "$REPO"` internally. Verify in the same call with `sleep 5; pgrep -af "train"` and a `tail` of the absolute log path.

**Reference:** [environment/known-failures.md#entry-id-poe-launch-001](../../../../../environment/known-failures.md#entry-id-poe-launch-001)

---

**Auto-update note:** This section is automatically regenerated by `/sync-plan-tree` after new errors are added to the catalogs via `/ingest-error-pattern`. Do not edit this section manually; changes will be overwritten on the next sync.
