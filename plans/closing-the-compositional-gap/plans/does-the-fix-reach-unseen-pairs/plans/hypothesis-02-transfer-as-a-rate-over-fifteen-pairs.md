# 🅰️ Hold out each pair in turn: transfer as a rate, not an anecdote

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 10 | [hypothesis-01-does-one-pooled-fix-transfer-at-all](hypothesis-01-does-one-pooled-fix-transfer-at-all.md) | Tests pooled fix on one pair |
| **11 (current)** | **this plan** | **Measures transfer rate across 15 held-out pairs** |
| 12 | [baseline-01-the-size-matched-control-pool](baseline-01-the-size-matched-control-pool.md) | Compares against control |

The one order is in the `## Running order` table in [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md). Verdicts live in [../review/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md](../review/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md).

---

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What the fix actually does (visual)](#what-the-fix-actually-does-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Error Matrix](#error-matrix)

---

## Recommended prompt (after run completes)

After you finish running this plan and want to ingest error patterns into the catalogs, use this prompt:

```
/ingest-error-pattern --from-run-log
```

This extracts error patterns from the run transcript, deduplicates against global and project catalogs, and adds new entries. New errors are automatically propagated to all affected plan files.

---

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment**: Leave-one-pair-out (LOPO) cross-validation across 15 animal pairs from the finalised training pool.

**The hypothesis**: The LoRA-based correction $r_t$ transfers to pairs the model never trained on, with a degradation curve that shows the relationship between held-out fraction and compose rate.

**If true:** The fix is pair-general (not pair-specific), so removing different pairs from training should yield a smooth degradation curve (compose-rate drops gradually as more data is held out). All 15 held-out evaluations compose with moderate-to-high rates.

**If false:** The fix is pair-specific or relies on data leakage, so removing even one pair breaks transfer. We'd see compose-rate drop sharply or hit zero for many held-out pairs.

**Rationale**: After hypothesis-01 confirms the pooled fix transfers to one held-out pair, the question is whether this success scales. LOPO answers this by testing 15 independent generalization points from the same training distribution, producing a rate curve instead of a binary answer.

**This scope's job**: Turn a single yes/no answer ("does transfer work?") into 15 transfer points with a rate and degradation curve.

**What this plan does**: Train 15 independent LoRAs, each omitting exactly one pair from the training pool, then evaluate each held-out pair on its matching LoRA. Collect compose-rate, direction-cosine, and distance-reached across all 15 runs to produce a leaderboard and degradation curve.

**Dataset details**:
- **Training pool**: 15 animal pairs (cat×dog, eagle×hawk, etc.)
- **Hold-out strategy**: Each run holds out exactly one pair; the other ~14 are used for training.
- **Eval pairs**: Each run evaluates on the one pair it never saw.
- **Known phenomenon**: Compose-rate typically plateaus at 40–60% on new pairs; direction-cosine should stay high (0.8+) if the fix is pair-general.

**Associated materials**:
- **Review questions**: [../review/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md](../review/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md)
- **Procedures**: (none yet; recommend creating [procedures/hypothesis-02-run-lopo-sweep.md](procedures/hypothesis-02-run-lopo-sweep.md) if manual steps emerge)
- **Assets/outputs**: Will be saved to `outputs/interaction_term/transfer_rate/`
  - **Figure organization**: Use [figure-coverage-prompt.md](diagrams/figure-coverage-prompt.md) to scan the repo, rename all related figures to the step-11 naming convention, and consolidate them into `outputs/interaction_term/transfer_rate/figures/`. This prompt generates a FIGURE_CATALOG.md that maps each figure to axes, meaning, and original location.
  - **Locations scanned**: `artifacts/results/ (per-question) and report/paper-evidence-index.md`, `outputs/interaction_term/`, `paper/iclr/figures/`, `/show-me` artifacts, results/ folders.

**For the full picture**: This plan serves Objective 2 (Transfer A) and Definition-of-Done item 3 of the scope. See [../MASTER_PLAN.md](../MASTER_PLAN.md) for the full experimental context.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime**: 15 sequential LoRA trainings, each to step budget (typically ~8-12 hours wall time per run, total ~5-7 days for the sweep).

**Prerequisites**:
- Finalised pair pool in `pair_pool.yaml`
- Base model checkpoint ready
- Eval hook wired (compose-rate + direction-cosine + distance-reached logging live during training)

**Environment facts this plan depends on**:
- See [environment/00-INDEX.md](../../../../../environment/00-INDEX.md) for cluster facts: partitions, walltime limits, Python paths, and disk guards.
- `/home-mscluster` disk has sufficient space for 15 checkpoints (~50GB total).
- Co3 cluster partition available for GPU allocation.
- W&B project: `prime_lab/poe-repair-animals-compose` (verify with `echo $WANDB_PROJECT`).
- Python 3.9+ required (DINOv2 and CLIP imports depend on it).

**Project tracking**: Results logged to W&B per run; leaderboard aggregated manually to `outputs/transfer-rate/leaderboard.json` after all 15 complete.

**Known issues**: See [Error Matrix](#error-matrix) section at the bottom for a full catalog of known issues and solutions.

---

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**15 LoRAs, each trained without one pair, will compose on their held-out pairs with a measurable rate (0–100%), producing a degradation curve that shows transfer robustness.**

**Why this matters right now:** A single transfer test cannot answer whether the fix is robust across the pair distribution. Fifteen points (one per held-out pair) yield a rate, a confidence band, and a curve showing how the fix breaks down as training data is removed.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-the-fix-actually-does-visual) ➡️

**The problem**: After hypothesis-01 shows the pooled fix transfers to one held-out pair, the next question is whether that success is a fluke or a durable property of the correction. Testing on a single pair gives a binary answer; testing on multiple pairs gives a rate and degradation curve.

**The solution**: Leave-one-pair-out (LOPO) is a standard validation technique. Train 15 independent LoRAs, each missing exactly one pair, then eval each LoRA on exactly the pair it never saw. This produces 15 independent transfer measurements from the same training distribution.

**Key insights**:
1. Each held-out pair represents a new generalization test, not a new pair in the training pool.
2. The 15 measurements are independent: each LoRA was trained without its test pair, so there is no data leakage.
3. Degradation curve (compose rate vs fraction held out) reveals whether the fix is brittle (sharp drop with small removals) or robust (gradual decline).
4. Direction-cosine and distance-reached per run show whether the correction's structure is stable across held-out pairs.

---

## What the fix actually does (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

This diagram shows the LOPO strategy:

```
BEFORE: Single transfer test (yes/no)
┌────────────────────────────────┐
│ Train LoRA on all 15 pairs     │
│ Test on one unseen pair        │
│ Result: composes? Y or N       │
│                                │
│ Problem: one data point,       │
│ no degradation curve           │
└────────────────────────────────┘

AFTER: Leave-one-pair-out (LOPO) cross-validation
┌────────────────────────────────┐
│ Iteration 1: train on 1–14     │
│            test on 15          │ → score₁
│ Iteration 2: train on 1,3–15   │
│            test on 2           │ → score₂
│ Iteration 3: train on 1–13,15  │
│            test on 14          │ → score₃
│ ...                            │
│ Iteration 15: train on 1–14    │
│             test on 15         │ → score₁₅
│                                │
│ Results: 15 transfer points    │
│ → degradation curve            │
│ → robust answer                │
└────────────────────────────────┘

Compose-rate curve (expected):
      1.0 ┤
          │                    ✓ robust
    0.75 ├────────╲
          │         ╲___
    0.50 ├              ╲
          │               ╲___
    0.25 ├                    ─ ─ ─
          │
      0 └──────────────────────────
            1/15  2/15  4/15  15/15
            (fraction held out)
```

If the fix is pair-general, the curve declines smoothly. If it's pair-specific or brittle, the curve drops sharply or hits zero early.

---

## Description: what to build

⬅️ [Previous](#what-the-fix-actually-does-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **Configure the 15 leave-one-pair-out runs**
   - Iterate over `pair_pool.yaml`; for each pair, create a config that trains on all other ~14 pairs.
   - Reuse `multi_pair_trainer.py` / `train_pooled.py` with a flag that specifies held-out pair.
   - Write configs to a sweep directory (e.g., `configs/lopo_sweep/`).

2. **Run the 15-run sweep with live eval**
   - Use wired eval hook that logs compose-rate, direction-cosine, distance-reached per step.
   - Each run trains to its step budget (e.g., 10,000 steps or convergence).
   - Runs may be parallel (if cluster allows) or sequential via Slurm array job.
   - Store outputs in `outputs/transfer-rate/run-{pair-name}/` per run.

3. **Eval each held-out pair on its LoRA**
   - After all 15 LoRAs are trained, eval each held-out pair on its corresponding LoRA.
   - Collect compose-rate (binary: yes/no), direction-cosine, and distance-reached.
   - Log results to W&B run summary + JSON file per pair.

4. **Build the leaderboard and degradation curve**
   - Leaderboard: one row per held-out pair (columns: pair name, compose y/n, distance-reached, direction-cosine, embedding space 1 score, embedding space 2 score).
   - Degradation curve: compose-rate (y-axis, 0–100%) vs fraction-of-pool-held-out (x-axis, 1/15 to 15/15), one point per pair, with error bars if multiple seeds exist.
   - Save outputs to `outputs/transfer-rate/leaderboard.json` and `outputs/transfer-rate/degradation_curve.png`.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose**: Answers the transfer-robustness question (Objective 2, Definition-of-Done item 3): does the correction transfer reliably, or only in lucky cases?

**Goals**:
1. All 15 LoRAs trained and checkpointed.
2. Each held-out pair evaluated on its LoRA.
3. Leaderboard table produced with complete metrics (compose, distance-reached, direction-cosine for both embedding spaces).
4. Degradation curve showing compose-rate vs held-out fraction.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

### 1. 🔧 Configure LOPO sweep

- [ ] Read `pair_pool.yaml` and extract all 15 pair names.
- [ ] For each pair, generate a training config that holds out that pair and trains on ~14 others.
- [ ] Write configs to `configs/lopo_sweep/` with naming scheme `lopo-{pair-name}.yaml`.
- [ ] Verify 15 configs exist and each references the correct held-out pair.

### 2. 🖥️ Run the 15-run sweep

- [ ] Launch all 15 runs via Slurm array job or sequential submission.
- [ ] Each run uses wired eval hook: compose-rate, direction-cosine, distance-reached logged to W&B per step.
- [ ] Monitor runs via W&B dashboard and `squeue`; mark delivery-null runs (see gate below) and skip full budget.
- [ ] Collect W&B run IDs and checkpoint paths after all complete.

### 3. 📊 Eval held-out pairs

- [ ] For each trained LoRA, load checkpoint and eval on its corresponding held-out pair.
- [ ] Collect compose-rate (binary), direction-cosine, distance-reached from eval output.
- [ ] Save per-pair results to `outputs/transfer-rate/{pair-name}-eval.json`.

### 4. 📊 Build leaderboard and degradation curve

- [ ] Aggregate all per-pair results into `outputs/transfer-rate/leaderboard.json`.
- [ ] Compute degradation curve: compose-rate (%) vs fraction-held-out (1/15, 2/15, ..., 15/15).
- [ ] Generate plot and save to `outputs/transfer-rate/degradation_curve.png`.

---

## The engagement gate

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> If transfer is real, 15 held-out evaluations will produce a rate (not a binary answer) and a degradation curve. This is the main lever for answering Definition-of-Done item 3. Failure here blocks the paper's core claim.

**Pass criteria**:
- All 15 LoRAs reach their step budget and produce a verdict.
- Leaderboard table is non-empty with one row per held-out pair.
- Both metric columns (compose y/n and distance-reached) are populated for all 15 pairs.
- Degradation curve is computable (at least 13/15 pairs with compose-rate data).

**Fail criteria (STOP)**:
- Fewer than 13 of 15 LoRAs complete training (insufficient data for a curve).
- Eval hook fails to log metrics, leaving leaderboard empty.
- Compose-rate is zero across all 15 pairs (indicates the correction does not transfer at all).

**Partial pass guidance**:
- If 13–14 of 15 LoRAs complete: compute curve with available data, mark missing runs as `null` on leaderboard, and note in review file that n=13–14 instead of 15.
- If a single run's distance-reached is zero: mark as `delivery-null` (not `no-transfer`) and exclude from curve calculation. Do not re-run a single cell to complete the grid.

**When you get results, answer** [../review/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md](../review/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md).

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

**Purpose**: Every figure related to this plan is tracked in a single catalog. Figures fall into two categories: (1) ones to be generated from diagram prompts (status: Pending), and (2) ones expected to be generated during plan execution (status: Generated during run).

### Pending: to be generated from diagram prompts

Run each `.prompt.md` file through Claude (or `/prompt-storyboard`) and save outputs to `diagrams/figures/` with the filenames below:

| Figure | Prompt file | What it shows | Save to |
|--------|-------------|---------------|---------|
| LOPO strategy visual | [lopo-strategy.prompt.md](diagrams/lopo-strategy.prompt.md) | The leave-one-pair-out design: how 15 LoRAs test generalization | `diagrams/figures/lopo-strategy.png` |
| Robustness vs brittleness | [robustness-curves.prompt.md](diagrams/robustness-curves.prompt.md) | Expected degradation curves: smooth (robust) vs sharp (brittle) | `diagrams/figures/robustness-curves.png` |

### Generated during plan execution

| Figure | Description | Generated by | Status | Axes |
|--------|-------------|--------------|--------|------|
| Degradation curve | Compose-rate (%) vs fraction-held-out (1/15–15/15), one point per pair | `/run-experiment` 15-run sweep, W&B aggregation | ⏳ Generated during run | X: fraction held out (0 to 1); Y: compose-rate (0.0 to 1.0) |
| Leaderboard table | One row per held-out pair: pair name, compose y/n, distance-reached, direction-cosine, space-1 score, space-2 score | Post-run aggregation script | ⏳ Generated during run | N/A (table); columns: pair, compose, distance, cosine, embeddings |

### Generated during plan execution

| Figure | Description | Generated by | Status | Axes |
|--------|-------------|--------------|--------|------|
| (to be filled after run) | | | | |

### Organization workflow

1. Generate pending figures: Run each prompt through Claude and save to `diagrams/figures/`.
2. Execute the plan: Run `/run-experiment` 15-run sweep.
3. After all 15 runs complete, pull W&B summaries for each run.
4. Aggregate metrics into `leaderboard.json` (script or manual).
5. Plot degradation curve using `leaderboard.json` as input.
6. Organize all figures: Run [figure-coverage-prompt.md](diagrams/figure-coverage-prompt.md) to scan the repo, rename figures to step-11 naming convention, move to `outputs/interaction_term/transfer_rate/figures/`, and generate [FIGURE_CATALOG.md](outputs/interaction_term/transfer_rate/FIGURE_CATALOG.md).

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

**After the run completes**:

1. **Ingest errors** (automatic propagation): Run `/ingest-error-pattern --from-run-log` to extract patterns from the run transcript. Deduplicates against `~/.claude/GLOBAL_ERROR_CATALOG.md` and `environment/known-failures.md`. New errors are appended to the appropriate catalog. The skill automatically triggers `/sync-plan-tree --update-error-matrices` when done.

2. **Update the Error Matrix section** (automatic): `/sync-plan-tree` reads both catalogs and regenerates the Error Matrix section of this plan file. New errors from this run are now visible in the section below.

3. **Organize figures** (manual, but guided): After the 15-run sweep finishes, W&B outputs metrics to your project. Use the [figure-coverage-prompt.md](diagrams/figure-coverage-prompt.md) to scan the repo for all related figures (existing artifacts/results/ (per-question) and report/paper-evidence-index.md, outputs/, paper/ figures plus new W&B plots). The prompt renames them to the step-11 naming convention and consolidates them into `outputs/interaction_term/transfer_rate/figures/`. It generates a `FIGURE_CATALOG.md` that maps each figure to its axes, meaning, and original location.

**Why this matters**: Without orchestration, the Error Matrix section becomes stale after a run completes, and figures scatter across the repo. With it, you run two commands post-run (`/ingest-error-pattern` and the figure-coverage prompt) and everything stays current. The next time you visit this plan file, you see what actually happened, not what was planned.

**Quick reference**:

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | Manual (after run) | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Organize figures | Run [figure-coverage-prompt.md](diagrams/figure-coverage-prompt.md) | Manual (after run) | Figures renamed, consolidated, cataloged |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

**File**: `scripts/train_pooled.py` or `multi_pair_trainer.py`  
**Function**: main training loop with `--hold-out-pair` flag  
**Relevant section**: Pair pool loading; filtering to exclude held-out pair from training data.

**File**: `scripts/eval_pair_on_lora.py`  
**Function**: eval_on_lora(lora_ckpt, test_pair, wired_scorer)  
**Relevant section**: Loads a LoRA checkpoint and scores the held-out pair using the wired scorer (compose-rate, direction-cosine, distance-reached).

**File**: `scripts/aggregate_lopo_results.py`  
**Function**: aggregate_leaderboard(run_dirs, output_file)  
**Relevant section**: Reads per-run W&B summaries and JSON outputs, builds the leaderboard table and degradation curve.

---

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

Move to [baseline-01-the-size-matched-control-pool.md](baseline-01-the-size-matched-control-pool.md): compares the LOPO transfer results against a size-matched control LoRA trained on random vector pairs instead of the learned correction.

---

## Error Matrix

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

**Purpose**: Known issues and solutions for this plan. This section is automatically updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`. New errors from failed runs are added to the catalogs and propagated here.

### From global catalog

Global patterns applicable across all projects. See [~/.claude/GLOBAL_ERROR_CATALOG.md](~/.claude/GLOBAL_ERROR_CATALOG.md) for the full catalog.

(none recorded yet)

---

### From project catalog

Patterns specific to poe_repair_min. See [environment/known-failures.md](../../../../environment/known-failures.md) for the full catalog.

(none recorded yet)

---

**Auto-update note:** This section is automatically regenerated by `/sync-plan-tree` after new errors are added to the catalogs via `/ingest-error-pattern`. Do not edit this section manually; changes will be overwritten on the next sync.
