# 🗂️ Retrofit poe_repair_min: one name per thing, everywhere

## Position in the plan tree

Standing work, sitting beside the plan tree rather than inside a scope, because it touches every
scope. Compiled 2026-08-23 from a walk that opened every artifact in the repository. Executed by
`/retrofit-repo` against this file.

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [Words this uses](#words-this-uses)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [Considerations](#considerations)
- [The scope tree](#the-scope-tree)
- [The piles](#the-piles)
- [The loose root files](#the-loose-root-files)
- [The stages for this repo](#the-stages-for-this-repo)
- [The rename table](#the-rename-table)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)
- [Still open](#still-open)
- [References](#references)
- [Next step](#next-step)

## Quick context: where you are

**What this is**

The whole repository is renamed and refiled so that one experiment has one name, on disk, in the
code, in the paper's register and on the cluster mount. Nothing has moved yet. Every decision below
was made by opening the thing and looking at it.

**Where it came from**

A `/retrofit-repo` sweep stopped at its census because `artifacts/` meant two contradictory things.
The walk that resolved it ran on 2026-08-23 across eight piles, opening 119 images in the repository
one at a time and judging the mount's 4,259 as eleven sets.

**What has already changed**

Seven convention documents under `~/.claude/`, listed in [Stages](#the-stages-for-this-repo). Nothing
inside this repository.

## Words this uses

**The naming rule.** `<what-is-measured>-as-<what-varies>`, plus a qualifier only where the figure
shows something extra. No claim, no paper slot letter, no pair suffix, no venue, no relative name,
no private label. A picture with no axes takes `-explainer` instead. The shortest phrase that stays
true if the result changes.

**Question folder.** `artifacts/results/<the-question-the-work-asked>/`. Grouping is by question,
never by medium, so one question's folder holds its figures, their sidecars and the runs behind them
together. The split inside it is what can be committed: a curve lives in the repository, a
checkpoint lives on the mount and the card names its path.

**The animal-pair boundary.** The 20 animal pairs are in. The 7 pairs from the earlier
cross-taxonomy era stay where they are. Anything cut by group stays, whatever it holds, which costs
833M and strands no per-pair directory. `a_butterfly__x__a_flower_meadow` is the deliberate
exception: it is the control that composes and the paper opens on it.

**Struck words**, in filenames and in prose alike: `rung`, `dose`, `dose_matched`, `cache_analysis`,
`readback`, `arm`, `adapter`, `oracle`, `full strength`. An arm is the corrected run or the
uncorrected run. The adapter is a **lora**. The oracle is **the joint-prompt correction**. Full
strength is **the earliest window where lambda = 1**.

**The pair form.** Prose says "a cat and a dog" everywhere it is read. The on-disk slug stays
`a_cat__x__a_dog`, because changing it costs 3,049 files and 456 directories and buys nothing a card
does not already give.

## The claim

One experiment currently has up to four names: one on `/home-mscluster`, one on `/datasets`, one in
`poe_repair/experiments/`, and one in the paper's register. After this plan there is one.

## Why this plan exists

A name that states a claim becomes a lie when the claim dies, and one already has: `F6`'s
shared-structure argument is recorded as dead in `report/paper-evidence-index.md` while four files still
carry its slot letter. A name that states a private label costs a lookup every time: nothing on disk
says what `rung2` or `group A` or `idea5a` was. And a name that exists twice sends a reader to
whichever copy their filesystem happens to hold, which `paper/iclr/figures.md` line 30 already does:
it names a backing curve that does not exist at the path it gives.

## Considerations

**Nothing moves until `/rename` is green.**

159 code files spell output paths by hand and 24 register rows name a builder script by path. The
first task replaces that with one module that owns the names, so a rename becomes one edit. Moving
first would break the paper build and the scoring scripts in the same commit.

**Two filesystems, one of them without undo.**

`/datasets` is not a git repository, so every move there is permanent. Nothing on it is deleted.
The repository is a git work tree with 49 uncommitted paths at compile time.

**No behaviour change during the sweep.**

`/retrofit-repo` moves files and writes cards. It does not edit source. Every code change in this
plan belongs to the `/rename` task and happens before the sweep runs.

## The scope tree

13 `MASTER_PLAN.md` files. One live parent at `plans/closing-the-compositional-gap/` with four
sub-scopes; three relocated-scope folders at the plan root (`completed/`, `shelved/`, `standing/`).

**One finding stays open**: `plans/completed/compose-scorer/` is a finished *scope* inside a folder
`PLAN_TREE_FORMAT.md` reserves for finished plan *files*. `sync-plan-tree` owns the fix.

**No scope carries the three state folders** (`in-progress/`, `staging/`, `completed/`), so 215 plan
files are filed by name only and no folder says whether work is left.

## The piles

| Pile | Verdict |
|---|---|
| 1. the run bytes, 35G on `/home-mscluster` | `artifacts/results/<question>/<what-was-held-out>/`; the `rung` level dropped, not renamed |
| 1a. `/datasets/mmolefe/poe_repair_min/`, 140G | `interaction_term` promoted to the same relative path; 6 families renamed after the look pass; the other 43 folders untouched |
| 1b. `outputs/` 9 real families | all nine renamed and moved; the "folder names are not renamed" rule replaced |
| 1c. the two built scenes, 7.1G | source to `artifacts/scenes/`; 6.9G of build output deleted, minus one held deletion |
| 1d. `results/` and `data/pilot/` | logs to `logs/`; `recap/` archived; `data/pilot/` stays as input, two of its images copied out |
| 2. the 33 image clusters | 119 opened one at a time across five sittings; the mount's eleven families judged as eleven sets |
| 3. `docs/` | decommissioned, here and as a category; twelve kinds, twelve homes |
| 4. the loose investigation folders | six top-level folders removed |
| 5. `dl-scene/` and `pressure-tests/` | `artifacts/scenes/` and `artifacts/notes/`; four documents given the source header they all lack |
| 6. the code | `follow`: experiment packages take the new vocabulary in underscore form |
| 7. the record folders | `inventory/` to its scope; `learning-captures/` and `todoist-staging/` archived |
| 8. the loose root files | `EXPERIMENTS.md` and `DECISION_TIMELINE.md` absorbed into `report/`, `RESEARCH_GUIDELINES.md` into `context/`, `PARKING_LOT.md` dropped; `pairs.py` to the package |

**Eight top-level folders disappear**: `docs/`, `evidence/`, `show-me/`, `recap/`, `captures/`,
`diagrams/`, `flow-map-images/`, `results/`, plus `learning-captures/`, `todoist-staging/` and
`inventory/` moving out.

The full old-to-new table, per figure and per family, is in
[The rename table](#the-rename-table) below, and is the input `/rename` reads.

## The loose root files

| File | Verdict |
|---|---|
| `CLAUDE.md`, `README.md`, `MASTER_PLAN.md`, `RETROFIT.md`, `pyproject.toml`, `.gitignore` | stay, already on the list of root files no check reports |
| `EXPERIMENTS.md`, `DECISION_TIMELINE.md`, `RESEARCH_GUIDELINES.md` | absorbed into `report/experiments-log.md`, `report/decision-timeline.md`, `context/research-guidelines.md`, matching that folder's existing style; no longer a root-file exception |
| `PARKING_LOT.md` | dropped; the routing practice it served (idea-runs land here) retires with it, striking results become a row or task in the plan tree directly |
| `pairs.py` | to `poe_repair/pairs.py`; one importer, and its contents describe a one-pair-one-seed world the project outgrew |
| `midrun_separation_example.png`, `text_orthogonality_probe.png` | to question folders, named for what they measure |

## The stages for this repo

| Stage | Skill | Runs? |
|---|---|---|
| 0 census | `retrofit-repo` | done 2026-08-23, in `RETROFIT.md` |
| 1 context | `context-pulse --build` | yes: `context/` is absent |
| 2 environment | `env-pulse` | done 2026-08-24: the old single-file environment doc (pre-folder shape) migrated into `environment/`, and `known-failures.md` populated |
| 3 plans | `sync-plan-tree`, three passes | yes: no scope carries the three state folders |
| 4 artifacts | `tidy-repo`, two halves | yes: this is the bulk of it, both filesystems |
| 5 runbook | `runbook-pulse` | yes: `runbook/` is absent |
| 6 context, second pass | `context-pulse` | yes: definitions carried from stages 2 and 3 |
| 6b the link | per `JOURNEY_FORMAT.md` | yes: three journeys, none linked, `JOURNEYS.md` absent |
| 7 proof | `retrofit-repo` | yes: rerun every check, grep every old path, run the test suite |


## The rename table

Every row was settled by opening the thing. A dropped name means the picture survives untouched on
disk; only the second name goes.

### The paper figures

| Now | Becomes |
|---|---|
| `what-the-product-misses-explainer` | `what-the-product-misses-explainer` |
| `correction-size-over-the-denoising-run` | `correction-size-over-the-denoising-run` |
| `samples-as-the-seed-changes` | `samples-as-the-seed-changes` |
| `F1b-two-regimes-seed42` | **drop**: the top row of `correction-size-over-the-denoising-run`, named by nothing |
| `F2-correction-strength` | **drop**: byte-identical to `compose-rate-as-correction-rises`, md5 `0aeb4016…` |
| `compose-rate-as-correction-rises` | `compose-rate-as-correction-rises` |
| `compose-rate-as-correction-rises-with-detector-boxes` | `compose-rate-as-correction-rises-with-detector-boxes` |
| `compose-rate-as-correction-rises-for-a-dissimilar-pair` | `compose-rate-as-correction-rises-for-a-dissimilar-pair` |
| `compose-rate-as-correction-rises-with-a-random-control` | `compose-rate-as-correction-rises-with-a-random-control` |
| `correction-size-over-the-denoising-run-across-17-pairs` | `correction-size-over-the-denoising-run-across-17-pairs` |
| `samples-as-the-correction-window-moves` | `samples-as-the-correction-window-moves` |
| `correction-size-per-step-beside-outcome-per-window` | `correction-size-per-step-beside-outcome-per-window` |
| `language-score-as-the-correction-window-moves` | `language-score-as-the-correction-window-moves` |
| `samples-as-window-and-strength-both-change` | `samples-as-window-and-strength-both-change` |
| `compose-rate-as-the-window-moves-at-matched-total` | `compose-rate-as-the-window-moves-at-matched-total` |
| `how-many-seeds-composed-as-the-window-moves` | `how-many-seeds-composed-as-the-window-moves`, **and a register row** |
| `samples-as-the-correction-runs-longer` | `samples-as-the-correction-runs-longer` |
| `samples-as-the-correction-starts-later` | `samples-as-the-correction-starts-later` |
| `F4g-overcorrection-grid` + `-seed12` | a two-cell set: `samples-as-the-window-moves-and-strength-goes-past-one/{seed-09,seed-12}.png` |
| `F5-one-dial-three-instruments` | **redraw as three**: `where-the-picture-sits-as-correction-rises`, `compose-rate-as-correction-rises-by-the-detector`, `direction-agreement-with-the-sampler-step` |
| `how-far-the-corrected-run-separates-from-the-uncorrected-one` | `how-far-the-corrected-run-separates-from-the-uncorrected-one` |
| `F6-spectrum-windowed` | **redraw as two**: `energy-captured-as-directions-are-added`, `outcome-correlation-by-direction-rank`; **and a register row** |
| `content-change-relative-to-attention-change-under-lora` | `content-change-relative-to-attention-change-under-lora` |
| `compose-rate-as-the-lora-trains` | `compose-rate-as-the-lora-trains` |
| `compose-rate-by-pair-for-lora-against-the-joint-prompt-correction` | `compose-rate-by-pair-for-lora-against-the-joint-prompt-correction` |
| `direction-agreement-between-consecutive-steps` | `direction-agreement-between-consecutive-steps` |
| `direction-agreement-between-consecutive-steps-when-it-alternates` | `direction-agreement-between-consecutive-steps-when-it-alternates` |
| `direction-agreement-for-random-vectors` | `direction-agreement-for-random-vectors` |
| `direction-agreement-between-two-seeds` | `direction-agreement-between-two-seeds` |
| `direction-agreement-as-the-starting-noise-is-moved` | `direction-agreement-as-the-starting-noise-is-moved` |
| `direction-agreement-between-two-pairs` | `direction-agreement-between-two-pairs` |
| `direction-agreement-by-what-the-two-runs-share` | `direction-agreement-by-what-the-two-runs-share` |
| `samples-per-step-with-the-correction-on-and-off` | `samples-per-step-with-the-correction-on-and-off` |

### The run families

| Now | Becomes |
|---|---|
| `artifacts/rung1-overfit/lora/` | `artifacts/results/can-lora-learn-a-residual-that-corrects-poe/one-pair-one-seed/` |
| `artifacts/rung2-survive-noise/cross_seed/` | same question folder, `held-out-seeds/` |
| `artifacts/rung3-group-wise/`, `artifacts/rung4-scale/` | **stay**, cut by group |
| `outputs/animals_compose_transfer/` | `does-the-fix-reach-unseen-pairs` |
| `outputs/group_a_failure/` | `residual-dynamics/correction-outside-the-unet` |
| `outputs/residual_diagnostics/` | `residual-dynamics/residual-between-mono-and-poe` |
| `outputs/conditioning_window/` | `when-the-correction-must-arrive/cfg-window-without-lora` |
| `outputs/conditioning_window_lora/` | `when-the-correction-must-arrive/cfg-window-with-lora` |
| `outputs/poe/` | `poe-blends-instead-of-composing/poe-baseline-samples` |
| `outputs/compose_scorer/` | `can-we-trust-the-compose-score/compose-scorer-validation` |
| `outputs/interaction_term/` | merges onto the mount; the zero-byte `dose` husk goes, `window` is diffed first |
| `outputs/presentation/` | **look first**, 6M of old slides, marked cold |

### The mount's eleven families

| Now | Becomes |
|---|---|
| `dose` | `how-much-correction-is-needed` |
| `window` | `window`, kept |
| `cross` | `samples-as-the-window-moves-one-step-at-a-time` |
| `experts` | `decoded-predictions-per-step-for-each-expert` |
| `reprobe` | `content-change-relative-to-attention-change` |
| `window_seeds` | `compose-rate-in-the-first-window-across-twelve-seeds` |
| `overcorrection_grid` | `samples-as-the-window-moves-and-strength-goes-past-one` |
| `live_curves_smoke_run` | `training-run-scored-while-it-trains` |
| `dose_matched` | `same-total-correction-different-window` |
| `xhat0_readback` | `predicted-clean-image-per-step` |
| `cache_analyses` | **dissolved** into the question folders its 16 figures answer; 5 of them already exist in the repo as `evidence/h05-three-sides/`, byte-identical |

### The evidence set

| Now | Becomes |
|---|---|
| `evidence/f2-lambda1-audit/` | `artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/` |
| `01-both-there/` | `both-animals-there/` |
| `02-two-of-one/` | `two-of-the-same-animal/` |
| `03-cannot-call/` | `cannot-decide/` |
| `04-look-alike-by-design/` | `the-two-animals-look-alike/` |
| `05-scored-failure/` | `scored-as-a-failure/` |
| the 32 cells | unchanged: `<pair>_seed<N>_n<count>.png` is already pair, seed and count |

### The experiment packages, and the code

| Now | Becomes |
|---|---|
| `poe_repair/experiments/correction_outside_the_unet/` | `correction_outside_the_unet/` |
| `poe_repair/experiments/cfg_window_without_lora{,_lora}/` | `cfg_window_without_lora/`, `cfg_window_with_lora/` |
| `poe_repair/experiments/residual_between_mono_and_poe/` | `residual_between_mono_and_poe/` |
| `poe_repair/experiments/does_the_fix_reach_unseen_pairs/` | `does_the_fix_reach_unseen_pairs/` |
| `poe_repair/experiments/compose_scorer_validation/` | `compose_scorer_validation/` |
| the 24 `scripts/make_*.py` | named for the figure each draws, not its paper slot |
| the 9 `scripts/build_*.py` | `scripts/build/`, names unchanged |
| the other 53 in `scripts/` | grouped by the experiment they probe |
| `pairs.py` | `poe_repair/pairs.py` |

### The duplicates found by opening, all four

| Copy | Original |
|---|---|
| `F2-correction-strength.png` | `compose-rate-as-correction-rises.png`, md5 `0aeb4016…` |
| `evidence/f5b-trajectory-divergence/how-far-the-corrected-run-separates-from-the-uncorrected-one.png` | `paper/iclr/figures/how-far-the-corrected-run-separates-from-the-uncorrected-one.png`, md5 `026b6a08…` |
| `evidence/h05-three-sides/` 5 images | the mount's `cache_analyses/`, all five identical |
| `outputs/interaction_term/dose` | a zero-byte husk of the mount's 6.3G `dose` |

## Tasks

For Claude to execute. Ask Claude to do these, in this order.

### 0. 🧭 Preflight

- [ ] **Read [The rename table](#the-rename-table)** end to end. It is the input, and every row was
      settled by opening the thing.
- [ ] **Confirm `/datasets` is reachable** and note the mark. `df -h /datasets` returned 348T free
      on 2026-08-23.

### 1. 🔤 Rename, before anything moves

- [ ] **Write the paths module.** `poe_repair/config.py` resolves the output root and stops there.
      It gains one constant per run family, and the hardcoded
      `/datasets/mmolefe/poe_repair_min/outputs/...` strings become that same root, so one script
      runs against either filesystem.
- [ ] Run the following prompt: `/rename` over the 159 code files that spell output paths, the 24
      register rows in `paper/iclr/figures.md` that name a builder script, the 12 experiment
      packages under `poe_repair/experiments/`, and the 24 `make_*.py` builders that carry paper
      slot letters.
- [x] **Proven green 2026-08-24.** The test suite passes (229 passed, 1 skipped, 1 xfailed) and
      `tests/test_paths_resolve.py` confirms every real path-construction site routes through
      `poe_repair/paths.py`. The literal grep below is **not** the gate: it still fires on
      `paths.py`'s own constant values (which correctly hold the old on-disk names, since nothing
      has moved yet) and on docstrings/`--help` text that name a path for a human reader, not for
      I/O. Both are confirmed harmless by direct inspection, file list in commit `f293bdf`. Do not
      re-run this grep as a stop condition; it will never go empty while `paths.py` exists.

### 2. 📦 Run the sweep

- [ ] Run the following prompt: `/retrofit-repo plans/retrofit-poe-repair-min.md`, stages 1 to 7 in
      the order above.
- [ ] **Carve out the audit set first** if a smaller first move is wanted: it is self-contained,
      already carries its card, and `calls.json` records every source path so it rebuilds from
      scratch.

### 3. 🔗 Link the three journeys

- [ ] **Write `JOURNEYS.md`** at `/home-mscluster/mmolefe/goal-setting/learning/`, generated from
      disk, one row per journey.
- [ ] **Write the project link both ways** for `poe-derivation-foundations`, which already names
      this repo in five plan files, and move `docs/IMMERSE_PoE_Foundations.md` into it.

### 4. 🧹 The ten skills carrying stale paths

- [ ] **Sweep the eight safe ones**: `show-me` (6), `recap-plan-tree` (4), `report-pulse` (3),
      `demonstrate` (3), `drip-report` (2), `evidence-ladder` (2), `tidy-repo` (1),
      `visual-research-director` (1). All are `artifacts/figures/` in prose.
- [ ] **Then the two that run unattended**, with a test: `ingest-error-pattern` (5 places) and
      `sync-plan-tree` (2). Fire the hook once against a known failure and confirm the entry lands
      at `environment/known-failures.md`.

## Instructions

For you to follow manually. Do these yourself.

### 5. 👀 Confirm the four irreversible deletions

Each is untracked, so there is no undo. Confirm one path at a time.

- [ ] `dl-scene/app/node_modules` (156M) and `scene*/node_modules` (196M). Check `package-lock.json`
      exists in each, then delete.
- [ ] `scene/dist` (3.4G). Check `scene/public/` holds its 9.1M of assets, then delete.
- [ ] **`scene-h04/dist` (3.5G): do not delete yet.** Open `scene-h04/loader/` and find how its 495
      images arrived, because `scene-h04/public/` is empty and they exist nowhere else in the repo.
      Write the copy step down as a runbook recipe, then delete.
- [ ] `dl-scene/app/dist` (2.4M). Same check as `scene/dist`.

### 6. 🔍 Re-judge one call in the audit set

- [ ] Open `evidence/f2-lambda1-audit/05-scored-failure/a_seal__x__a_walrus_seed9_n1.png`. It shows
      two distinct pinnipeds and the detector returned one. The card calls both scored-failure cells
      correct. If this one is the scorer undercounting, the bound in that card moves **up** as well
      as down, and the card says so.

### 7. 📐 Decide the two redraws

- [ ] `F5-one-dial-three-instruments` is three figures in one strip, three x axes and three y axes
      under one caption. Decide whether it becomes three.
- [ ] `F6-spectrum-windowed` is four panels answering two questions, and no register row names any
      of it. Decide whether it becomes two, and whether the slot survives at all given
      `report/paper-evidence-index.md` records its argument as dead.

## Outputs

| Deliverable | Where |
|---|---|
| every run family under one name on both filesystems | `artifacts/results/<question>/` and the mount's mirror |
| a card per question folder | `artifacts/results/<question>/README.md` |
| the old-to-new table, so a stale reference still resolves | `RENAMES.md` |
| the environment folder, with the failure catalog in it | `environment/` |
| the report folder | `report/` |
| the context folder and the runbook | `context/`, `runbook/` |
| the journeys register and the project link | `goal-setting/learning/JOURNEYS.md` and both `MASTER_PLAN.md` files |

## Still open

**Two figures have no referrer at all.** `midrun_separation_example.png` and
`how-many-seeds-composed-as-the-window-moves.png`. Each needs a register row or a `Held since` line.

**Five figures carry their claim rendered into the pixels**, as a title: "More of the correction,
more composition" on two, "the same cliff, scored in language", "the cliff is not a dose effect",
"Geometry says not shared, the adapter transfers anyway". Renaming does not touch those. Redrawing
does.

**Two illustrations are owed**, with their prompts already written and their slots visible:
`figures/rt-interior-peak.png` and `figures/posterior-narrowing-covariance.png` under
`pressure-tests/`.

**Three pieces of work judge the paper and the paper does not name them**: the lambda=1 audit that
bounds the headline rate, `F6-what-the-spectrum-measures` which records a dead claim, and the
pressure test carrying per-claim verdicts on section 5.

**The scorer question is open in a review file** and surfaced four times during the walk, drawn by
the author against their own figures each time.

**`pairs.py` is stale.** One pair, one seed, in a project running twenty pairs across seeds 9 to 12.

**`plans/completed/compose-scorer/`** is a finished scope inside a folder reserved for finished plan
files.

## References

- `RETROFIT.md` — the census and the sitting ledger
- `~/.claude/ARTIFACT_TREE_FORMAT.md` v3 — the shape and the naming forms
- `~/.claude/ERROR_MATRIX_SYSTEM.md` — the failure catalog's new home
- `report/paper-evidence-index.md` — the second conforming card, and where F6's claim is recorded as dead
- `evidence/f2-lambda1-audit/README.md` — the first conforming card, and the model for the rest

## Next step

Task 1: write the paths module, then `/rename`. Nothing moves until its grep comes back empty.
