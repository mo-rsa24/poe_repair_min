# 📊 Three rules on one amount axis

What happens to four composition rules when they are all driven along the same dial, from the rule
running alone to the joint prediction exactly?

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-02-three-rules-on-one-dose-axis.md — <what the lambda=1 column classified>
```

## Recommended skill

▶ `/design-figure` ✅ two grids on a shared axis, with rows that deliver different absolute amounts
   and pictures that cannot reach `λ=1`, is a layout decision that has to be made before the render,
   not fixed in a caption after it.
   alt: `/analyze-figure` on the existing amount grid `strength-grid-with-three-controls-and-compose-curve.png`,
   so this one inherits a layout the paper already reads.

## Position in the plan tree

**Step 29 of 30.** Waits on steps 25 and 28. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 25 | [instrument-01: the-corrector-and-the-step-size-it-runs-at](../tools/02-the-corrector-and-the-step-size-it-runs-at.md) ⚠️ | supplies the composer behind the two corrector rows |
| 28 | [baseline-01: what-changes-when-superdiff-leaves-its-own-defaults](05-what-changes-when-superdiff-leaves-its-own-defaults.md) ⚠️ | supplies the per-step prediction behind the SuperDiff row |
| **29 (current)** | **baseline-02: three-rules-on-one-amount-axis** ⚠️ | **puts product-of-experts, SuperDiff and the corrector on one generalised amount axis, in two grids** |
| 30 | [idea-01: feynman-kac-correctors-gated](../ideas/07-feynman-kac-correctors.md) ⚠️ | runs only if this plan and the answer at step 26 give it a reason to |

Design only. Verdicts and run state live in
[the paired review file](../../review/06-three-rules-on-one-amount-axis.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Words this plan uses](#words-this-plan-uses)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [The two traps this grid is known to hit](#the-two-traps-this-grid-is-known-to-hit)
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

Give every composition rule the same dial, by defining the correction relative to that rule rather
than only relative to plain product-of-experts. Then render two grids, so four rows can be read
against each other instead of as four unrelated pictures.

## Words this plan uses

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

**The generalised amount axis.** For any composition rule `M` with a per-step prediction `eps_M`,
define `r_t^M = eps_J - eps_M` and inject `eps_M + λ·r_t^M`. At `λ=0` the rule runs alone. At `λ=1`
the prediction is `eps_J` exactly. Every rule then travels the same axis, so a `λ` column is a
matched comparison across rules.

**The four rows.** Plain product-of-experts with `r_t`, SuperDiff with `r_t^SD`, the corrector at
`k=1` with `r_t^(1)`, and the corrector at `k=5` with `r_t^(5)`.

**Not-an-identity.** A label on a picture in the grid where the rule cannot reach the joint render at `λ=1`, even
though the arithmetic says the prediction is `eps_J`. It is expected for the corrector rows and it
is information rather than a failure.

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** A generalised injection driver over four composition rules, and two grids drawn
from it.

**What it does.** It removes the thing that makes rule comparisons unreadable. Four methods each
rendered at their own settings produce four pictures with no shared axis. Defining the correction
relative to each rule gives every one of them the same dial from "the rule alone" to "the joint
prediction exactly", so a column means the same thing across rows.

**Key components.** The injection itself, the two grid drivers, and the labelling that handles the
two known traps rather than discovering them.

**Dataset details.** Grid one: four rows by four columns, seeds 9 to 12 at `λ=0.75`, plus the same
grid at `λ=0` rendered in the same pass, which gives the methods-alone figure for free. Grid two:
four rows by five columns, `λ ∈ {0, 0.25, 0.5, 0.75, 1}` at seed 9. Thirty-six renders in total.

**Associated materials.** [The review questions](../../review/06-three-rules-on-one-amount-axis.md),
[the whole corrector design](../../source/the-whole-corrector-design.md), and
[F5, the existing one-dial figure](../../../paper/iclr/figures/F5-one-dial-three-instruments.png)
whose `λ=0.75` this grid reuses.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**A baseline freezes on landing and may not change a claim.**

What this produces is a comparison column, not a result about the correction. If a row here looks
striking, it earns the right to propose an experiment and nothing more.

**The framing depends on how step 26 came out, even though the runs do not.**

Under a split at [step 26](../hypothesis/03-what-is-left-once-the-chain-settles.md), these grids
illustrate a diagnosis. Under a null they are a baselines table. Same renders, different sentences,
and the difference is settled before the captions are written rather than after.

> A null at step 26 means the correction's size at the end of the run came out the same whether the
> chain ran or not.

**These figures are about how much is added, so they do not go in the timing folder.**

They file under `paper/iclr/figures/how-much-is-added/across-composition-rules/`. The folder
`when-the-correction-arrives/` asks a question about timing, and one of these grids filed there
would be answering the wrong question by its own path.

**Two traps are known in advance.** See
[the section below](#the-two-traps-this-grid-is-known-to-hit).

**The cache cannot be used.** Three of the four rows follow their own trajectories.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Four composition rules travel one amount axis, and the `λ=1` column separates rules that combine
scores from rules that do something else.**

**Independent variables.** The composition rule (four rows), and `λ` (five values in grid two). In
grid one `λ` is fixed at 0.75 and the seed varies instead.

**Dependent variable.** Whether the render composes, by the detector and by eye, and whether the row
reaches the joint render at `λ=1`.

**Falsify condition.** This is a baseline and a figure, so there is no threshold that closes the scope.
What is pre-registered is the labelling: the two product-of-experts-family rows are expected to
land on the joint render at `λ=1` and the corrector rows are not. A row behaving other than
expected is recorded as a finding about what that rule is, not adjusted into agreement.

**Why this matters right now.** The `λ=1` column is a free classifier. Any rule that genuinely just
combines scores must reproduce `eps_J` when the correction is fully added back. A rule that cannot
is doing something beyond combining scores, and that is worth a sentence in the paper whether or
not step 26 came back with a split.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Comparing composition rules by rendering each at its own settings produces
pictures nobody can read against each other. The differences that show up are differences in
everything at once.

**The approach.** Generalise the amount axis this project already uses. `r_t` was always
`eps_J - eps_PoE`; the same definition against any rule's own prediction gives every rule the same
journey from itself to the joint prediction.

**Key insights.**

1. A shared axis is what makes a column a comparison. Without it four rows are four anecdotes.
2. The end of the axis classifies the rules for free, because reaching `eps_J` at `λ=1` is a
   property only a score-combining rule has.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#the-two-traps-this-grid-is-known-to-hit) ➡️

```
  grid two: the amount rises left to right, at seed 9

  λ =                0      0.25    0.5     0.75    1
  plain PoE        [ ]     [ ]     [ ]     [ ]     [=joint]     <- reaches it
  SuperDiff        [ ]     [ ]     [ ]     [ ]     [=joint]     <- reaches it
  corrector k=1    [ ]     [ ]     [ ]     [ ]     [not-an-identity]
  corrector k=5    [ ]     [ ]     [ ]     [ ]     [not-an-identity]
                    ^                               ^
              the rule alone               the free classifier:
              (the methods-alone           who can reach eps_J
               figure, for free)           and who cannot
```

## The two traps this grid is known to hit

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

Both are known in advance and both are handled rather than discovered.

**Rows deliver different absolute amounts at the same `λ`.**

`λ` is a fraction of that row's own `r_t^M`, and `‖r_t^M‖` differs per row. So a `λ=0.5` column is
matched in fraction and unmatched in absolute size, and a row that composes at `λ=0.5` may simply
have received more.
[The timing plan in the causal scope](../../../03-does-the-correction-cause-composition/plans/hypothesis/05-when-in-the-run-it-matters.md)
met exactly this and answered it by running a matched condition, `--mode matched`, implemented at
[interaction_term_dose_matched.py](../../../scripts/interaction_term_dose_matched.py). Either put
the delivered absolute total on each row's label, or run matched. Deciding which is task 1.3, and
it is decided before the render rather than after the grid looks confusing.

**The corrector rows will not reproduce the joint render at `λ=1`.**

The chain has already moved the latent off the joint trajectory, so injecting the full correction
at a point the joint model never visits does not land on the joint render. Those pictures are
labelled not-an-identity, on the picture itself, so a reader does not read them as failures. This is what makes the
`λ=1` column a free classifier of what each rule is: a row that fails to converge there is doing
something beyond combining scores.

## Description: what to build

⬅️ [Previous](#the-two-traps-this-grid-is-known-to-hit) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The generalised injection.** For rule `M`, inject `eps_M + λ·r_t^M` where
   `r_t^M = eps_J - eps_M`. Four rows: plain product-of-experts with `r_t`, SuperDiff with
   `r_t^SD`, the corrector at `k=1` with `r_t^(1)`, and the corrector at `k=5` with `r_t^(5)`.
2. **Grid one.** Columns are seeds 9 to 12, `λ` fixed at 0.75, the value
   [F5](../../../paper/iclr/figures/F5-one-dial-three-instruments.png) already reads at. Render the
   same grid at `λ=0` in the same pass, which gives the methods-alone figure for free.
3. **Grid two.** Columns are `λ ∈ {0, 0.25, 0.5, 0.75, 1}`, seed fixed at 9.

Both figures land in `paper/iclr/figures/how-much-is-added/across-composition-rules/`, each with a
`.json` sidecar and a `README.md` entry naming which rule produced which row.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objectives 5 and 7 of [the scope's direction](../../MASTER_PLAN.md): compare three composition
rules along one amount axis. The delivered absolute amount per row is made visible or matched, and
the pictures no rule can reach at `λ=1` are labelled rather than left looking like failures.

**Goals**

1. The generalised injection implemented and verified on one render per row.
2. Grid one rendered at `λ=0.75` and at `λ=0`, four rows by four seeds.
3. Grid two rendered across five `λ` values at seed 9.
4. Both figures filed under `across-composition-rules/` with sidecars and README entries.
5. The `λ=1` column classified per row, by eye, and recorded.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Renders write under
  `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/dose_across_rules/`, with a
  disk guard on `/datasets`, the filesystem actually written to, per
  [environment/storage.md](../../../../environment/storage.md). Only the finished figures and their
  sidecars go into the repo.
- biggpu allows one job per user, so the two grids run under `nohup` outside Slurm and `squeue` is
  blind to them. Harvest by `pgrep` on the session node, per
  [environment/hpc/execution-protocol.md](../../../../environment/hpc/execution-protocol.md).
- **The cached trajectories cannot be used.** SuperDiff and both corrector rows follow their own
  paths.
- The models run in fp16, and every norm (including `‖r_t^M‖` for the row labels) upcasts to fp32
  before it is taken.
- SDXL base, DDIM, 50 steps, guidance 7.5, latents 4×128×128 at 1024².
- No system LaTeX here, so figure PDFs come from matplotlib, per
  [environment/paper.md](../../../../environment/paper.md).

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-02-three-rules-on-one-dose-axis.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Confirm all four rows can produce a per-step prediction before any grid is planned.
  - Plain product-of-experts from [poe.py](../../../poe_repair/composers/poe.py), SuperDiff from
    [step 28's hook](05-what-changes-when-superdiff-leaves-its-own-defaults.md), and both corrector rows
    from `poe_repair/composers/poe_langevin.py`.
  - **Done when:** `r_t^M` is formed on one render for each of the four rows and its per-step norm
    printed, so a missing hook is found now rather than mid-grid.

▶ **Next: [task 1.1](#1--implement-the-generalised-injection)**.

### 1. 🔧 Implement the generalised injection

◀ **Needs: [task 0.2](#0--check-this-plan-before-working-from-it)**, so every row has a
prediction to build the correction from.

- [ ] **1.1** Implement the injection: for rule `M`, inject `eps_M + λ·r_t^M` where
      `r_t^M = eps_J - eps_M`.
  - **Done when:** at `λ=0` each row is byte-identical to that rule run alone, checked the same way
    [step 25](../tools/02-the-corrector-and-the-step-size-it-runs-at.md) checks its composer.
- [ ] **1.2** Verify at `λ=1` on the two product-of-experts-family rows that the prediction is
      `eps_J` to numerical tolerance.
  - **Done when:** the per-step difference is at fp32 noise level for those two rows. The corrector
    rows are expected to differ, and that expectation is recorded rather than treated as a failure.
- [ ] **1.3** Decide and record how the different absolute amounts per row are handled.
  - Either put the delivered absolute total on each row's label, or run the matched condition,
    `--mode matched`, from
    [interaction_term_dose_matched.py](../../../scripts/interaction_term_dose_matched.py),
    which [the timing plan](../../../03-does-the-correction-cause-composition/plans/hypothesis/05-when-in-the-run-it-matters.md)
    already used for this exact problem.
  - **Done when:** the choice and its reason are written into the review file **before** any grid
    renders, since it changes what each picture in the grid is.

▶ **Next: [task 2.1](#2--render-the-two-grids)**.

### 2. 🚀 Render the two grids

◀ **Needs: [task 1.3](#1--implement-the-generalised-injection)**, the matched-or-labelled decision,
because it changes what every picture is.

- [ ] **2.1** Grid one: columns are seeds 9 to 12, `λ` fixed at 0.75, four rows. Render the same
      grid at `λ=0` in the same pass.
  - Output goes to:
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/dose_across_rules/grid_one/`
  - **Done when:** 32 renders exist (4 rows × 4 seeds × 2 `λ` values), counted.
- [ ] **2.2** Grid two: columns are `λ ∈ {0, 0.25, 0.5, 0.75, 1}`, seed fixed at 9, four rows.
  - Output goes to:
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/dose_across_rules/grid_two/`
  - **Done when:** 20 renders exist, counted.
- [ ] **2.3** Label the pictures the corrector rows cannot reach at `λ=1` as not-an-identity, on
      the picture itself rather than in a legend or a caption.
  - **Done when:** the two corrector rows' `λ=1` pictures carry the label in the rendered figure.
- [ ] **2.4** Draw both figures into
      `paper/iclr/figures/how-much-is-added/across-composition-rules/`, not into the timing folder.
  - `rules-at-one-dose.png` from grid one, `rules-as-the-dose-rises.png` from grid two.
  - Each carries a `.json` sidecar recording rows, seeds, `λ` values, the `k` and `c` of the
    corrector rows, the delivered absolute amount per row, and the border rule.
  - Add a `README.md` entry per figure saying which rule produced which row.
  - **Done when:** both PNGs, both sidecars and the README entries exist. These figures are about
    how much is added, and the timing folder's name is a question about timing, which is why they
    do not go there.

▶ **Next: [instruction 3.1](#3--judge-the-two-grids-by-eye)**, the read the detector cannot
do on this pair.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-02-three-rules-on-one-dose-axis.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-02-three-rules-on-one-dose-axis.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [the check before moving on](#the-check-before-moving-on).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 3. 🖱️ Judge the two grids by eye

◀ **Needs: [tasks 2.1 and 2.2](#2--render-the-two-grids)**, both grids.

- [ ] **3.1** For grid two, walk the `λ=1` column across all four rows.
  - Expected result: the two product-of-experts-family rows land on the joint render. The corrector
    rows do not, and that is expected.
  - Record for each row whether it converged. That column is a free classifier of what each rule is
    doing, and it is the cheapest real finding in this plan.
- [ ] **3.2** For grid one, count composed renders per row by eye and compare against the
      detector's count.
  - [The timing verdict](../../../03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md)
    records that the detector and the eye disagree on cat and dog often enough that the eye read is
    the one cited, so do both and cite the eye where they differ.
- [ ] **3.3** Write both counts into the review file, side by side, naming every picture the two
      reads disagree on.
  - A summary count hides exactly the pictures a reviewer would ask about.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**, then
[step 30](../ideas/07-feynman-kac-correctors.md), which runs only if the corrector rows here
moved something.

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** these two grids are what a reviewer looks at when asking whether
> this paper's rule was compared against anything. A grid whose rows deliver different amounts, or
> whose unreachable pictures look like failures, argues against itself.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
DR=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/dose_across_rules

nohup bash scripts/mechanism_study/run_dose_across_rules.sh > "$DR/dose.log" 2>&1 &
pgrep -af 'dose_across_rules'          # squeue is blind to this

$PY -c "import json;d=json.load(open('$DR/grid_one/cells.json'));print(len(d['cells']),'cells')"
# expect 4 rows x 4 seeds x 2 lambda = 32
$PY -c "import json;d=json.load(open('$DR/grid_two/cells.json'));print(len(d['cells']),'cells')"
# expect 4 rows x 5 lambda = 20

ls -l paper/iclr/figures/how-much-is-added/across-composition-rules/
```

**Pass criteria**

- 32 renders in grid one, 20 in grid two.
- Every row is byte-identical to its rule alone at `λ=0`.
- The matched-or-labelled decision is recorded before the render, not after.
- Both figures, both sidecars and the README entries exist under `across-composition-rules/`.
- Instruction 3.3 has recorded both counts, naming every picture the two reads disagree on.

**Fail criteria (STOP)**

- A row is not byte-identical to its rule alone at `λ=0`. The injection is leaking and the whole
  axis is meaningless.
- The figures landed in `when-the-correction-arrives/`. Move them: that folder's name is a question
  about timing and these answer a question about how much is added.

**Partial pass guidance**

- A product-of-experts-family row failing to reach the joint render at `λ=1` is not a partial pass,
  it is a finding, and it means the injection is not doing what the arithmetic says. Investigate
  before drawing any conclusion from the grid.

**When you get results, answer**
[the review file](../../review/06-three-rules-on-one-amount-axis.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`, so there is no illustrated map to draw from.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `across-composition-rules/rules-at-one-dose.png` | — | rows are the four composition rules, columns are seeds 9 to 12 at `λ=0.75`, plus the same grid at `λ=0`. Each entry in the grid is the final generated picture, green border where the detector scored two separate animals | task 2.4 | ⏳ | **Supplementary, not main text.** It varies the seed, which the paper already establishes elsewhere. Sidecar records rows, seeds, `λ`, the corrector rows' `k` and `c`, and the delivered absolute amount per row |
| `across-composition-rules/rules-as-the-dose-rises.png` | — | rows are the four rules, columns are `λ ∈ {0, 0.25, 0.5, 0.75, 1}` at seed 9. The `λ=1` pictures the corrector rows cannot reach are labelled not-an-identity on the picture itself | task 2.4 | ⏳ | **Main text, if step 26 passed.** The `λ=1` column separates rules that combine scores from rules that do something else, which is a statement about the dynamics rather than a leaderboard |

**Two sentences these captions owe.** `λ` is a fraction of each row's own correction, so rows
deliver different absolute amounts unless the matched condition was run; whichever was chosen is named.
And the not-an-identity pictures are expected rather than failed, because the chain has already moved
off the joint trajectory.

### Organization workflow

1. Render both grids under `/datasets`.
2. Draw both figures into `paper/iclr/figures/how-much-is-added/across-composition-rules/`, each
   with its `.json` sidecar.
3. Add a `README.md` entry per figure naming which rule produced which row.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the `λ=1` classification | the review file, then [step 30](../ideas/07-feynman-kac-correctors.md), which reads it to decide whether it runs |
| a figure lands in `across-composition-rules/` | that folder's `README.md` gains an entry naming the rule and what produced it, per this repo's artifact rule |
| the matched-or-labelled decision | both captions, since it changes what a column means |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, both by `sync-plan-tree` rather than by hand |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-02-three-rules-on-one-dose-axis.md` | **task 0.1**, before any work | Conformance and thin instructions reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-02-three-rules-on-one-dose-axis.md` | **the close out**, after any red run | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-02-three-rules-on-one-dose-axis.md` | **the close out** | Statuses, running order and Error Matrix match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [scripts/interaction_term_dose_matched.py](../../../scripts/interaction_term_dose_matched.py) | the matched condition, `--mode matched`, which already answered the different-absolute-amounts problem once for this project |
| [poe_repair/composers/poe.py](../../../poe_repair/composers/poe.py) | the plain product-of-experts row, and the interface all four rows share |
| `poe_repair/composers/poe_langevin.py` | the two corrector rows, at `k=1` and `k=5` |
| `poe_repair/composers/superdiff.py` | the SuperDiff row and its per-step prediction, from [step 28](05-what-changes-when-superdiff-leaves-its-own-defaults.md) |
| [paper/iclr/figures/F5-one-dial-three-instruments.png](../../../paper/iclr/figures/F5-one-dial-three-instruments.png) | the existing one-dial figure, whose `λ=0.75` grid one reuses so the two are comparable |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[Step 30, the Feynman-Kac correctors](../ideas/07-feynman-kac-correctors.md). It runs only if
the corrector condition moved something; otherwise it closes unrun with the reason recorded.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>3 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🔴 rows compared at the same `λ` received different absolute amounts

**When it happens:** any grid comparing rules across correction amounts, because `λ` is a fraction of each row's own
correction and `‖r_t^M‖` differs per row.
**What you see:** a row that composes earlier than another, read as the rule being better.
**Why:** the axis is matched in fraction and unmatched in absolute size.
**How to fix:** task 1.3, decided before the render. Put the delivered total on the row label, or
run the matched condition, `--mode matched`.

#### 🟡 not-an-identity pictures read as failures

**When it happens:** the corrector rows at `λ=1`.
**What you see:** two pictures that do not match the joint render, in a column where the other two do.
**Why:** the chain has already moved the latent off the joint trajectory.
**How to fix:** put the label on the picture itself, per task 2.3, and say in the caption that it is
expected and what it classifies.

#### 🟡 figures about how much is added, filed in the timing folder

**When it happens:** filing by "which scope made it" rather than by what the figure asks.
**What you see:** one of these grids under `when-the-correction-arrives/`.
**Why:** both folders hold this scope's output, so the wrong one is one keystroke away.
**How to fix:** `across-composition-rules/` under `how-much-is-added/`. The timing folder's name is
a question about timing.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
