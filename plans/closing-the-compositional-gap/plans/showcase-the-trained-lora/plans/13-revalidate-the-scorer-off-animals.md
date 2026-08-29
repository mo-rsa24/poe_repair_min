# 🔬 Revalidate the scorer off animals

**Step 43 in the root running order. Waits on: nothing. Next: back to [05-assemble-the-showcase-figures](05-assemble-the-showcase-figures.md), which closes the scope.**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 42 (previous) | [12-close-f8a-and-the-oracle-panel](12-close-f8a-and-the-oracle-panel.md) | The tail and the ceiling |
| **43 (current)** | **13: revalidate-the-scorer-off-animals** | The validation round that opens tier-three captions |
| 35 (closes scope) | [05-assemble-the-showcase-figures](05-assemble-the-showcase-figures.md) | The wall, assembled last |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The instrument question:** the instance-count scorer (GroundingDINO, count ≥ 2) is validated for two-animal scenes only. The population ladder's tier three, "SDXL composition generally", is claimable only once the scorer's verdicts are trusted on scenes that are not two animals: the four non-animal cells already generated (dog × oil-painting-style, dolphin × ocean-wave, mailbox × snowfield, typewriter × cactus).

**What validation means here, copying the pattern that produced `scorer_validated.json`:** a labelled mini-set, the scorer's read against the eye's, and a bar in code before the reading.

**If it validates:** tier-three captions open; the four cells get scored and enter the population figures.

**If it does not:** tier three stays closed, the captions keep their animal-pairs reach, and that is a recorded scope decision, not a failure.

**Associated materials:**
- **Review questions:** [../review/13-revalidate-the-scorer-off-animals.md](../review/13-revalidate-the-scorer-off-animals.md)
- **Ledger entry:** [the population ladder](../decisions-taken-here.md#what-every-figure-claims-over-population)
- **The cells:** under `/datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/` (`a_dog__x__oil_painting_style`, `a_dolphin__x__an_ocean_wave`, `a_mailbox__x__a_snowfield`, `a_typewriter__x__a_cactus`)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** one validation round: labelling a mini-set by eye (the human half), one scoring pass (minutes). No training, no sweep.

**The hard part is conceptual, not computational:** "compose" for dog × oil-painting-style is not an instance count; the labelling step must first write down what a pass looks like per non-animal pair kind, and that definition is the real deliverable.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The `co3` python; scorer weights as in `scorer_validated.json`'s provenance; outputs to `/datasets` ([overview](../../../../../environment/overview.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A validation verdict that either opens tier-three captions or closes them with a reason.** Bars in code, per the repo's rule, before any verdict is read.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** The generated non-animal cells are unusable evidence while the scorer's reach stops at two-animal scenes; scoring them anyway would put uncheckable numbers in the paper.

**The solution.** The same validation pattern that earned `scorer_validated.json`, run on the new scene kinds.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The pass definition** per non-animal kind (object × style, object × scene), written before labelling; one page beside this plan.
2. **The labelled mini-set**: N renders per kind labelled by eye against the definition [owner: human].
3. **The scoring pass and the agreement read**: scorer vs labels, agreement rate per kind, the bar in code (copy the `MIN_MEDIAN_RATIO` pattern: threshold in source, visible in diff).
4. **The verdict artifact**: `scorer_validated_offanimals.json`, mirroring the original's shape.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 4 of the Definition of Done (re-validation verdict recorded; tier-three opened or declined). Checkable outcomes:

1. The pass definition exists per kind.
2. The agreement rates computed against an in-code bar; the verdict artifact written.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/13-revalidate-the-scorer-off-animals.md`**

▶ **Next: [task 1.1](#1--define-score-compare)**.

### 1. 🔬 Define, score, compare

◀ **Needs: nothing.**

- [ ] **1.1 Write the pass definition per kind** (`plans/13-pass-definitions.md` beside this file): what "composed" means for object × style and object × scene, with one example render named per outcome.
- [ ] **1.2 Build the labelling sheet**: sample renders per kind from the four cells' folders into one contact sheet with blank label columns.
- [ ] **1.3 Score the same renders** with the scorer (and any adapted prompt/class configuration it needs off animals); record the configuration verbatim in the sidecar.
- [ ] **1.4 Compute agreement against the labels once they exist; bar in code** (`scripts/showcase/offanimal_validation.py`, threshold constant at top). Write `scorer_validated_offanimals.json`.

▶ **Next: [instruction 2.1](#2--label-and-judge)** (the labels are the human half).

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

### 2. 👁️ Label and judge

◀ **Needs: [tasks 1.1 to 1.2](#1--define-score-compare)** done, so the definition and sheet exist.

2.1 **Label the contact sheet** against the pass definition, one label per render, no peeking at scorer output [owner: human]. ✅ every render labelled; ambiguous cases get their own mark and are excluded from the bar, counted in the sidecar.

2.2 **Read the agreement rates** from task 1.4's output against the in-code bar. ✅ bar met per kind: tier three opens for that kind; ❌ bar missed: tier three stays closed for that kind, recorded as a decision.

2.3 **Write the verdict** into the [review file](../review/13-revalidate-the-scorer-off-animals.md).

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> A caption's reach is bought here or not at all. Scoring the non-animal cells without this verdict would put numbers in the paper nobody can check.

**Pass criteria:**
- Definitions written before labels; labels before agreement; bar in code before reading.

**Fail criteria:**
- Any ordering violation above (the whole point of the pattern), or the scorer configuration off animals left unrecorded.

**When you get results, answer the open questions in the [review file](../review/13-revalidate-the-scorer-off-animals.md).**

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| (none; the deliverable is a verdict artifact) | — | — | — |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| scorer_validated_offanimals.json | — | agreement per kind, bar, verdict (sidecar) | task 1.4 | ⏳ |
| the labelled contact sheet | — | the human labels the bar was judged against | instruction 2.1 | ⏳ |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.3](#2--label-and-judge)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/`**

▶ **Next: [05-assemble-the-showcase-figures](05-assemble-the-showcase-figures.md)**, the wall, which closes the scope.

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**Pattern:** `scorer_validated.json`'s provenance and the `MIN_MEDIAN_RATIO` in-code-bar idiom from the mechanism re-probe scorer.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (opens tier-three captions):

```
Execute plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/13-revalidate-the-scorer-off-animals.md: tasks 1.1 to 1.3 only (pass definitions, the labelling contact sheet, the scoring pass with its configuration recorded); stop before agreement. The labels are the user's to do blind; afterwards one follow-up prompt computes agreement against the in-code bar and writes scorer_validated_offanimals.json.
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[05-assemble-the-showcase-figures](05-assemble-the-showcase-figures.md) runs last and closes the scope; its captions take whatever tier this plan's verdict allows.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
