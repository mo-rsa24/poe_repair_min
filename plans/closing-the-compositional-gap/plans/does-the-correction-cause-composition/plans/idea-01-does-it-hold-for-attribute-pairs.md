# 🗺️ The composition-type scatter

**No step number: nothing waits on this.** Background, listed in the background-experiments pool of the [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md), which also holds the one `## Running order` table. This plan asks whether attribute pairs behave like object pairs. Its scaffold waits at `plans/shelved/composition-type-cells`.

## What this asks, in one line
Does the correction behave the same when the two concepts are an object and an attribute rather than two animals? Background: widens the claim's reach, blocks nothing.

## Description
One dot per pair: normalized correction size (x) against plain PoE's own
compose-rate (y), colored by regime (intersection is the target; a blend
exists but is not the target; the intersection is empty). The cached 76 pairs
cover two and a half regimes; the missing attribute×object cells and their
separate success instrument are a sub-scope.

## Purpose
The predictive second claim (Goal 5): the size of the dropped term predicts
which composition types PoE can do. Turns the paper's framing into one
measured plot. Serves DoD 7.

## Goal
The money scatter with regime coloring plus the per-regime exemplar strip at
λ=0.

## Environment Facts This Plan Depends On
- Requires plan 01's committed normalization; this plot must not exist before
  that memo does.
- The validated compose-scorer counts objects: correct for co-occurrence
  pairs, WRONG for attribute cells ("a red cube" succeeding is one object).
  The sub-scope builds and validates the separate instrument.
- Scatter itself computes in-session from cache once cells exist.

## Tasks
- [ ] regime labels for the 76 cached pairs (rule-based from prompt
      structure, spot-checked by eye)
- [ ] λ=0 compose-rate per cached pair over its seeds (scorer on cached or
      cheaply regenerated λ=0 outputs)
- [ ] build the attribute×object cells and their separate success
      instrument  → decomposed: see `plans/shelved/composition-type-cells/MASTER_PLAN.md`
- [ ] the scatter with regime coloring; report the ordering result against
      Goal 5's rule
- [ ] per-regime exemplar strip at λ=0

## Success/Failure Outcomes
- **the scatter**
  - Success: types order along a falling curve under the pre-registered
    normalization (or the null is reported under that same normalization).
  - Failure: the relationship appears in only one normalization. Report both,
    adopt neither (Goal 5's inconclusive arm).

## Recommended skill
▶ `/design-figure` ✅ for the scatter's final form; `/demonstrate` ✅ for the
   exemplar strip.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY scripts/composition_scatter.py   # requires docs/normalization_preregistration.md
# expect: scatter written + Spearman rho printed per regime ordering
```
