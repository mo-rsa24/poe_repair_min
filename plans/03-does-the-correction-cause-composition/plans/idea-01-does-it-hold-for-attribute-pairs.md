# 🗺️ The composition-type scatter

This plan asks whether the size of the dropped correction predicts which kinds of two-concept
prompt plain PoE can already handle, including pairs made of an object and an attribute.

**No step number: nothing waits on this.** Background, listed in the background-experiments pool of the [repo root MASTER_PLAN.md](../../../MASTER_PLAN.md), which also holds the one `## Running order` table. This plan asks whether attribute pairs behave like object pairs. Its scaffold waits at `artifacts/plans/parked/composition-type-cells`.

## What this asks, in one line
Does the correction behave the same when the two concepts are an object and an attribute rather than two animals? This is background work: it widens the claim's reach and blocks nothing.

## Description
One dot per pair. The x axis is the normalized correction size and the y axis is plain PoE's own
[compose rate](../../../context/world/compose-rate.md), with the dot colored by regime: the
intersection is the target, a blend exists but is not the target, or the intersection is empty.
The 76 cached pairs cover two and a half of those regimes. The attribute-and-object runs that are
still missing, along with the separate way of judging whether they succeeded, are a sub-scope.

## Purpose
Goal 5 is the paper's predictive claim: the size of the dropped term predicts which composition
types PoE can do. This plan turns that framing into one measured plot. Serves DoD 7.

## Goal
The scatter with regime coloring, plus the strip of example images at λ=0, one per regime.

## Environment Facts This Plan Depends On
- Requires plan 01's committed normalization; this plot must not exist before
  that memo does.
- The validated compose-scorer counts objects. That is correct for co-occurrence pairs and wrong
  for attribute pairs, because "a red cube" succeeding still shows one object. The sub-scope
  builds and validates the separate measuring tool those pairs need.
- The scatter itself computes in-session from the cache once those runs exist.

## Tasks
- [ ] regime labels for the 76 cached pairs (rule-based from prompt
      structure, spot-checked by eye)
- [ ] λ=0 compose rate per cached pair over its seeds (scorer on cached or
      cheaply regenerated λ=0 outputs)
- [ ] build the attribute-and-object runs and the separate way of judging
      whether they succeeded  → decomposed: see `artifacts/plans/parked/composition-type-cells/MASTER_PLAN.md`
- [ ] the scatter with regime coloring; report the ordering result against
      Goal 5's rule
- [ ] the strip of example images at λ=0, one per regime

## Success/Failure Outcomes
- **the scatter**
  - Success: types order along a falling curve under the pre-registered
    normalization (or the null is reported under that same normalization).

    > A null here means the three composition types sit at the same height on the
    > scatter, so knowing the type tells you nothing about the correction's size.
  - Failure: the relationship appears in only one normalization. Report both and
    adopt neither, which is what Goal 5 counts as inconclusive.

## Recommended skill
▶ `/design-figure` ✅ for the scatter's final form; `/demonstrate` ✅ for the
   strip of example images.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY scripts/composition_scatter.py   # requires report/normalization_preregistration.md
# expect: scatter written + Spearman rho printed per regime ordering
```
