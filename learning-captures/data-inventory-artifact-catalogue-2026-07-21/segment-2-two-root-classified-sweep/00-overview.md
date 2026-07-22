# Segment 2 Overview: two-root classified sweep

**Type**: freeform data-inventory
**Ask**: sweep `/datasets/mmolefe` + the repo; classify every artifact into
datasets / LoRA-checkpoints / caches (training, eval, group) / saved-results,
grouped by owning experiment (lora, cross_seed_lora_pooling,
cross_pair_lora_pooling); one table path → type → group → owning-experiment →
size → last-modified; flag duplicates and orphans.

**Key outputs**:
- Only `cross_seed_lora_pooling` (7.6G) and `training_cache` (22G) of the three
  experiments live under `/datasets/mmolefe`; no `lora` / `cross_pair` there.
- `/datasets/mmolefe/poe_repair_min/outputs` is a separate larger output tree,
  not a mirror of repo outputs.
- `training_cache` (22G) = shared training + group (`manifest_A/B/H`) + eval
  (`heldout/`) cache read by both pooling experiments.
- `manifold_cache` owned by `scripts/manifold/`, an orphan w.r.t. the three.
- Duplicates: cross_seed pair-name collisions across roots are different-stage
  artifacts (heavy banks vs light eval), not byte-copies.
- Orphans: 3 empty `lora` pair stubs, empty `training_cache_overfit_catdog`,
  and `manifold_cache`.
- Deliverable: `inventory/02-two-root-classified.md`.

Files: `prompt.md` (verbatim ask), `response.md` (full method + classified table
+ duplicates + orphans + out-of-scope + deliverable + caveats).

**Load-bearing for downstream**: this segment defines the set of "kept" artifacts
and where they physically live across two roots — the exact input a
`/data-integrity-check` step verifies (checkpoint load, shard completeness,
result-vs-manifest match).
