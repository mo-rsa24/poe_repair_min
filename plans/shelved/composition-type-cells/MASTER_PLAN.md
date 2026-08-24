# Composition-Type-Cells

> Shelved: an empty scaffold for background work (zero plan files). Comes back when its parent
> plan is promoted out of the background pool into the paper table; /populate-plans it then.

## Mission
The composition-type scatter needs the one regime the cache lacks: pairs where
both-at-once IS the target, like a red cube. This scope curates those cells,
caches their residual trajectories in the training_cache schema, and builds
and validates the success instrument that can read them, because the existing
instance-count scorer wrongly fails a one-object attribute success.

## Objectives
1. Curate: ~10 attribute×object pairs plus a few regime-3 extremes (no
   plausible single thing is both), prompts finalized.
2. Cache: residual trajectories per cell (inference-only: Mono + marginal
   passes), same schema as training_cache.
3. Instrument: a regime-1 success reader (VQA or CLIP-match), built and
   validated before any scatter point uses it.

## Goals
1. Pair list finalized: prompts, regime labels, seeds; written to a yaml.
2. Cache complete: every cell carries the four eps branches per step,
   bulk-smoke passes.
3. Instrument validated: agreement with a hand-labeled set of ~30 outputs at
   or above 90%, rejection of weaker candidate reads recorded (the
   compose-scorer null protocol pattern).
4. Contract emitted: instrument_validated.json, the machine-checkable file
   plan 07 halts on if absent (the scorer_validated.json seam pattern).

## Expected Outcome
Plan 07 can place every regime on the scatter with a defensible instrument
per regime, and the missing-regime cells exist in the same cache the rest of
the program reads.

## Definition of Done
1. pair yaml committed with prompts and regime labels.
2. All cells cached and bulk-smoke-checked.
3. Instrument validation report written, weaker candidates' rejection recorded.
4. instrument_validated.json present and referenced by plan 07's tasks.

## Sub-Scopes
(none)

## Plans
(to be populated)

## Environment Context
See `environment/00-INDEX.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.
