# Session Overview: data-inventory-artifact-catalogue-2026-07-21
Captured: 2026-07-21

## What was studied
A data-inventory / artifact-reconciliation session over the `poe_repair_min`
repo and the shared `/datasets/mmolefe` storage. Two exchanges: (1) catalogue
every artifact under `poe_repair/experiments/` and reconcile it against the
project's W&B runs, producing an experiment → runs → status → surviving-artifact
table; (2) sweep two roots and classify every artifact into
datasets / LoRA-checkpoints / caches / saved-results, grouped by the three
LoRA-pooling experiments, with duplicates and orphans flagged.

This is not a learning session (no learn-concept / unpack / reconcile / socratic).
Both segments are freeform data-inventory work. Captured so a downstream session
can reference the prompts and the full responses when running a
`/data-integrity-check` + keep/re-run/discard workflow on the same artifacts.

## Sequence of skills
1. `/data-inventory` (freeform) — catalogue `poe_repair/experiments/` + reconcile vs W&B
2. freeform — two-root sweep + classified table + duplicates/orphans

## Cross-references
- Segment 2 builds directly on segment 1: segment 1 established the four W&B
  projects and per-run status (finished / died-early / false-start / benign
  sync-fatal); segment 2 reused that status to decide which artifacts are the
  "kept" ones and where duplicates/orphans sit across the two roots.
- Both segments produced on-disk deliverables in `inventory/`:
  - `inventory/01-artifact-inventory.md` + `inventory/scripts/01_inventory.py`
    (re-runnable) — from segment 1.
  - `inventory/02-two-root-classified.md` — from segment 2.
- The W&B run-status table in segment 1 is the load-bearing input for any
  downstream integrity check: the "died-early" and "false-start" runs are the
  ones whose artifacts should be treated as suspect.

## Open ends
- Reconciliation was against **local** W&B run dirs, not a live server query. A
  server-side diff (`prime_lab/<project>`) was offered but not run.
- Duplicate detection used name + size + mtime, not checksums. A hash /
  `rsync -n` diff of the cross_seed pair dirs across the two roots was offered
  but not run.
- The 22G `training_cache` was not walked in full (recursive `du` times out);
  its size and subdir dates came from `ls`/`stat`.
- No integrity verification was performed yet (checkpoint load test, shard
  completeness, result-vs-manifest match). That is the intended next step and
  the subject of the downstream `/data-integrity-check` prompt.
