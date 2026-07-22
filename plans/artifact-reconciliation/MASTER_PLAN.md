# Artifact Reconciliation

## Mission

Keep every saved file this project produces (LoRA checkpoints, training and eval caches, saved results and figures) easy to find, trustworthy, and organised one consistent way, across the two storage roots: the repo (`/home-mscluster/mmolefe/Playground/PhD/poe_repair_min`) and `/datasets/mmolefe/poe_repair_min/outputs`. Any file we keep should be findable by experiment/rung/pair/seed, should load, and should carry a clear keep / re-run / discard label.

## Objectives

1. Maintain a current inventory of what artifacts exist, where they live, and which run produced them, reconciled against the W&B run history.
2. Verify integrity: checkpoints load, caches are complete (shards vs manifests), result sets match their run manifests.
3. Keep a single canonical naming and directory scheme keyed by rung → experiment → pair → seed, with backward-compat symlinks so nothing breaks on move.
4. Re-reconcile as new runs land, so the inventory, integrity report, and canonical layout never go stale.

## Goals

1. Inventory tables (`inventory/01-artifact-inventory.md`, `inventory/02-two-root-classified.md`) reflect the current disk + W&B state, regenerable via `inventory/scripts/01_inventory.py`.
2. An integrity-and-disposition report (`inventory/03-integrity-and-disposition.md`) load-tests every kept run's headline checkpoint plus all suspects, checks cache completeness, and assigns keep / re-run / discard per artifact.
3. The canonical `artifacts/` layout (`rung1-overfit` … `rung4-scale`, `caches/`, `_shared/`, `_quarantine/`) is applied on both roots via `inventory/scripts/04_apply_layout.sh`, with compat symlinks at every old path and a re-run reporting 0 residual actions.
4. A standing re-sweep exists that folds new W&B runs and new checkpoints into the inventory, integrity report, and canonical layout with no artifact left unfiled.

## Expected Outcome

The artifact tree is navigable by rung/experiment/pair/seed regardless of which root holds the bytes; every surviving artifact has been load-tested and dispositioned; discards are quarantined (reversible), not deleted; and new runs get reconciled and filed rather than accumulating as an untracked sprawl.

## Definition of Done

1. ✅ `python inventory/scripts/01_inventory.py` regenerates the inventory, and `01`/`02` tables match current disk + the four W&B projects (`poe-repair-lora`, `poe-repair-cross-seed`, `poe-repair-cross-pair`, `poe-repair-group-a`).
2. ✅ `python inventory/scripts/03_integrity.py` reports every tested checkpoint loading (LoRA keys present, non-empty shapes) and `03b_cache_check.py` reports cache completeness; `inventory/03-integrity-and-disposition.md` carries a per-artifact keep / re-run / discard row.
3. ✅ `bash inventory/scripts/04_apply_layout.sh` (dry-run) reports 0 residual actions on the applied tree; the `artifacts/` scheme exists on both roots with compat symlinks resolving.
4. ⚠️ The standing re-sweep plan (`plans/05-resweep-on-new-runs.md`) is in place and has been run at least once against the current run set with no unfiled artifact.

## Sub-Scopes

(None yet.)

## Plans

- ✅ 01-data-inventory.md
- ✅ 02-two-root-classified-sweep.md
- ✅ 03-data-integrity-check.md
- ✅ 04-canonical-layout-reorg.md
- ⚠️ 05-resweep-on-new-runs.md (standing / recurring)
