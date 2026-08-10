# 🔍 Data Integrity Check + Disposition

## Description

Verify the "kept" artifacts identified by `01`/`02` are actually clean: checkpoints load, caches are complete, result sets match their run manifests. Reconcile against the W&B run status, mark suspect artifacts, and assign a keep / re-run / discard decision per artifact. Read-only: recommend, never delete or re-run.

## Purpose

An inventory says what exists; this says whether it can be trusted. It is the check before any move or delete. A `died-early` run's checkpoint may still load fine, and a "0-byte" folder may actually be a working symlink. Look before you act is the whole point.

## Goal

`inventory/03-integrity-and-disposition.md`: one table of artifact → integrity check (load / shards / manifest) → pass/fail → W&B status → suspect? → decision (keep / re-run / discard) with a one-line reason per suspect. Backed by re-runnable `inventory/scripts/03_integrity.py` (checkpoints) and `03b_cache_check.py` (caches).

## Tasks

- [x] ✅ Load-test the kept checkpoints + all suspects; check caches vs manifests; check run-dir JSON consistency.
- [x] ✅ Reconcile against W&B suspects and 0-byte stubs; assign keep / re-run / discard.
- [x] ✅ Write `inventory/03-integrity-and-disposition.md` + the two scripts.

Fully-qualified prompt (invoke via `/data-integrity-check`):

```
/data-integrity-check the "kept" artifacts from this repo's inventory session
(inventory/01-artifact-inventory.md, inventory/02-two-root-classified.md). Scope:
experiments lora, cross_seed_lora_pooling, cross_pair_lora_pooling + shared
training_cache, across two roots (repo + /datasets/mmolefe/poe_repair_min/outputs).

Verify per artifact:
1. Checkpoints load — torch.load(weights_only) each lora_step_*.pt / best.pt /
   last.pt; confirm LoRA tensor keys present and shapes non-empty. Load-test the
   final checkpoint of each kept run dir plus every suspect (not all ~230).
2. Caches complete — for training_cache (22G) check each cell has embeddings.pt +
   meta.json + residuals/step_000..049.pt (50 shards); spot-open a sample of
   shards; check manifold_cache against inventory.json. Note: build_eval_cache.py
   cells legitimately have ONE zeroed residual shard (held-out-pair eval stubs) —
   not truncation.
3. Result sets match run manifests — config.json / dataset_meta.json /
   history.json / verdict.json / lora_attach.json internally consistent (task_b
   verdict.json = "ok"; within_group g6 eval_crossbar manifest n_cells claim =
   sampled images).

Reconcile against W&B and mark suspect any run that did not cleanly finish:
0y9un0o4 (cross_pair all_groups/main, died-early), ow1jo0xq (within_group/g6,
died-early), lu7g7svh (lora typewriter, false-start step 0), 9ux1sm67 / 6enlob54
(all_groups false-start / early-stop), d5b2706v (task_c benign sync-fatal — load-
test step 16510 explicitly). Also treat known 0-byte stubs as suspect-by-default
(outputs/lora/{a_dog__x__oil_painting_style,a_dolphin__x__an_ocean_wave,
a_mailbox__x__a_snowfield}, training_cache_overfit_catdog) — but LOOK at them:
du/find -type f do not traverse symlinks, so a symlink view reads as 0 bytes.

Produce one table: artifact → integrity (load / shards / manifest) → pass/fail →
W&B status → suspect? → decision (keep / re-run / discard) with a one-line reason.
Read-only; recommend only. Save to inventory/03-integrity-and-disposition.md.
```

## Recommended skill

`/data-integrity-check` ✅

## Engagement Instructions

```
$ CUDA_VISIBLE_DEVICES="" python inventory/scripts/03_integrity.py
# Expect: every tested checkpoint PASS (420 lora keys, shape (8,640)), incl. the
# four suspects (bytes intact; suspect = run completeness, not corruption).
$ CUDA_VISIBLE_DEVICES="" python inventory/scripts/03b_cache_check.py 2>&1 | tail -3
# Expect: 645 cells, 24 "1-shard" cells (the by-design eval stubs), 0 real gaps.
$ grep -c "keep\|re-run\|discard" inventory/03-integrity-and-disposition.md   # per-artifact dispositions present
```
