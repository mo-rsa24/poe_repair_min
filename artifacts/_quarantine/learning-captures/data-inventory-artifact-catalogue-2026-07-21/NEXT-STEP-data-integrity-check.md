# Next step: /data-integrity-check (paste into the downstream session)

This prompt continues the inventory session captured in this folder. Read the
capture first so you have the artifact list, W&B statuses, and the two roots.

---

/data-integrity-check the "kept" artifacts identified in this repo's inventory
session. Context is captured at
`learning-captures/data-inventory-artifact-catalogue-2026-07-21/` — read
`00-session-overview.md`, then `segment-1-.../response.md` (W&B run-status table)
and `segment-2-.../response.md` (two-root classified table). The machine-readable
inventories are `inventory/01-artifact-inventory.md`,
`inventory/02-two-root-classified.md`, and the re-runnable
`inventory/scripts/01_inventory.py`.

Two roots hold the kept artifacts: the repo
(`/home-mscluster/mmolefe/Playground/PhD/poe_repair_min`) and
`/datasets/mmolefe/poe_repair_min/outputs`. Scope the check to the three
experiments `lora`, `cross_seed_lora_pooling`, `cross_pair_lora_pooling` plus the
shared `training_cache`.

Verify, per artifact:
1. **Checkpoints load** — `torch.load` (weights-only) each `lora_step_*.pt` /
   `best.pt` / `last.pt`; confirm the LoRA tensor keys are present and shapes are
   non-empty. Kept checkpoints include: `outputs/lora/{cat_dog,
   a_typewriter__x__a_cactus, a_camel__x__a_desert_landscape}/…/lora_step_080000.pt`,
   `outputs/cross_pair_lora_pooling/{all_groups,within_group}/…/lora_step_030000.pt`,
   `outputs/cross_seed_lora_pooling/task_c_per_seed_ceiling/…/lora_step_016510.pt`,
   task_b checkpoints, and the datasets-root cross_seed seed banks
   (`/datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/*`).
2. **Caches complete (no partial/truncated shards)** — for `training_cache`
   (22G) check every `.npy`/`.pt` shard opens and matches the count/keys in its
   manifest (`manifest.json`, `manifest_A/B/H.json`, `manifest_all.json`); check
   `train/` (8 pairs) and `heldout/` shard counts against the manifests; do the
   same for `outputs/manifold_cache/` against `inventory.json`.
3. **Result sets match their run manifests** — for each run dir, confirm
   `config.json` / `dataset_meta.json` / `history.json` / `verdict.json` /
   `lora_attach.json` are internally consistent (e.g. task_b `verdict.json` =
   "ok"; cross_pair `within_group` `eval_crossbar/step_020000/manifest.json`
   claims 43 cells → confirm 43 sampled images exist).

Reconcile against the W&B run status from segment 1 and mark as **suspect** any
artifact whose run did not cleanly finish:
- `0y9un0o4` — cross_pair `all_groups/main`, died-early.
- `ow1jo0xq` (train copy, run-20260528_044614) — cross_pair `within_group/g6`,
  died-early.
- `lu7g7svh` — lora `a_typewriter__x__a_cactus`, false-start (step 0).
- `9ux1sm67` / `6enlob54` — cross_pair `all_groups`, false-start / early-stop.
- `d5b2706v` — cross_seed `task_c`, benign W&B-upload sync-fatal; checkpoint
  reached step 16510 so load-test it explicitly to confirm intact.
Also treat the known 0-byte stubs as suspect-by-default:
`outputs/lora/{a_dog__x__oil_painting_style, a_dolphin__x__an_ocean_wave,
a_mailbox__x__a_snowfield}` and
`/datasets/mmolefe/poe_repair_min/outputs/training_cache_overfit_catdog`.

Produce one table: artifact path → integrity check (load / shards / manifest) →
pass/fail → W&B run status → suspect? → **decision (keep / re-run / discard)**
with a one-line reason per suspect. Keep it read-only; do not delete or
re-run anything, just recommend. Save the result to
`inventory/03-integrity-and-disposition.md` so it chains with 01 and 02.

‖ decide keep / re-run / discard per suspect artifact ‖

---

## Why these edits (vs the original one-liner)
- **Points at the capture + inventory files** so the new session inherits the
  artifact list, W&B statuses, and roots without re-deriving them.
- **Names the two roots and scopes to the 3 experiments + training_cache** — the
  original "kept artifacts from step 2" is meaningless in a fresh session.
- **Makes each check concrete** (what "load", "complete shards", "match
  manifest" mean here, with the actual manifest filenames and the 43-cell
  example).
- **Enumerates the suspect runs by id** from segment 1 instead of the vague "any
  artifact whose run crashed", and folds in the 0-byte orphans.
- **Names the output file** (`inventory/03-…`) so 01 → 02 → 03 form a workflow.
- **Read-only guardrail** so the check itself can't destroy an artifact before
  the keep/re-run/discard decision is made.
