# Coverage Manifest: data-inventory-artifact-catalogue-2026-07-21

Every checkbox traces to a unit in the captured markdowns. Downstream
(`/data-integrity-check`, keep/re-run/discard workflow) should touch every item.

## Segment 1: /data-inventory — experiments + W&B reconcile
- [ ] Verbatim prompt (with unfilled W&B placeholder) — `segment-1-.../prompt.md`
- [ ] Method: W&B project + status recovered from local run dirs — `segment-1-.../response.md`
- [ ] Four W&B projects named (`poe-repair-group-a/-lora/-cross-seed/-cross-pair`, entity `prime_lab`) — same file
- [ ] Status taxonomy: finished / died-early / false-start / benign sync-fatal — same file
- [ ] Reconciliation table (16 experiment rows, per-run id·step·state) — same file
- [ ] Suspect runs identified: 0y9un0o4, ow1jo0xq(train), lu7g7svh, 9ux1sm67, 6enlob54, d5b2706v — `segment-1-.../00-overview.md`
- [ ] Finding: 3 empty `lora` pair stubs used downstream as heldout targets — `segment-1-.../response.md`
- [ ] Finding: `group_a_failure` runs record old module name `group_a_corrector` — same file
- [ ] Finding: W&B `_step` ≠ checkpoint step in cross_pair — same file
- [ ] Finding: no `.safetensors`/`.bin`; all repo checkpoints are `.pt` — same file
- [ ] Deliverable: `inventory/01-artifact-inventory.md` — same file
- [ ] Deliverable: `inventory/scripts/01_inventory.py` (re-runnable, verified) — same file
- [ ] Caveat: local reconciliation, not live W&B server query — same file

## Segment 2: two-root classified sweep
- [ ] Verbatim prompt — `segment-2-.../prompt.md`
- [ ] Scope finding: only cross_seed (7.6G) + training_cache (22G) of the 3 under `/datasets/mmolefe` — `segment-2-.../response.md`
- [ ] Finding: `/datasets/mmolefe/poe_repair_min/outputs` is a separate tree, not a mirror — same file
- [ ] Cache ownership: `training_cache` read by both pooling experiments (grep-confirmed) — same file
- [ ] Cache ownership: `manifold_cache` read only by `scripts/manifold/*` — same file
- [ ] Classified table (path → type → group → owning-exp → size → last-mod) — same file
- [ ] training_cache 8 train pairs enumerated — same file
- [ ] Duplicate 1: cross_seed pair-name collision across roots (not byte-dups) — same file
- [ ] Duplicate 2: task_b `k04` (datasets) vs `k01` (repo) — same file
- [ ] Duplicate 3: lora pair names reappear as cross_seed names — same file
- [ ] Orphan: 3 empty `lora` pair stubs — same file
- [ ] Orphan: `training_cache_overfit_catdog` empty dir — same file
- [ ] Orphan: `manifold_cache` (owned by scripts/manifold) — same file
- [ ] Out-of-scope list (synthesizer, veracity*, idea*, e_*; other-project roots) — same file
- [ ] Deliverable: `inventory/02-two-root-classified.md` — same file
- [ ] Caveats: 22G training_cache not fully walked; duplicates by name+size+mtime not checksum — same file

## Session-level
- [ ] Session overview + cross-references (segment 2 builds on segment 1) — `00-session-overview.md`
- [ ] Open ends: no server-side W&B diff, no checksum dedup, no integrity check yet — same file
