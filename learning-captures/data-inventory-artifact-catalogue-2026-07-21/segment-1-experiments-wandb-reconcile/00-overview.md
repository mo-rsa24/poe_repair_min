# Segment 1 Overview: experiments + W&B reconcile

**Type**: freeform data-inventory (invoked via `/data-inventory`)
**Ask**: catalogue every artifact under `poe_repair/experiments/` (run outputs,
LoRA checkpoints, caches, eval results) and reconcile against the project's W&B
projects; produce one table experiment → runs → status → surviving-artifact-path.

**Key outputs**:
- Four W&B projects recovered from local run dirs: `poe-repair-group-a`,
  `poe-repair-lora`, `poe-repair-cross-seed`, `poe-repair-cross-pair` (entity
  `prime_lab`).
- Per-run status method (finished / died-early / false-start / benign
  sync-fatal), derived from debug logs.
- Reconciliation table for 16 experiment rows.
- Deliverables: `inventory/01-artifact-inventory.md`,
  `inventory/scripts/01_inventory.py`.

Files: `prompt.md` (verbatim ask), `response.md` (full method + table + findings
+ deliverables + caveat).

**Load-bearing for downstream**: the run-status column here is what a
`/data-integrity-check` step consumes to mark artifacts from crashed / died-early
/ false-start runs as suspect. The named suspect runs are `0y9un0o4`
(cross_pair all_groups), `ow1jo0xq` train copy (cross_pair within_group),
`lu7g7svh` (lora typewriter×cactus false start), `9ux1sm67` + `6enlob54`
(cross_pair all_groups false-start / early-stop), and `d5b2706v` (cross_seed
task_c — benign sync-fatal, checkpoint intact but worth a load test).
