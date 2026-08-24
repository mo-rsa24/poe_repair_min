# Segment 1 — Response (full)

## Method
The W&B project names in the command still contained the `<paste...>`
placeholder, so the inventory was built from disk first, then reconciled against
W&B by reading the 15 local run directories under `outputs/**/wandb/run-*` rather
than blocking on the user.

**outputs/ total**: 22 GB, ~13,200 files (5,636 png, 4,427 json, 2,608 pt,
438 npy, 15 `.wandb`).

**Four W&B projects, all under entity `prime_lab`:**
- `poe-repair-group-a`
- `poe-repair-lora`
- `poe-repair-cross-seed`
- `poe-repair-cross-pair`

Four experiments have no W&B runs (local-only): `conditioning_window`,
`conditioning_window_lora`, `residual_diagnostics`, `internal_force_failure`.

Run status inferred from each run's debug log: a clean finish reaches the run
footer ("restore done"); an early death stops at "Redirects installed"; a
"filestream: fatal error" is a benign W&B upload failure at shutdown, not a
training crash (the checkpoint still landed). This is a **local** reconciliation,
not a live server query.

## Main table: experiment → runs → status → surviving artifact

| Experiment | W&B project | Runs (id · step · state) | Status | Surviving artifact |
|---|---|---|---|---|
| group_a_failure / latent_cnn | poe-repair-group-a | ikvn0qzu·30k·fin; 7mop5hcl·150k·fin | worked | `outputs/group_a_failure/latent_cnn/…` (2.4G); `checkpoints/direct_eps/*/best.pt,last.pt` |
| group_a_failure / latent_unet | poe-repair-group-a | 9qvtt1zv·30k·fin | worked | `outputs/group_a_failure/latent_unet/…` (1.5G) |
| group_a_failure / frozen_feature_mlp | poe-repair-group-a | z17dxf2w·30k·fin | worked | `outputs/group_a_failure/frozen_feature_mlp/…` (958M) |
| lora / camel×desert | poe-repair-lora | 8p1spi5b·80k·fin | worked | `…/checkpoints/lora_step_080000.pt` (945M) |
| lora / typewriter×cactus | poe-repair-lora | lu7g7svh·0·false-start; wag4z592·80k·fin | worked (1 false start) | `…/133632/checkpoints/lora_step_080000.pt` (1.5G) |
| lora / cat_dog | (no wandb dir) | 1 local run | worked | `outputs/lora/cat_dog/…/lora_step_080000.pt` (3.4G, +MDS caches) |
| lora / dog×oil, dolphin×ocean, mailbox×snowfield | poe-repair-lora | none on disk | not run / empty | `outputs/lora/<pair>/seed_42/` empty |
| cross_seed / step0_prescreen | (local) | ×2 (default+seed42) | worked | `step0_prescreen/{summary.json,seed_pool.json,contact_sheet.png}` |
| cross_seed / task_b_learning_curve | poe-repair-cross-seed | hbpotmnk·80k·fin | worked (verdict "ok") | `k01_pick1__ep1600/checkpoints/…`, `verdict.json` (284M) |
| cross_seed / task_c_per_seed_ceiling | poe-repair-cross-seed | d5b2706v·16510·fin (benign sync-fatal) | worked | `per_seed_s9__ep1600/checkpoints/lora_step_016510.pt` (66M) |
| cross_pair / all_groups/main | poe-repair-cross-pair | 9ux1sm67·false-start; 6enlob54·1246·early-stop; **2em6frqv·10k·fin**; 0y9un0o4·died-early | worked (2 false starts + 1 crash) | `all_groups/main/checkpoints/lora_step_030000.pt` (477M) |
| cross_pair / within_group/g6 | poe-repair-cross-pair | ow1jo0xq(train)·died-early; ow1jo0xq(eval)·21976·fin | worked (train crashed once) | `…/checkpoints/lora_step_030000.pt`; `eval_crossbar/step_020000/manifest.json` (402M) |
| conditioning_window | none (local) | 6 pairs×seed42 | worked | `outputs/conditioning_window/…` (697M, 384 png) |
| conditioning_window_lora | none (local) | cat_dog×seed42 | worked | `outputs/conditioning_window_lora/…` (5.0G, 2952 png) |
| residual_diagnostics / delta_structure | none (local) | guided + unguided | worked | `delta_structure{,_unguided}/{results.json,tensors.pt,figures/}` (855M) |
| internal_force_failure | none | none | not run | code only; no `outputs/` dir |

**Status meanings:** *worked* = finished + usable artifact on disk (for
`group_a_failure` the experiment's finding is a failure, but the training runs
completed); *false-start* = died in seconds and relaunched; *died-early* = log
stops at "Redirects installed", no summary; *benign sync-fatal* = `d5b2706v`
logged a W&B upload `fatal` at shutdown but training hit step 16510 and the
checkpoint survived (network failure, not a training crash); *not run/empty* =
dir stub, no artifact.

## Findings flagged (descriptive)
- **3 of 6 `lora` pairs are empty stubs** yet appear as held-out targets in
  `outputs/presentation/heldout_summary/` and in the cross-seed/cross-pair runs.
- **`group_a_failure` runs record the old module name**
  `poe_repair.experiments.group_a_corrector.__main__`; code now lives under
  `group_a_failure/`. Renamed after the runs.
- **W&B `_step` ≠ checkpoint step** in cross_pair (logs to 10k, checkpoints to
  30k). Trust the filename.
- No `.safetensors`/`.bin` — all checkpoints are `.pt`.

## Deliverables written
- `inventory/01-artifact-inventory.md` — full report with caches, coverage notes,
  legend.
- `inventory/scripts/01_inventory.py` — re-runnable, read-only; regenerates the
  run table, checkpoint counts, and disk footprint. Verified to run.

## Caveat stated
Reconciliation is against **local** run dirs, not a live W&B server query. Offer
made to run `wandb` against `prime_lab/<project>` to confirm server-side states,
or to diff against a pasted run list — not executed.
