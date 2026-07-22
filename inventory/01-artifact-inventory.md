# Artifact Inventory: poe_repair_min

**Repo root**: /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
**Generated**: 2026-07-21
**outputs/ total**: 22 GB, 13,214 files (5,636 png, 4,427 json, 2,608 pt, 438 npy, 15 .wandb)
**Re-run**: `python inventory/scripts/01_inventory.py`

## Reconciliation method

W&B project names were not supplied, so they were read from the 15 local W&B run
directories under `outputs/**/wandb/run-*`. Four projects appear, all under entity
`prime_lab`:

- `poe-repair-group-a`
- `poe-repair-lora`
- `poe-repair-cross-seed`
- `poe-repair-cross-pair`

Four experiments have no W&B runs (local-only): `conditioning_window`,
`conditioning_window_lora`, `residual_diagnostics`, `internal_force_failure`.

Run status is inferred from each run's debug log: a clean finish reaches the run
footer ("restore done"); an early death stops at "Redirects installed"; a
"filestream: fatal error" is a benign W&B upload failure at shutdown, not a
training crash (the checkpoint still landed). This is a **local** reconciliation —
it reflects the run dirs on disk, not a live query of the W&B server. If a run was
deleted locally but exists on W&B (or vice versa), that will not show here.

## Main table: experiment -> runs -> status -> surviving artifact

| Experiment (module) | W&B project | Runs (id · step · state) | Status | Surviving artifact path |
|---|---|---|---|---|
| **group_a_failure** / latent_cnn | poe-repair-group-a | ikvn0qzu · 30k · finished; 7mop5hcl · 150k · finished | worked | `outputs/group_a_failure/latent_cnn/...` (2.4G); ckpts `outputs/group_a_failure/checkpoints/direct_eps/*/best.pt,last.pt`,snapshots |
| **group_a_failure** / latent_unet | poe-repair-group-a | 9qvtt1zv · 30k · finished | worked | `outputs/group_a_failure/latent_unet/...` (1.5G) |
| **group_a_failure** / frozen_feature_mlp | poe-repair-group-a | z17dxf2w · 30k · finished | worked | `outputs/group_a_failure/frozen_feature_mlp/...` (958M) |
| **lora** / a_camel__x__a_desert_landscape | poe-repair-lora | 8p1spi5b · 80k · finished | worked | `.../lora_.../checkpoints/lora_step_080000.pt` (945M dir) |
| **lora** / a_typewriter__x__a_cactus | poe-repair-lora | lu7g7svh · 0 · false-start; wag4z592 · 80k · finished | worked (after 1 false start) | `.../133632/checkpoints/lora_step_080000.pt` (1.5G dir) |
| **lora** / cat_dog | (no W&B dir) | 1 local run (history.json) | worked | `outputs/lora/cat_dog/seed_42/results/checkpoints/lora_step_080000.pt` (3.4G; + MDS caches, 596 .pt) |
| **lora** / a_dog__x__oil_painting_style | poe-repair-lora | none on disk | not run / empty | `outputs/lora/a_dog__x__oil_painting_style/seed_42/` empty (no artifact) |
| **lora** / a_dolphin__x__an_ocean_wave | poe-repair-lora | none on disk | not run / empty | `outputs/lora/a_dolphin__x__an_ocean_wave/seed_42/` empty (no artifact) |
| **lora** / a_mailbox__x__a_snowfield | poe-repair-lora | none on disk | not run / empty | `outputs/lora/a_mailbox__x__a_snowfield/seed_42/` empty (no artifact) |
| **cross_seed_lora_pooling** / step0_prescreen | (no W&B dir) | local prescreen (×2: default + seed42) | worked | `outputs/cross_seed_lora_pooling/step0_prescreen/{summary.json,seed_pool.json,contact_sheet.png}` |
| **cross_seed_lora_pooling** / task_b_learning_curve | poe-repair-cross-seed | hbpotmnk · 80k · finished | worked (verdict.json = "ok") | `.../k01_pick1__ep1600/checkpoints/lora_step_*.pt`, `verdict.json` (284M) |
| **cross_seed_lora_pooling** / task_c_per_seed_ceiling | poe-repair-cross-seed | d5b2706v · 16510 · finished (benign sync-fatal) | worked | `.../per_seed_s9__ep1600/checkpoints/lora_step_016510.pt` (66M) |
| **cross_seed_lora_pooling** / trajectory_diagram | (no W&B dir) | local | worked | `.../trajectory_diagram/seed_42/{mono.pt,poe.pt,*_meta.json}` |
| **cross_pair_lora_pooling** / all_groups/main | poe-repair-cross-pair | 9ux1sm67 · 11s false-start; 6enlob54 · 1246 · stopped early; 2em6frqv · 10k · finished (main); 0y9un0o4 · died-early | worked (after 2 false starts + 1 later crash) | `.../all_groups/main/checkpoints/lora_step_030000.pt` (477M) |
| **cross_pair_lora_pooling** / all_groups/dryrun | poe-repair-cross-pair | smoke | smoke only | `.../all_groups/dryrun/checkpoints/lora_step_000005.pt` |
| **cross_pair_lora_pooling** / within_group/g6 | poe-repair-cross-pair | ow1jo0xq(train) · died-early; ow1jo0xq(eval) · 21976 · finished | worked (train crashed once, eval completed) | `.../within_group/g6/main/checkpoints/lora_step_030000.pt`; `.../eval_crossbar/step_020000/manifest.json` (43 cells) (402M) |
| **conditioning_window** | none (local) | 6 pairs × seed_42 | worked | `outputs/conditioning_window/<pair>/seed_42/` (697M, 384 png) |
| **conditioning_window_lora** | none (local) | cat_dog × seed_42 | worked | `outputs/conditioning_window_lora/cat_dog/seed_42/` (5.0G, 2,952 png) |
| **residual_diagnostics** / delta_structure | none (local) | guided + unguided | worked | `outputs/residual_diagnostics/delta_structure{,_unguided}/{results.json,tensors.pt,figures/}` (855M) |
| **internal_force_failure** | none | none | not run (no outputs) | code only (`poe_repair/experiments/internal_force_failure/`); no `outputs/` dir |

## Status legend

- **worked** — run(s) finished and left a usable artifact on disk. For `group_a_failure` the experiment's *finding* is a failure (the corrector cannot fix Group A), but the training runs themselves completed.
- **false-start** — a run that died within seconds/minutes and was immediately relaunched (step 0 or tiny runtime). The relaunch is the real run.
- **died-early** — a launched run whose log stops at "Redirects installed" (killed/crashed before training loop logged). No summary written.
- **benign sync-fatal** — `d5b2706v` logged a `filestream: fatal error` (W&B upload gave up, "context canceled") at shutdown. Training reached step 16510 and the checkpoint survives; this is a network/sync failure, not a training crash.
- **not run / empty** — directory stub exists but holds no artifact.

## Non-obvious findings (descriptive only)

- **3 of 6 `lora` pairs are empty stubs**: `a_dog__x__oil_painting_style`,
  `a_dolphin__x__an_ocean_wave`, `a_mailbox__x__a_snowfield` have a `seed_42/`
  dir with no `.pt`, `.png`, or `history.json`. Yet these three pairs *do* appear
  as held-out targets in `outputs/presentation/heldout_summary/` and in
  `cross_seed`/`cross_pair` runs, so the pairs are used downstream even though the
  standalone per-pair LoRA was not saved here.
- **Program path renamed since the runs**: `group_a_failure` runs record
  `-m poe_repair.experiments.group_a_corrector.__main__`, but the code now lives
  at `poe_repair/experiments/group_a_failure/`. The module was renamed after those
  runs; artifacts are under the new name.
- **W&B `_step` != checkpoint step**: `cross_pair all_groups/main` logs `_step`
  up to 10000 but writes checkpoints up to `lora_step_030000.pt` (logging counter
  vs optimizer step differ). Trust the checkpoint filename for training progress.
- **Duplicate run-id `ow1jo0xq`**: the `within_group/g6` train run and its
  crossbar-eval run share the same W&B id across two run dirs (resumed run). The
  044614 dir died early; the 114544 dir is the eval that finished.
- **cat_dog is the instrumented flagship**: `outputs/lora/cat_dog` (3.4G, 596 .pt)
  carries the MDS inspector caches (`mds_cache`, `mds_cache_semantic`,
  `mds_probes*`) that the other pairs do not.

## Caches (not experiment outputs, but on disk)

| Cache | Path | Size |
|---|---|---|
| Manifold eps/z banks | `outputs/manifold_cache/` (`a_cat__x__a_dog`, `seed_42_phase1`) | 838M; `inventory.json` present |
| Pilot trajectory arrays | `data/pilot/seed_42/*/grid_assets/trajectory_flat_*.npy` | part of `data/` (42M) |
| Composition finetuning tensors | `composition/focus/datasets/data_finetuning/*.pt` (flux/sd3) | part of `composition/` (118M) |

## Coverage notes

- Run status inferred from local debug logs of 15 W&B run dirs; not cross-checked
  against the live W&B server. To confirm server-side state (e.g. a run marked
  "crashed" on W&B that looks finished locally), query `prime_lab/<project>`.
- Checkpoint counts include nested probe/delta-overlay `.pt` tensors, not only
  headline LoRA checkpoints (`group_a_failure` 1031, `lora` 1156, `residual_
  diagnostics` 400 total `.pt`). Headline LoRA checkpoints are the
  `lora_step_*.pt` / `best.pt` / `last.pt` files cited in the table.
- No `.safetensors`/`.bin` adapters exist; all checkpoints are PyTorch `.pt`.
