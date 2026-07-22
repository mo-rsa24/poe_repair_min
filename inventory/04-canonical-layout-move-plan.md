# Canonical Artifact Layout and Move Plan (dry-run)

**Generated**: 2026-07-21
**Scope**: the surviving (keep) artifacts of `lora`, `cross_seed_lora_pooling`, `cross_pair_lora_pooling` + shared `training_cache`, across `R:` repo and `D:` `/datasets/mmolefe/poe_repair_min`. Disposition inherited from `03-integrity-and-disposition.md`.
**Status**: APPLIED 2026-07-21 (within-root moves + compat symlinks done; see `plans/artifact-reconciliation/plans/04-canonical-layout-reorg.md`). Re-runnable idempotently with `inventory/scripts/04_apply_layout.sh` (dry-run by default; `APPLY=1`).

## Design

One relative scheme, applied under a new `artifacts/` base on **each** root. Keyed **rung → experiment → pair → seed/run**. An artifact stays on the filesystem it currently lives on (no cross-root copies); the relative path is identical under whichever root holds the bytes, so both roots are navigable the same way.

```
<root>/artifacts/
  rung1-overfit/          # plans 04, 09   single-seed, single-pair LoRA
    lora/<pair>/seed_42/run__<tag>/         {checkpoints, probes, results, inspector_manifest.json}
  rung2-survive-noise/    # plans 08, 10, 12   cross-seed pool, one pair, held-out seeds
    cross_seed/<pair>/
        taskB__<pool-tag>/                  {checkpoints, samples, verdict.json}
        taskC__<seed-tag>/
        step0_prescreen/{heldout,seed42}/
        heldout_pair/<sibling>/             within-group cross-pair probe
        trajectory_diagram/seed_42/
  rung3-group-wise/       # plan 16   within-group cross-pair, pair axis held out
    cross_pair/within_group/<group>/main__<tag>/   {checkpoints, samples, eval_crossbar}
  rung4-scale/            # plans 11, 14, 15   all-groups cross-pair, both axes held out
    cross_pair/all_groups/main__<tag>/      {checkpoints, samples, eval_crossbar}
  _shared/cross_pair_pool_configs/          pair_pool / seed_pool / pair_prompts YAMLs
  caches/
    training_cache/{train,heldout}/<pair>/seed_<N>/
    manifold_cache/<pair>/
  _quarantine/<reason>/<original>/          discards, moved not deleted
```

### Naming rules

- **Pair slug is always `a_x__x__a_y`.** The historical short slug `cat_dog` is canonicalised to `a_cat__x__a_dog` (see external-contract risks).
- **Run tag** encodes task + pool + provenance: `taskB__k04_ep2000_resumed__wandb-pueuo7bl`, `taskC__s9_ep1600__wandb-d5b2706v`, `run__wandb-wag4z592`, `run__local` (no W&B).
- **Seed** lives in the path (`seed_42/`, `seed_<N>/`) for single-seed and cache cells; for pools it is inside the tag (`k04` = 4 pooled seeds).
- **Rung** is the top level, so the pyramid is the primary axis of navigation.

### W&B project mapping (reference only, not renamed)

W&B projects are external containers; renaming them orphans run history, so this plan does not touch them. Legend for cross-reference: `poe-repair-lora` → rung1, `poe-repair-cross-seed` → rung2, `poe-repair-cross-pair` → rung3+rung4, `poe-repair-group-a` → out of scope, `poe-repair-m5-lora` → legacy/out of scope.

## Move plan (old → new)

`→⇲` = move + leave a backward-compat symlink at the old path. `→Q` = move to quarantine. `⌫` = remove (0-byte empty dir).

### Rung 1 — overfit (lora, single seed) · repo root

| Old (R:) | New (R:) | Op |
|---|---|---|
| `outputs/lora/cat_dog/seed_42/results` | `artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local` | →⇲ |
| `outputs/lora/a_typewriter__x__a_cactus/seed_42/lora_sdxl__…__20260520-133632` | `artifacts/rung1-overfit/lora/a_typewriter__x__a_cactus/seed_42/run__wandb-wag4z592` | →⇲ |
| `outputs/lora/a_camel__x__a_desert_landscape/seed_42/lora_sdxl__…__20260520-131542` | `artifacts/rung1-overfit/lora/a_camel__x__a_desert_landscape/seed_42/run__wandb-8p1spi5b` | →⇲ |
| `outputs/lora/a_typewriter__x__a_cactus/seed_42/lora_sdxl__…__20260520-131542` | `artifacts/_quarantine/false-start/lora_typewriter_131542__wandb-lu7g7svh` | →Q |
| `outputs/lora/a_dog__x__oil_painting_style` | — | **KEEP** (not empty: live alias → cross_seed bank; see correction below) |
| `outputs/lora/a_dolphin__x__an_ocean_wave` | — | **KEEP** (live alias → cross_seed bank) |
| `outputs/lora/a_mailbox__x__a_snowfield` | — | **KEEP** (live alias → cross_seed bank) |

### Rung 2 — survive-noise (cross_seed) · datasets = heavy banks, repo = eval/results

| Old | New | Op |
|---|---|---|
| D: `outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed` | D: `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl` | →⇲ |
| D: `…/a_dog__x__oil_painting_style/task_b_learning_curve/k04__ep2000` | D: `artifacts/rung2-survive-noise/cross_seed/a_dog__x__oil_painting_style/taskB__k04_ep2000` | →⇲ |
| D: `…/a_dolphin__x__an_ocean_wave/task_b_learning_curve/k04__ep2000` | D: `artifacts/rung2-survive-noise/cross_seed/a_dolphin__x__an_ocean_wave/taskB__k04_ep2000` | →⇲ |
| D: `…/a_mailbox__x__a_snowfield/task_b_learning_curve/k04__ep2000` | D: `artifacts/rung2-survive-noise/cross_seed/a_mailbox__x__a_snowfield/taskB__k04_ep2000` | →⇲ |
| D: `…/a_typewriter__x__a_cactus/task_b_learning_curve/k04__ep2000` | D: `artifacts/rung2-survive-noise/cross_seed/a_typewriter__x__a_cactus/taskB__k04_ep2000` | →⇲ |
| D: `outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep200` | D: `artifacts/_quarantine/superseded/cross_seed_catdog_k04_ep200` | →Q |
| R: `outputs/cross_seed_lora_pooling/task_b_learning_curve/k01_pick1__ep1600` | R: `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k01_pick1_ep1600__wandb-hbpotmnk` | →⇲ |
| R: `outputs/cross_seed_lora_pooling/task_c_per_seed_ceiling/per_seed_s9__ep1600` | R: `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskC__s9_ep1600__wandb-d5b2706v` | →⇲ |
| R: `outputs/cross_seed_lora_pooling/step0_prescreen` (seeds 9-12) | R: `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/step0_prescreen/heldout` | →⇲ |
| R: `outputs/cross_seed_lora_pooling/step0_prescreen_seed42` | R: `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/step0_prescreen/seed42` | →⇲ |
| R: `outputs/cross_seed_lora_pooling/trajectory_diagram/seed_42` | R: `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/trajectory_diagram/seed_42` | →⇲ |
| R: `outputs/cross_seed_lora_pooling/a_cat__x__a_dog` (heldout_pair eval) | R: `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/` (merge) | →⇲ |
| R: `outputs/cross_seed_lora_pooling/{a_dog__x__oil_painting_style,a_dolphin__x__an_ocean_wave,a_mailbox__x__a_snowfield,a_typewriter__x__a_cactus}` (heldout_pair + step0) | R: `artifacts/rung2-survive-noise/cross_seed/<same-pair>/` (merge) | →⇲ |

Note: repo per-pair dirs (light eval samples) and datasets per-pair dirs (heavy seed banks) share the same canonical `cross_seed/<pair>/` path but on different roots. They are different-stage artifacts (confirmed in `02`), so merging by root keeps both.

### Rung 3 — group-wise (cross_pair within_group) · repo root

| Old (R:) | New (R:) | Op |
|---|---|---|
| `outputs/cross_pair_lora_pooling/within_group/g6/main` | `artifacts/rung3-group-wise/cross_pair/within_group/g6/main__wandb-ow1jo0xq` | →⇲ |
| `outputs/cross_pair_lora_pooling/within_group/{g1,g2,g3,g4}` (pair_pool.yaml only, no run) | `artifacts/rung3-group-wise/cross_pair/within_group/{g1,g2,g3,g4}` | →⇲ |
| `outputs/cross_pair_lora_pooling/within_group/seed_pool.yaml` | `artifacts/rung3-group-wise/cross_pair/within_group/seed_pool.yaml` | →⇲ |

### Rung 4 — scale (cross_pair all_groups) · repo root

| Old (R:) | New (R:) | Op |
|---|---|---|
| `outputs/cross_pair_lora_pooling/all_groups/main` | `artifacts/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv` | →⇲ (re-run flag: trained but not crossbar-evaluated) |
| `outputs/cross_pair_lora_pooling/all_groups/dryrun` | `artifacts/_quarantine/smoke/cross_pair_all_groups_dryrun` | →Q |
| `outputs/cross_pair_lora_pooling/{pair_pool.yaml,seed_pool.yaml,pair_prompts.yaml}` | `artifacts/_shared/cross_pair_pool_configs/` | →⇲ |

### Caches

| Old | New | Op |
|---|---|---|
| D: `outputs/training_cache` (train + heldout, 645 cells) | D: `artifacts/caches/training_cache` | →⇲ (symlink is load-bearing: `POE_REPAIR_TRAINING_CACHE` + absolute `cell_dir` paths in manifests resolve through it) |
| D: `outputs/training_cache_overfit_catdog` | `artifacts/_quarantine/broken-symlink-view/…` | →Q (symlink view, links already broken pre-reorg; see correction below) |
| R: `outputs/manifold_cache` | R: `artifacts/caches/manifold_cache` | →⇲ (STATUS: 0/7 complete per its own `inventory.json`; kept, not trusted) |

## External-contract risks (must read before APPLY)

Moving artifact dirs breaks path references unless compat symlinks are kept. Every `→⇲` move leaves a symlink at the old path, which neutralises most of these. Residual risks:

1. **`cat_dog` → `a_cat__x__a_dog` canonicalisation.** `scripts/lora_inspector.py` (hardcoded `cat_dog`, incl. the `trajectory_diagram/seed_42/mono.png` fallback at line 626), `scripts/build_lora_manifest.py` default results-root, and `inspector_manifest.json` internal keys all name `cat_dog`. The compat symlink `outputs/lora/cat_dog → artifacts/…/a_cat__x__a_dog` keeps the app working, but the *canonical* name differs from what the code prints. Recommend a follow-up `rename` pass on the code strings if you want full consistency.
2. **`POE_REPAIR_TRAINING_CACHE`.** Points at `…/outputs/training_cache`. The symlink keeps it valid. If you later delete the symlink, update the env default in `training_cache.py:38-43` and every plan file.
3. **Absolute `cell_dir` paths inside cache manifests.** `manifest_all.json` stores absolute `…/outputs/training_cache/…` cell dirs. These resolve through the symlink; a hard cutover requires regenerating the manifests (already flagged stale in `03`).
4. **Plan files and run scripts** reference `outputs/lora/…`, `outputs/cross_seed_lora_pooling/…`, `outputs/cross_pair_lora_pooling/…` throughout `plans/` and `scripts/`. Symlinks cover reads; a clean cutover is a separate code-side `rename`.
5. **W&B run dirs** under moved run dirs (`…/wandb/run-*`) contain absolute paths in their metadata. Harmless (upload already done), but do not expect W&B resume to work from the new path without `--resume-from` pointing at the new checkpoint.

## Apply

```bash
# dry-run (prints every planned mv / ln / rmdir, changes nothing):
bash inventory/scripts/04_apply_layout.sh
# execute on the cluster after approval:
APPLY=1 bash inventory/scripts/04_apply_layout.sh
```

The script: makes within-root moves only, leaves a compat symlink at each old path, routes discards to `_quarantine/`, and `rmdir`s only verified-empty 0-byte dirs. It refuses to overwrite an existing destination.

## Post-apply corrections (2026-07-21)

Applied on the cluster (`APPLY=1`). All 32 move+symlink and 3 quarantine actions succeeded; old paths are compat symlinks; checkpoints load via both old and new paths; `training_cache` resolves through the env-compatible symlink. Two inventory mislabels were caught **during** apply, before any data loss:

- **The three `outputs/lora/{a_dog__x__oil_painting_style,a_dolphin__x__an_ocean_wave,a_mailbox__x__a_snowfield}` dirs are not empty stubs.** Each holds `seed_42/results` → a symlink into that pair's cross_seed seed bank. They resolve after the reorg (10/10/8 checkpoints). The planned `⌫` removals **failed safely** (a dir containing a symlink is not empty) and the entries were removed from the apply script. Decision: **keep**.
- **`training_cache_overfit_catdog` is a symlink view, not an empty dir**, and its links were **already broken before the reorg** (they target a repo `outputs/training_cache` that does not exist in this checkout). Hand-quarantined to `artifacts/_quarantine/broken-symlink-view/`. Decision: **discard (quarantined, reversible)**.

Root cause of the mislabel: the inventory sized these with `du -sh` / `find -type f`, neither of which traverses symlinks, so a symlink-only directory read as 0 bytes. No artifacts were deleted; the only removals were the reversible quarantine moves.

## Code-side canonicalisation (2026-07-21)

Repointed the six standalone LoRA-artifact path constants from the legacy
`outputs/lora/cat_dog/seed_42/results` (compat symlink) to the canonical
`artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local`:
`build_lora_manifest.py`, `build_lora_inspector_mds.py`,
`build_lora_inspector_mds_semantic.py`,
`cross_seed_lora_pooling/smoke_dino_distance.py`,
`manifold/seed42_phase1.py`, `manifold/inventory_trajectories.py`. All compile;
`build_lora_manifest.py` was run end-to-end (47 epochs / 183 cells) and writes
to the same `inspector_manifest.json` the inspector reads through the symlink.

**Deliberately NOT renamed: `scripts/lora_inspector.py`'s `cat_dog` slug and the
conditioning_window references.** The inspector derives the pair slug from the
`outputs/lora/` directory name and uses it as a join key into the
conditioning_window tree, which was not reorganised (it is not one of the three
in-scope experiments). Renaming the slug without also moving the CW artifacts
would break the cat×dog mono fallback and the CFG-mask tabs. Finishing the job
means reorganising `conditioning_window` + `conditioning_window_lora` cat_dog
artifacts, then a single slug-rename pass. Recorded in memory
`catdog-slug-shared-key`.
