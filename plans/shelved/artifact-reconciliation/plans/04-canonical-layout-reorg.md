# 🧭 Canonical Layout + Reorg

## Description

Propose one naming and directory scheme for the surviving artifacts, keyed by experiment/rung/seed/pair, so `/datasets/mmolefe` and the repo are navigable identically. Emit a dry-run move plan (old → new), then apply it on the cluster with backward-compat symlinks.

## Purpose

`03` says what to keep; this makes the kept set easy to find. Right now the run-folder names are all over the place (`lora_sdxl__…__timestamp`, `k01_pick1__ep1600`, `main`), no path tells you which pyramid rung a run belongs to, and the split across the two roots is arbitrary. One rung → experiment → pair → seed/run scheme fixes that, and leaving a symlink at every old path means no code, environment variable, or manifest breaks when things move.

## Goal

`plans/standing/artifact-reconciliation/inventory/04-canonical-layout-move-plan.md` (scheme + old→new table + external-contract risks) and a guarded `plans/standing/artifact-reconciliation/inventory/scripts/04_apply_layout.sh` (dry-run by default, `APPLY=1` to execute). Applied result: the `artifacts/` tree on both roots, compat symlinks at every old path, discards quarantined (not deleted).

## Tasks

- [x] ✅ Design the rung→experiment→pair→seed scheme; write the dry-run move plan.
- [x] ✅ Apply on the cluster (within-root moves + compat symlinks; quarantine discards).
- [x] ✅ Do the code-side `cat_dog` → `a_cat__x__a_dog` canonicalisation (paths + inspector slug + docs).

Fully-qualified prompt (invoke via `/rename`):

```
Propose a canonical layout for the surviving artifacts: one naming + directory
scheme keyed by experiment/rung/seed/pair so /datasets/mmolefe and the repo caches
are navigable. Scheme: artifacts/rung1-overfit/lora, rung2-survive-noise/cross_seed,
rung3-group-wise/cross_pair/within_group, rung4-scale/cross_pair/all_groups,
caches/, _shared/, _quarantine/. Pair slug canonical a_cat__x__a_dog (retire
cat_dog). Output a dry-run move plan (old path → new path), no moves yet; within-
root moves only (no cross-filesystem copies); leave a backward-compat symlink at
every old path; quarantine discards (reversible), do not delete; remove only
verified-empty 0-byte dirs. After approval, apply with APPLY=1 and verify the
dry-run reports 0 residual actions. Then a code-side pass repointing the LoRA path
constants and the inspector slug to a_cat__x__a_dog.
```

## Recommended skill

`/rename` ✅

## Engagement Instructions

```
$ bash plans/standing/artifact-reconciliation/inventory/scripts/04_apply_layout.sh 2>&1 | grep -cE "^(MOVE\+LINK|QUARANTINE|RMDIR \(empty\))"
# Expect: 0  (idempotent — everything already filed; a fresh run finds no real action).
$ find artifacts -maxdepth 2 -type d | sort            # rung1-overfit … rung4-scale, caches, _shared, _quarantine
$ ls -l outputs/lora/cat_dog                            # compat symlink -> a_cat__x__a_dog
$ CUDA_VISIBLE_DEVICES="" python -c "import torch; sd=torch.load('artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_062500.pt',map_location='cpu',weights_only=True); print('lora keys', len([k for k in sd.get('lora_state',sd) if 'lora' in k.lower()]))"
# Expect: lora keys 420  (loads via the canonical path)
```
