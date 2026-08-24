---
type: recurring
category: admin, data-hygiene
---

# 🔁 Re-Sweep on New Runs (standing)

## Description

A standing check: whenever new training runs finish (new W&B runs, new `lora_step_*.pt` checkpoints, or new cache cells), fold them into the inventory, integrity report, and canonical layout so none of `01`–`04` goes stale. This node is never "done"; it has a cadence, not a definition of done.

## Purpose

Files pile up between sessions. Without a re-sweep the inventory goes stale, new checkpoints sit outside the `artifacts/` scheme unfiled, and broken runs go unnoticed. This keeps everything current with almost no effort per run, because `01`–`04` are already scripts.

## Goal

After each re-sweep: the `01`/`02` inventories match disk + W&B, every new checkpoint has been load-tested and dispositioned in `03`, and every kept artifact is filed under `artifacts/` (a dry-run of `04_apply_layout.sh` shows 0 unfiled real actions). No artifact left unfiled or unchecked.

## Tasks

- [ ] ⚠️ Detect new runs since last sweep (new W&B run dirs, new `lora_step_*.pt`, new `training_cache` cells).
- [ ] ⚠️ Re-run `01_inventory.py`; diff `01`/`02` against the previous version; note additions.
- [ ] ⚠️ Load-test new checkpoints + any new suspects; update `plans/standing/artifact-reconciliation/inventory/03-integrity-and-disposition.md`.
- [ ] ⚠️ Extend `04_apply_layout.sh` with any new run dirs; dry-run, then `APPLY=1` to file them; confirm compat symlinks.
- [ ] ⚠️ If new pairs appear, canonicalise their slug (retire any short form) the same way `cat_dog` → `a_cat__x__a_dog` was done.
- [ ] ⚠️ Append a decision-timeline gate for each new landing to `DECISION_TIMELINE.md` (append-only; supersede with a banner, never rewrite) so the spine stays current.

## Pending backlog (surfaced by sync 2026-07-22)

Concrete unfiled items owed to the next sweep. These are why the scope's DoD-4 is ⚠️:

- [x] ✅ ~~Classify `a_camel__x__a_desert_landscape` LoRA: keep / re-run / discard, add its disposition row and name its rung-1 owner.~~  ✓ verified (inventory `01` row + `03` keep-disposition; filed at `artifacts/rung1-overfit/lora/a_camel__x__a_desert_landscape/seed_42/run__wandb-8p1spi5b`)
- [ ] ⚠️ File the four G1–G4 cross-seed pool runs (W&B `aoj3oz7s` G1, `yrfw5dio` G2, `ig20iqul` G3, `xcp40234` G4): load-test, disposition, confirm canonical path.
- [ ] ⚠️ Record the `koy9gjis` cat×dog failure run as a `discard`/`reference` row (the negative example for Survive-Noise).
- [ ] ⚠️ Close the cross-root navigability gap: leave a repo-side symlink into `/datasets/.../artifacts/` for every datasets-only run (e.g. the `pueuo7bl` pool) so bare `artifacts/...` paths resolve from the repo. Alternative already partly applied: plans use absolute `/datasets` paths (02/03). Pick one convention and make it uniform.

Standing prompt (run each sweep):

```
Re-sweep the artifact roots for runs that landed since the last reconciliation.
Re-run plans/standing/artifact-reconciliation/inventory/scripts/01_inventory.py and diff 01/02 against the prior version.
Load-test every new lora_step_*.pt / best.pt (torch weights-only, LoRA keys +
shapes) and any run W&B marks unfinished; append rows to
plans/standing/artifact-reconciliation/inventory/03-integrity-and-disposition.md with keep / re-run / discard. Add any
new run dirs to plans/standing/artifact-reconciliation/inventory/scripts/04_apply_layout.sh, dry-run (expect the existing
entries to REFUSE as already-filed), then APPLY=1 to file the new ones under
artifacts/ with compat symlinks. Canonicalise any new short pair slug. Read-only
on source data except the reversible quarantine/symlink moves. Report what changed.
```

## Recommended skill

`training-analyst` ⚠️ (for the W&B-run sweep half) · `/data-integrity-check` ✅ + `/rename` ✅ (for the check + file halves) — custom glue otherwise.

## Engagement Instructions

Adherence is read from run history, not a done-box. A healthy sweep leaves:

```
$ CUDA_VISIBLE_DEVICES="" python plans/standing/artifact-reconciliation/inventory/scripts/01_inventory.py   # inventory regenerated
$ bash plans/standing/artifact-reconciliation/inventory/scripts/04_apply_layout.sh 2>&1 | grep -cE "^(MOVE\+LINK|QUARANTINE)"
# Expect: 0 when everything new has already been filed this sweep (non-zero => unfiled artifacts remain).
$ git -C . status --porcelain plans/standing/artifact-reconciliation/inventory/   # plans/standing/artifact-reconciliation/inventory/0{1,2,3}.md updated when runs were added since last sweep
```
