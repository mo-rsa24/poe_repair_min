# Re-sweep detection: 2026-08-04

Standing re-sweep node (`plans/05-resweep-on-new-runs.md`), detection half only. This
file records what the read-only scripts found on 2026-08-04, so the next session's
dispositioning starts from a precise, dated list. It assigns **no** keep/re-run/discard
labels and moves nothing: those steps are parked for a human (see `plans/05` and the
scope `parked.md`).

## How this was produced (read-only)

```
$ CUDA_VISIBLE_DEVICES="" python inventory/scripts/01_inventory.py      # prints, writes nothing
$ bash inventory/scripts/04_apply_layout.sh | grep -cE "^(MOVE\+LINK|QUARANTINE)"   # 0
```

Plus a set difference of three lists: experiment top-dirs on disk, top-dirs named in
`inventory/01-artifact-inventory.md`, and top-dirs referenced by `04_apply_layout.sh`.

## Layout script status: clean for the runs it knows about

`04_apply_layout.sh` dry-run: **0** `MOVE+LINK`/`QUARANTINE` actions, 32 `REFUSE (dest exists)`
(all previously-filed rung1-4 entries resolve), 3 `SKIP (missing)` (quarantined false-starts).
So nothing the script tracks is unfiled. The gaps below are dirs the script does not mention.

## Gap 1: inventoried on 2026-07-21 but never filed into `artifacts/`

Named in `inventory/01-artifact-inventory.md`, but not referenced by `04_apply_layout.sh`
(which only ever covered `lora`, `cross_seed_lora_pooling`, `cross_pair_lora_pooling`,
`training_cache`, `manifold_cache`):

| top-dir | disk | note (mechanical) |
|---|---|---|
| `group_a_failure` | 7G | 1031 ckpt files; 4 runs, all 2026-05-14, project `poe-repair-group-a` (`z17dxf2w`, `ikvn0qzu`, `7mop5hcl`, `9qvtt1zv`) |
| `conditioning_window_lora` | 5G | no local `wandb/run-*` dir seen by `01` |
| `residual_diagnostics` | 855M | 400 ckpt files; no local `wandb/run-*` dir seen by `01` |
| `conditioning_window` | 697M | no local `wandb/run-*` dir seen by `01` |
| `presentation` | 6M | no local `wandb/run-*` dir seen by `01` |

## Gap 2: landed since the inventory was authored, tracked nowhere

On disk, absent from both `inventory/01` and `04_apply_layout.sh`:

| top-dir | disk | note (mechanical) |
|---|---|---|
| `animals_compose_transfer` | 6G | project `poe-repair-animals-compose`, run `1d3qy31e`: one `died-early` dir (2026-07-30 03:54) then a `finished` dir (2026-07-30 05:04, `_step=100000`, runtime 34320s); 20 ckpt files under `pooled_lora/phase1_r8_100k/checkpoints/lora_step_*.pt` |
| `compose_scorer` | 15M | no local `wandb/run-*` dir seen by `01` |
| `poe` | 184M | no local `wandb/run-*` dir seen by `01` |

## What is parked for the human (not done here)

Everything that changes state, because each step needs the output read:

1. Decide whether the diagnostics / mechanism-study dirs (`residual_diagnostics`,
   `group_a_failure`, `conditioning_window`, `conditioning_window_lora`) belong in the
   rung1-4 canonical scheme at all, or in a separate bucket / separate scope.
2. Load-test the new checkpoints (`animals_compose_transfer` headline
   `lora_step_100000.pt`, plus suspects) and assign keep / re-run / discard in
   `inventory/03-integrity-and-disposition.md`.
3. Author `inventory/01` + `02` rows for the Gap-2 dirs and reconcile against W&B.
4. Extend `04_apply_layout.sh` with the chosen destinations, dry-run, then `APPLY=1`.
5. Append a `docs/DECISION_TIMELINE.md` entry for this landing.

The pre-existing backlog in `plans/05` (G1-G4 pool disposition rows, the `koy9gjis`
failure row, the cross-root symlink convention) is unchanged by this detection and
remains parked.
