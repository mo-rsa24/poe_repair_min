# Storage: where things live, sizes, and the disk guard

Navigation: 📋 [Index](00-INDEX.md) | [Overview](overview.md#access-paths)

## Access paths

- **Repo and code:** `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min`.
- **Large artifacts** (checkpoints, caches, sweep outputs): `/datasets/mmolefe/poe_repair_min/`.
  These are separate filesystems with different quotas; there is no automatic mirroring between
  them, so a path that exists on one is not implied to exist on the other.
- **Cached residual trajectories** (the r_t training targets, all four eps branches per step):
  `/datasets/mmolefe/poe_repair_min/outputs/training_cache/` (18 train + 58 heldout pairs, ~12
  seeds, 50 steps each; contents verified by direct filesystem inspection and a loaded residual
  file on 2026-08-04).
- **W&B project for the animals program:** `prime_lab/poe-repair-animals-compose`.

## Size, retention, and the quota gap

Both filesystems are NFS mounts. Verified live via `df -h /datasets /home-mscluster` on
2026-08-24:

| Mount | Size | Used | Free | What belongs there |
|---|---|---|---|---|
| `/datasets` | 377T | 30T (8%) | 348T | Every checkpoint, cache, and sweep output. |
| `/home-mscluster` | 73T | 29T (39%) | 45T | The repo, the conda envs, small results and logs. |

(A prior check on 2026-08-09 read `/datasets` at 201T size / 24T used / 12% and `/home-mscluster`
at 73T / 26T / 35%; both mounts have grown or filled further since. `/datasets`'s reported total
size moved from 201T to 377T between those two checks, which is a real change in what the mount
reports, not a measurement error; treat the size column as something to re-check rather than a
fixed ceiling.)

**There is no per-user `quota` command on these nodes** (confirmed absent from PATH live,
2026-08-24), so a user's own share of either mount is not queryable. The only guard available is
the `df` check every job script runs before writing.

## The disk guard rule, and the failure that motivates it

**`/home-mscluster` hit 100% once and silently killed checkpointing mid-run.** Checkpoints go to
`/datasets/...` only, and every job script keeps a `df` guard that aborts before writing if the
target filesystem is too full (see `hpc/execution-protocol.md` step 4).

**The guard must check the filesystem the script actually writes to, not an assumed one.**
`scripts/mechanism_study/run_dose_sweep.sh` set its output root under the repo on
`/home-mscluster` while its disk guard read `df /datasets/mmolefe`, so 3.4GB of sweep cells
landed on the wrong mount with the guard reporting healthy the whole time. This is logged as
`poe-disk-001` in `known-failures.md`. A recent series of commits (`54b4b79`, `8522459`,
`c3f8bb4`, `f293bdf`) routed output paths in roughly 100 files through a shared
`paths.resolve()` helper specifically to stop this class of drift between "where output is
configured to land" and "where the guard checks"; whether every script's guard now reads the
same resolved root as its output root has not been re-audited this sitting.

## Version control

Neither filesystem is under version control and neither is snapshotted. Every `✓ verified (...)`
evidence tag in the plan tree points at a path on one of them, so an artifact deleted or moved
silently breaks the claim it backs. `plans/shelved/artifact-reconciliation/plans/05-resweep-on-new-runs.md`
exists to catch this, and is a standing recurring plan rather than a finished one.
