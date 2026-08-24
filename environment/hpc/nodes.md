# HPC: partitions, nodes and GPUs

Navigation: 📋 [Index](../00-INDEX.md) | [Overview](../overview.md#compute--runtime)

The working session (VS Code + Claude) runs ON a compute node, not the login node. Claude
never runs on the login node. So the real execution split this project cares about is
"in-session on the current node" versus "submitted or interactive job on a bigger GPU," not
"login vs compute."

**The session node right now is `mscluster85`, on `bigbatch`** (RTX 3090, 24GB VRAM, 123GB
RAM, 28 cores). Verified live via `nvidia-smi` on 2026-08-04, and its partition re-verified
live via `sinfo -N -o "%N %P %t"` on 2026-08-24 (state: `alloc`).

## Partitions

Verified live via `sinfo -o "%P %a %l %D"` on 2026-08-24 (columns: partition, availability,
time limit, node count).

| Partition | Nodes | Time limit | Notes |
|---|---|---|---|
| `biggpu` | mscluster[106-112], 7 nodes | 3 days | First choice for GPU jobs. `mscluster106`: 2x Quadro RTX 8000, 49GB each (verified via `nvidia-smi`, 2026-08-19). `mscluster110`: Blackwell-class (user-stated, not machine-verified). `sinfo` GRES reads `(null)` for every partition, so per-node GPU models are only discoverable by SSH + `nvidia-smi`, not by querying Slurm. |
| `bigbatch` | mscluster[42-89], 48 nodes | 3 days | Fallback when no `biggpu` node is idle. |
| `batch` | mscluster[120-219], 100 nodes | 1 day | Short jobs only. |
| `stampede` | mscluster[22-41], 20 nodes | 3 days | Rarely used for this project. |
| `gpuexpress` | mscluster116, 1 node | 1 hour | Smoke tests only. |

**`sinfo` never shows GPU GRES on this cluster.** Confirmed again live on 2026-08-24
(`sinfo -o "%P %N %G"`): every partition reads `(null)` in the GRES column. A job script that
trusts Slurm to pick a GPU-capable node will not get one; node capability is tribal knowledge,
recorded in the table above, not queryable.

**Many nodes show down or drain at any given time.** An idle-looking partition (available in
the `%a` column) can still queue a job, because individual nodes inside it are unavailable.
Check node STATE (`sinfo -N -o "%N %P %t"`), not just partition-level availability, before
assuming a submission will start promptly.

## Python environments

`/home-mscluster/mmolefe/miniforge3/envs/` (mamba/miniforge). Verified live via `ls` on
2026-08-24: `co3`, `co3_bw`, `compvis_ldm`, `cxr`, `jax`, `jax115`, `ldm`, `mamba`, `rosalia`,
`score-sde`, `superdiff`, `superdiffusion`, `tex`.

**`co3` is the project environment.** Absolute path:
`/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python` (a symlink to `python3.10`, verified
live via `ls -la` on 2026-08-24). Every job script must use this absolute path, never a bare
`python`, because nothing in the execution model activates an environment implicitly.

**`co3_bw` also appears in at least one launcher** (`scripts/animals_compose_transfer/smoke_live_curves.sh`,
per `plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/plans/instrument-02-three-live-curves-while-training.md`).
Confirmed to exist via the `ls` above; what specifically differs from `co3` has not been
probed this sitting, so treat `co3_bw` as a second real environment whose exact purpose is
still open (see `known-failures.md` if a run against it behaves unexpectedly).

`superdiff` exists for SuperDiff-related work if its dependencies conflict with `co3`
(stated, not re-verified this sitting).
