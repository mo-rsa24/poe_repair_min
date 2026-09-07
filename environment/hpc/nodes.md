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
| `biggpu` | mscluster[106-112], 7 nodes | 3 days | First choice for GPU jobs. `mscluster106`: 2x Quadro RTX 8000, 49GB each. `mscluster108`: 2x Quadro RTX 8000, 49GB each. `mscluster109`: 2x RTX A6000, 49GB each. `mscluster110`, `111`, `112`: 1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition each, ~96GB VRAM (Blackwell generation, newer and larger than every other `biggpu` card). `mscluster107`: down. `mscluster111`: visible to `nvidia-smi` but unusable on 2026-09-05, pstate reads `[GPU requires reset]`, utilisation `[N/A]`, and `torch.cuda.is_available()` is False under `co3_bw`, so a launch there runs on the CPU with no error (a 320-render job crawled at 14 GB RSS for ten minutes before it was caught); every launcher now asserts `torch.cuda.is_available()` after pinning the device. All others verified live via `nvidia-smi` on 2026-09-01. `sinfo` GRES reads `(null)` for every partition, so per-node GPU models are only discoverable by SSH + `nvidia-smi`, not by querying Slurm. |
| `bigbatch` | mscluster[42-89], 48 nodes | 3 days | Fallback when no `biggpu` node is idle. `mscluster50` and `mscluster51` sit `idle` in `sinfo` but their GPU is hardware-faulted (`nvidia-smi`: "Unable to determine the device handle for GPU0: 0000:17:00.0: Unknown Error"), so a job lands there and the GPU guard aborts it in under a second (jobs 49277 and 49278, 2026-09-03). A probe of every idle `bigbatch` node over SSH on 2026-09-06 (`ssh <node> nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv`) found the same fault on `mscluster44`, `45`, `65` and `83` (`Unable to determine the device handle for GPU0: 0000:17:00.0`), and `mscluster74` reading utilisation `[N/A]`, which the launchers treat as faulted; job 50336 landed on `mscluster65` and the guard refused it in one second. The healthy idle cards that day were RTX 3090 24 GB on `mscluster46`, `53`, `55`, `56`, `60`, `61`, `62`, `63`, `71`, `72`, `73`, `75`, `88`, `89`. So either submit with `--exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83`, or run that probe over `sinfo -p bigbatch -N -h -o "%N %t" | awk '$2=="idle"'` first and pin a healthy node with `--nodelist=<node>`, until an administrator clears them. `sinfo` lists the session node `mscluster85` as idle while it carries this project's own runs, so idle is not free there either. |
| `batch` | mscluster[120-219], 100 nodes | 1 day | Short jobs only. |
| `stampede` | mscluster[22-41], 20 nodes | 3 days | Rarely used for this project. |
| `gpuexpress` | mscluster116, 1 node | 1 hour | Short wiring checks only. |

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

**`co3_bw` is the environment for the Blackwell nodes (mscluster110, 111, 112).** Measured
2026-09-03 on mscluster112: `co3_bw` is torch 2.11.0+cu128 with kernels for `sm_100` and
`sm_120`, and the same diffusers 0.29.2 and transformers 4.44.1 as `co3`, plus peft 0.19.1.
`co3` is torch 2.5.1+cu118 with kernels up to `sm_90`, so on an RTX PRO 6000 Blackwell a
`co3` CUDA op produces no output at all rather than an error. Rule: `co3` on the RTX 8000 and
A6000 nodes (106, 108, 109) and on the session node; `co3_bw` on 110 to 112. Launchers take
the python path as `PY=` for this reason (`scripts/launch_sdlora.sh`,
`scripts/animals_compose_transfer/smoke_live_curves.sh`).

**`mscluster111`'s card is not usable from torch (2026-09-05).** `nvidia-smi` lists it with 2 MiB used
and `utilization.gpu=[N/A]`, and `torch.cuda.is_available()` under `co3_bw` returns `False`, so a
sampler launched there runs on the CPU at about 40 times the GPU time with no error in its log.
Consequence: treat the node as faulted until an administrator resets it; the shared-device launch
script refuses a device whose utilisation reads `[N/A]` or whose CUDA probe fails, per
[poe-launch-002](../known-failures.md). `mscluster110` and `mscluster112` were fine the same day.


`superdiff` exists for SuperDiff-related work if its dependencies conflict with `co3`
(stated, not re-verified this sitting).

## Cross-references

- The cross-device spread named in [the finding "does a Langevin corrector remove part of the correction"](../../report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md): one uncorrected cell reads 0.153, 0.169 and 0.180 on the RTX 3090, the RTX PRO 6000 Blackwell and the Quadro RTX 8000, so a comparison across corrector counts stays on one device.
