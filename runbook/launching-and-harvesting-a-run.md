# Launching and harvesting a run

Navigation: 📋 [Index](00-INDEX.md)

Contents: [1. Decide where a run goes](#1-decide-where-a-run-goes) ·
[2. Launch on a shared device](#2-launch-on-a-shared-device) ·
[3. Harvest what is running](#3-harvest-what-is-running)

The full reasoning behind each step, the mandatory safety rules for the shared-device path, and
the disk-guard and `co3`-path preflight requirements are in
[environment/hpc/execution-protocol.md](../environment/hpc/execution-protocol.md). This theme
holds only the commands; that file holds the why.

## 1. Decide where a run goes

`unverified`, transcribed 2026-08-24 from `environment/hpc/execution-protocol.md`, not run this
sitting

In order, stop at the first match:

```bash
squeue -u mmolefe              # already-allocated interactive node? run there directly
sinfo -p biggpu                # idle biggpu node? target it
ssh <node> nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # shared device free?
```

✅ Expected: `squeue` lists your own allocations; `sinfo` shows node states; the `nvidia-smi`
query shows one row per GPU, a free device reads roughly 0MiB used and 0% utilisation.

❌ **No interactive node, no idle node, no free shared device**: fall back to `bigbatch` via
`sbatch`, per the same execution-protocol file.

## 2. Launch on a shared device

`unverified`, transcribed 2026-08-24, not run this sitting

⚠️ Starts a real process on a node another user may be sharing. Read the safety rules in
`execution-protocol.md` first (never start on a device carrying a foreign process, re-check the
device inside the launch script itself, keep VRAM within the free device only, record node/device/PID
in the log header). Do step 1 above first.

```bash
ssh <node> 'GPU=<free index> nohup bash /abs/path/to/launch.sh > /abs/path/to/logs/<name>.log 2>&1 &'
sleep 5; ssh <node> 'pgrep -af "train"; tail -20 /abs/path/to/logs/<name>.log'
```

**Every path must be absolute.** A non-interactive SSH command starts in `$HOME`, not the repo,
and `cd <repo> &&` is not reliable (`known-failures.md`, entry `poe-launch-001`, three identical
silent failures on 2026-08-19). The launch script does `cd "$REPO"` internally so the caller
needs no working directory.

✅ Expected: the `pgrep` line shows the new process; the log tail shows the preflight block
passing (disk guard, `co3` path check, `nvidia-smi` guard) followed by real training/generation
output.

❌ **`pgrep` shows nothing**: the launch failed before backgrounding; read the full log, not just
the tail, for the preflight guard that aborted it.

## 3. Harvest what is running

`unverified`, transcribed 2026-08-24, not run this sitting

Two execution modes, and Slurm only sees one of them:

```bash
squeue -u mmolefe                     # normal sbatch jobs
ssh <node> "pgrep -af 'sweep|train'"  # shared-device runs launched with nohup; invisible to squeue
```

**Why both.** `biggpu` allows one job per user, so long sweeps on a shared half-used node are
started with `nohup` outside Slurm rather than via `sbatch`, and Slurm is blind to them. A
harvest that only checks `squeue` will report "nothing running" while a real sweep is mid-flight
on another node.

✅ Expected: the output count against what the plan expected, and the tail of the log, per
`CLAUDE.md`'s "Harvest reads three execution modes, not one."

No ❌ branch beyond an unreachable node, which is an SSH/network problem, not a harvest problem.
