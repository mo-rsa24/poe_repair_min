# HPC: how a run actually gets launched

Navigation: ⬅️ [Nodes](nodes.md) | 📋 [Index](../00-INDEX.md) | [Overview](../overview.md#execution-model)

"Running an experiment" means one of two paths, decided by this protocol, in order. This is
the section a plan is most likely to silently get wrong, because "submit a job" reads as one
action and is actually a several-step decision.

1. **Check for an already-allocated interactive node.** `squeue -u mmolefe`: if an interactive
   node is already allocated on `biggpu`, run directly there, no `sbatch`.
2. **Check for an idle node.** Else `sinfo -p biggpu`: if a `biggpu` node is idle, target
   `biggpu`. Otherwise check the shared-device path (next step) before falling back to
   `bigbatch`. **`sinfo`'s `idle` is not proof the GPU is free**: work launched over SSH with
   `nohup` (the shared-device path below) is invisible to Slurm, so a node can read `idle` while
   a training saturates its GPU (2026-09-03: mscluster112 showed `idle` with a rank-8 LoRA run at
   94% on it). Always confirm with the per-device `nvidia-smi` read in step 3 before pinning a
   device, whatever `sinfo` says.
3. **The shared-device path.** An `alloc` `biggpu` node is often only half used, because Slurm
   here allocates whole nodes while knowing nothing about GPUs (GRES is `(null)`, see
   `nodes.md`) and most users run on one device. SSH in and read per-device state:
   `ssh <node> nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv`.
   A device at roughly 0MiB used and 0% utilisation is free. Two readings that look wrong and
   are not: a device at under 100 MiB can read 100% utilisation for a minute or so after a
   process on it was killed, because `nvidia-smi`'s utilisation is a windowed sample and its
   memory column is not (2026-09-03, mscluster112), so under 100 MiB the memory column alone
   decides; and a device holding a few GB at 0% utilisation for hours is another user's idle
   allocation, which this project has shared safely (mscluster109 device 0, 2.9 GB, all of
   2026-09-03), the rule being no *active* foreign process rather than no foreign memory.
   `scripts/launch_sdlora.sh`'s guard encodes both. **A device reading `[N/A]` or
   `[GPU requires reset]` in the utilisation or temperature column is not free either — it is
   hardware-faulted**, and a memory-only check passes it just as readily as a healthy idle
   device. A launch onto a faulted device hangs producing zero progress with no error
   (`poe-launch-002`); the only recovery is `kill -9` and relaunch elsewhere, since the fault
   needs an administrator or node reboot to clear. Slurm cannot schedule onto that node (it is
   already allocated, and `biggpu` allows one job per user anyway), so the run goes there
   directly instead: SSH in, pin `CUDA_VISIBLE_DEVICES=<free index>`, launch with `nohup`, tee
   to a log the session node can read.

   **Every path on the SSH launch line must be absolute** (the script path and the log
   redirect): a non-interactive SSH command starts in `$HOME`, not the repo, and a `cd <repo>
   &&` prefix is not reliable either, because Claude Code's permission layer can strip or block
   `cd` inside compound Bash commands. This produced three identical silent failures in a row
   on 2026-08-19 (`known-failures.md`, entry `poe-launch-001`). The launch script does
   `cd "$REPO"` internally, so the caller needs no working directory at all:
   ```bash
   ssh <node> 'GPU=<idx> nohup bash /abs/path/to/launch.sh > /abs/path/to/logs/<name>.log 2>&1 &'
   ```
   Verify the launch in the same SSH call: `sleep 5; pgrep -af "train"` plus `tail` of the
   absolute log path.

   **The script itself must live on a shared filesystem** (`/datasets` or `/home-mscluster`),
   never in this session's `/tmp` scratchpad: `/tmp` is node-local, so a script written there
   does not exist on the launch node. The symptom is an SSH launch returning exit code 2 with
   no log file at all, because python never found the file to run (2026-09-03, mscluster109).
   Copy the script under the run's output root on `/datasets` first, and launch that path.

   **Safety rules for the shared-device path, all mandatory:**
   - Never start on a device carrying a foreign process.
   - Re-check the device inside the launch script itself (a guard that aborts if the chosen
     device has more than 1GB in use), not only in the check run minutes earlier before the
     launch: state can change between the check and the launch.
   - Keep VRAM usage within the free device only.
   - Record node, device index, and PID in the log header.
   - These runs are invisible to `squeue` (Slurm does not know about them), so harvesting a
     shared-device run means running `pgrep -af 'sweep|train'` on the node; `squeue` will never
     show it.

   **The admin node cap trips this path too.** `biggpu` carries an administrator cap on how
   many nodes a user may hold at once, separate from the one-job rule. When `sbatch` is denied
   by that cap even though idle nodes exist, do not wait for the queue: take the shared-device
   path onto an idle node instead. Device 1 has been the free device in practice (provenance
   below), so prefer it, but still verify with `nvidia-smi` before pinning
   `CUDA_VISIBLE_DEVICES`; the guard rules below stay mandatory.

4. **Every job script opens with a block of checks that must pass before the work starts:** a `df` disk guard on the checkpoint target
   (abort at 90% full; must check the filesystem the script actually writes to, see
   `storage.md`), a `co3` python path check, and an `nvidia-smi` guard that aborts in seconds if
   no GPU is visible.
5. **Submit, then poll.** `squeue`/`sacct` for a normal Slurm job, or tail the `nohup` log and
   `pgrep` on the node for a shared-device run. **Read the log and count the outputs on the
   launch node, over SSH, not from the session node**: the session node's NFS view of
   `/datasets` (and of `/home-mscluster`) can lag the launch node's writes by minutes, so a
   `grep -c` or an `ls` run locally undercounts, and a figure assembled locally from those files
   draws finished cells as missing (2026-09-03: 32 renders present on mscluster109, absent from
   mscluster85's view for over ten minutes). Assemble figures on the launch node for the same
   reason, and copy them into the repo from there. On failure, read the log and classify: OOM
   (reduce batch size or resolution, or move to `biggpu`), wrong environment (fix the path),
   missing GPU or bad `#SBATCH` directives (fix and resubmit), node failure (resubmit
   elsewhere). Retries are bounded; never silently loop.
6. **In-session tier.** Cache-only analyses (SVD, SNR curves, language-space tests, scoring cached
   PNGs) and light GPU inference run directly on the current session node (`mscluster85`, RTX
   3090 24GB), no queue, while bigger jobs wait. Reference point: `phase1_r8_100k` TRAINING
   peaked at 22.95GB VRAM, so training-scale work goes to `biggpu`; SDXL inference-only runs
   across many settings generally fit the 3090, but check `nvidia-smi` for co-tenants before
   launching.

## Skill wiring

`/run-experiment` drives GPU tasks. Every experiment logs to W&B, including the qualitative
Mono vs PoE vs LoRA triptych panels, so `/analyze-run` can go back over runs later. `/execute-plan-tree`
may run tasks unattended, using each plan's pre-registered falsification rules and
`/demonstrate` checkpoints as its stop conditions.

**Provenance.** This protocol, the shared-device path, and the launch-failure pattern were
established live over SSH on 2026-08-19 on `mscluster106`: GPU 1 sat at 1MiB/0% while GPU 0
carried another user's 8GB process, and a torch matmul from `co3_bw` on GPU 1 succeeded
without touching GPU 0.

## Cross-references

- The mention of **launching outside Slurm and harvesting by process** in [does a Langevin corrector remove part of the correction](../../report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md) and [does a corrector alone produce two animals](../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md), whose grids ran this way on three nodes.
- The mention of **a device taken between the check and the launch**, which this protocol's guard refused twice on 2026-09-05, in [running the Langevin corrector](../../runbook/running-things-on-the-cluster/running-the-langevin-corrector.md).
