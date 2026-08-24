# HPC: how a run actually gets launched

Navigation: ⬅️ [Nodes](nodes.md) | 📋 [Index](../00-INDEX.md) | [Overview](../overview.md#execution-model)

"Running an experiment" means one of two paths, decided by this protocol, in order. This is
the section a plan is most likely to silently get wrong, because "submit a job" reads as one
action and is actually a several-step decision.

1. **Check for an already-allocated interactive node.** `squeue -u mmolefe`: if an interactive
   node is already allocated on `biggpu`, run directly there, no `sbatch`.
2. **Check for an idle node.** Else `sinfo -p biggpu`: if a `biggpu` node is idle, target
   `biggpu`. Otherwise check the shared-device path (next step) before falling back to
   `bigbatch`.
3. **The shared-device path.** An `alloc` `biggpu` node is often only half used, because Slurm
   here allocates whole nodes while knowing nothing about GPUs (GRES is `(null)`, see
   `nodes.md`) and most users run on one device. SSH in and read per-device state:
   `ssh <node> nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`. A device
   at roughly 0MiB used and 0% utilisation is free. Slurm cannot schedule onto that node (it is
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

   **Safety rules for the shared-device path, all mandatory:**
   - Never start on a device carrying a foreign process.
   - Re-check the device inside the launch script itself (a guard that aborts if the chosen
     device has more than 1GB in use), not only in the preflight check minutes earlier: state
     can change between the check and the launch.
   - Keep VRAM usage within the free device only.
   - Record node, device index, and PID in the log header.
   - These runs are invisible to `squeue` (Slurm does not know about them), so harvesting a
     shared-device run means `pgrep -af 'sweep|train'` on the node, not `squeue`.

4. **Every job script carries a preflight block:** a `df` disk guard on the checkpoint target
   (abort at 90% full; must check the filesystem the script actually writes to, see
   `storage.md`), a `co3` python path check, and an `nvidia-smi` guard that aborts in seconds if
   no GPU is visible.
5. **Submit, then poll.** `squeue`/`sacct` for a normal Slurm job, or tail the `nohup` log and
   `pgrep` on the node for a shared-device run. On failure, read the log and classify: OOM
   (reduce batch size or resolution, or move to `biggpu`), wrong environment (fix the path),
   missing GPU or bad `#SBATCH` directives (fix and resubmit), node failure (resubmit
   elsewhere). Retries are bounded; never silently loop.
6. **In-session tier.** Cache-only analyses (SVD, SNR curves, language probes, scoring cached
   PNGs) and light GPU inference run directly on the current session node (`mscluster85`, RTX
   3090 24GB), no queue, while bigger jobs wait. Reference point: `phase1_r8_100k` TRAINING
   peaked at 22.95GB VRAM, so training-scale work goes to `biggpu`; SDXL inference-only sweeps
   generally fit the 3090, but check `nvidia-smi` for co-tenants before launching.

## Skill wiring

`/run-experiment` drives GPU tasks. Every experiment logs to W&B, including the qualitative
Mono vs PoE vs LoRA triptych panels, so `/analyze-run` can sweep runs later. `/execute-plan-tree`
may run tasks unattended, using each plan's pre-registered falsification rules and
`/demonstrate` checkpoints as its stop conditions.

**Provenance.** This protocol, the shared-device path, and the launch-failure pattern were
established live over SSH on 2026-08-19 on `mscluster106`: GPU 1 sat at 1MiB/0% while GPU 0
carried another user's 8GB process, and a torch matmul from `co3_bw` on GPU 1 succeeded
without touching GPU 0.
