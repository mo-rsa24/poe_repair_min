# 🧭 Probing a pooled LoRA checkpoint, and training one

A `biggpu` node over SSH, or one Slurm job on an idle node: render the held-out cat x dog grid
for one checkpoint, measure how well a checkpoint fits the cached correction, resume a pooled
training from a checkpoint, and train two ranks side by side. Picking the device and the guards
every launch line carries are in
[launching and harvesting a run](launching-and-harvesting-a-run.md); this file does not repeat
them. Which node has which GPU and which python to use on it is in
[the cluster nodes](../../environment/hpc/nodes.md). How long a step, a checkpoint save and an
eval pass take is in [throughput](../../environment/hpc/throughput.md). What the training run
logs and how to read it is in
[reading a training run](../looking-at-what-a-run-produced/reading-a-training-run.md).

<details>
<summary><b>What these scripts touch, in three minutes</b></summary>

**The cache is the ground truth**

Every recipe here reads `/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/`. A
cell there is one pair and one seed: the pinned initial latent, the two single-prompt encodings,
and at every one of the 50 DDIM steps the state and the four branch predictions (each concept
alone, the joint prompt, unconditional). The correction the adapter learns is the joint prediction
minus the product-of-experts prediction at that state.

**A probe renders from a checkpoint**

`lambda_boundary_probe.py` loads one `lora_step_*.pt` into the adapter, takes each held-out seed's
pinned latent from the cache, and samples 50 steps adding λ times the adapter's correction on the
steps inside a window. It scores the renders and draws one grid per window. It never touches the
training run.

**A training run writes checkpoints and a W&B run**

`train_pooled.py` saves `lora_step_<N>.pt` every 5k steps under the run's `checkpoints/` folder,
renders the 4-cell tracking set every 10k, and logs to W&B project
`prime_lab/poe-repair-animals-compose`. Resuming means pointing it at one of those `.pt` files
with a new run id; the W&B run is new, the step counter continues.

</details>

## Table of contents

The recipes are numbered so the index and the pictures can point at them. The numbers are
addresses, not an order to follow: start at whichever one answers your question.

- [Words this file uses](#words-this-file-uses)
- [Before any of these](#before-any-of-these)
- [The reference block](#the-reference-block)
- [1. Render the 8-seed held-out grid for one checkpoint](#1-render-the-8-seed-held-out-grid-for-one-checkpoint)
- [2. Measure how well a checkpoint fits the cached correction](#2-measure-how-well-a-checkpoint-fits-the-cached-correction)
- [3. Resume a pooled training from a checkpoint on a free device](#3-resume-a-pooled-training-from-a-checkpoint-on-a-free-device)
- [4. Train two ranks on one idle node with one Slurm job](#4-train-two-ranks-on-one-idle-node-with-one-slurm-job)
- [If something looks wrong](#if-something-looks-wrong)
- [Where this came from](#where-this-came-from)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#before-any-of-these) ➡️

- **checkpoint**: one `lora_step_<N>.pt` file holding the adapter weights (`lora_state`) and the
  optimizer step, saved by the trainer every 5k steps.
- **held-out grid**: the 8 cat x dog seeds (9 to 16) rendered from one checkpoint, one row per
  seed, one column per λ, with the joint-prompt render and plain PoE as reference columns.
- **window**: which of the 50 DDIM steps get the correction. `full` is all 50, `early` is steps 0
  to 9 with plain guided PoE after.
- **fit cosine**: the cosine between the adapter's correction and the cached correction at the
  exact cached state, so the state is the one the trainer saw.
- **kill criterion**: the trainer aborts a fresh run if the commit-bucket loss has not halved from
  its step-200 value by step 5000. It is not saved in the checkpoint, so a resumed run is never
  aborted by it.

## Before any of these

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-reference-block) ➡️

- [ ] 🖥️ **A free device on a `biggpu` node**, found and checked the way
  [decide where a run goes](launching-and-harvesting-a-run.md#1-decide-where-a-run-goes) says.
  ```bash
  ssh mscluster108 nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv
  ```
  ✅ **A row like `1, 1 MiB, 0 %, 43`**: that device is free.
  ❌ **`[N/A]` or `[GPU requires reset]` in the last two columns**: the device is faulted, not free
  (`poe-launch-002` in [known failures](../../environment/known-failures.md)). Pick another node.
- [ ] 🖥️ **The script you launch lives on `/datasets` or `/home-mscluster`**, never in `/tmp`,
  which is node-local. Every path on the SSH line is absolute.

## The reference block

Navigation: ⬅️ [Before any of these](#before-any-of-these) | 📋 [TOC](#table-of-contents) | [Next](#1-render-the-8-seed-held-out-grid-for-one-checkpoint) ➡️

| Where the checkpoints are | Steps saved | Watch out |
|---|---|---|
| `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/` | rank 8, 5k to 100k, `lora_step_030000.pt` is the showcase pick | on `/home-mscluster`; read from it, never write new checkpoints there |
| `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_200k/checkpoints/` | rank 8, 105k to 200k | |
| `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_450k/checkpoints/` | rank 8, 205k to 450k | renders are hazy from about 240k on |
| `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r16_100k/checkpoints/` | rank 16; steps end in `018` after the resume (`lora_step_050018.pt`) | pass `--rank 16` to the probe |
| `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/` | rank 32; steps end in `050` after the resume (`lora_step_030050.pt`) | pass `--rank 32` to the probe |

The python to use: `co3` on the RTX 8000 and A6000 nodes (106, 108, 109), `co3_bw` on the
Blackwell nodes (110 to 112), per [the cluster nodes](../../environment/hpc/nodes.md).

## 1. Render the 8-seed held-out grid for one checkpoint   `ran 2026-09-03`

Navigation: ⬅️ [The reference block](#the-reference-block) | 📋 [TOC](#table-of-contents) | [Next](#2-measure-how-well-a-checkpoint-fits-the-cached-correction) ➡️

**When you need this**

You have a checkpoint and want to see whether it composes cat x dog on the held-out seeds, at
which λ, and how the renders look, as one grid with the scorer's verdict on every tile.

**Fill in**

| What | Example | Where to get it |
|---|---|---|
| the checkpoint | `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_090050.pt` | the reference block above |
| `--rank` | `32` | must match the checkpoint's rank |
| `--out-root` | `/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_090050` | one folder per checkpoint; the grid and `results.json` land there |
| `--windows` | `full` | `full`, `early`, or `full,early` |
| `--lambdas` | `1.0,1.2` | λ 0 (plain PoE) is always rendered as the reference column |

Write the launch script beside the output on `/datasets`, then run it over SSH on the free
device. This is the script that produced `figure_r32_090050` on mscluster109 device 1:

```bash
#!/bin/bash
set -uo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python   # co3 on 106/108/109 works too
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export CUDA_VISIBLE_DEVICES=1                                    # change this
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i 1)
if [ "$USED" -gt 1024 ] || [[ "$UTIL" == *"N/A"* || "$UTIL" == *"reset"* ]]; then echo "ABORT: device 1 used=${USED}MiB util='$UTIL'" >&2; exit 1; fi
$PY scripts/showcase/lambda_boundary_probe.py --rank 32 \
  --checkpoint /datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_090050.pt \
  --out-root /datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_090050 \
  --windows full --lambdas 1.0,1.2
echo "[$(date -u +%H:%M:%S)] DONE figure render r32_090050."
```

```bash
ssh mscluster109 'nohup bash /datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_090050/probe.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/figure_r32_090050.log 2>&1 &'
```

✅ **One line per render, then a summary line and the grid path.** Real output, 2026-09-03:
```
[probe] LoRA attached+loaded: n_matched=210 n_loaded=420 checkpoint_step=90050
[probe] seed=9 window=full lambda=0.0 -> seed_9_lambda_0.0.png max||delta||=68.4
...
  full  l=0.0: 0.000 cat+dog 0.000 (drift +0.575, max|d| 68.4)  l=1.0: 0.875 cat+dog 0.875 (drift -0.013, max|d| 53.8)  l=1.2: 0.875 cat+dog 0.875 (drift -0.035, max|d| 43.1)
[probe] wrote /datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_090050/grid_full_window.png
[11:32:28] DONE figure render r32_090050.
```
The summary line reads: compose fraction over the 8 seeds, then the stricter cat-and-dog
fraction, then the mean DINOv2 drift (negative = nearer the joint-prompt image than plain PoE)
and the largest per-step correction norm. About 25 minutes for 8 seeds x 3 columns on an A6000,
plus about 5 minutes of scoring. Read the log on the launch node over SSH; the session node's
view of `/datasets` can lag by minutes.

❌ **`CUDA out of memory` in the first seconds**: another process holds the device (this happened
on the session node's 3090 with a 12.8 GB co-tenant). Move to a free `biggpu` device.

❌ **Shape mismatch while loading `lora_state`**: `--rank` does not match the checkpoint.

**Variations**

- Re-score existing renders without rendering again (after a scorer change):
  `... lambda_boundary_probe.py --rank 32 --checkpoint <same> --out-root <same> --windows full --lambdas 1.0,1.2 --measure --grid`.
  Pass the same `--lambdas` the renders were made with, or the grid draws only the λ columns
  you named.
- Rebuild the grid from `results.json` only: add `--grid` alone.
- A λ sweep on both windows: `--windows full,early --lambdas 1.0,1.2,1.3,1.4,1.5,2.0` (about
  2 hours on an A6000 for 8 seeds).

<details>
<summary><b>Why the early window renders plain PoE after step 9, not the unconditional model</b></summary>

**Two samplers exist and only one is the tracking sampler**

`run_lora_residual_inject_windowed_poe` in `lambda_window_grid.py` adds the correction inside the
window and runs plain guided PoE outside it, which is what a partial correction of PoE means.
The older `run_lora_residual_inject_masked` runs the unconditional model outside the window,
which answers a different question. The probe uses the first.

</details>

## 2. Measure how well a checkpoint fits the cached correction   `ran 2026-09-03`

Navigation: ⬅️ [1. Render the 8-seed held-out grid](#1-render-the-8-seed-held-out-grid-for-one-checkpoint) | 📋 [TOC](#table-of-contents) | [Next](#3-resume-a-pooled-training-from-a-checkpoint-on-a-free-device) ➡️

**When you need this**

You want the adapter's fit on the states it trained on, and on held-out cells, without any
sampling: the cosine between its correction and the cached one at every cached state, per pair
and per step bucket.

**Fill in**

| What | Example | Where to get it |
|---|---|---|
| `--checkpoint` | `.../phase1_r8_100k/checkpoints/lora_step_030000.pt` | the reference block |
| `--rank`, `--alpha` | `8 8` | alpha equals rank in every run here |
| `--out` | `/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_r8_030000` | one folder per checkpoint |
| `--step-stride` | `5` | every fifth cached step (10 per cell); `1` reads all 50 and takes five times longer |

```bash
#!/bin/bash
set -u
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_r8_030000
GPU=${GPU:-1}
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$GPU")
if [ "$USED" -gt 1024 ] || echo "$UTIL" | grep -q "N/A\|reset"; then echo "ABORT device $GPU busy/faulted: ${USED}MiB $UTIL"; exit 3; fi
PCT=$(df --output=pcent /datasets | tail -1 | tr -dc 0-9); [ "$PCT" -ge 90 ] && { echo "ABORT /datasets at ${PCT}%"; exit 4; }
cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/showcase/fit_cosine_on_cache.py \
  --checkpoint "$REPO/artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_030000.pt" \
  --out "$OUT" --rank 8 --alpha 8 --step-stride 5
echo "DONE fit-cosine r8_030000 exit=$? $(date)"
```

```bash
ssh mscluster108 'GPU=1 nohup bash /datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_r8_030000/run.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_r8_030000/run.log 2>&1 &'
```

✅ **One line per cell, then a per-pair table.** Real output, 2026-09-03, 14 minutes for 152
cells on an RTX 8000:
```
[fit] checkpoint step=30000 rank=8 commit_window=(5, 25)
[fit] train a_wolf__x__a_husky seed=1 done (10 rows, 31s)
...
| **train mean** | | 880 | **0.969** | 0.985 | 0.975 | 0.961 | 0.987 |
| **heldout mean** | | 640 | **0.800** | 0.925 | 0.832 | 0.748 | 0.874 |
DONE fit-cosine r8_030000 exit=0
```
Columns after the count are the mean cosine over all sampled steps, then over steps 0 to 4,
5 to 24, 25 to 49, then the norm ratio of the adapter's correction to the cached one. The table
is also written as `fit_cosine_table.md` and the per-row data as `fit_cosine.json` in `--out`.

❌ **`[fit] MISSING <pair> seed=<n>`**: that cell is not in the cache for that split. The run
continues without it; the pair's `n` in the table shows how many rows it got.

**Variations**

- A different pool (other pairs or seeds): pass `--pair-pool`, `--seed-pool-path` and
  `--pair-prompts` pointing at YAML files of the same shape as the ones under
  `artifacts/results/does-the-fix-reach-unseen-pairs/`. This is how the seen-words-unseen-pair
  tiers were read (their pool files sit beside `fit_cosine_tiers_r8_030000/`).

## 3. Resume a pooled training from a checkpoint on a free device   `ran 2026-09-01`

Navigation: ⬅️ [2. Measure how well a checkpoint fits](#2-measure-how-well-a-checkpoint-fits-the-cached-correction) | 📋 [TOC](#table-of-contents) | [Next](#4-train-two-ranks-on-one-idle-node-with-one-slurm-job) ➡️

**When you need this**

A run stopped (finished its step budget, or the kill criterion aborted it) and you want to
carry the same adapter on to more steps, on whichever device is free, without changing anything
else about the recipe.

**Fill in**

Edit these at the top of `scripts/showcase/experiment_a_resume.sh`, which is the resume launcher
for the rank 8 lineage and the template for the others:

| What | Example | Where to get it |
|---|---|---|
| `RESUME_FROM` | `$SCOPE/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt` | the reference block |
| `RUNID` | `phase1_r8_200k` | new id per resume; the W&B run takes this name |
| `--total-epochs` (inside the script) | `4000` for 200k steps at 50 steps per epoch | the step budget you want, divided by `--epoch-size` |
| `PY` | `co3_bw` for a Blackwell node, `co3` otherwise | [the cluster nodes](../../environment/hpc/nodes.md) |
| the device index | `0` | the first argument to the script |

```bash
ssh mscluster110 'nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/experiment_a_resume.sh 0 > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/experiment_a_resume.log 2>&1 &'
```

✅ **The header says which node and device, the guards pass, and the adapter attaches.** Real
output, 2026-09-01:
```
[23:17:32] node=mscluster110.ms.wits.ac.za device=0 === phase1_r8_200k (resume from 100k) ===
[23:17:32] guards passed: device 0 used=2MiB, /datasets=23%, resume checkpoint present
23:17:43 INFO __main__: run_dir=/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_200k
23:18:10 INFO __main__: LoRA attached: n_matched=210 trainable_params=4956160
```
The W&B run appears within a minute at
`https://wandb.ai/prime_lab/poe-repair-animals-compose` under the run id. The first checkpoint
lands at the next multiple of 5k, the first tracking-set images at the next multiple of 10k.
100k steps of rank 8 took 6.8 hours on the Blackwell (`8vl2uzak`) and 18.1 hours for 250k
(`n1w4fw5b`).

❌ **The log stops after "LoRA attached" and never prints a step**: the device is hardware
faulted even though the memory guard passed (mscluster111, 2026-09-02, `poe-launch-002`).
`kill -9` the process and relaunch on another node.

❌ **`ABORT: device 0 has <N>MiB in use`**: someone took the device between your check and the
launch. Re-check and pick another.

**Variations**

- Rank 16 or 32: the fresh-run launcher is `scripts/showcase/experiment_b_rank.sh <rank> <device>`
  (shared-device path). The two resumes after the kill-criterion aborts (`phase1_r16_100k` from
  step 37,418 on mscluster108 device 1, `phase1_r32_100k` from 6,500 on mscluster106 device 1)
  were launched with a copy of that script pointed at the abort checkpoint via `--resume-from`;
  their logs are `logs/experiment_b_rank16_resumed.log` and `logs/experiment_b_rank32_resumed.log`
  under the showcase outputs.
- Rank 32 with AdamW weight decay on (scope 01 plan 15): `scripts/showcase/experiment_d_weight_decay.sh <wd> <device> [python]`,
  the same flags as the rank-32 run plus `--weight-decay` and the kill criterion disabled. Its
  memory guard is 4096 MiB and it adds a utilisation guard (under 5%) and a `torch.cuda` check.
  Run id `phase1_r32_wd<wd>_100k` under the showcase outputs; log `logs/experiment_d_wd<wd>.log`.

<details>
<summary><b>Why a resumed run cannot be killed by the kill criterion</b></summary>

**The reference loss lives only in memory**

The trainer records the commit-bucket loss at step 200 of a fresh run and aborts if it has not
halved by step 5000. That reference value is not saved in the checkpoint, so a resumed run has
none and the check never fires. This is why both experiment B runs that aborted (rank 32 at
6,500, rank 16 at 37,418) ran to 100k once resumed. The ETA line in the log is computed from
epochs run since the resume, not since step 0.

</details>

## 4. Train two ranks on one idle node with one Slurm job   `ran 2026-09-01`

Navigation: ⬅️ [3. Resume a pooled training](#3-resume-a-pooled-training-from-a-checkpoint-on-a-free-device) | 📋 [TOC](#table-of-contents) | [Next](#if-something-looks-wrong) ➡️

**When you need this**

A whole `biggpu` node is idle and you want both of its GPUs for two independent trainings that
nobody else can land on. Slurm allocates whole nodes here, so one job reserves both devices and
the script runs one training per device in the background.

**Fill in**

| What | Example | Where to get it |
|---|---|---|
| `#SBATCH --nodelist` | `mscluster106` | `sinfo -N -p biggpu -o "%N %t"`, then `nvidia-smi` over SSH to confirm both devices are free |
| the two `run_rank` lines | `run_rank 0 16 16 &` and `run_rank 1 32 32 &` | device index, rank, alpha |

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
sbatch scripts/showcase/experiment_b_rank.sbatch
squeue -u mmolefe
```

✅ **One job in the queue, two logs growing.** Real output, 2026-09-01, job 48965:
```
Submitted batch job 48965
  JOBID PARTITION       NAME     USER ST  TIME  NODES NODELIST(REASON)
  48965    biggpu experime  mmolefe  R  0:41      1 mscluster106
```
with `logs/experiment_b_rank16.log` and `logs/experiment_b_rank32.log` each printing the same
header block as recipe 3, and two W&B runs both named by their run id (`jii2mtb0` and `le2tp2ti`
on that day). The Slurm log is `logs/experiment_b_rank_slurm_<jobid>.log`.

❌ **`slurm_script: line 50: RANK: unbound variable` and the job exits at once** (job 48964):
`set -u` with several variables declared on one `local` line, where a later one referenced an
earlier one. Declare them on separate lines.

❌ **One log ends with `kill: commit-bucket loss 0.0033 did not halve from initial 0.0066 by
step 6500`** (rank 32, `le2tp2ti`, 2.3 hours in; rank 16 the same at 37,418 after 12.3 hours):
the kill criterion fired on a fresh run whose loss started low. The last checkpoint is on disk
under the run's `checkpoints/`; resume from it with recipe 3, which the criterion cannot stop.

❌ **`sbatch` is refused although nodes are idle**: the administrator cap on nodes per user.
Take the shared-device path in [launch on a shared device](launching-and-harvesting-a-run.md#2-launch-on-a-shared-device).

## If something looks wrong

Navigation: ⬅️ [4. Train two ranks](#4-train-two-ranks-on-one-idle-node-with-one-slurm-job) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**A count or a listing on the session node is short.** The session node's NFS view of `/datasets`
lags the launch node by minutes. Count and tail over SSH on the node that wrote the files.

**A `pkill -f` or `pgrep -f` returns exit 144 or kills your own shell.** The pattern matched the
command line of the shell running it. Bracket the first letter (`[l]ambda_boundary_probe`) or
kill by PID.

**The renders look right but the scorer says one animal.** Seed 16's plain-PoE tile is an
example: two animals to the eye, one instance to the detector. The disagreement is recorded in
the finding, not resolved.

## Where this came from

Navigation: ⬅️ [If something looks wrong](#if-something-looks-wrong) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| Recipe 1's script and output | run as `/tmp/probe_r32_090050.sh` on mscluster109 device 1; log `logs/figure_r32_090050.log` | 2026-09-03 |
| Recipe 2's script and output | run as `fit_cosine_r8_030000/run.sh` on mscluster108 device 1; log `run.log` beside it | 2026-09-03 |
| Recipe 3's launch line and output | `logs/experiment_a_resume.log`, W&B `8vl2uzak` | 2026-09-01 |
| Recipe 4's job and both failures | `logs/experiment_b_rank_slurm_48964.log`, `..._48965.log`, `logs/experiment_b_rank32.log`, `logs/experiment_b_rank16.log` | 2026-09-01 to 09-02 |
| The faulted-device hang | mscluster111, `poe-launch-002` in [known failures](../../environment/known-failures.md) | 2026-09-02 |
| Durations | W&B `_runtime` for each run, read with the `wandb` API | 2026-09-05 |
