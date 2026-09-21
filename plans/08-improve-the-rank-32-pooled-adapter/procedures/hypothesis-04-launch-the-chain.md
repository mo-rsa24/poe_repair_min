# 🧭 Claim a Blackwell, smoke the parent, and launch the chain of five runs

[Plan 04's instruction 4.1](../plans/hypothesis/04-the-parent-and-the-four-children.md#4--claim-the-device-and-launch-the-chain) sends you here. When you are done, five trainings are running one after another on one device, the log header names the node, the device and the process id, and the review file's Runs table records all of it. Nothing else needs doing until the first checkpoint lands.

## Recommended prompt (when you finish)

```
/analyze-run <the parent's W&B run id>
```
(If a launch failed in a way worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/04-the-parent-and-the-four-children.md) | the five configurations and the verdict rule |
| [review](../review/04-the-parent-and-the-four-children.md) | the Runs table this procedure fills, and the questions the runs answer |
| **this file** | **the steps, and the reason behind each one** |

## Table of contents

- [Why you are doing this, and where it lands](#why-you-are-doing-this-and-where-it-lands)
- [Words this file uses](#words-this-file-uses)
- [Before you start](#before-you-start)
- [What you need to understand first](#what-you-need-to-understand-first)
- [1. Find out which Blackwell is actually free today](#1-find-out-which-blackwell-is-actually-free-today)
- [2. Copy the launcher onto the shared disk](#2-copy-the-launcher-onto-the-shared-disk)
- [3. Smoke the parent for 200 steps, then kill it](#3-smoke-the-parent-for-200-steps-then-kill-it)
- [4. Launch the chain, and verify it took](#4-launch-the-chain-and-verify-it-took)
- [5. Harvest, on the node](#5-harvest-on-the-node)
- [If it does not work](#if-it-does-not-work)
- [What you produce](#what-you-produce)
- [Next step](#next-step)

## Why you are doing this, and where it lands

Navigation: 📋 [TOC](#table-of-contents) | [Next](#words-this-file-uses) ➡️

**What is at stake**

About twelve hours on the only free high-end device, and the answer to whether the rank-32 adapter can be trained into something that draws a clean cat and a clean dog. A launch that goes wrong quietly costs the night and is not noticed until morning.

**What is already settled, and what is not**

The five configurations, the order they run in, and the rule that judges them are all fixed before you start, and none of them is decided here. What is not settled is which device is free today, which is a fact with a date on it and has to be re-checked rather than read from a plan.

**Why now**

Nothing in the scope after this can be read until the checkpoints exist, and the chain wants a whole night.

## Words this file uses

Navigation: ⬅️ [Why you are doing this](#why-you-are-doing-this-and-where-it-lands) | 📋 [TOC](#table-of-contents) | [Next](#before-you-start) ➡️

- **The chain**: one script that runs the five trainings in order, moving to the next if one dies rather than stopping, and saying so in the log.
- **`nohup`**: a way of starting a program so it keeps running after you close the connection that started it. The chain is launched this way because the partition allows only one Slurm job per user, so these runs are started outside Slurm and are invisible to the queue.
- **The device guard**: a check inside the launcher that refuses to start if the target GPU already has more than 1 GB in use, or if it reports no readable state at all. Both cases mean somebody else's work or broken hardware.
- **The disk guard**: a check that aborts if the filesystem being written to is over 90% full.
- **`co3_bw`**: the python environment built for the Blackwell cards. The other one, `co3`, will run there and produce no output rather than an error, which is the worst kind of failure because it looks like success.
- **`pgrep`**: lists running processes by name. It is how you see a run the job queue cannot see.

## Before you start

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-you-need-to-understand-first) ➡️

- [ ] 🖥️ **The switches exist and the dry run passed.**

  ```bash
  cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
  grep -n "exclude-cells\|train-step-range\|orth-weight" poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py
  ```

  ✅ **All three appear in the argument parser.** [Plan 01](../plans/tools/01-the-three-switches-and-the-guidance-interval.md) is done.
  ❌ **They do not**: stop. Nothing here can run and plan 01 is where to go.

- [ ] 🖥️ **The five configurations differ from the parent by one line each.**

  ```bash
  for c in C1 C2 C3 C4; do echo "== $c"; diff /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/configs/P.args \
    /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/configs/$c.args; done
  ```

  ✅ **Each diff is one changed line.** The one-axis rule holds mechanically, not just in the plan's prose.
  ❌ **Any diff is longer**: that child's comparison is already mixed and reading it would attribute a gap to the wrong thing. Fix the configuration before launching.

- [ ] 🖥️ **Nothing of yours is already running that this would compete with.**

  ```bash
  squeue -u mmolefe
  pgrep -af 'sweep|train_pooled'
  ```

  ✅ **Both come back empty, or name only work you know about and expect.**
  ❌ **Something is running**: the queue allows one job per user on this partition, and two SDXL trainings on one card will not fit. Decide which one you want before continuing.

## What you need to understand first

Navigation: ⬅️ [Before you start](#before-you-start) | 📋 [TOC](#table-of-contents) | [Next](#1-find-out-which-blackwell-is-actually-free-today) ➡️

**Why this runs outside the job queue.**

The partition holding these cards allows one queued job per user. A chain of five trainings submitted as five jobs would run one at a time with a wait between each, and a chain submitted as one long job risks losing everything to a single walltime limit. Starting it directly on a node with `nohup` sidesteps both. The cost is that the job queue cannot see the run at all, so every later check on it uses `pgrep` on the node rather than the queue.

**Why the device has to be re-checked rather than read.**

The plan names `mscluster112` as free, verified on a specific date. Another user's job can claim it at any time, and a card that was healthy last week can fault. The guard inside the launcher is the backstop, not the plan's sentence.

**Why a relative path in the launch line is dangerous here.**

A command sent over a connection to another machine resolves relative paths against your home directory, not against the repository you are standing in. The failure is silent: the command finds nothing, or finds something else with the same name. Every path on the launch line is absolute for this reason, and it is a mistake this project has already made and catalogued.

**Why the launcher is copied to the shared disk first.**

Each node has its own temporary directory, so a launcher written to one is not visible from another. It goes under the run's own output directory on the shared data disk, where every node can read it and where it sits beside the outputs it produced.

**⚠️ Why you must not edit the launcher once it is running.**

The shell reads a script by byte position as it goes, not all at once. Editing a running script shifts everything after the point it has reached, and what it executes next is whatever now sits at that offset, which is usually a fragment of a line. This has already cost a run in this project. If the launcher is wrong, kill it, edit it, and start again.

## 1. Find out which Blackwell is actually free today

Navigation: ⬅️ [What you need to understand first](#what-you-need-to-understand-first) | 📋 [TOC](#table-of-contents) | [Next](#2-copy-the-launcher-onto-the-shared-disk) ➡️

The plan's device is a dated fact, so check it rather than trusting it.

```bash
for n in mscluster110 mscluster111 mscluster112; do
  echo "== $n"
  ssh $n 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
done
```

✅ **A node reporting a few MiB used and 0% utilisation on a device.** That is the one to use, and you note its index.

❌ **A node reporting `[N/A]` for utilisation**: the card is hardware-faulted and torch will not see it. Leave it alone and record it in [the node facts](../../../environment/hpc/nodes.md) if it is not already there.

❌ **Every node is busy**: do not launch. The chain wants twelve uninterrupted hours and starting it beside somebody else's job will run both out of memory. Wait, or check the other partitions' cards knowing the run will take four times as long.

- [ ] Write down the node and the device index. Every command below uses them.

## 2. Copy the launcher onto the shared disk

Navigation: ⬅️ [1. Find the free Blackwell](#1-find-out-which-blackwell-is-actually-free-today) | 📋 [TOC](#table-of-contents) | [Next](#3-smoke-the-parent-for-200-steps-then-kill-it) ➡️

So every node can read it, and so it sits beside the outputs it produces.

The source is the file [task 2.2](../plans/hypothesis/04-the-parent-and-the-four-children.md#2--build-the-five-configurations-and-the-launcher) writes into the repository. If it is not there, that task is not done and this procedure cannot start.

```bash
ls -l /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/improve_r32_launch_chain.sh
mkdir -p /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs
cp /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/improve_r32_launch_chain.sh \
   /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/launch_chain.sh
grep -c "^" /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/launch_chain.sh
```

✅ **The copy exists and its line count matches the source.** From here on, the copy is the one that runs, and the source is the one you edit.

❌ **The copy is missing or truncated**: the shared disk may be full. Check it before anything else, since a full disk is also what stops checkpoints from being written later.

## 3. Smoke the parent for 200 steps, then kill it

Navigation: ⬅️ [2. Copy the launcher](#2-copy-the-launcher-onto-the-shared-disk) | 📋 [TOC](#table-of-contents) | [Next](#4-launch-the-chain-and-verify-it-took) ➡️

⚠️ This is the step that costs five minutes and saves a night. It proves four things at once: the environment resolves, the tracker receives the run, a checkpoint reaches the shared disk, and the step time is what the estimate assumed.

```bash
ssh <node> 'GPU=<idx> SMOKE=1 nohup bash /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/launch_chain.sh \
  > /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/smoke.log 2>&1 &'
```

Wait about three minutes, then read it on the node:

```bash
ssh <node> 'tail -40 /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/smoke.log'
```

✅ **The log header names the node, the device and the process id; a W&B run URL appears; a checkpoint file lands under the smoke output directory; the reported step time is near 0.24 s.** Everything the chain depends on is proven.

- [ ] 🌐 **Open the printed W&B URL in a browser and confirm the run is actually receiving data**: the charts tab shows `train/loss` with points on it, and the media panel shows the tracking thumbnails. A run that was created and is logging nothing looks identical in the log.
      ❌ **The run page exists but every panel is empty**: the tracker was reached at startup and nothing is arriving since. Do not launch the chain; twelve hours would produce no readable curves.

❌ **The step time is far above 0.24 s**: the job is not on the card you think it is. Check the device index in the header against the one you chose. Twelve hours of chain would otherwise land somewhere unexpected and take four times as long.

❌ **No W&B URL appears**: the run is training into nothing observable. Fix the tracker settings before launching, because the strips and curves are how the runs are read.

❌ **The device guard refused**: another user's job claimed the card between step 1 and now. Go back to step 1 and pick again. Do not disable the guard.

Then kill it, so the real chain starts from a clean device:

Read the smoke's process id out of its own log header rather than matching on a pattern, then kill that one process:

```bash
ssh <node> 'grep -m1 "^PID" /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/smoke.log'
ssh <node> 'kill <the pid that printed>; sleep 2; pgrep -af train_pooled'
```

✅ **`pgrep` returns nothing.** The card is yours and empty.

❌ **The process is still there after the kill**: give it `kill -9 <pid>` and check again. Do not fall back to a pattern match.

- [ ] ⚠️ Kill by process id, never with `pkill -f` on a pattern. A pattern sent over `ssh` appears in the remote shell's own command line, so `pkill -f` matches the shell running it and kills the connection that issued the command. That is the catalogued failure that exits 144, and the launcher writes its process id into the log header precisely so this step never needs a pattern.

## 4. Launch the chain, and verify it took

Navigation: ⬅️ [3. Smoke the parent](#3-smoke-the-parent-for-200-steps-then-kill-it) | 📋 [TOC](#table-of-contents) | [Next](#5-harvest-on-the-node) ➡️

```bash
ssh <node> 'GPU=<idx> nohup bash /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/launch_chain.sh \
  > /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/chain.log 2>&1 &'
sleep 5
ssh <node> 'pgrep -af train_pooled; tail -20 /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/chain.log'
```

✅ **`pgrep` shows one training process and the log's header names the node, the device and the process id.** The chain is running and will keep running after you disconnect.

❌ **`pgrep` shows nothing and the log holds a guard message**: read which guard fired. Disk over 90%, wrong python, or a busy device each have their own fix, and none of them is to run it again unchanged.

- [ ] Record the node, the device, the process id and the launch time in [the review file's Runs table](../review/04-the-parent-and-the-four-children.md#runs), and add each W&B run id as it appears.

## 5. Harvest, on the node

Navigation: ⬅️ [4. Launch the chain](#4-launch-the-chain-and-verify-it-took) | 📋 [TOC](#table-of-contents) | [Next](#if-it-does-not-work) ➡️

In the morning. Read it on the node, because the session node's view of the shared disk lags by minutes and a checkpoint count read from there can be wrong.

```bash
ssh <node> 'pgrep -af train_pooled; \
  for r in P C2 C1 C3 C4; do echo -n "$r: "; \
    ls /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/$r/checkpoints/ 2>/dev/null | wc -l; done; \
  tail -60 /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/chain.log'
```

✅ **Five directories each holding a checkpoint at 30,000 steps, and `pgrep` empty.** The chain is done.

❌ **Fewer than five**: the log names which run died and that the chain moved on. This is the designed behaviour, not a failure of the night. Record which died in the Runs table, and read the runs that finished; a partial chain still answers the parent's question if the parent is among them.

- [ ] Record the checkpoint counts and any death in the Runs table.

## If it does not work

Navigation: ⬅️ [5. Harvest](#5-harvest-on-the-node) | 📋 [TOC](#table-of-contents) | [Next](#what-you-produce) ➡️

**Every Blackwell is busy.** Do not squeeze the chain beside another job. Either wait, or run it on the older cards knowing each run takes four times as long and needs gradient checkpointing to fit, and say in the Runs table which hardware was used, since the estimate no longer applies.

**A run runs out of memory.** Rank 32 needs about 25 GB. On a 24 GB card the trainer's gradient-checkpointing flag makes it fit at some cost in speed. On the Blackwell it should not happen; if it does, something else is on the card.

**The chain dies part-way and the log ends abruptly.** Check whether the launcher was edited while running, which corrupts what the shell reads next. If it was, that is the cause and the fix is to relaunch the runs that did not complete from the unedited copy.

**A run finished but its W&B record is empty.** The checkpoints are still on disk and the probe pass can read them, so the run is not lost. Note it in the Runs table and point the probe at the checkpoint directly.

## What you produce

Navigation: ⬅️ [If it does not work](#if-it-does-not-work) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What | Where it lands | What it answers |
|---|---|---|
| five checkpoints at 30,000 steps | `improve_r32/{P,C2,C1,C3,C4}/checkpoints/` | everything [plan 04](../plans/hypothesis/04-the-parent-and-the-four-children.md) goes on to judge |
| five W&B runs with curves and thumbnails | `prime_lab/poe-repair-animals-compose` | whether training behaved, before the renders are looked at |
| the filled Runs table | [the review file](../review/04-the-parent-and-the-four-children.md#runs) | which run is which, on what hardware, and what it cost |

## Next step

Navigation: ⬅️ [What you produce](#what-you-produce) | 📋 [TOC](#table-of-contents)

Go back to [plan 04's task 3.1](../plans/hypothesis/04-the-parent-and-the-four-children.md#3--probe-every-checkpoint-once-the-chain-is-done) and run the probe pass over each checkpoint.
