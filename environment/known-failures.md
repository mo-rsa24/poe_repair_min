# Project Error Catalog: poe_repair_min

Navigation: 📋 [Index](00-INDEX.md) | [Overview](overview.md)

Patterns specific to this project: codebase structure, model sizes, dataset characteristics,
cluster constraints observed in practice. Per `~/.claude/ERROR_MATRIX_SYSTEM.md`, this file is
the project layer of the error matrix (the global, environment-independent layer is
`~/.claude/GLOBAL_ERROR_CATALOG.md`). An entry's symptom and root cause are the environment
fact; where the solution is a procedure someone drives by hand, it belongs in a `runbook/`
recipe once that folder exists (it does not yet, this sitting).

**Migrated from `environment/known-failures.md` this sitting**, with every `Environment
reference` repointed from `docs/ENVIRONMENT.md` to this folder. Nothing in the six migrated
entries was reworded beyond that repointing; `poe-disk-001` is new, added this sitting from a
failure already documented in the project's own `CLAUDE.md` but not previously catalogued here.

**Last updated:** 2026-08-24
**Total entries:** 7
**Seed entries from:** step-09 (three-live-curves-while-training), plus one added during this
folder's migration

---

## Error patterns (organized by subsystem)

### Scoring and evaluation

#### Entry ID: poe-score-001
**Name:** Scorer returns all zeros or NaNs despite valid inputs

**Symptom:** W&B shows `eval/compose_rate/mean = 0.0` or `NaN` for all eval steps, even though
generated images look visually correct.

**Root cause:** Scorer receives tensors with wrong dtype (int instead of float), wrong device
(CPU vs GPU), or an invalid value range (e.g. pixel values 0-255 instead of 0-1).

**Solution:** Before calling the scorer, validate eval outputs:
```python
assert eval_output.dtype == torch.float32, f"Expected float32, got {eval_output.dtype}"
assert eval_output.device.type == 'cuda', f"Expected GPU tensor, got {eval_output.device}"
assert eval_output.min() >= -0.1 and eval_output.max() <= 1.1, f"Range error: [{eval_output.min()}, {eval_output.max()}]"
```
Check the normalization pipeline in `experiments/one_pair_one_seed/main.py::eval_hook`.

**First discovered:** poe_repair_min, step-09

**Affects steps:** step-09, step-10, step-11 (any step using the `compose_rate` metric)

**Category:** 🔴 critical

**Environment reference:** `overview.md` (needs a section: tensor dtype and normalization for
the scorer; not yet written)

---

#### Entry ID: poe-score-002
**Name:** Compose-rate stuck at 0.0 even though the fix is active

**Symptom:** Correction is being applied (loss decreases), but `eval/compose_rate/mean = 0.0`
throughout training.

**Root cause:** The scorer was trained on PoE (product-of-experts) outputs in a particular
style or value range. The fix is changing the images, but not in the direction the scorer
recognizes.

**Solution:**
1. Manually inspect generated images (save a batch to disk, visualize in Jupyter).
2. Compare to the PoE reference images used to train the scorer.
3. If images look different (color space, brightness), recalibrate the scorer or adjust the
   correction target.
4. If images look correct but the scorer is wrong, the scorer model may need retraining on this
   data distribution.

**First discovered:** poe_repair_min, step-09

**Affects steps:** step-09, step-10 (diagnostic metric reliability)

**Category:** 🟡 warning

**Environment reference:** `overview.md` (needs a section: scorer calibration and reference
images; not yet written)

---

### Memory and batch sizes

#### Entry ID: poe-mem-001
**Name:** Eval hook OOM on `bigbatch` with full-size quadrants

**Symptom:** `RuntimeError: CUDA out of memory` during `eval/compose_rate` computation on a
V100 32GB.

**Root cause:** Full eval batch for cat x dog (11 training pairs x 4 seeds = 44 images) plus
DINOv2 embeddings (40M) plus CLIP embeddings (400M) exceeds V100 memory. A100 does not have
this issue.

**Solution:**
- Use A100 nodes (`mscluster110`) for evals with the full batch.
- OR: shard the eval batch to 8-16 images per forward pass, loop and aggregate.
- OR: reduce eval frequency (every 5 steps instead of every step).

Check [hpc/nodes.md](hpc/nodes.md) for which partitions and nodes carry which GPU models.

**First discovered:** poe_repair_min, step-09, V100 runs

**Affects steps:** step-09, step-10, step-11 (if the eval batch is not reduced)

**Category:** 🟡 warning

**Environment reference:** [hpc/nodes.md](hpc/nodes.md)

---

### LoRA and correction dynamics

#### Entry ID: poe-lora-001
**Name:** Fraction-of-distance-reached plateaus at 20% instead of an expected 40%

**Symptom:** `eval/frac_distance_reached/mean` climbs to about 0.2 by epoch 1, then stalls.
Expected plateau is about 0.4 based on prior runs.

**Root cause:** LoRA rank too low (r=4 or r=8) to capture the full correction magnitude. The
correction saturates the LoRA space before reaching the full PoE-to-Mono distance.

**Solution:**
1. Check the LoRA rank used (in config; default is r=16).
2. If r < 16, increase to r=32 or r=64 and re-run the smoke test.
3. If r >= 16, the plateau really is 20% for this particular correction. Document it in run
   notes and adjust the hypothesis rather than the rank.

**First discovered:** poe_repair_min, step-09

**Affects steps:** step-09 (goal metric validation)

**Category:** 🟡 warning

**Environment reference:** `overview.md` (needs a section: LoRA rank selection by correction
magnitude; not yet written)

---

#### Entry ID: poe-lora-002
**Name:** Direction-cosine diverges (< 0.3) despite low training loss

**Symptom:** Training loss decreases smoothly, but `eval/direction_cosine/mean` stays below 0.3
or goes negative (anti-aligned).

**Root cause:** The LoRA correction is optimizing for a different objective than the pool-mean
direction. Possible causes: a stale pool-mean cache, a different loss function (L1 vs L2), or
the correction is legitimately different on this pair.

**Solution:**
1. Check that the pool-mean cache was built before training started.
2. Verify the loss function in `train_pooled.py` matches the loss used to compute the pool-mean.
3. If the loss is correct, the correction is pair-specific. That is data, not an error;
   document it in run notes.

**First discovered:** poe_repair_min, step-09

**Affects steps:** step-09, step-10 (transfer diagnosis)

**Category:** 🟡 warning

**Environment reference:** `overview.md` (needs a section: pool-mean computation and cache
invalidation; not yet written)

---

### Launching runs

#### Entry ID: poe-launch-001
**Name:** SSH-direct launch dies instantly: relative paths resolve in `$HOME`, not the repo

**Symptom:** `bash: line 1: logs/<name>.log: No such file or directory` the moment the SSH
launch command runs; `pgrep` on the node shows no process; no log file is ever created.

**Root cause:** A non-interactive SSH command starts in `$HOME`, not the repo, so every
relative path in the launch line (the script path, the `nohup` log redirect) resolves against
`$HOME`. The redirect fails before `nohup` even starts the script, so the run dies silently.
Prefixing `cd <repo> &&` is not reliable either: Claude Code's permission layer can strip or
block `cd` inside compound Bash commands, which produced three identical failures in a row on
2026-08-19.

**Solution:** Make every path in an SSH launch line absolute: the script, the log redirect, and
anything else on the line. The launch script itself does `cd "$REPO"` internally, so the caller
needs no working directory at all:
```bash
ssh <node> 'GPU=<idx> nohup bash /abs/path/to/launch.sh > /abs/path/to/logs/<name>.log 2>&1 &'
```
Verify the launch in the same SSH call: `sleep 5; pgrep -af "train"` plus `tail` of the absolute
log path.

**First discovered:** poe_repair_min, step-09 (smoke run launch on `mscluster106`, shared-device
path)

**Affects steps:** any step launched over SSH on a node this session is not on (step-09, step-10,
step-11 sweeps)

**Category:** 🟡 warning

**Environment reference:** [hpc/execution-protocol.md](hpc/execution-protocol.md)

---

### Disk and output paths

#### Entry ID: poe-disk-001
**Name:** Disk guard checks a different filesystem than the script writes to

**Symptom:** A job completes normally, the disk guard reports the target filesystem healthy
throughout, and large output is later found on `/home-mscluster` instead of the intended
`/datasets` mount.

**Root cause:** `scripts/mechanism_study/run_dose_sweep.sh` set its output root under the repo
on `/home-mscluster` while its disk guard read `df /datasets/mmolefe`. The two paths were
configured independently and drifted apart; the guard was checking a filesystem the script was
not writing to, so it could never catch the problem it exists to catch. 3.4GB of sweep cells
landed on the wrong mount this way.

**Solution:** A disk guard must resolve the same root the script's output path resolves to and
`df` exactly that mount, never a hardcoded or assumed one. A series of commits (`54b4b79`,
`8522459`, `c3f8bb4`, `f293bdf`) routed roughly 100 files' output paths through a shared
`paths.resolve()` helper for this reason. Whether every remaining script's guard now reads the
same resolved root as its output root has not been re-audited; treat a new script's guard as
unverified until its `df` target is checked against its actual output root by hand.

**First discovered:** poe_repair_min, `scripts/mechanism_study/run_dose_sweep.sh`, recorded in
the project's own `CLAUDE.md`

**Affects steps:** any step with a job script that writes large output (checkpoints, caches,
sweep cells)

**Category:** 🔴 critical

**Environment reference:** [storage.md](storage.md)

---
