# Project Error Catalog: poe_repair_min

Patterns specific to this project: codebase structure, model sizes, dataset characteristics, cluster constraints observed in practice.

**Last updated:** 2026-08-19  
**Total entries:** 6  
**Seed entries from:** step-09 (three-live-curves-while-training)

---

## Error patterns (organized by subsystem)

### Scoring and evaluation

#### Entry ID: poe-score-001
**Name:** Scorer returns all zeros or NaNs despite valid inputs

**Symptom:** W&B shows `eval/compose_rate/mean = 0.0` or `NaN` for all eval steps, even though generated images look visually correct

**Root cause:** Scorer receives tensors with wrong dtype (int instead of float), wrong device (CPU vs GPU), or invalid value range (e.g., pixel values 0-255 instead of 0-1)

**Solution:** Before calling scorer, validate eval outputs:
```python
assert eval_output.dtype == torch.float32, f"Expected float32, got {eval_output.dtype}"
assert eval_output.device.type == 'cuda', f"Expected GPU tensor, got {eval_output.device}"
assert eval_output.min() >= -0.1 and eval_output.max() <= 1.1, f"Range error: [{eval_output.min()}, {eval_output.max()}]"
```
Check normalization pipeline in `experiments/one_pair_one_seed/main.py::eval_hook`.

**First discovered:** poe_repair_min, step-09

**Affects steps:** step-09, step-10, step-11 (any step using compose_rate metric)

**Category:** 🔴 critical

**Environment reference:** environment/overview.md (needs a section: tensor dtype and normalization for the scorer; not yet written)

---

#### Entry ID: poe-score-002
**Name:** Compose-rate stuck at 0.0 even though fix is active

**Symptom:** Correction is being applied (loss decreases), but `eval/compose_rate/mean = 0.0` throughout training

**Root cause:** Scorer was trained on PoE (mixture of experts) outputs in a particular style or value range. The fix is changing the images, but not in the direction the scorer recognizes.

**Solution:** 
1. Manually inspect generated images (save a batch to disk, visualize in Jupyter).
2. Compare to PoE reference images used to train the scorer.
3. If images look different (color space, brightness), recalibrate scorer or adjust the correction target.
4. If images look correct but scorer is wrong, the scorer model may need retraining on this data distribution.

**First discovered:** poe_repair_min, step-09

**Affects steps:** step-09, step-10 (diagnostic metric reliability)

**Category:** 🟡 warning

**Environment reference:** environment/overview.md (needs a section: scorer calibration and reference images; not yet written)

---

### Memory and batch sizes

#### Entry ID: poe-mem-001
**Name:** Eval hook OOM on biggpu with full-size quadrants

**Symptom:** `RuntimeError: CUDA out of memory` during `eval/compose_rate` computation on V100 32GB

**Root cause:** Full eval batch for cat×dog (11 training pairs × 4 seeds = 44 images) + DINOv2 embeddings (40M) + CLIP embeddings (400M) exceeds V100 memory. A100 does not have this issue.

**Solution:** 
- Use A100 nodes (mscluster110) for evals with full batch. 
- OR: Shard eval batch to 8-16 images per forward pass, loop and aggregate.
- OR: Reduce eval frequency (eval every 5 steps instead of every step).

Check `environment/hpc/nodes.md` for which partitions have A100 vs V100.

**First discovered:** poe_repair_min, step-09, V100 runs

**Affects steps:** step-09, step-10, step-11 (if eval batch not reduced)

**Category:** 🟡 warning

**Environment reference:** [environment/hpc/nodes.md: Partitions](../environment/hpc/nodes.md#partitions)

---

### LoRA and correction dynamics

#### Entry ID: poe-lora-001
**Name:** Fraction-of-distance-reached plateau at 20% instead of expected 40%

**Symptom:** `eval/frac_distance_reached/mean` climbs to ~0.2 by epoch 1, then stalls. Expected plateau is ~0.4 based on prior runs.

**Root cause:** LoRA rank too low (r=4 or r=8) to capture full correction magnitude. The correction saturates the LoRA space before reaching the full PoE→Mono distance.

**Solution:** 
1. Check LoRA rank used (in config, default r=16).
2. If r < 16, increase to r=32 or r=64 and re-run smoke.
3. If r >= 16, the plateau is at 20% for this particular correction. Document in run notes and adjust hypothesis.

**First discovered:** poe_repair_min, step-09

**Affects steps:** step-09 (goal metric validation)

**Category:** 🟡 warning

**Environment reference:** environment/overview.md (needs a section: LoRA rank selection by correction magnitude; not yet written)

---

#### Entry ID: poe-lora-002
**Name:** Direction-cosine diverges (< 0.3) despite low training loss

**Symptom:** Training loss decreases smoothly, but `eval/direction_cosine/mean` stays < 0.3 or becomes negative (anti-aligned)

**Root cause:** LoRA correction is optimizing for a different objective than the pool-mean direction. Possible causes: stale pool-mean cache, different loss function (L1 vs L2), or the correction is legitimately different on this pair.

**Solution:**
1. Check that pool-mean cache was built before training started (see dist-001 for synchronization issues).
2. Verify loss function in `train_pooled.py` matches the loss used to compute pool-mean.
3. If loss is correct, the correction is pair-specific. This is data, not an error; document in run notes.

**First discovered:** poe_repair_min, step-09

**Affects steps:** step-09, step-10 (transfer diagnosis)

**Category:** 🟡 warning

**Environment reference:** environment/overview.md (needs a section: pool-mean computation and cache invalidation; not yet written)

---

### Launching runs

#### Entry ID: poe-launch-001
**Name:** SSH-direct launch dies instantly: relative paths resolve in $HOME, not the repo

**Symptom:** `bash: line 1: logs/<name>.log: No such file or directory` the moment the SSH launch command runs; `pgrep` on the node shows no process; no log file is ever created

**Root cause:** A non-interactive SSH command starts in `$HOME`, not the repo, so every relative path in the launch line (the script path, the nohup log redirect) resolves against `$HOME`. The redirect fails before nohup even starts the script, so the run dies silently. Prefixing `cd <repo> &&` is not reliable either: Claude Code's permission layer can strip or block `cd` inside compound Bash commands, which produced three identical failures in a row on 2026-08-19.

**Solution:** Make every path in an SSH launch line absolute: the script, the log redirect, and anything else on the line. The launch script itself does `cd "$REPO"` internally, so the caller needs no working directory at all:
```bash
ssh <node> 'GPU=<idx> nohup bash /abs/path/to/launch.sh > /abs/path/to/logs/<name>.log 2>&1 &'
```
Verify the launch in the same SSH call: `sleep 5; pgrep -af "train"` plus `tail` of the absolute log path.

**First discovered:** poe_repair_min, step-09 (smoke run launch on mscluster106, shared-device path)

**Affects steps:** any step launched over SSH on a node this session is not on (step-09, step-10, step-11 sweeps)

**Category:** 🟡 warning

**Environment reference:** environment/hpc/execution-protocol.md (shared-device path)

---
