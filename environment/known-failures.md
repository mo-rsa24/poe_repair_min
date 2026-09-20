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

**Last updated:** 2026-09-20
**Total entries:** 11
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

#### Entry ID: poe-mem-002
**Name:** `_Embedders()` instantiated with no device defaults to CPU, and DINOv2's xformers
attention kernel does not support CPU

**Symptom:** `NotImplementedError: No operator found for memory_efficient_attention_forward`,
the trace running through `dinov2/layers/attention.py` and `xformers/ops/fmha/dispatch.py`,
listing every dispatched backend as unsupported for `device=cpu`, `dtype=torch.float32`.

**Root cause:** `poe_repair.experiments.compose_scorer_validation.scorer._Embedders.__init__`
defaults `device` to `torch.device("cpu")` when the caller passes none. Its `.dino()` method
loads DINOv2 via `xformers`-accelerated attention, whose available kernels
(`fa2F@v2.5.7-pt`, `cutlassF-pt`) only support `device=cuda` with `dtype` in
`{float16, bfloat16}` — CPU + float32 has no matching operator, so the forward pass raises
instead of falling back to eager attention.

**Solution:** Always pass `device=torch.device("cuda")` (or the run's actual device) explicitly
when constructing `_Embedders`; never rely on its CPU default on a GPU job. `.clip()` on the
same class tolerates CPU fine (CLIP's attention path doesn't route through xformers the same
way), which makes this easy to miss until `.dino()` is called.

**First discovered:** poe_repair_min, `scripts/showcase/lambda_window_grid.py`, plan 07 of
`01-showcase-the-trained-lora` (slurm job 48608)

**Affects steps:** any step calling `_Embedders()` on a GPU node
(`poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py` and
`poe_repair/experiments/compose_scorer_validation/validate.py` also construct it — worth
checking both pass `device` explicitly)

**Category:** 🟡 warning

**Environment reference:** none yet; belongs beside `_Embedders` in
`poe_repair/experiments/compose_scorer_validation/scorer.py`, not yet documented there.

---

### LoRA and correction dynamics

#### Entry ID: poe-lora-001
**Name:** Fraction-of-distance-reached stops rising at 20% instead of the expected 40%

**Symptom:** `eval/frac_distance_reached/mean` climbs to about 0.2 by epoch 1, then stalls.
Prior runs stop rising at about 0.4.

**Root cause:** LoRA rank too low (r=4 or r=8) to capture the full correction magnitude. The
correction saturates the LoRA space before reaching the full PoE-to-Mono distance.

**Solution:**
1. Check the LoRA rank used (in config; default is r=16).
2. If r < 16, increase to r=32 or r=64 and re-run the short one-epoch check that proves the
   wiring works.
3. If r >= 16, this particular correction really does stop at 20%. Document it in run
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

#### Entry ID: poe-lora-003
**Name:** A control condition run through a different sampler than the condition it's compared
against, so the comparison mixes two axes

**Symptom:** A "should do nothing" control (wrong-seed or shuffled correction) shows a compose
rate, or an AUC, well *above* the real condition it is supposed to be a null for — the opposite
of what a size-matched control should ever do.

**Root cause:** `run_lora_residual_inject_masked` (the masked sampler `composition_mode="with_prompt"`
uses) runs its off-window steps as a single adapter-off forward on the *unconditional* (empty
prompt) branch only. `run_constant_residual_inject`, the project's existing precomputed-residual
injector, runs full PoE on the real prompt at *every* step, including off-window ones, whether or
not that step has an injected residual. Routing the real condition through the first function and
a control through the second changes the injection source *and* the off-window sampling at the
same time, so any gap between them cannot be attributed to the injection source alone. In
`scripts/showcase/lora_dose_sweep.py`'s first attempt (slurm job 48596, plan 03 of
`01-showcase-the-trained-lora`), this produced AUC 0.734 and 0.875 for two controls against 0.203
for the real condition, tripping the plan's own "a control rises above luck" fail criterion — as
a comparison-fairness artefact, not a result.

**Solution:** When a control needs to inject a *different* Δ̂ than the one a masked sampler would
compute live, extend that same sampler to accept an externally-supplied per-step residual
(`run_lora_residual_inject_masked`'s `external_delta_by_step` parameter) rather than switching to
a differently-behaved sampler. Every condition in a comparison must run through the identical
sampler function and window/composition-mode configuration, differing only in the one axis under
test. Before trusting a control result, ask whether it was even possible for it to differ from
the real condition in more than the intended way — if a different code path was used, this is not
provable from the numbers alone.

**First discovered:** poe_repair_min, `scripts/showcase/lora_dose_sweep.py`, plan 03 of
`01-showcase-the-trained-lora` (slurm job 48596 → corrected in job 48619)

**Also seen:** plan 07's own λ-window grid (`scripts/showcase/lambda_window_grid.py`, jobs
48607-48611) hit the same root cause in a different guise: the plan's pre-registered λ=0
identity check compared a `run_lora_residual_inject_masked`-style render against `poe.png`
(rendered with full guided PoE at all 50 steps), and the off-window unguided-only branch gave a
DINOv2 distance of 0.43 to that reference (should be small). Confirms the "affects steps"
prediction below. Fixed the same way this entry's Solution describes in spirit — not by
supplying an external residual, but by writing a sampler whose off-window steps run the same
full guided-PoE forward as on-window steps, adapter disabled, so λ=0 matches the baseline by
construction (`run_lora_residual_inject_windowed_poe`, same file). After the fix, all 5 cells'
identity checks passed (largest distance 0.13, down from 0.43).

**Affects steps:** any step building a new control condition against an existing masked or
windowed sampler, not just this one (step 07, the λ-and-window series, has the same shape)

**Category:** 🔴 critical

**Environment reference:** none yet; the fairness principle belongs beside the sampler functions
in `poe_repair/methods/_sampling.py`, not yet documented there.

---

#### Entry ID: poe-lora-004
**Name:** A hand-written sampler function omits `@torch.no_grad()`, OOMing on the very first
forward pass

**Symptom:** `torch.OutOfMemoryError: CUDA out of memory` on the second UNet forward of the
first denoising step (23.5 of 24GB already in use before that call), even though an
identically-shaped forward pass in an existing, working sampler function completes fine.

**Root cause:** Every inference-only sampler in `poe_repair/methods/_sampling.py` (there are
nine of them) is decorated `@torch.no_grad()`. A new sampler written by copying that file's
per-step pattern (three-branch forward, `guided_eps`, `poe_eps`, `ddim_prev_from_x0_eps`) but
typed out fresh, not copy-pasted whole, can silently drop the decorator. Without it, every UNet
forward builds and retains a full autograd graph; at SDXL's size and batch=3, two forward passes
in the same step (a frozen pass plus an adapter-on pass) exhausts a 24GB card before the loop
even reaches its second step.

**Solution:** Any new inference-only sampler function must carry `@torch.no_grad()` (or run
inside a `with torch.no_grad():` block). Before trusting an OOM as a genuine memory-budget
problem, check whether the failing function has the decorator that every sibling function in the
same module already carries — a missing decorator produces an OOM trace that looks identical to
a real capacity problem.

**First discovered:** poe_repair_min, `scripts/showcase/lambda_window_grid.py`
(`run_lora_residual_inject_windowed_poe`), plan 07 of `01-showcase-the-trained-lora` (slurm job
48610)

**Affects steps:** any step writing a new sampler function against this codebase's
`poe_repair/methods/_sampling.py` conventions rather than calling an existing one

**Category:** 🟡 warning

**Environment reference:** none yet; belongs beside the sampler functions in
`poe_repair/methods/_sampling.py`.

---

#### Entry ID: poe-lora-005
**Name:** A pooled adapter trained on the broader cell pools draws a person into a picture of two
animals

**Symptom:** A rank-32 adapter renders `a_cat__x__a_dog` and the picture contains a human, usually
crouching beside a single animal, on a seed where the same prompt pair without the adapter draws
one fused cat-dog creature and no person. The detector's instance count does not flag it, because
a person is not one of the animal queries.

**Root cause:** Not established. It is reproducible and it tracks the training pool rather than the
objective: on `a_cat__x__a_dog` seed 1, four adapters do it (`pool43-early25` and
`pool43-all50-orth3` at 40,000, `v54d_P` and `v54d_C1_allseeds` at 30,000) and every adapter
trained on the original 88-cell pool over 11 look-alike animal pairs does not, at matched steps.
The broader pools carry object pairs and scene pairs the original did not, so a prompt-driven
prior toward a photographed scene with a person in it is the first thing to check.

**Solution:** None yet. Until there is one, a figure using a broader-pool adapter is read by eye
before it is used, because no scorer in this repository reports an extra person. Recorded 2026-09-20
from the fourteen-adapter sheet at
`artifacts/results/composing-unseen-pairs/`, which renders one seed through every adapter this
project has trained.

**Environment reference:** none; this is a model behaviour rather than a system fact.

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

**First discovered:** poe_repair_min, step-09 (the first short run launched on `mscluster106`,
shared-device path)

**Affects steps:** any step launched over SSH on a node this session is not on (step-09, step-10,
and step-11's runs across many settings)

**Category:** 🟡 warning

**Environment reference:** [hpc/execution-protocol.md](hpc/execution-protocol.md)

---

#### Entry ID: poe-launch-002
**Name:** A device passes the pre-launch guard but is in a hardware fault state — the process hangs producing nothing, no error

**Symptom:** A training process starts, loads models, encodes prompts, and logs "training
plan: target=N epochs" — then produces zero epoch progress for many minutes, while an
identical launch on a different node reaches its first epoch in under a minute. No exception,
no traceback, no OOM: the process just sits there. A second form (`mscluster111`, 2026-09-05): the
process does not hang but runs on the CPU, because `torch.cuda.is_available()` returns `False` on
the faulted device while `nvidia-smi` still lists it. A 16-second sampler run then takes about
15 minutes, files keep appearing, and nothing in the log says why. A third form (`bigbatch`
nodes `mscluster44`, `mscluster45` and `mscluster65`, 2026-09-06): the node sits `idle` in
`sinfo`, and the first `nvidia-smi` call in the job prints `Unable to determine the device handle
for GPU0: 0000:17:00.0: Unknown Error` and exits, so a job that guards on `nvidia-smi` fails in
one second and a job that does not would run on the CPU. Idle `bigbatch` nodes are idle for a
reason often enough that a one-second probe job before the real one is worth it.

**Root cause:** The device's pre-launch `nvidia-smi --query-gpu=memory.used` guard (the
`>1024MiB in use` check) only catches a device already busy with another process. It does not
catch a device in a hardware/driver fault state. `nvidia-smi` on the same device mid-hang
reported `temperature.gpu=[GPU requires reset]` and `utilization.gpu=[N/A]` — a fault state
distinct from "idle" (which reads `0 %`) and distinct from "busy" (which reads a real
percentage and real memory use). The guard's `<1024MiB` check passes a faulted device just as
readily as a healthy idle one, because a fault reports near-zero memory too.

**Solution:** After the memory guard, also check that `utilization.gpu` and `temperature.gpu`
return numeric values, not `[N/A]` / `[GPU requires reset]`:
```bash
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader -i <idx>
```
A row containing `[N/A]` or `[GPU requires reset]` means the device is faulted, not free — do
not launch there even though the memory-used guard alone would pass it. A launch script should
also run `python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 7)'` under
the pinned `CUDA_VISIBLE_DEVICES` before starting real work, which catches the CPU-fallback form. If a launch is already
hung with this symptom (progress log lines stop appearing well past the point comparable runs
on other nodes reach their first checkpoint), `kill -9` the process and relaunch on a different
device; the faulted GPU itself needs a reset only an administrator or a node reboot can do, no
user-level recovery exists.

**First discovered:** poe_repair_min, 01-showcase-the-trained-lora, the experiment A/B timing
smoke run: `mscluster111` device 0 (an RTX PRO 6000 Blackwell node, confirmed free by the
memory guard at launch) produced zero epoch progress for 9+ minutes while identical-config runs
on `mscluster106` and `mscluster108` (RTX 8000) both logged their first epoch within a minute.

**Affects steps:** any shared-device SSH launch (`execution-protocol.md`'s step 3), on any node,
since the fault is per-device hardware state, not tied to this project's code.

**Category:** 🔴 blocking (for the affected launch; the fix is routine once recognized)

**Environment reference:** [hpc/execution-protocol.md](hpc/execution-protocol.md),
[hpc/nodes.md](hpc/nodes.md)

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
not writing to, so it could never catch the problem it exists to catch. 3.4GB of output from
that run across many settings landed on the wrong mount this way.

**Solution:** A disk guard must resolve the same root the script's output path resolves to and
`df` exactly that mount, never a hardcoded or assumed one. A series of commits (`54b4b79`,
`8522459`, `c3f8bb4`, `f293bdf`) routed roughly 100 files' output paths through a shared
`paths.resolve()` helper for this reason. Whether every remaining script's guard now reads the
same resolved root as its output root has not been re-audited; treat a new script's guard as
unverified until its `df` target is checked against its actual output root by hand.

**First discovered:** poe_repair_min, `scripts/mechanism_study/run_dose_sweep.sh`, recorded in
the project's own `CLAUDE.md`

**Affects steps:** any step with a job script that writes large output (checkpoints, caches, and
the per-setting output of a run across many settings)

**Category:** 🔴 critical

**Environment reference:** [storage.md](storage.md)

---

### Training data and splits

#### Entry ID: poe-data-001
**Name:** A cell named in `--cells` is loaded from the held-out split without a warning

**Symptom:** None. The run launches, trains, and reports normal losses. `dataset_meta.json`
records `split: None` for every cell, so nothing afterwards can say which split a cell came
from, and a contaminated run is indistinguishable from a clean one in W&B.

**Root cause:** `CellPath.from_root` at
[training_cache.py:63](../poe_repair/training_cache.py) resolves `split=None` by searching
`["heldout", "train"]` in that order, so a cell present in both directories is taken from
`heldout/`. [train_pooled.py:434](../poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py)
calls `resolve_cells(pair, seeds)` with no `split` argument, so every training cell is
resolved held-out-first. The cache holds 12 cells across 4 pairs
(`a_lion__x__a_dog` seeds 1, 2, 4, 6, 8; `a_cat__x__a_lion` seeds 1 to 4;
`a_dog__x__a_horse` seeds 1 and 3; `a_butterfly__x__a_flower_meadow` seed 4) that sit in both
directories and would be taken from `heldout/`.

**Solution:**
- Pass `split="train"` explicitly from `train_pooled.py`, so a cell that is missing from
  `train/` raises instead of silently falling back.
- Record the resolved split per cell in `dataset_meta.json`, which today writes `None`, so a
  finished run can be audited without re-deriving the lookup.
- Until both land, check a pool before launching:
  ```bash
  python3 -c "
  import json, pathlib
  T = pathlib.Path('/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache')
  d = json.load(open('artifacts/_shared/cross_pair_pool_configs/cells_v54.json'))
  bad = [(p, s) for p, seeds in d.items() for s in seeds
         if (T / 'heldout' / p / f'seed_{s}' / 'meta.json').exists()]
  print('cells that would resolve to heldout:', len(bad), bad[:10])
  "
  ```
  Expect `0`. Anything above zero is a cell the adapter would train on and then be scored
  against.

**First discovered:** poe_repair_min, 2026-09-16, traced from the plan-09 walk rather than from
a failed run. No run has hit it: every cell of `cells_v54`, `cells_v55` and `cells_v57_animals`
resolves to `train/`, and the training seeds (1 to 8, 17 to 19) and held-out seeds (9 to 16) are
disjoint.

**Affects steps:** step 71 onward, every plan in
[designing the correction loss](../plans/09-designing-the-correction-loss/MASTER_PLAN.md) that
launches a run, and any future pool built by hand rather than from `seed_pool.train_pool`. Both
solutions above are owned by
[task 1.5 of the instrument fixes](../plans/09-designing-the-correction-loss/plans/tools/01-the-three-instrument-fixes.md).

**Category:** 🔴 critical

**Environment reference:** [storage.md](storage.md) for where the cache lives

---

## Cross-references

- The instrument finding in [the corrector's step-size review](../plans/06-is-the-gap-the-samplers-or-the-models/review/02-the-corrector-and-the-step-size-it-runs-at.md): a Langevin step size can pass both numeric divergence guards while its renders are texture noise, because the latent norm cannot rise until the Euler step stops contracting. The picture is rung 3 of [the finding](../report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md).
- The mention of **the adapter left attached after a windowed run** (why the two adapter stages run in their own process) in [can a corrector or a clean tail sharpen the adapter's renders](../report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) and [running the Langevin corrector](../runbook/running-things-on-the-cluster/running-the-langevin-corrector.md).
