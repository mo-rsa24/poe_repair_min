# Running the Langevin corrector

Navigation: 📋 [Index](../00-INDEX.md)

Contents: [Before any of these](#before-any-of-these) |
[1. Prove the corrector is off when it is switched off](#1-prove-the-corrector-is-off-when-it-is-switched-off) |
[2. Fix the step size before reading any curve](#2-fix-the-step-size-before-reading-any-curve) |
[3. Run the corrector-count grid and print its verdict](#3-run-the-corrector-count-grid-and-print-its-verdict) |
[4. Slide a corrector window across the run](#4-slide-a-corrector-window-across-the-run) |
[5. Render the eight-seed sheets, the adapter tail and the clean tail](#5-render-the-eight-seed-sheets-the-adapter-tail-and-the-clean-tail) |
[6. Draw the figures and log the set to W&B](#6-draw-the-figures-and-log-the-set-to-wb) |
[If something looks wrong](#if-something-looks-wrong) |
[Where this came from](#where-this-came-from)

Every recipe behind the three corrector findings:
[does a Langevin corrector remove part of the correction](../../report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md),
[does a corrector alone produce two animals](../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md),
and [can a corrector or a clean tail sharpen the adapter's renders](../../report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md).

One launcher drives all of them, `scripts/mechanism_study/run_corrector_curve.sh`, picked by a
`STAGE` variable. It carries its own disk guard, its own device guard and a CUDA probe, so a
faulted card or a device another user just took is refused rather than run on. Which card takes
which python build is owned by [nodes](../../environment/hpc/nodes.md); how to pick a free device
is [launching a run § 1](launching-and-harvesting-a-run.md#1-decide-where-a-run-goes).

**What the corrector is, said once.** At each of the 50 noise levels it takes `k` small steps
along the product-of-experts score with a little fresh noise each time, then takes the ordinary
reverse step from wherever that lands. The step size is `c` times the scheduler's beta at that
level. With `k = 0` it is plain product-of-experts, byte for byte. Everything else here is a
measurement or a render taken on the path it produces.

## Before any of these

Navigation: 📋 [TOC](#running-the-langevin-corrector) | [Next](#1-prove-the-corrector-is-off-when-it-is-switched-off) ➡️

- [ ] 🖥️ **The composer and its two drivers are on this branch**
  ```bash
  ls /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/poe_repair/methods/_poe_langevin.py \
     /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/poe_repair/composers/poe_langevin.py \
     /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/corrector_residual_curve.py \
     /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/corrector_window_sweep.py
  ```
  ✅ **Four paths print.** They landed on `overhaul/tree-2026-08-10` on 2026-09-05.
  ❌ **`No such file`**: you are on an older branch.

- [ ] 🖥️ **The cache and the rank-32 checkpoint exist** (the checkpoint only for recipe 5)
  ```bash
  ls /datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/heldout/a_cat__x__a_dog/seed_9/residuals/step_000.pt \
     /datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt
  ```
  ✅ **Both print.** `step_000.pt` holds the pinned initial noise every condition starts from.

- [ ] 🖥️ **The output root is set in any shell that runs these by hand**
  ```bash
  export POE_REPAIR_OUTPUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs
  ```
  The launcher sets it itself; a bare `python scripts/...` call does not.

## 1. Prove the corrector is off when it is switched off

`ran 2026-09-05` (in-session on mscluster85 device 0, about 25 seconds per render)

Navigation: ⬅️ [Before any of these](#before-any-of-these) | 📋 [TOC](#running-the-langevin-corrector) | [Next](#2-fix-the-step-size-before-reading-any-curve) ➡️

**When you need this**

Before believing any measurement taken on the corrector's path, and after any edit to the
composer or the sampler.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export POE_REPAIR_OUTPUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs
$PY scripts/corrector_residual_curve.py --check-identity --pair a_cat__x__a_dog --seed 9
$PY scripts/corrector_residual_curve.py --check-identity --window off --k 200 --pair a_cat__x__a_dog --seed 9
```

✅ **Both print** `corrector off is byte-identical to plain PoE`, and the first also prints
`against run_cfg_poe (the three-branch reference sampler): byte-identical`.
❌ **`IDENTITY FAILED: max |diff| = ...`**: the composer changes something it should not, and
nothing measured on its path may be believed. Two checks exist because the first cannot see a
corrector that ignores its own window: at zero steps there is nothing to ignore.

## 2. Fix the step size before reading any curve

`ran 2026-09-05` (mscluster108 device 1, about 9 to 19 minutes per value)

Navigation: ⬅️ [1. Prove the corrector is off](#1-prove-the-corrector-is-off-when-it-is-switched-off) | 📋 [TOC](#running-the-langevin-corrector) | [Next](#3-run-the-corrector-count-grid-and-print-its-verdict) ➡️

**When you need this**

Once, before the grid, and again if the model or the scheduler changes. At the wrong step size
the chain either never moves or wrecks the sample, and both imitate a scientific answer.

```bash
ssh mscluster108 'STAGE=search GPU=1 nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/mechanism_study/run_corrector_curve.sh > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/logs/step_size_search.log 2>&1 &'
```

Widen the range in place when the pick lands at an edge, appending rows to the same file:

```bash
ssh mscluster108 'STAGE=search GPU=1 EXTRA="--search-c 3.0,10.0,30.0,100.0,300.0" nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/mechanism_study/run_corrector_curve.sh > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/logs/step_size_search_wider.log 2>&1 &'
```

Read the table, and then read the pictures:

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY -c "import json;d=json.load(open('/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/step_size_search.json'));print('picked',d['picked_c'],'at edge',d['picked_c_at_edge_of_range']);[print(r['c'],round(r['median_chain_disp_rel'],3),round(r['max_latent_norm_rel'],3),r['verdict']) for r in d['rows']]"
ls /datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/pairs/a_cat__x__a_dog/seed_9/
```

✅ **A picked value that is not at either end of the range**, and its render is a photograph.
⚠️ **The picked value's render is texture noise.** This happened, and it is the trap the numeric
guards do not catch: an unadjusted Langevin step contracts toward the score's mean whatever its
size until the step passes 2, so the latent-norm bound cannot trip while the sample is destroyed.
Look at the image before trusting the pick. The working value on SDXL at 50 steps is `c = 3`,
the largest whose composing-pair curve does not rise in recipe 3.
❌ **Every value stalls or diverges**: the corrector cannot be run stably here, which is a finding
rather than a bug.

## 3. Run the corrector-count grid and print its verdict

`ran 2026-09-05 and 06` (mscluster110 device 0 for one step size, mscluster108 and mscluster85 for the others)

Navigation: ⬅️ [2. Fix the step size](#2-fix-the-step-size-before-reading-any-curve) | 📋 [TOC](#running-the-langevin-corrector) | [Next](#4-slide-a-corrector-window-across-the-run) ➡️

**When you need this**

To measure how much of the correction survives once the chain has settled: six corrector counts
by two pairs by 50 steps, 600 rows, no images and no detector.

**Fill in**

| What | Example | Where to get it |
|---|---|---|
| node and device | `mscluster110`, `0` | a free device, per [launching a run § 1](launching-and-harvesting-a-run.md#1-decide-where-a-run-goes) |
| step size | `--c 3` | recipe 2's pick, checked against its render |
| pair | `--pairs a_cat__x__a_dog` | one pair per process keeps a whole pair on one device, which the fp16 spread makes necessary |

```bash
ssh mscluster110 'STAGE=grid GPU=0 EXTRA="--pairs a_cat__x__a_dog --c 3" nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/mechanism_study/run_corrector_curve.sh > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/logs/grid_cat_dog.log 2>&1 &'
```

It is resumable: cells are keyed by pair, seed, step size and count, and an existing cell is
skipped. Harvest and judge:

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
$PY scripts/corrector_residual_curve.py --verdict --c 3
$PY scripts/corrector_residual_curve.py --plot --c 3 --fig-name how-much-of-the-correction-a-corrector-removes-c3
```

✅ **`rows: 600 of 600 expected`** and one of `BRANCH: support`, `null` or `inconclusive`, with
the numbers it was judged against.
⚠️ **`BRANCH: inconclusive`** naming the instability bar: the two largest counts disagree by more
than 5%. On one seed this is expected, because the read-out's own scatter is larger than the bar;
the fix is seeds, not a looser bar.
❌ **Fewer than 600 rows**: a count crashed and the loop swallowed it. Rerun the same command; it
fills only what is missing.

**Timing on the cards this ran on.** A count of 200 takes about 16 minutes per pair on the RTX PRO
6000, 34 on the RTX 3090 and 73 on the Quadro RTX 8000. The whole grid for one pair is roughly one
hour on the Blackwell card and four on the others.

## 4. Slide a corrector window across the run

`ran 2026-09-05 and 06` (mscluster108 device 1, about 1.6 hours)

Navigation: ⬅️ [3. Run the grid](#3-run-the-corrector-count-grid-and-print-its-verdict) | 📋 [TOC](#running-the-langevin-corrector) | [Next](#5-render-the-eight-seed-sheets-the-adapter-tail-and-the-clean-tail) ➡️

**When you need this**

To ask whether the corrector changes the outcome in the window where composition is decided, in a
layout matched to the injected-correction figure.

```bash
ssh mscluster108 'STAGE=window GPU=1 EXTRA="--c 3 --k 20" nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/mechanism_study/run_corrector_curve.sh > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/logs/window.log 2>&1 &'
```

✅ **`per column composed of 4:`** with ten entries, and `window_curves_mcmc.json` holding 40
scored cells.
❌ **`no picked c in the step-size search`**: recipe 2 has not run, or its file is elsewhere.

Each window render is about 2 minutes on that card and the all-50 column about 7.6.

## 5. Render the eight-seed sheets, the adapter tail and the clean tail

`ran 2026-09-06` (sheets on mscluster108 device 1, both tails on mscluster85 device 0)

Navigation: ⬅️ [4. Slide a corrector window](#4-slide-a-corrector-window-across-the-run) | 📋 [TOC](#running-the-langevin-corrector) | [Next](#6-draw-the-figures-and-log-the-set-to-wb) ➡️

**When you need this**

The read-out every parallel session reports on: eight held-out seeds, both pairs, the joint prompt
and plain product-of-experts beside whatever this session tests.

```bash
L=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/mechanism_study/run_corrector_curve.sh
O=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/logs
ssh mscluster108 "STAGE=sheet GPU=1 EXTRA='--c 3 --k 20' nohup bash $L > $O/sheet.log 2>&1 &"
ssh mscluster85  "STAGE=tail  GPU=0 EXTRA='--c 3'        nohup bash $L > $O/tail.log 2>&1 &"
ssh mscluster85  "STAGE=clean GPU=0 EXTRA='--c 3'        nohup bash $L > $O/clean_tail.log 2>&1 &"
```

⚠️ **Run `tail` and `clean` in their own process, never beside `sheet` or `window`.** Both attach
the adapter, and the windowed adapter sampler leaves it enabled when it returns, so a reference
rendered afterwards in the same process silently carries the correction. This cost one re-render
on 2026-09-05 and is catalogued in [known failures](../../environment/known-failures.md).

✅ **`sheet_scores.json` holds 16 rows, `tail_fidelity.json` 48, `clean_tail.json` 96**, and the
two tail stages print `BRANCH:` with the numbers behind it.
❌ **`NameError`** in the scoring pass: a threshold was renamed and its old name survives in the
verdict function. The renders are on disk and the stage re-scores without re-rendering.

**Timing.** A sheet cell is 8.3 minutes on the Quadro (two references plus one corrector render).
A tail render is 42 seconds at 0 corrector steps, 105 at 5 and 293 at 20 on the 3090. The clean
tail's 96 renders take about 2.4 hours there.

## 6. Draw the figures and log the set to W&B

`ran 2026-09-06` (session node, CPU for the drawing, about 4 minutes)

Navigation: ⬅️ [5. Render the sheets](#5-render-the-eight-seed-sheets-the-adapter-tail-and-the-clean-tail) | 📋 [TOC](#running-the-langevin-corrector) | [Next](#if-something-looks-wrong) ➡️

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
export POE_REPAIR_OUTPUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs
$PY scripts/corrector_window_sweep.py --figures --wandb --c 3
```

✅ **Seven paths print** (the window strip into `paper/iclr/figures/when-the-correction-arrives/mcmc/`
and six sheets into `artifacts/results/is-the-gap-the-samplers-or-the-models/`), then
`wandb run: prime_lab/poe-repair-animals-compose/<id>`, also written to
`corrector/wandb_run_id.txt`.

The scoring needs a GPU for the detector and the embedder; drawing alone does not. Run it on the
node that holds the renders, because the session node's view of `/datasets` can lag a launch
node's writes by minutes.

## If something looks wrong

Navigation: ⬅️ [6. Draw the figures](#6-draw-the-figures-and-log-the-set-to-wb) | 📋 [TOC](#running-the-langevin-corrector) | [Next](#where-this-came-from) ➡️

**The launcher exits immediately with `guard:` on the first line.** It refused the device, and the
message says which check: more than 1 GiB in use (another user took it between your check and the
launch, which happened twice on 2026-09-05), a utilisation reading of `[N/A]` or
`[GPU requires reset]` (the card is faulted), or a failed CUDA probe. Pick another device; none of
these is worth waiting out.

**The run produces files at about forty times the expected wall time.** The process is on the CPU.
A Blackwell card can list in `nvidia-smi` and still fail `torch.cuda.is_available()`, which is
[poe-launch-002](../../environment/known-failures.md). The launcher probes for this, so a run that
gets past the probe and is still slow is a different problem.

**The same cell reads a different number on a different card.** Expected, and it is why a pair's
whole count sweep stays on one device: fp16 arithmetic over 50 steps spreads the uncorrected cell
by 15% across the three card types here.

**`pkill -f` kills your own shell.** The pattern matches the ssh command line that carries it. Match
a pattern that cannot appear in the caller, as
[cluster shell pitfalls](../../environment/known-failures.md) records.

## Where this came from

Navigation: ⬅️ [If something looks wrong](#if-something-looks-wrong) | 📋 [TOC](#running-the-langevin-corrector)

Every recipe was run on 2026-09-05 and 2026-09-06 while executing steps 25 to 27 of
[the sampler-against-model scope](../../plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md),
and the commands here are the ones that ran, with the node and device each was launched on. The
verdicts they produced are the three corrector findings linked at the top; the constants every
branch is judged against sit in `scripts/corrector_residual_curve.py` and
`scripts/corrector_window_sweep.py`.
