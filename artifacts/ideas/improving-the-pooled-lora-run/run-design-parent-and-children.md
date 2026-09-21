# The run set: one parent with everything on, four children that each switch one thing off

Compiled from the walk on 2026-09-08. Five training runs on W&B, one no-training baseline, two
samplers applied to all of them, one strip per seed. Each child differs from the parent by exactly
one axis, so any gap between a child and the parent is that axis and nothing else. This is the
one-axis rule (claim 6) run as a leave-one-out design.

## The problem, with examples

On most held-out cat x dog seeds the adapter produces some composition but not a clean cat and a
clean dog. Where both do appear, the picture carries details the training distribution would
never produce. The count scorer calls most of these "composed", which is why it is not the read.

| seed | the joint prompt's picture | the adapter's render | what is wrong |
|---|---|---|---|
| 9 | a clean cat and dog | a cat and a dog | something unnatural around the tail; limbs not clearly resolved |
| 10 | two dogs | a blended pair with some sense of cat and dog | the reference is wrong; features blend across the two animals |
| 12 | a clean cat and dog | two animals | cannot say which animals they are |
| 13 | two dogs | two fluffy animals facing each other | neither is clearly a cat or a dog |
| 14 | two cats | a cat beside a human-like figure | an object outside anything the pool contains |
| 16 | a clean cat and dog | two animals, plausible at a glance | odd proportions and features on inspection |

## Words this uses

**Cell.** One pair on one seed: a 50-step trajectory in the cache with the correction target at
every step. 11 train pairs by 8 seeds is 88 cells.

**Joint target.** The picture the joint prompt ("a cat and a dog") draws on that seed, `mono.png`
in the cache. It is what the adapter is trained to reach. On three of the eight cat x dog seeds
it shows two dogs or two cats, so on those seeds the adapter is trained to reach the wrong picture.

**Nearness.** How close a render looks to its seed's joint target, measured by the compose
scorer's image embedding; 0 would be identical. The bar set before any of this ran is a move of
0.05 nearer. Only meaningful on seeds whose joint target is right.

**Composed.** The validated count says two animal-shaped regions. It cannot tell a cat and a dog
from two cats, and it cannot see softness.

**Both names present.** A second read: each detected region is cropped and CLIP chooses between
the pair's two names; the render passes when both names win at least one region. It finds two-of-
the-same on visually separable pairs and over-flags on similar ones.

**Hand-off at 25.** The adapter runs for the first 25 of 50 denoising steps and the frozen model
finishes. Run 7 located the adapter's softness in steps 25 to 50 and found 25 to be the step
where composition is already set.

## The parent, P: everything on

| Axis | Setting | Where it comes from |
|---|---|---|
| pairs | the 11 train pairs plus lion x horse (12 seeds), wolf x horse (12), bear x salmon (seeds 1 to 4; its fifth cached seed is 42, outside the train seed list), 28 new cells | the only cached pairs with 4 or more train seeds that contain neither cat nor dog. Their correction sizes are measured and reported beside the results, not used as a reason. The two horse pairs live under `heldout/` in the cache; the loader searches both splits, so no move is needed |
| cells | drop every train cell whose joint target fails the both-names read | claim 7; the read over-flags similar pairs, which child C1 measures |
| training steps | loss on steps 0 to 24 only | task 8.2: softness lives in steps 25 to 50 |
| loss | error in the part of the correction orthogonal to the two experts' own predictions weighted up (factor 3) | report: the adapter reproduces the re-weightable part and only half the new direction |
| optimizer | weight decay 1e-2, EMA 0.999 of the adapter weights, constant LR 1e-4 | claim 3; not ablated, because it adds safety for a longer run rather than fidelity |
| rank, length | rank 32, alpha 32, 30k steps | the checkpoint being improved |

Rendered two ways, like every run below: **shipped** (adapter on all 50 steps, deterministic
DDIM, guidance 7.5 throughout) and **stacked** (adapter on steps 0 to 24, frozen model after,
stochastic DDIM at eta 1, guidance 7.5 on steps 5 to 35 and 1.0 outside). The stacked sampler
is the intended use of P; the shipped render is kept so P can be compared with 30050 on 30050's
own terms.

## The children

| Run | Same as P except | What a gap against P says |
|---|---|---|
| C1 | keep every cell | whether dropping wrong-target cells helps, which run 6 could not settle |
| C2 | loss on all 50 steps | whether training the adapter on the late steps is what makes it soft |
| C3 | plain loss, no orthogonal weighting | whether spending capacity on the new direction is what improves the render |
| C4 | the 11 pairs only | whether adding those three pairs changed anything on cat x dog and elephant x penguin |

Plus **B**, the baseline: checkpoint 30050 as it is, rendered both ways. B under the stacked
sampler is the zero-training improvement; every training run has to beat that, not the shipped
render.

## What every run logs to W&B

Project `prime_lab/poe-repair-animals-compose`, one run per row above, tagged `parent`, `child`
or `baseline`. Each run logs, every 200 epochs and at the end: the strip per held-out seed
(joint target | plain PoE | 30050 | this run, under both samplers), the count-composed rate, the
both-names rate, the nearness on the five clean-reference cat x dog seeds (10, 13, 14 excluded and
said so), and held-out fit as a drift monitor only. The strips are the artifact. The numbers are
the ordering.

## The read, written before launch

The verdict is by eye, per seed, on the strips, and nothing numeric decides it. Every strip covers
all 17 cached cat x dog seeds and all 8 elephant x penguin seeds.

Each render gets one of three labels, assigned blind: the condition names are hidden and the
order within a seed is shuffled before labelling.

1. **clean**: a cat and a dog, each clearly itself, nothing unnatural on inspection
2. **unclear**: two animals, but not clearly a cat and a dog, or clearly both with unnatural
   details (blended features, odd limbs or tail, an object from outside the pool)
3. **not two**: one animal, a chimera, or the same animal twice

A run is better than the baseline under the same sampler when it has more seeds labelled clean
and no seed that the baseline had clean falls out of that label. A child's switch mattered when
the parent is better than the child by that same rule. If the baseline under the stacked sampler
is already better than the baseline under the shipped sampler, that is reported first and the
training runs are read against it.

The count scorer and the both-names read are logged beside the labels as secondary reads. They
order the strips for viewing and are never the verdict. The correction size per pair is reported
beside the results as a measurement, and any claim about it waits until clean, unclear and
not-two examples have been laid against it.

What would surprise: C2 better than P. That says training on the late steps helps even though
running the adapter there hurts, and 8.3's unknown becomes the finding.

## Code before launch, three small changes to `train_pooled.py`

1. `--exclude-cells <json>`: a list of (pair, seed) to skip when the dataset is built. The list
   comes from `/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json`,
   rows with `both_present == false` on train pairs.
2. `--train-step-range LO HI`: sample training steps from [LO, HI) only. P uses 0 24.
3. `--orth-weight W`: per sample, project the target and the prediction onto the span of
   (eps_a - eps_uncond, eps_b - eps_uncond) and its complement; loss = in-span error + W x
   complement error. The projection is what `scripts/showcase/correction_span_common.py` already
   computes.

And one change to the sampler used for the stacked render: a guidance interval (guidance 7.5 on
steps [5, 35), 1.0 outside) beside the existing `lambda_window` and eta flags.

## Cost

Rank 32 needs about 25 GB, so on a 3090 pass `--gradient-checkpointing`; on the Blackwell it
fits as is. Roughly 4 to 6 hours per run at 30k steps on 117 cells. bigbatch allows six jobs per
user, so all five training runs go up together and finish the same evening. Rendering both
samplers over the held-out seeds adds about 40 minutes per run.

## Not in this batch, and why

The better teacher (an attention-guided joint sampler whose prediction replaces the joint
prompt's as the target) is the strongest data idea and it needs a new cache plus a validation
that the guided sampler composes where the plain one does not. It is claim 9 and it is the
follow-on if C1 shows wrong targets matter.

## Where it runs, and in what order

Read live on 2026-09-08 against `environment/hpc/nodes.md`, `execution-protocol.md` and
`throughput.md`.

**The device.** `mscluster112`, an RTX PRO 6000 Blackwell with about 96 GB, reads 2 MiB used
and 0% utilisation with no foreign process. `mscluster110`, the other healthy Blackwell, is
carrying another user's job at 97%. `mscluster111` is hardware-faulted (`[N/A]` utilisation;
torch sees no CUDA) and stays off the list until an administrator resets it. So there is one
Blackwell available, and `biggpu` allows one job per user in any case.

**Why the Blackwell and not five 3090s in parallel.** Rank 32 steps at about 0.24 s on the
Blackwell against 1.14 s on an RTX 8000, and rank does not change step time. Thirty thousand
steps is about two hours there plus ten minutes of eval passes; the same run is over nine hours
on an RTX 8000 and needs `--gradient-checkpointing` on a 3090 to fit 25 GB. Five runs in a chain
on `mscluster112` finish in roughly twelve hours, overnight, on one device with one environment
and no cross-device spread in the reads. That beats five parallel slower runs on mixed hardware.

**The chain.** One launcher on `/datasets` runs P, then C2, then C1, then C3, then C4, each with
its own run id and W&B run, so the parent and the child that tests the late-step mask (the one
most likely to matter) land first. If a run dies the chain moves on and says so in the log.

**Environment and paths.** `co3_bw` on the Blackwell, never `co3` (a `co3` CUDA op there
produces no output rather than an error). Every path on the launch line absolute. The launcher
copied under the run's output root on `/datasets` before launch, because `/tmp` is node-local.
Launch shape, from the runbook:

```bash
ssh mscluster112 'GPU=0 nohup bash /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/launch_chain.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/chain.log 2>&1 &'
sleep 5; ssh mscluster112 'pgrep -af train_pooled; tail -20 /datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/logs/chain.log'
```

**Guards inside the launcher, all mandatory.** Disk guard on `/datasets` (the filesystem it
writes to), abort at 90%. `co3_bw` python path check. `nvidia-smi` guard that refuses a device
over 1 GB in use or reading `[N/A]`. Node, device and PID written to the log header.

**Harvest.** These runs are invisible to `squeue`. Read the log and count checkpoints over SSH on
`mscluster112`, not from the session node, whose view of `/datasets` lags by minutes.

## What goes to W&B, and from where

The trainer already logs, per run, at every eval pass: `eval/compose_rate/{quadrant}/{pair}/seed_NN`
and its mean (the count read), `eval/direction_cosine/...` and `eval/frac_distance_reached/...`
(drift monitors only, since neither predicts composition), the tracking-set thumbnails (the joint
target, plain PoE and the adapter side by side, the triptych `/analyze-run` reads),
`train/lora_weight_norm` and its EMA (the decay watch), and step and checkpoint timings. Those
come for free with `--wandb-mode online --wandb-project prime_lab/poe-repair-animals-compose
--sample-every-epochs 200`.

What the design adds, logged into the same W&B run by id after each checkpoint lands, from a
probe pass on the session node's 3090 (inference only): the strip per held-out seed under the
**stacked** sampler (joint target | plain PoE | 30050 | this checkpoint), `eval/both_names/...`
from the forced-choice read, and `eval/nearness_clean/...` on the five clean-reference cat x dog
seeds. The probe is `lambda_boundary_probe.py` with the guidance interval added, pointed at the
checkpoint, with `--resume-wandb-id`.

## Before the launch line, in order

1. Three trainer flags (`--exclude-cells`, `--train-step-range`, `--orth-weight`) and the
   guidance interval in the windowed sampler. `--dry-run` on the parent config to see the cell
   count it builds (expected: 88 + 28 minus the excluded cells) and that every flag parses.
2. The exclusion list, `exclude_cells.json`, written from `target_quality.json` (train pairs,
   `both_present == false`), with its count printed beside it. It over-flags similar pairs; that is
   what C1 measures.
3. The 14-pair pool and prompts yaml under `artifacts/_shared/cross_pair_pool_configs/`, and a
   check that every cell the pool names exists in the cache.
4. Baseline B under the stacked sampler on the session node, eight cat x dog seeds, one sheet
   against the shipped render. If it reaches the read on its own, say so before spending a
   training run.
5. A 200-step smoke of P on `mscluster112` under `co3_bw` with `--sample-every-epochs 2`, to see
   the W&B run appear with its panels, the checkpoint land on `/datasets`, and the step time
   near 0.24 s. Kill it, then launch the chain.
6. The launch line above, then the verify line. Then nothing until the first checkpoint.
