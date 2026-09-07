# Does an adapter trained on SuperDiff's own residual compose inside SuperDiff?   ❓ inconclusive, with a clear direction · verified 2026-09-05

**The claim**

An adapter trained on SuperDiff's own residual does what the PoE-trained one could not: by
10k to 20k optimizer steps every rank puts a cat beside a dog on three or four of four seeds
inside SuperDiff at full strength, and rank 8 at 30k also puts a clear butterfly on all four
meadow seeds. Then every rank gets worse with more training at the same strength, the bigger
the adapter the sooner: rank 32 separates nothing by 30k, rank 16 nothing by 50k, rank 8 still
three of four seeds at 60k but softer each checkpoint. The training loss falls and plateaus
through all of this and never sees the smear. The three runs were stopped at 61k, 62k and 36k
of 100k on that reading, so the pre-registered sweep never ran and the verdict stays
inconclusive by the plan's own criterion.

**What would have counted**

Pre-registered in [the plan](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md):
per pair and rank, the first separating λ per seed on the finished adapters, beside the
no-adapter sheet and the PoE-adapter sheet. Pass if lower than the baseline on more seeds than
not for at least one rank; fail if equal at every rank; inconclusive for any rank whose
training did not converge. Judged in
[the review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/08-adapters-that-learn-superdiffs-own-residual.md),
where every checkpoint read below is recorded.

## 1. It separates early, then smears

![Rank 8 on cat × dog: rows are the mono target, SuperDiff alone, then the adapter at each 10k checkpoint; columns seeds 9 to 12](../../artifacts/results/is-the-gap-the-samplers-or-the-models/superdiff-lora-rank8-cat-dog-checkpoint-timeline.png)
*Every tile is SuperDiff at 200 steps, κ 0.5, the adapter injected on every step at λ 1. What to
notice: SuperDiff alone is one animal on every seed; the 30k row is two distinct animals on all
four; the 40k to 60k rows keep two bodies but lose the fur and the background.*

![Rank 16 on cat × dog, same layout](../../artifacts/results/is-the-gap-the-samplers-or-the-models/superdiff-lora-rank16-cat-dog-checkpoint-timeline.png)
*What to notice: two silhouettes at 10k to 30k, then a painterly smear, then by 50k and 60k one
animal per seed again.*

**The number.** Seeds of four, eye read, showing two bodies on cat × dog, per checkpoint in
thousands of optimizer steps. Rank 8: 3, 4, 4, 4, 3, 3 at 10k to 60k. Rank 16: 4, 4, 4, 3, 0, 0.
Rank 32: 4, 0, 0 at 10k to 30k. Butterfly × meadow, seeds with a clear butterfly: rank 8 2, 2,
4, 0, 0, 1; rank 16 1, 0, 0, 0, 0, 0; rank 32 0, 0, 0. Recorded in the review file's run table
as each strip landed, drawn in `superdiff-lora-separation-by-checkpoint.png` and listed in
`superdiff-report-figures.json` beside it. Strips under
`corrector/superdiff/sdlora_r{8,16,32}_100k/samples/superdiff/`, also on W&B in the companion
runs [rank 8](https://wandb.ai/prime_lab/poe-repair-animals-compose/runs/sdlora_r8_samples),
[rank 16](https://wandb.ai/prime_lab/poe-repair-animals-compose/runs/sdlora_r16_samples),
[rank 32](https://wandb.ai/prime_lab/poe-repair-animals-compose/runs/sdlora_r32_samples)
against the checkpoint step.

![Seeds of four that separate, per checkpoint, per rank](../../artifacts/results/is-the-gap-the-samplers-or-the-models/superdiff-lora-separation-by-checkpoint.png)
*y is the count of seeds 9 to 12 that separate, x is the checkpoint in thousands of steps, one
line per rank, left cat × dog and right butterfly × meadow. What to notice: the bigger rank
peaks earlier and falls to zero sooner.*

## 2. The loss does not see it

![Training loss per rank, total and per step bucket, log scale](../../artifacts/results/is-the-gap-the-samplers-or-the-models/superdiff-lora-loss-curves.png)
*y is mean-squared error against the cached residual on a log axis, x is the optimizer step, one
line per rank, a 25-point moving mean; the buckets are SuperDiff steps 0 to 19, 20 to 99 and 100
to 199. What to notice: every curve falls and flattens from about 30k to 40k, and nothing rises
where the strips above degrade.*

**The number.** Total loss, 25-point moving mean at the stop: rank 8 0.0008 at step 61,000,
rank 16 0.0025 at 62,000, rank 32 0.0017 at 36,000. Rank 8 is lowest in every panel. Read
2026-09-04 from the W&B histories of
[t5b00h56](https://wandb.ai/prime_lab/poe-repair-animals-compose/runs/t5b00h56),
[5oz1bsa1](https://wandb.ai/prime_lab/poe-repair-animals-compose/runs/5oz1bsa1) and
[ue205g6e](https://wandb.ai/prime_lab/poe-repair-animals-compose/runs/ue205g6e), keys
`train/loss` and `train/loss_bucket/*`.

## 3. Half strength is usable where full strength smears

![Rank 16 at step 30k across λ, rows seeds 9 to 12](../../artifacts/results/is-the-gap-the-samplers-or-the-models/cat-dog-rank16-step30k-lambda-sweep.png)
*Same layout as the no-adapter sheet in the first finding. What to notice: the λ 0.5 column has
two clean animals on seeds 10 and 12 while the λ 1 column is a smear on every seed.*

**The number.** Seeds of four with two bodies at λ 0.5, rank 16: 2 at step 30k, and 3 at step
10k on the same sweep (`cat-dog-rank16-step10k-lambda-sweep.png`); at λ 1 the counts are 4
(silhouettes in a smear) and 3. Every column at 30k is hazier than the same column at 10k, so
part of the smear is dose and part is learned. Eye read, recorded in the review file; renders
under `corrector/superdiff/preview_sdlora/`.

## 4. The cache is what the composer computed

**The number.** The training target formed from one cached cell (`a_wolf__x__a_husky` seed 1)
matches the composer's own `r_t_sd_norms` for that render to a relative error of 3.6 × 10⁻⁵ at
steps 0, 50, 100 and 199 (5.358 vs 5.359, 14.546 vs 14.546, 29.943 vs 29.943, 9.614 vs 9.614).
The cache is 88 cells, 17,600 step files, 11 GB, every one of the 200 steps kept. From the review
file's run table. No picture is owed for this rung; it is the check that the adapters trained on
the right target.

## What this cannot tell you

Why the rendered image worsens while the loss keeps falling. Two readings fit and nothing here
separates them: full strength over-applies a correction fit on trajectories the adapter never
steered, or the cached residual carries high-frequency texture the adapter learns late. Every
read above is an eye read on four seeds; the detector was part of the un-run final set. The
comparison with the PoE-trained adapters is fair on pool, ranks, alpha and modules but not on
training length, since these were judged on earlier checkpoints, and that cuts in their favour
because they were at their best early. The verdict is inconclusive because the pre-registered
number, the first separating λ on finished adapters, does not exist.

## Where this came from

| What | Source | Mark |
|---|---|---|
| The loss values and curves | W&B histories of the three runs above, read 2026-09-04 by the plotting script recorded in the review | verified |
| The per-checkpoint separation counts | eye reads of the 15 strips, recorded in the review file's run table as each landed, 2026-09-03 to 09-04 | stated |
| The cache check and size | review file run table, 2026-09-03 | inferred |
| The trainings | rank 8 and 16 on mscluster109 devices 0 and 1 (`co3`, about 42 s per 50-step epoch), rank 32 first on mscluster112 (`co3_bw`, 24 s per epoch) then resumed on mscluster106 device 1 (64 s per epoch) after a foreign process took the Blackwell; launched 2026-09-03 12:38, restarted twice for stall-guard recalibration (window 5 to 24 → 20 to 99, halving deadline 5k → 20k), stopped by the user 2026-09-04 01:30 | verified |
| The strips | `scripts/superdiff/sdlora_sample_watcher.py` on mscluster112 device 0, 4 renders per pair per checkpoint at about 45 s each, 2026-09-03 17:40 to 2026-09-04 01:30 | verified |
| Regenerate with | [runbook § train an adapter on SuperDiff's residual](../../runbook/running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#2-launch-one-training) and [§ watch its checkpoints](../../runbook/running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#3-render-a-strip-from-every-checkpoint-while-it-trains) | |
| The plan | [adapters that learn SuperDiff's own residual](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md), root step 50 | |

## Depends on

- The SuperDiff blend inside the trainer's step, `ε_u + g · ((ε_b − ε_u) + κ (ε_a − ε_b))`, selected by `--compose superdiff --kappa 0.5`: the scope [changelog entry of 2026-09-03](../../plans/06-is-the-gap-the-samplers-or-the-models/CHANGELOG.md)
- Why the stall guard was rescaled twice before these runs could get past step 5k: the same changelog entry and the review file
- Which node needs which python (`co3_bw` on the Blackwell nodes): [the environment index](../../environment/00-INDEX.md)

## Still open

- [ ] The pre-registered λ sweep on the stopped checkpoints (`scripts/superdiff/run_sdlora_sheets.py` reads the latest checkpoint per rank once its final-name check is relaxed), if a SuperDiff-side adapter is ever wanted.
- [ ] A per-step ‖Δ̂‖ against ‖r_t^SD‖ on rank 8's 30k and 60k checkpoints, which would separate the dose reading from the learned-texture reading.
- [ ] Whether rank 8 at 30k and λ 0.5 is the adapter to keep.
- [ ] The cache check and size are lifted from the review; re-running `scripts/superdiff/check_cache_cell.py` makes them `verified`.
