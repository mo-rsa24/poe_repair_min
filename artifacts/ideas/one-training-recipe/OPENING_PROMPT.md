# Opening prompt: one training recipe for a clean, composing adapter

For `/drip-idea`. Written 2026-09-25 from a read of all 22 folders under `artifacts/results/`, the
`report/` findings, the trainer code, and W&B project `prime_lab/poe-repair-animals-compose`.

## The idea to walk

I want one training recipe for the correction adapter that, run for a stated number of steps,
reliably produces an adapter that does two things at inference on pairs it never trained on:

1. **Composes.** Product-of-experts sampling (PoE) through the adapter renders both concepts as
   separate objects in one scene: a cat sitting beside a dog, never a single fused animal.
2. **Keeps fidelity.** Each object reads as its own species, cleanly, at the base model's photo
   quality: no haze, no low resolution, no third object, no hybrid on one of the two, no wrong
   species, and no large swing in the picture between neighbouring checkpoints.

Co-occurrence already emerges in most runs. Fidelity is where every run fails. The walk should
find the recipe, or say plainly which part of it cannot be had and why.

## What the system is (the facts a diffusion researcher pins first)

- **Base model and space.** SDXL base 1.0, 1024 px, 4x128x128 latents, SDXL's own VAE upcast to
  fp32. The network predicts epsilon (`prediction_type` is never overridden). fp16 weights, LoRA
  weights in fp32 (`poe_repair/experiments/one_pair_one_seed/trainer.py:247-259`).
- **Sampler.** DDIM, eta 0, 50 steps for caches and final renders, 20 for inline previews during
  training (`poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py:215,299`).
- **How PoE combines the branches.** `eps = guided(a) + guided(b) - u` with `guided(c) = u + w(c - u)`
  and w = 7.5, so each concept enters at +7.5 and the empty (unconditional) branch at -14
  (`poe_repair/methods/_sdxl/metrics.py:12-17`).
- **Where the adapter acts.** A LoRA on attn2 q,k,v (optionally attn1 too, `--lora-targets
  cross+self`), switched on for all three branches, the empty branch included. At render time
  `eps = eps_frozen + λ(eps_lora - eps_frozen)`, λ = 1 (`poe_repair/methods/_poe_langevin.py:397-406`).
- **What it is trained on.** Distillation: the adapted PoE prediction is regressed onto the frozen
  model's guided joint-prompt prediction ("a cat and a dog"), which is the same as fitting
  r_t = ε̃_joint − ε̃_PoE. States come from cached plain-PoE trajectories, so training is off the
  path the adapter later runs on (`scripts/build_training_cache.py:168-170`). The v58 runs use
  the clean-image (x0) weighted loss with the weight cap lifted, and train on cached steps 0 to 10.
- **The loss cannot see one direction.** It sees only `a + b − u`. Adding the same tensor to a
  concept branch and to the empty branch leaves the loss unchanged while sampling reads it at
  −6.5x. Whether trained runs use that direction was never measured (the undialled error in
  `plans/09-designing-the-correction-loss/MASTER_PLAN.md`).
- **Optimiser.** AdamW 1e-4, weight decay 0, grad clip 1.0, batch 1. v58 cohort: cosine decay
  over 50k, EMA 0.9999 saved as `lora_state_ema`, checkpoints every 2,500 steps.
- **Data.** `artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml`: 48 training pairs
  (animals plus some objects), about 182 cells from `cells_v58.json`. Every training target was
  inspected by hand for clean co-occurrence: the joint-prompt render where it drew both concepts,
  and a CO3 render where the plain joint prompt failed and CO3 drew both. Which cells carry a CO3
  target is not recorded in the pool file. Held out: cat×dog, piano×candle, skateboard×helmet,
  elephant×penguin, seeds 9 to 16.
- **Unknown, and the walk should say so where it matters:** an intrinsic-dimension estimate for
  this data; which seed each "good" picture below was judged on; which refl-03 step drew the clean
  elephant and penguin; whether a repeated render of one checkpoint is bit-identical (a
  `showcase/determinism_repeat_1/2` folder exists on the cluster, unread).

## The pictures worth keeping

| Run | W&B | Checkpoint | What it drew |
|---|---|---|---|
| v58-05-self-contrast (x0 loss, cross+self, contrast 0.01) | `1bzw25hh` | 15,000 | a clear penguin and a clear elephant; also a clean tiger and dog on seed 10 |
| v58-03-self (same, no contrast term; one-axis comparison with the row above) | `qtk5gi37` | 27,500 | a clear elephant and penguin, worse before and after |
| refl-03-self-contrast-15k (reward fine-tune from v58-05's step-15,000 raw weights) | `95o900mc` | unknown, 250 to 1,250 exist | the first clean elephant and penguin |
| v57-09-x0-seeds | `ek4w6lbu` | 45,000 | a sharp tabby and white dog on cat×dog seed 9 |
| v57-11b-x0-self-decay (first run with cosine decay) | `tiiz0c0c` | 40k to 45k | near-identical renders across three checkpoints; clean on elephant×penguin seeds 11 and 15 |
| the original rank-32 pool adapter | `6xc2l8ix` | | a cat and a dog, blurry |

Figures: `artifacts/results/which-checkpoint-to-take-each-figure-cell-from/latest/`,
`artifacts/results/which-fidelity-fix-wins/`, `artifacts/results/which-renders-show-it-working-and-failing/elephant-and-penguin-seeds-and-checkpoints.png`.

## The failures, and where each is on disk

| Failure | Where it shows |
|---|---|
| one object clean, the other a hybrid (penguin-bodied elephant, lion with a meerkat face, pug-tiger) | `which-fidelity-fix-wins/`, `does-the-fix-reach-unseen-pairs/`, `what-full-step-coverage-changes/` |
| wrong species or two of one (cat turns into a puppy at 30k on cat×dog seed 42; meerkat drawn as a lion cub) | `which-checkpoint-to-take-each-figure-cell-from/latest/` |
| a stray third animal or props | cat×dog seeds 9 and 42 across many runs; guidance 10 |
| grey mush and haze | rank 8 at 450k; v58f at 10k; the adapter's early contrast collapse, −56% over steps 0 to 9 (`plans/09-designing-the-correction-loss/review/06-*`) |
| flat line art or clip art | every corrector; the adapters trained on the 54, 72 and 182-cell pools |
| invented people | v57-10-x0-contrast on cat×dog seed 9; four broad-pool adapters |
| the evaluation reference is wrong on held-out seeds (training targets are clean) | the joint prompt drops or adds a concept on 11 of 32 held-out renders; cat×dog seeds 10 and 14 draw two of one animal, so any distance-to-reference score is unreadable there (`report/is-the-adapter-just-copying-the-joint-prompt/does-the-joint-prompt-always-draw-both-concepts.md`) |

## Already tried: do not propose these again without a new reason

Each line is settled in the file named. Its failure counts as evidence in the walk.

- **Samplers that add no correction** (twisted SMC, steering on plain PoE, noise search on plain
  PoE, a Langevin corrector on its own): never produce two animals. `report/is-the-gap-the-samplers-or-the-models/`
- **Feynman-Kac steering**: on plain PoE, 0 of 128 particles compose. On the adapter at λ 0.5
  (`dfrceppl`) it composes 8 of 8, but best-of-16 with no steering also composes 8 of 8, so the
  draws compose on their own and the weights only pick which picture. The count scorer steering
  on cannot tell a cat and a dog from two dogs. `report/is-the-gap-the-samplers-or-the-models/does-steering-the-corrected-sampler-produce-clean-pairs.md`
- **Render-time fixes on v58-05 @15k**: guidance 10 and 12 change what is in the scene; a fixed
  3/3 expert reweight smears both subjects on 8 of 8 cells; repaint at 0.25 and 0.40 and a
  20-step handover cannot be told apart from the adapter by eye. `report/is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md`
- **Samplers bolted onto the v58 adapters**: Langevin on the tail, CO3, SuperDiff with the adapter,
  best-of-8. None beat the adapter alone; CO3 turns everything into clip art, SuperDiff into
  noise texture. `artifacts/results/which-sampling-strategy-helps/`
- **Training longer**: fidelity decays from about 70k (rank 32) and 160k (rank 8); the weight norm
  grows without bound at weight decay 0. `report/does-training-longer-help-the-pooled-lora/`
- **Rank 8/16/32/64 and adding self-attention**: no trend, and every comparison so far is mixed
  (two or more axes differ). `report/what-the-adapter-ablations-show/`
- **Energy penalty**: renders drift toward plain PoE; the adapter already spends about half the
  true correction's energy. `plans/01-showcase-the-trained-lora/review/22-*`
- **Adapting only the empty branch**: one-animal clip art. **Training on real photographs**: the
  co-occurrence filter is wrong a third of the time. **CFG++, APG, keeping PoE's norm, manifold
  projection**: null.
- **Growing the pool** (43, 54, 72, 182 cells): brighter pictures with fused faces and sketch style.
- **Scores that cannot be used as the judge**: the instance-count compose rate (it cannot tell a
  cat and a dog from two dogs, and sits at 1.0 on every v58 eval); Laplacian variance (it counts
  edges, prefers sketches, passed guidance 12); DINOv2 distance to the joint-prompt render (the
  reference is wrong on many seeds, and it ranks both flagged good checkpoints 17th and 18th of 20).

## What the W&B read found, which the walk should weigh first

- **The saved EMA weights have never been rendered.** Every sheet and every probe loads the raw
  `lora_state` (`scripts/figure_candidates.py` needs `--ema`, which `scripts/sweep_probe.sbatch`
  never passes). refl-03 also started from the raw weights.
- **Late renders swing while the weights barely move.** In v58-03 between 40k and 50k the learning
  rate is below 1e-5, the raw weight norm goes 92.55 to 92.69, and the elephant×penguin seed 9
  score goes −0.41, +0.19, −0.37, +0.11, +0.32. So part of what looks like training instability
  may be the render itself flipping on tiny weight changes, near where the sample chooses between
  modes.
- **Training loss falls smoothly** and shows nothing at the good steps.
- **refl-03's reward rose through the distinctness term alone** (0.40 to 0.54). Identity stayed at
  about 0.27. The ImageReward picture-quality term had weight 0 (it did not exist yet). The run
  trained on exactly the two pairs being judged, at constant lr 1e-5 with no EMA, and its gradient
  sat above the clip on nearly every step. It crashed at 1,331 steps. refl-04-fidelity, the run
  that adds the ImageReward term, crashed at step 49. **The reward idea has not had a fair test.**
- **Most v58 and v58f runs crashed** before their planned length (v58f-05 at 27,516, v58f-06
  huber at 7,406, v58f-07 out-of-span at 541).
- **Unread runs that bear on the null-branch question:** freeze-null `kdzx03ji`, adapt-null-only
  `qbvj0wsm`, null anchor `72as6yk3`, v58-04-freezenull `mwrgppyw` (crashed at 45,979).

## Directions the walk should test, not assume

My hunches, plus candidates added from the read. Each is a claim for the walk to mark, not a
premise:

1. **Some of the checkpoint instability is in the render.** Render one checkpoint twice, and the
   EMA weights beside the raw ones, before blaming the objective.
2. **The empty branch is the lever.** The adapter sits on a branch weighted −14, and the loss is
   blind to a direction sampling amplifies. Does freezing or anchoring the empty branch, or
   logging the undialled error, change fidelity? Read the four unread null-branch runs first.
3. **Clean targets still leave room for hybrids.** Every v58 target shows clean co-occurrence, so
   the hybrids are not copied from the targets. Two things the walk should separate: whether the
   CO3-sourced targets carry CO3's flat illustration style into the adapter (CO3 renders poster art
   on our pairs, `artifacts/results/what-co3-does-on-our-pairs/`), and whether a clean target
   reached only through an early-window regression still lets the second object borrow the first
   one's features at render time.
4. **Training off the adapter's own path costs fidelity.** The adapter trains on plain-PoE states
   and runs on corrected ones. The `--adapter` cache (commit b98673f) steps with a fixed earlier
   adapter and was never used in a run.
5. **A picture reward belongs in training**, with a fair test this time: a per-concept identity
   term that scores the weaker of the two objects rather than their sum (so a clean elephant
   cannot pay for a hybrid penguin), the ImageReward term switched on, an anchor to the
   distillation adapter so the reward cannot drag it off, EMA and decay, and pairs held out from
   the reward.
6. **Attention-level supervision against the hybrids.** Attend-and-Excite already exists here as
   an inference-time intervention (`poe_repair/experiments/mechanism_study/aae_intervene.py`,
   `guide_lora_attention.py`), result unread. The training-time version is a token-level
   attention loss that keeps each word's cross-attention on its own object, with masks from the
   target renders (the TokenCompose line of work). This goes at the failure where one object
   borrows the other's features.
7. **Stopping needs a judge that sees species.** A recipe that is "reliable after N steps" needs a
   per-checkpoint score that can tell a hybrid from a clean pair, fixed before the run, on
   held-out pairs and more than one seed.

## Constraints

- One recipe to train, one number of steps, one stopping rule. Every change is a single axis
  against the current v58-05 recipe, so each comparison stays clean.
- The correction only works in steps 0 to 10 (`artifacts/results/when-the-correction-must-arrive/`).
  The early window is fixed, and whatever the adapter does after step 10 must not cost fidelity.
- The bar goes in code before the run, in the pattern of `MIN_MEDIAN_RATIO`.
- Fidelity is judged by eye on a fixed sheet of held-out pairs and seeds as well as by any score,
  and the seed is named for each judgement, because the seed sets photo against drawing.
- Cluster runs go through `scripts/cluster.sh`. Rank 32 needs `--gradient-checkpointing` on a
  3090. Disable the commit-bucket kill with `--kill-halve-after-steps`.

## Route weave (guidance the walk can override)

- Early: a `run` of the two free checks before any new training, the double render and the EMA
  render of v58-03 at 25k to 30k and v58-05 at 15k.
- On directions 5 and 6: a `known` search for training-time attention losses (TokenCompose,
  Attend-and-Excite as a loss, Divide-and-Bind) and for reward fine-tuning that keeps fidelity
  (ReFL, DRaFT, anchored or KL-regularised reward fine-tuning), then a `/pressure-test` on
  whichever survives.
- Near compile: `/frame-hypothesis` on the chosen recipe, so the bar and the null are written
  before it launches.
- Each route's result comes back to the walk through `integrate`.
