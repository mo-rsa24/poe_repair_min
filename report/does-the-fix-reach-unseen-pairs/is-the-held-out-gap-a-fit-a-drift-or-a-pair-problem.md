# Is the held-out gap a fit, a drift, or a pair problem?   ✅ support: it is a pair problem · ⚪ null on drift · verified 2026-09-05

**The claim**

The rank 8 adapter trained on 11 animal pairs matches its target on the training pairs (cosine
0.97) and less well on pairs it never trained on (0.80). Three explanations were on the table.

It does not fit: ruled out. The fit on training pairs is near perfect.

It is evaluated on states it never trained on, since the cache holds states from the
uncorrected run and inference visits the corrected one: ruled out. The adapter fits its own
corrected trajectory at 0.82, within 0.04 of the cache.

It learns one correction per training pair and does not carry it to new pairs: supported. Pairs
made of two words the adapter saw in training, in a pairing it never saw, fit at 0.84, no better
than pairs of words it never saw at all.

The plan's original number, a render-time cosine of 0.40 that never moved, measured neither fit
nor drift and is retired.

**What would have counted**

Three criteria were written in the idea map's `Checks outstanding` before each run, in
[the idea map](../../artifacts/ideas/improving-the-pooled-lora-run/IDEA_MAP.md).

For drift: if the on-policy cosine starts near 0.9 and decays toward 0.4 along the run, the 0.40
is drift. If it is flat from step 0, it measures something else.

For the pair question: if pairs of seen words in an unseen pairing fit near the training 0.97,
the gap is about new words. If they fit near 0.80, it is about new pairs.

The run-to-run noise floor was set at 0.081, the across-cell spread from run 1.

No criterion was pre-registered in a plan file. The idea walk was the register.

## 1. The adapter fits its own trajectory about as well as the cache

![Fit cosine against denoising step for cat x dog, on the corrected trajectory and on the cached states, with the state drift on the right axis](../../artifacts/results/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem/on-policy-vs-cached-fit-cat-x-dog.png)
*Red line: cosine between the adapter's correction and the true correction at the state the
adapter actually reached, mean over seeds 9 to 16, shaded one standard deviation. Blue points:
the same adapter on the cached, uncorrected states, every fifth step. Dashed grey, right axis: how
far the corrected state has moved from the cached one, as a fraction of the cached state's norm.
What to notice: the red line never leaves the 0.75 to 0.9 band while the grey line climbs to 0.8.
The dotted line at 0.40 is the number the plan had been chasing.*
📊 Drawn in [Figure 1 of the figure explainer](../../artifacts/results/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem/figure-explainer.md#figure-1-fit-on-the-adapters-own-trajectory-against-fit-on-the-cache).

**What you are looking at.** Two measurements of one adapter on one pair. The blue measurement
asks the adapter the question it was trained on. The red measurement asks it the question it
faces at inference, where every step's state includes every earlier correction. If drift were the
cause of the held-out gap, red would fall away from blue as grey rises. It does not.

**The number.** On-policy cosine, mean over 8 seeds and 50 steps: 0.817, with per-window means
0.855 (steps 0 to 4), 0.810 (5 to 24), 0.815 (25 to 49) and per-seed means from 0.732 to 0.884.
Cached-state cosine for the same pair and adapter: 0.860. Relative state drift at steps 10, 25
and 49: 0.04 to 0.08, 0.14 to 0.27, 0.69 to 0.86. The true correction at the reached state
against the cached correction at the same step index: cosine 0.015 in the late window, so a read
that compares the adapter's output against the cached target along a corrected run reads near
zero by construction. From
`/datasets/mmolefe/poe_repair_min/outputs/showcase/on_policy_fit_r8_030000/on_policy_fit.json`,
fields `rows[].cos`, `rows[].state_drift_rel`, `rows[].cos_true_vs_cached_target`, summarised in
`on_policy_fit_table.md` beside it; and
`/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_r8_030000/fit_cosine.json`,
rows with `pair == a_cat__x__a_dog`. Written by `scripts/showcase/on_policy_fit_cosine.py`
(Slurm job 49421, mscluster72, 2026-09-03) and `scripts/showcase/fit_cosine_on_cache.py`.

**What this figure leaves out.** It is one pair. Whether the drift matters more on a pair where
the correction is larger is not measured. It also says nothing about the compose rate: seed 14
fits at 0.77 here and still renders one animal, so fit and outcome are different questions.

## 2. Seeing the words does not help; seeing the pair does

![Fit cosine per pair, grouped by how much of the pair the adapter saw in training](../../artifacts/results/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem/fit-by-held-out-tier-rank8-30k.png)
*One bar per pair, mean cosine over cells, whisker one standard deviation across cells. Green:
the 11 training pairs. Blue: both words seen in training, never together. Purple: one word seen.
Red: the pool's held-out pairs, no word seen. Dashed lines: group means. What to notice: the blue
bars sit with the red bars, a full 0.13 below the green ones.*
📊 Drawn in [Figure 2 of the figure explainer](../../artifacts/results/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem/figure-explainer.md#figure-2-fit-by-how-much-of-the-pair-the-adapter-saw).

**What you are looking at.** The adapter's fit sorted by exposure. If the adapter had learned a
correction per animal, wolf x horse would inherit what it learned from wolf x husky and horse x
zebra, and the blue group would sit near green. It sits near red.

**The number.** Mean fit cosine on cached states, rank 8 at 30k, 10 steps per cell: training
pairs 0.969 (11 pairs, 880 cells); both words seen, pair unseen 0.839 (wolf x horse 0.815, lion x
horse 0.862, 12 seeds each); one word seen 0.769 (six pairs, 0.705 to 0.827); no word seen 0.800
(eight pairs, 0.677 to 0.877). The blue group sits 0.13 below training, well past the 0.081
noise floor, and 0.04 above the red group, inside it. From
`/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_tiers_r8_030000/fit_cosine.json`
and `fit_cosine_table.md` (Slurm job 49851, mscluster52, 2026-09-03, pool configs under `pool/`
beside them) and `fit_cosine_r8_030000/fit_cosine.json` for the green and red groups.

**What this figure leaves out.** Two pairs make the blue group, both sharing the word horse, so
the group mean is two numbers rather than a distribution. The purple pairs mix two things, a seen
word and an unseen one, so their position between blue and red is what one would expect and not
evidence on its own.

## 3. The number the plan was built on

No figure.

The render-time cosine of 0.54 on training cells and 0.40 on held-out cells, flat from 10k
steps, was the quantity the idea walk set out to move with more training pairs. It is neither of
the two measurements above.

It is `direction_metrics` in `poe_repair/experiments/cross_pair_lora_pooling/_inline_sampling.py`:
the cosine between a cell's correction and the pool-mean correction at the same step, averaged
over the steps where the correction was recorded. The docstring names 0.4 as "the observed
plateau".

A state-specific target cannot align with a pool mean. So this number measures how much of the
correction is shared across cells, and nothing about fit.

It is retired from the map. It appears here only so the retirement is traceable.

## 4. The pictures behind the numbers

![Eight seeds by four columns: plain product of experts, the showcase sampler with the correction, the run 2 rollout with the correction, the joint prompt](../../artifacts/results/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem/cat-x-dog-eight-seeds-poe-corrected-joint.png)
*Rows are held-out seeds 9 to 16. Columns: the cache's own product-of-experts render, the
showcase sampler with the rank 8 correction at λ 1, run 2's rollout with the same correction, the
joint prompt. What to notice: columns two and three are identical on every row, and seed 14 is the
one row where the corrected columns show a single animal.*
📊 Drawn in [Figure 3 of the figure explainer](../../artifacts/results/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem/figure-explainer.md#figure-3-the-eight-seeds-four-ways).

**The number.** Compose verdicts from the instance-count scorer on the showcase renders, rank 8
at 30k, λ 1, full window: 7 of 8 seeds compose, seed 14 does not (one instance); at λ 0 all eight
count one instance. From
`/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r8_030000/results.json`, fields
`rows[].compose` and `rows[].n_instances`, and `summary.full.1.0.compose_rate` 0.875. Renders:
`training_cache/heldout/a_cat__x__a_dog/seed_<n>/{poe,mono}.png`,
`figure_r8_030000/renders/full/seed_<n>_lambda_1.0.png`,
`on_policy_fit_r8_030000/seed_<n>_corrected.png`.

**What this rung leaves out.** Seed 16's product-of-experts render looks like two animals to the
eye and counts as one instance; that disagreement is noted and not resolved. Style drifts between
columns on three seeds and the scorer is blind to it.

## What this cannot tell you

**Fit is not composition.** Seed 14 of cat x dog fits inside the pack on both measurements and
still renders one animal. Nothing here says what makes a seed fail; that belongs to the basin
work in [when does the outcome lock in](../../plans/05-when-does-the-outcome-lock-in/MASTER_PLAN.md).

**One rank, one checkpoint.** Whether rank 32 or a later checkpoint generalizes per concept is
not measured.

**One λ, one window.** The on-policy read is at λ 1 with the correction at every step, the
setting the showcase renders use, and no other.

## Where this came from

| What | Source | Mark |
|---|---|---|
| On-policy cosine, drift, true-vs-cached target cosine | `/datasets/mmolefe/poe_repair_min/outputs/showcase/on_policy_fit_r8_030000/on_policy_fit.json`, read 2026-09-05 | verified |
| Cached-state cosine, training and pool held-out pairs | `/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_r8_030000/fit_cosine.json`, read 2026-09-05 | verified |
| Cached-state cosine, the two middle tiers | `/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_tiers_r8_030000/fit_cosine.json`, read 2026-09-05 | verified |
| The renders in rung 4 | the three folders named in rung 4, read 2026-09-05; the scorer verdicts from `figure_r8_030000/results.json` | verified |
| The adapter | rank 8, alpha 8, step 30000, `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_030000.pt` | verified |
| The cache | `/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/`, states on the uncorrected product-of-experts trajectory, 50 DDIM steps, guidance 7.5 | verified, read off `scripts/build_training_cache.py` |
| The 0.54 and 0.40 as pool-mean alignment | `poe_repair/experiments/cross_pair_lora_pooling/_inline_sampling.py`, `direction_metrics` docstring and body, read 2026-09-05 | verified |
| The runs | runs 1 to 4 in the `Runs` table of [the idea map](../../artifacts/ideas/improving-the-pooled-lora-run/IDEA_MAP.md); the adapter itself from the task "Train one pooled adapter on the eleven training pairs" in [does one pooled fix transfer at all](../../plans/04-does-the-fix-reach-unseen-pairs/plans/hypothesis/03-does-one-pooled-fix-transfer-at-all.md) | verified |
| Regenerate with | [decide where a run goes](../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#1-decide-where-a-run-goes), then the two scripts above with the `run.sbatch` files kept beside each output | |

## Depends on

- What the cached correction is and how it is defined: the interaction term entry in
  [the context index](../../context/00-INDEX.md).
- Which trajectory the cache stores: `scripts/build_training_cache.py`, the latent is advanced
  with the product-of-experts prediction and the joint prediction is queried at those states.
- Which nodes ran it: [the cluster nodes](../../environment/hpc/nodes.md).
- Why the question was asked, and the literature it was tested against:
  [the pressure-test route](../../artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md).

## Still open

- [ ] The same tier read at rank 32 and at a late checkpoint, to say whether per-pair learning is
      a property of rank 8 at 30k or of the architecture.
- [ ] The next training pairs, chosen as new pairings of the 22 seen words from the high band of
      [does interaction strength predict which pairs blend](does-interaction-strength-predict-which-pairs-blend.md),
      then this figure again with a third blue group.
