# Experiments: does commitment timing explain PoE failure and adapter failure?

Written before any run. The falsification rule for each experiment is fixed here and is not
revised after seeing results. Schema is the experiment-planner provisional block shape; reconcile
against a canonical format if one is adopted.

## The axes

Every experiment below is one choice on each of these. Nothing is planned per-cell.

| Axis | Values | Notes |
|---|---|---|
| pair | 17 in the current pool, plus a new spread set built in EXP-02 | the current pool is selected on the outcome, see the warning below |
| seed | 8 cached per pair | the sampling unit for anything about means |
| step | 0 to 49 | the schedule is 50 steps everywhere in the cache |
| arm | PoE, Mono (joint prompt), PoE plus oracle r_t over a window, PoE plus the trained adapter | one arm per comparison, never two changes at once |
| space | DINOv2, CLIP | a nuisance axis, not a science axis: DINOv2 is pre-committed, CLIP is reported as a robustness check |
| adapter strength | number of training pairs, training steps, rank | only used in EXP-05 to manufacture failures |

## The selection warning that governs the whole file

`outputs/animals_compose_transfer/fail_rate.json` records `train_min_fail: 0.5`. Pairs entered the
pool by failing. 15 of 17 sit at fail-rate 1.00, donkey × pony at 0.75 and crocodile × alligator at
0.62. The dissimilar control, elephant × penguin, also fails 8 of 8.

Two consequences, both binding:

1. No analysis inside this pool can say what predicts PoE failure, because the pool has no
   successes to contrast against. Any such analysis is conditioned on the outcome.
2. The prediction "similar pairs fail, dissimilar pairs succeed" is already contradicted by
   elephant × penguin at 8 of 8. It enters the plan as a pre-registered null, not as the
   hypothesis.

## EXP-01: does the commitment step vary from pair to pair?

- claim_id: local: the step at which a run's outcome stops being changeable is a property of the
  pair, not a constant of the sampler.
- independent_var: pair.
- dependent_var: commitment step, defined once here and reused everywhere. For each (pair, seed),
  form the model's running estimate of the finished image at each step in latent space, by
  Tweedie's formula from the cached state and the PoE prediction:
  `x0(t) = (x_t - sqrt(1 - abar_t) * eps_PoE(t)) / sqrt(abar_t)`. Record how far that estimate has
  settled, as the cosine between `x0(t)` and the run's final `x0`. The commitment step is the
  first step after which that cosine stays at or above 0.90 for the rest of the run. Sensitivity
  at 0.80 and 0.95 is reported beside it so the verdict cannot rest on one threshold.

  This is measured in latent space, not in DINOv2, and the reason is coverage rather than
  preference: the DINOv2 reading exists for 3 pairs and 9 cells, where this one covers all 17
  pairs at 8 seeds from the cache alone with no decoding and no GPU. The cost is that latent
  distance is not perceptual distance, so the measure is validated against the DINOv2 reading on
  the 3 pairs where both exist, and the agreement is reported with the result. If they disagree,
  the DINOv2 reading wins and this experiment is re-run with decoding.

  What the proxy assumes: that once the model's estimate of the finished image has settled, the
  alternative outcome is no longer reachable. That is the speciation claim itself, so the proxy is
  descriptive. Only the handover sweep can make it causal.
- ablation_rows: none. This is a measurement, not a comparison.
- metric: the commitment step per (pair, seed), then the median per pair. Report the between-pair
  standard deviation of those medians against the pooled within-pair standard deviation across
  seeds. Both numbers, always, because the second is what makes the first mean anything.
- sample_size: 17 pairs x 8 seeds = 136 cells. Cache only.
- falsify_condition: **varies** if the range of per-pair medians is at least 5 steps AND the
  between-pair standard deviation is at least 1.5x the within-pair. **Does not vary** if the range
  is under 2 steps OR the ratio is under 1.0. **Inconclusive** between those, which means add seeds
  rather than interpret. The 5-step anchor is half the composing window (steps 0 to 10): a spread
  smaller than that cannot plausibly make one fixed schedule miss a pair.
- what would surprise us: every pair committing within 2 steps of the same point. The correction's
  size profile varies a lot by pair, so a constant commitment step would mean timing is set by the
  sampler and not by the content.
- figures: commitment step per pair, one point per pair at the median with its 8 seeds behind it,
  ordered by median. Qualitative half: the running estimate at the commitment step for the earliest
  and latest pair, beside their two endpoints.
- compute: in-session, no GPU, about 8 minutes over 162 cells.
- status: ✅ done, with the perceptual check owed. `python scripts/commitment_step.py`, written up
  in `docs/evidence/EXP01-commitment-step/QUERY.md`. **Varies**: per-pair medians span 18 steps
  (dolphin × porpoise 18, cat × dog 36), between-over-within 1.90 against a bar of 1.5, and the
  verdict holds at both sensitivity thresholds. The unregistered reading matters more: every pair
  settles at step 18 or later while the correction only works over steps 0 to 10, so the
  correction stops working 8 to 26 steps before the picture settles. Either the decision happens
  well before settling, or this measure tracks the wrong event.
- gates: EXP-04 then answered the question this one could not, and answered it against the measure:
  the window sits at steps 0 to 10 for every pair regardless of when that pair settles. So failure
  mode (c) of EXP-05 is removed, and the settling step is not the event that decides composition.
- qualitative half built (`/pair-figure`, `scripts/commitment_step_frames.py`): two rows, cat × dog
  seed 2 (individual step 36) and dolphin × porpoise seed 4 (individual step 20, pair median 18),
  three decoded Tweedie estimates per row (step 0, commitment step, step 49). In both rows the
  commitment-step frame already reads as the finished animal, close enough to the final frame that
  the remaining steps look like cleanup rather than a decision. Small decode speckle sits on the
  commitment frame near the dog's face and the dolphin's eye/snout, consistent with the estimate not
  being fully denoised yet. `docs/evidence/EXP01-commitment-step/commitment-step-frames.png` (+ .pdf,
  sidecar `.json` with per-row provenance). Perceptual validation of the 0.90 threshold is still owed
  and is tagged on the figure itself, not just in prose.

## EXP-02: build a pair set that actually spans success and failure

- claim_id: local: prerequisite. Without outcome variance there is nothing for a predictor to
  predict.
- independent_var: pair, chosen to span a similarity range rather than to fail.
- dependent_var: plain PoE compose rate per pair, from the validated instance-count scorer.
- ablation_rows: one arm only (plain PoE). No correction, no adapter.
- metric: compose rate over 8 seeds per pair. The pair set is admissible only if it spans the
  range: at least 8 pairs above 0.5 and at least 8 below 0.5.
- sample_size: 40 candidate pairs x 8 seeds = 320 samples, 50 steps each. Candidates chosen to
  span text-embedding distance between the two nouns, deliberately including pairs expected to
  succeed (different families, different scales, different habitats), because the current pool has
  none.
- falsify_condition: this experiment cannot fail, it can only come out unusable. **Usable** if at
  least 8 pairs land above 0.5 compose. **Unusable** if fewer than 4 do, which would mean PoE
  fails on essentially everything and Claim 1 is not a question about pairs at all, at which point
  EXP-03 is cancelled rather than run on a degenerate axis.
- figures: compose rate against pair, sorted, with the two-animal and blended samples shown for
  the pairs at each end.
- compute: GPU, plain sampling with no training. Estimate a few hours on one card, plus scorer
  time. Write samples to /datasets.
- status: ⚠️ pending

## EXP-03: does endpoint separation at the commitment step predict PoE failure?

- claim_id: local: PoE fails when the blended outcome and the two-animal outcome are still close
  together at the moment the run commits.
- independent_var: the predictor, computed per pair with no access to the outcome: the distance
  between that pair's PoE endpoint and its Mono endpoint in DINOv2, divided by the noise scale at
  that pair's commitment step from EXP-01.
- dependent_var: plain PoE compose rate from EXP-02.
- ablation_rows: three predictors, each its own row, so the comparison is one axis at a time.
  (a) endpoint separation alone, no noise scaling. (b) separation divided by noise at the
  commitment step, the live hypothesis. (c) text-embedding distance between the two nouns, the
  semantic-similarity version, entered as a pre-registered null because elephant × penguin already
  contradicts it.
- metric: Spearman correlation between predictor and compose rate across pairs, with a 95%
  bootstrap interval over pairs. Spearman rather than Pearson because compose rate is a bounded
  proportion.
- sample_size: however many pairs EXP-02 yields, target 40. Correlations on fewer than 20 pairs are
  not reported as evidence.
- falsify_condition: **support** if row (b) reaches |rho| of at least 0.5 with the interval
  excluding zero AND the sign is negative, meaning closer endpoints predict more failure. **null**
  if |rho| is at most 0.2 with the interval containing zero. **inconclusive** between, which means
  extend the pair set, not reinterpret. Row (b) must also beat row (a) by at least 0.15 in |rho|,
  otherwise the noise scaling is decoration and the honest claim is the simpler one.
- what would surprise us: row (c) winning. Semantic similarity predicting failure would contradict
  the elephant × penguin measurement and would need that measurement re-examined first.
- figures: predictor on x, compose rate on y, one point per pair, with the pairs at both extremes
  shown as their actual PoE and Mono images.
- compute: cache only once EXP-02 has produced the samples.
- status: ⚠️ pending, blocked on EXP-01 and EXP-02.

## EXP-04: does the window where the correction works move with the pair?

- claim_id: local: the effective correction window sits at the pair's own commitment step, so it
  moves when the commitment step moves.
- independent_var: window position, the 9 sliding windows already used, crossed with pair.
- dependent_var: compose rate under oracle r_t injected over that window.
- ablation_rows: one row per pair. The existing window sweep covers cat × dog only, so every other
  pair is new.
- metric: the window centre with the highest compose rate, per pair. Correlate that against the
  commitment step from EXP-01.
- sample_size: 6 pairs x 9 windows x 4 seeds = 216 cells. Six pairs chosen to span the commitment
  range EXP-01 reports, which is why EXP-01 must come first.
- falsify_condition: **support** if the best window centre spans at least 5 steps across pairs AND
  correlates with the commitment step at Spearman rho of at least 0.5. **null** if every pair's
  best window centre lands within 2 steps of the others, which would mean one fixed schedule fits
  all pairs and mode (c) of EXP-05 is dead. **inconclusive** between.
- figures: compose rate against window centre, one curve per pair, with each pair's commitment step
  marked on its own curve.
- compute: none needed. The sweep already existed at 8 pairs x 9 windows x 4 seeds, 288 scored
  cells in `interaction_term/window/window_curves.json`, so this was a read rather than a run.
- status: ✅ done. `python scripts/window_vs_commitment.py`, written up in
  `docs/evidence/EXP04-window-vs-commitment/QUERY.md`. **Does not move.** All 8 pairs peak at
  window centre 5 (steps 0 to 10), a span of 0 steps against a bar of 5, while their settling steps
  span 13 (23 to 36). The registered correlation is undefined because the best window never varies,
  which is the finding. Secondary summaries are weak: rho +0.16 for the compose-weighted centre and
  +0.26 for the latest window that still works. Left-censored: centre 5 is the earliest the grid
  holds, so the window is bounded rather than located.

## EXP-05: when the adapter fails, which of three things went wrong?

- claim_id: local: adapter failure on a pair is one of aims-wrong or delivers-too-little, and the
  two are distinguishable from measurements already implemented. **The third mode, the adapter's
  window sitting in the wrong place for a pair, is removed: EXP-04 measured every pair's window at
  the same place, so there is no per-pair window to miss.**
- independent_var: pair, within a set of runs weak enough to produce failures.
- dependent_var: three per-cell measures. Direction cosine and fraction-of-distance-reached, both
  already in `_inline_sampling.py::direction_metrics`. Plus the gap between that pair's commitment
  step and the step where the adapter's correction is largest.
- ablation_rows: the leave-one-pair-out runs already planned in the transfer scope, which produce
  degradation by construction. If those have not run, deliberately weakened adapters (fewer
  training pairs, fewer steps) serve the same purpose, and this is marked mixed because two things
  change at once between a weakened adapter and the full one.
- metric: for each failing cell, which of the three measures is out of range. Report the count in
  each mode and the count that fits none, because a classification that explains nothing must be
  visible as such.
- sample_size: gated. **Print the number of failing cells before classifying anything.** Fewer than
  5 failing cells means stop and report "no failures available to classify" rather than splitting
  noise. Today that count is zero: the worst transfer pair is 0.9375 and none sit at the floor.
- falsify_condition: **support** if at least 70% of failing cells fall into exactly one of the two
  remaining modes. **null** if fewer than 40% do, meaning aiming and delivery are not the right
  decomposition. **inconclusive** between, which means add failing cells from more leave-one-out
  runs.
- what would surprise us: a large group fitting neither mode. Both remaining modes are about the
  correction the adapter emits, so failures fitting neither would mean the adapter emits a fine
  correction and the run fails anyway, which nothing in the current picture explains.
- figures: one row per failing pair showing what the adapter produced against the oracle correction
  at matched steps, beside the two numbers.
- compute: GPU-light, re-scoring existing checkpoints with the metrics already wired.
- status: ⚠️ pending, blocked on failures existing at all.

## Order, and why

1. EXP-01, because it is free, it defines the quantity both claims share, and it can delete a
   failure mode before anyone builds a classifier around it.
2. EXP-02, because Claim 1 is untestable until an outcome axis exists.
3. EXP-04 and EXP-03 in either order once their inputs land.
4. EXP-05 last, and only when failing cells exist.

## What must be printed on every run

The count each selection actually made. The pair and seed beside every number. The measurement
space, since DINOv2 is the pre-committed one and CLIP is the check. Anything read in fp16 upcast to
fp32 before accumulation.
