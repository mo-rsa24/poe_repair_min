# Does interaction strength, read from the cache, predict which pairs blend?   ⚪ null · verified 2026-09-05

**The claim**

Mitra et al. 2026 bound the error of adding two experts' scores by 2√(GM). G is how large the
interaction between the two concepts is at a given state, and M is how curved it is.

This project's cached correction is that error in noise-prediction units. So every cached state
gives a lower bound on G·M with no model call.

The bound does not predict which pairs blend under plain product-of-experts. Every pair the
scorer marks as failing on 8 of 8 seeds fails whether its bound is 100 or 3000.

What the bound does separate is how far apart the two experts are. Pairs of near-synonyms sit
ten times lower than pairs of distinct animals from step 10 onward. The reference test pair,
cat x dog, sits in the high group, while most of the training pool sits in the low one.

**What would have counted**

The prediction written before the run, in the session that proposed the figure: pairs with a
large bound are the pairs score addition loses, so the bound should rise with the plain-PoE fail
rate across pairs. Support if the failing pairs sit high and the composing control sits low;
null if the bound and the fail rate are unrelated. The two-band observation was made after the
run and is reported as a post-hoc read, not as the tested claim.

## 1. The bound does not track the fail rate

![The lower bound on G·M along the run, one curve per pair, and its mean over steps 10 to 40 per pair sorted and coloured by fail rate](../../artifacts/results/does-interaction-strength-predict-which-pairs-blend/interaction-bound-per-pair-from-the-cache.png)
*Left: ‖r_t/σ_t‖²/4 at each cached state, median over seeds, log y, against denoising step, one
curve per pair, steps 46 to 49 cut because 1/σ_t² dominates there; cat x dog, elephant x penguin
and butterfly x flower meadow drawn thick and named. Right: the same quantity averaged over steps
10 to 40, one bar per pair, sorted; red bars fail 8 of 8 seeds under plain product-of-experts,
orange fail 5 or 6 of 8, grey have no measured fail rate. What to notice: red bars fill the whole
x range, and the two bands on the left are a factor of ten apart from step 10.*
📊 Drawn in [Figure 1 of the figure explainer](../../artifacts/results/does-interaction-strength-predict-which-pairs-blend/figure-explainer.md#figure-1-the-lower-bound-on-the-interaction-per-pair-and-per-step).

**What you are looking at.** A quantity a published theorem says must be small wherever adding
scores works, computed from data this project already had. If the theorem's quantity explained
the blend, the red bars would cluster at the right and the composing control at the left.

**The number.** Mean bound over steps 10 to 40, in squared score-norm units: dolphin x porpoise
102, crow x raven 117, butterfly x flower meadow 118, seal x walrus 189, eagle x hawk 206, wolf x
husky 213, rabbit x hare 234, frog x toad 257, then a gap to cheetah x cougar 1458, cat x dog
1837, lion x tiger 2007, elephant x penguin 2237, tiger x dog 3206. Fail rates from the 8-seed
instance-count scorer are 1.00 for every named pair that has one, from
`artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md`. Bound values from
`/datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/interaction_bound.json`,
field `pairs.<slug>.median_bound_by_step`, written by
`scripts/showcase/interaction_bound_from_cache.py` on the session node's CPU, 2026-09-03.

**What this figure leaves out.** Only two pairs have a fail rate below 1, so the x axis of the
prediction has almost no spread to correlate against; the null is against a nearly constant
outcome. The bound is a lower bound on a product of two unknowns, and neither G nor M is
recovered. The late spike is the σ_t scaling of the theorem's units, not a late interaction.

## 2. What the bound does separate

No second figure; the reading is off the same panel.

The low band from step 10 holds wolf x husky, rabbit x hare, seal x walrus, eagle x hawk, frog x
toad, crow x raven, dolphin x porpoise and the composing control butterfly x flower meadow.

The high band holds cat x dog, lion x tiger, cheetah x cougar, cat x lion, tiger x dog, lion x
dog, elephant x penguin, bear x salmon and dog x duck.

The low band is pairs whose two single-prompt predictions are nearly the same vector. The joint
prediction is then close to either, and the correction is small. The product of two
near-identical experts is a faithful picture of a wolf-husky, and the scorer counts it as a blend
all the same.

The high band is pairs whose experts disagree. There the joint is far from the product and the
correction has something to do.

**The number.** Ratio of band medians over steps 10 to 40: about 10 (low band roughly 100 to 400,
high band roughly 1300 to 3200), same file and field as rung 1.

## What this cannot tell you

**Whether the low-band blends are failures at all.** The fail-rate scorer counts instances and
cannot distinguish one honest wolf-husky from one cat-dog chimera. This figure cannot either. It
says only that the interaction term sees them differently.

**The fail rates come from a different run than the cache.** The pairing of x and colour is
across two experiments.

**Five pairs have no fail rate at all.** The held-out pairs from the wider cache (bear x salmon,
dog x duck, cat x lion, tiger x dog, lion x dog) were never scored.

## Where this came from

| What | Source | Mark |
|---|---|---|
| The bound per pair and step | `/datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/interaction_bound.json`, read 2026-09-05 | verified |
| The fail rates | `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md`, read 2026-09-05 | verified |
| The cache | `/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/`, `residuals/step_NNN.pt` per cell, fields `eps_a_raw`, `eps_b_raw`, `eps_j_raw`, `eps_uncond`, `timestep` | verified |
| σ_t | `sqrt(1 - alphas_cumprod[timestep])` from the SDXL DDIM scheduler, read in the script | verified |
| The theorem | Mitra et al. 2026, arXiv 2605.22596, Theorem 1 and Assumption 1, read 2026-09-05 from the HTML version | verified |
| The run | run 3 in the `Runs` table of [the idea map](../../artifacts/ideas/improving-the-pooled-lora-run/IDEA_MAP.md); the pairs from the task "Write `pair_pool.yaml` and confirm it loads" in [the clean pair pool](../../plans/04-does-the-fix-reach-unseen-pairs/plans/tools/01-the-clean-pair-pool.md); the cache itself by `scripts/build_training_cache.py` | verified |
| Regenerate with | the script above, CPU only, about four minutes; `run.log` beside the output | |

## Depends on

- What the correction is: the interaction term entry in [the context index](../../context/00-INDEX.md).
- What the compose rate and fail rate measure: [compose rate](../../context/world/compose-rate.md).
- Why the theorem applies here, with their symbols mapped to this project's:
  [the pressure-test route](../../artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md)
  and the paper-scout selection under the learning tree, named in that route.

## Still open

- [ ] Fail rates for the five wider-cache pairs, so the colour column is complete.
- [ ] Whether the low-band blends are honest products: a read of the wolf x husky renders against
      the cat x dog ones with a scorer that separates a plausible hybrid from a chimera, which the
      instance counter cannot.
- [ ] The next training pairs chosen from the high band, which is where the correction is large;
      the pair-axis result in
      [is the held-out gap a fit, a drift, or a pair problem](is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md)
      says they can be new pairings of seen words.
