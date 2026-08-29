# 💡 Which variable explains what PoE is missing

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. some pairs blend under PoE and some compose, and which is which is measured | needs a check | whether the scene pairs blend or compose under PoE, scored on `poe.png` already sitting in the training cache |
| 2. the four directions are four candidate causes, so isolating one is well-posed | ✅ **settled: wrong as stated** | the repair stands and now has a run behind it. The cause is `r_t` as a function of state and step; the trajectory is one of its arguments, not a consequence. The sampler-against-model split it produced is designed and ordered in [its own scope](../../../plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md), steps 24 to 30 |
| **7 (current)** | **needs a check** | **two named checks below: a matched random push applied early only, and the parked prompt-window experiment** |
| 3. linear score addition has a signature over the run that separates a blending pair from a composing pair | open | |
| 4. a 2D projection of the run shows what the 1D manifold slide throws away | open | |
| 5. reading each step back as text separates a blending run from a composing run | needs a check | whether the averaged caption bank survives being applied to mid-run x̂_0, not just to finished renders |
| 6. the joint prompt carries scene context the two solo prompts do not | open | |

Load-bearing: claim 3, and claim 7 is a candidate to replace it. If linear score addition carries no
step-wise signature that splits the two groups, claims 4 and 5 become instruments with nothing to
instrument. Claim 7 asks a different question (when, not which pair) and the window sweep has
already answered most of it.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [What the ground check found](#what-the-ground-check-found)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

Product-of-experts composition blends some concept pairs into one hybrid animal and composes
others into two separate things. Four different variables have been offered as the explanation.
This walk finds which one, isolated and varied over the fifty denoising steps, actually accounts
for the split, and kills the ones that do not.

**Where the walk is**

Claim 1, round 1. Nothing has been settled by a run yet.

**Where this returns**

Section 4 of the paper, "Restoring the Plurality Term Restores Composition", walked in
[the draft map](../../../paper/iclr/DRAFT_MAP.md) and listed there in its routes table. That
section runs a figure test: a measurement earns a main-text figure only if it advances
understanding of the dynamics of failure and correction. Beat 8 of its skeleton is held to a
prose mention until claim 3 has a verdict, so the compile of this walk is what decides whether
any mechanism figure enters the main text.

**What compile would produce today**

Destination A, thin. The cost order is already settled by the ground check and reverses what the
idea assumed, but no claim has been tested, so a compile now would name the cheap first check and
nothing more.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

SDXL conditions on two text encoders. Product-of-experts composition evaluates the network under
each prompt separately, adds the two guided predictions, subtracts the unconditional one, and
reverse-diffuses the sum. For some pairs that produces one fused animal instead of two. Four
variables have been named as the reason: the linearity of the score addition itself, the geometry
of the trajectory the composed sampler follows, what the run reads back as in language, and scene
context that the joint prompt carries and the two solo prompts do not.

Three of those four are ways of *reading* the composed trajectory. One of them, the score
addition, is the only thing that is actually different between the arms. So the honest form of the
question is not "which of four causes" but "does the one candidate cause carry a step-wise
signature, and which of the three readouts can see it".

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. 🔍 Some pairs blend under PoE and some compose, and which is which is measured  ← current

Mark: needs a check. Every other claim compares a blending group against a composing group, so the
groups have to exist before anything is measured across them. The idea's own words call the choice
of pairs "vague and ambiguous", which is the admission that this is open.

The check is cheap and named in [Checks outstanding](#checks-outstanding).

### 2. ❌ The four directions are four candidate causes, so isolating one is well-posed

Mark: wrong as stated. Score addition is the only thing that differs between the PoE arm and the
joint arm. The trajectory geometry, the language readback, and the decoded pictures are all
consequences of that difference, measured in different spaces. Varying them independently is not
possible, because there is nothing to vary: they are readouts, not knobs.

Repaired wording, second version and the one that stands: **the cause is `r_t` as a function of
state and step, so the trajectory is one of the cause's two arguments rather than a consequence of
it.** The rule's form is constant across pairs and therefore cannot be what separates them; only
its error evaluated along two different paths can. Claim 6 remains a genuine second cause, because
the prompt is the other free choice.

The first repair offered ("one cause, three instruments") was overturned in round 3 by the window
sweep: the sliding-window experiment already intervenes on *when*, which is a property of the
trajectory, so trajectory cannot be purely downstream.

**The split the literature adds.** Du et al. separate two errors this project has been measuring as
one. Error A, the sampler: the sum of diffused scores is not the score of the diffused product,
because noising and multiplying do not commute. It vanishes at t=0 and is worst at high noise.
Error B, the model: `p(cat)·p(dog)` is not `p(cat and dog)`, and it does not vanish at t=0.
`r_t = ε_J − ε_PoE` is A and B added together, and nothing in this repo separates them.

**The dose axis generalises to any composition rule.** For any rule producing a per-step prediction
`ε_M`, define `r_t^M = ε_J − ε_M` and inject `ε_M + λ·r_t^M`. At λ=0 that is the method alone, at
λ=1 it is `ε_J` exactly, so every rule travels the same journey and a λ column is a matched
comparison again. `‖r_t^M‖` per step then measures how much of the missing term the rule already
supplies.

### 3. 🔍 Linear score addition has a signature over the run that separates a blending pair from a composing pair

Mark: open. **Load-bearing.**

### 4. 🔍 A 2D projection of the run shows what the 1D manifold slide throws away

Mark: open. Three sub-decisions have to land before this is walkable: which points get projected,
what the two axes are, and which embedding space.

### 5. 🔍 Reading each step back as text separates a blending run from a composing run

Mark: needs a check. The finished-render version of this instrument already works
(`caption_readback.json`). What is untested is whether it survives being pointed at mid-run x̂_0.

### 6. 🔍 The joint prompt carries scene context the two solo prompts do not

Mark: open. The text-intervention experiment that would test this is designed and parked, never run.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#what-the-ground-check-found) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| getting text back from an image | CLIP inversion, four families in rising cost | retrieval over a caption bank, decoder captioning, gradient hard-prompt inversion, textual inversion | named from the user's recall, not yet read in this repo |
| an image embedding does not sit where its caption's text embedding sits | the modality gap | the two encoders' outputs occupy separate cones, so image-to-text-embedding comparison measures the gap plus the content | named from the user's recall, not yet read |
| the model's guess at the finished picture at step k | x̂_0, the Tweedie mean | `(x_t − √(1−ᾱ_t)·ε) / √ᾱ_t`, implemented as `tweedie_mean` in `poe_repair/_sdxl/metrics.py` | confident, this repo's own code |
| the leftover the joint prompt has and PoE does not | `r_t`, the interaction term, narrowed here to plurality | `ε̃_J − ε̃_PoE` per step | confident, this project's own term |

## What the ground check found

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

Five findings, all read from disk before any mark was assigned. Each one changes a claim.

**The cost order is the reverse of what the idea assumed.**

The 2,200 `latent_trajectory.pt` files under `outputs/interaction_term/` hold latents only, and
they cover eight animal-animal pairs with zero scene pairs. The cache that matters is
`/datasets/mmolefe/poe_repair_min/outputs/training_cache`, 790 cells over 77 pairs, each holding
`eps_a_raw`, `eps_b_raw`, `eps_j_raw`, `eps_uncond`, `x_t` and `timesteps` for all 50 steps, plus
`mono.png` and `poe.png`. It already contains every pair the grid wants:
`a_dolphin__x__an_ocean_wave` (13 seeds), `a_picnic_table__x__a_snowstorm` (12),
`a_park_bench__x__a_sand_dune` (12), `a_camel__x__a_desert_landscape` (13),
`a_deer__x__a_forest_clearing` (12), alongside `a_butterfly__x__a_flower_meadow` (12),
`a_cat__x__a_dog` (17) and `an_eagle__x__a_hawk` (8). **The grid costs no new sampling.**

**The readback's winning-caption lead is not discriminative.**

In `readback.json` the joint arm's winning caption is "a cat" at step 10 (0.2863), step 25 (0.2877)
and step 40 (0.2777). The joint arm never picks "a cat and a dog" either. So "the PoE arm reads as
one solo and never as the pair" describes both arms equally and separates nothing.

**The distance collapse inverts once the Tweedie prefactor is divided out.**

All four arms are built from the same cached `x_t`, so their x̂_0 values differ only by
`√(1−ᾱ_t)/√ᾱ_t` times the difference in ε. That prefactor is 4.69 at step 10, 1.52 at step 25 and
0.53 at step 40, so it alone falls to 32% then 11% of its step-10 value. The observed joint-against-PoE
picture distance falls to 28% then 15%. Divided through, the ε-space difference goes 1.00 → 0.85 →
1.35: flat, then rising. Solo-A-against-solo-B goes 1.00 → 1.71 → 1.99. Neither collapses. The VAE
decode is nonlinear so the division is approximate, which is the second reason not to use this
quantity: `‖r_t‖` per step is measured directly from the same cache and needs no correction.

**The averaged caption bank does rescue the discrimination, and it is already proven.**

`caption_readback.json` uses four caption kinds with three wordings each for `two` and `blend`, all
prefixed "a photo of", and it includes the two solo captions in the contest. On 32 oracle cells the
winner goes from `blend` 19 / `b_only` 8 / `a_only` 4 / `two` 1 at λ=0 to `two` 18 / `blend` 8 /
`a_only` 5 / `b_only` 1 at λ=0.75. The solos are present and they lose. `xhat0_readback.py`'s
single-wording, no-prefix bank is the weaker instrument, and swapping the proven bank in is an edit
to one function.

**The off-axis mass the 2D scatter would show is already stored.**

`manifold_slide_clip.json` records `mean_off_axis` beside `mean_projection` per dose per arm. The
real correction runs on-axis +0.031 / +0.210 / +0.376 / +0.654 / +0.992 against off-axis 0.984 /
0.909 / 0.836 / 0.660 / 0.085 at λ = 0, 0.25, 0.5, 0.75, 1. The wrong-pair control stays off-axis at
0.93 throughout while its on-axis travel reaches +0.225. So the off-axis magnitude is a number the
1D read already has, and what a scatter adds is direction, not magnitude.

## Held claims

Navigation: ⬅️ [What the ground check found](#what-the-ground-check-found) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|
| 2 | vary the four directions independently and see which moves the outcome | three of them are readouts of the score difference rather than knobs on it, so there is nothing to vary |
| 5 | read claim 3 off the joint-against-PoE picture distance collapsing over the run | the collapse is the Tweedie prefactor, and the ε-space quantity underneath it rises rather than falls |

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 1 | run the validated instance-count scorer over `poe.png` for the five scene pairs and the two animal pairs in `training_cache`, 8 to 13 seeds each. No sampling, no queue | a clean split (scene pairs high, animal pairs near zero) gives claims 3 to 5 two groups to compare. A muddled split means the grouping is the finding and every later comparison is mixed |
| 2 | read the per-step correction size as t approaches 0 off curves already on disk: `paper/iclr/figures/correction-size-over-the-denoising-run.json` (50 steps, 3 seeds, two pairs) and `cache_analyses/step_collapse.json` with `snr_collapse.json` across pairs, the latter carrying a `peak_at_edge` field that speaks to high-noise concentration. Error A vanishes at t=0 and error B does not, so a size bounded away from zero at the end of the run bounds B from below. No sampling, no queue. Two things the read states out loud or it is not worth having: the cached measure is the ratio `‖r_t‖/‖ε_PoE‖` where this question needs the unnormalised quantity, and late-step values are confounded unless both arms are evaluated at the same state, because by then the trajectories have diverged and the difference carries accumulated path as well as the rule | mostly A means the correction is largely a sampler artifact, the sampler comparison stops being optional, and section 7 carries it as a limitation. Mostly B means the paper's framing strengthens. This bar holds before the numbers are seen |
| 5 | rerun `xhat0_readback.py` with `caption_readback.json`'s four-kind averaged bank on all 50 steps, one blending cell and one composing cell | if the mid-run pick tracks the finished-render pick, the readback is an instrument. If it stays flat mid-run while the finished render separates, x̂_0 is too blurry to read and the route needs real inversion |
| 7 | a random push of matched size applied to **steps 0 to 10 only**. The existing random control runs at every step, so nothing yet separates "the early window works because the direction is right" from "the early window works because early perturbations have more leverage on the final image" | if the random early push also composes, the early window is sampler leverage and has nothing to do with composition. If it does not, the effect is direction-specific and the commitment reading survives |
| 7 | the parked prompt-window experiment: run PoE for steps 0 to 10, then switch to the **joint prompt** for steps 11 to 50. Designed and deliberately not run, named as future work in F4's caption | if the joint prompt cannot rescue it either, the outcome is fixed early no matter what you do, which is the ordinary fact that low-frequency structure is set early and needs no basin story. If the joint prompt does rescue it, something specific to PoE is committing and "commitment" earns its name |
| 2 | the free bound on the model's share: `‖r_t‖` at the last steps of the run, read from cached cells. At `t→0` the sampler error is defined to vanish, so what is left is the model's share alone. Costs nothing, step 24 of the running order | a small number means the model agrees with the product on the product's own support and most of `r_t` is sampler error. A large one means the reverse, and the corrector work is capped before it starts |
| 2 and 7 | `‖r_t^(k)‖` per step against corrector count `k ∈ {1, 5, 20, 100}`, where `r_t^(k) = ε_J(x_t^(k)) − ε_PoE(x_t^(k))` and both are evaluated at the point reached after `k` Langevin corrector steps. No images, no detector | a curve that falls then flattens above zero splits the two errors: the drop is the sampler's share (error A), the leftover is the model's share (error B). A curve flat from the start says the corrector changes nothing and the failure is all error B. **The rate or norm at any single `k` is not a result; only the trend is** |

## Routes

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

A route runs in another session. The row is written when the prompt is emitted and deleted only
once the result is folded back in.

| Routed to | Which claim it serves | Where the result lands | State |
|---|---|---|---|
| `/reconstruct-prompt --skill submerge` on MCMC, the non-commutation fact, predictor-corrector, Langevin, Hamiltonian correctors, annealed marginals and Feynman-Kac | claim 2, so the sampler-side story can be judged rather than quoted | a submerge journey under the learning root, registered in `JOURNEYS.md` | out |
| `/frame-hypothesis` on claim 7: sampler-side correction with no oracle, three figures and a gate run | claim 7, and claim 2's A-against-B split | [is-the-gap-the-samplers-or-the-models](../../../plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md), a scope of seven plans at steps 24 to 30, each carrying its own pre-registered review file; step 21 of the paper waits on its gate at step 26. Its `source/` holds the design and the pre-registered questions. **Its `plans/` and `review/` folders are empty**: the seven plans are named in the master plan and not yet written | **landed, half done.** `/populate-plans` has to write the seven plan files before anything can execute. Delete this row once the measured size is folded back in |

## Runs

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|

## Sources

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| `hypothesis-05-the-same-story-from-three-sides.md` | the two language-side nulls bound every text framing here, and its open row is the same question claim 5 asks | confident, this repo's own review file |
| `caption_readback.json` | the averaged caption bank that separates blend from two-animal on finished renders | confident, read from disk |
| `manifold_slide_clip.json` | the 1D read and the off-axis magnitude claim 4 wants to see | confident, read from disk |
| Du et al., *Reduce, Reuse, Recycle*, [arXiv 2302.11552](https://arxiv.org/abs/2302.11552), ICML 2023 PMLR 202:8489-8510 | its abstract concludes "the sampler (not the model) is responsible for this failure" and proposes MCMC-corrected samplers. The score sum equals the product's score exactly at t=0 and not for t>0, because noising and multiplying do not commute | in `plans/standing/literature/reading-register.md` at abstract level since 2026-08-12; abstract re-verified by fetch. The method's noise-level behaviour still needs the full read |
| Skreta et al., *SuperDiff*, [arXiv 2412.17762](https://arxiv.org/abs/2412.17762), ICLR 2025 Spotlight | a composition rule derived from the continuity equation rather than from naive score addition. Working SDXL pipeline at [superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0), source at [necludov/super-diffusion](https://github.com/necludov/super-diffusion) | register row at abstract level, 2026-08-12. The pipeline defaults to 200 inference steps against this repo's 50, which has to be matched |
| Skreta et al., *Feynman-Kac Correctors in Diffusion*, [arXiv 2503.02819](https://arxiv.org/pdf/2503.02819), ICML 2025 Spotlight | a sequential-Monte-Carlo corrector for sampling correctly from product distributions | register row at abstract level, 2026-08-12, and parked in `paper/iclr/DRAFT_MAP.md` under section 2 |
| Soiffer et al., *Catastrophic Compositional Generation*, arXiv 2606.23920 | **binds claim 7.** No inference-time technique alone produces the target distribution once the composed condition is out of distribution. Existing correctors, Feynman-Kac included, reduce the gap but do not close it | register row with the deepest engagement of the four, and already cited in `gate-01-two-literature-checks-before-print` as this paper's strongest reason to train a fix rather than trust a better sampler |
| [Reduce-Reuse-Recycle project page](https://energy-based-model.github.io/reduce-reuse-recycle/) | repo, Colab, HMC implemented, energy parameterisation is what buys Metropolis adjustment | fetched, not yet run |
| *MCMC-Correction of Score-Based Diffusion Models for Model Composition*, [arXiv 2307.14012](https://arxiv.org/abs/2307.14012) | a Metropolis-Hastings acceptance rule built from line integration of the score | title and claim from search, not read, no register row |
| *Product of Experts for Visual Generation*, [arXiv 2506.08894](https://arxiv.org/html/2506.08894v2) | a 2025 treatment of the same composition problem | title only, not read, no register row |
| Liang et al., "Mind the Gap", ~arXiv 2203.02053 | the modality gap that blocks naive image-to-text-embedding comparison | named from recall, not read |
| Gal et al., textual inversion, ~arXiv 2208.01618 | the only inversion family returning something SDXL could be re-conditioned on | named from recall, not read |
| PEZ, "Hard Prompts Made Easy", ~arXiv 2302.03668 | gradient hard-prompt inversion | named from recall, not read |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Walk claim 1. `run` executes its check, which is the cheapest thing on the board and gates
claims 3, 4 and 5.

Claim 2's A-against-B read is the other free one and does not wait behind claim 1. It is the
open threat to the paper's framing rather than to a figure: `r_t` is error A plus error B, the
paper's timing result (correction matters early) is also what a sampler artifact looks like,
since error A is worst at high noise, and nothing measured so far tells the two apart. The
`/frame-hypothesis` route on claim 7 serves the same split from the design side; this read is
what the cached numbers can say before that plan comes back.
