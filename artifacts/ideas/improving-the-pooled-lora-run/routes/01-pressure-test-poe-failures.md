# Pressure test: the three remaining failures of the pooled LoRA, against the literature

Route 01 of [the idea map](../IDEA_MAP.md). Written 2026-09-03 from the session record plus the
trainer source. Confidence on every reference is flagged: **confident** (would cite without
checking), **verify** (matches memory, check before it enters the paper), **vague** (family
known, no reference to hand).

## Part 1: the idea, distilled

One LoRA on the cross-attention projections of the SDXL UNet (attn2 to_q, to_k, to_v; rank r in
{8, 16, 32}; alpha = r) is trained so that the product-of-experts combination of its three single-prompt
outputs matches the cached joint-prompt output:

```
eps_PoE^lora(x_t, t) = w (eps_A^lora + eps_B^lora - eps_0^lora)          three separate forwards on "a cat", "a dog", ""
loss                 = || eps_PoE^lora - eps_J^cached ||^2               eps_J from the joint prompt "a cat and a dog"
inference            = eps_PoE + lambda * r_hat,   r_hat = eps_PoE^lora - eps_PoE
```

The target is the interaction term the product drops, r_t = eps_J - eps_PoE = -sigma_t * grad_x
log [p(c_A, c_B | x_t) / (p(c_A | x_t) p(c_B | x_t))], the gradient of the pointwise mutual
information between the two concepts at state x_t. The design claim under test is that this map
from (x_t, t, one concept) to a correction is learnable from 11 pairs and reaches unseen pairs.

The one fact the session record did not state, read from the trainer at
`poe_repair/experiments/one_pair_one_seed/trainer.py:263-357`: **each LoRA branch sees only its
own prompt.** The A branch gets "a cat" and x_t. It never sees "dog". Whatever it learns about
the interaction has to be read off the latent.

## Part 2: the load-bearing claim

**Load-bearing.** The interaction term is a function of the state that a branch-blind adapter can
approximate: r_t ≈ f_theta(x_t, t, c_A) + f_theta(x_t, t, c_B) - f_theta(x_t, t, ∅). If the term needs
both concepts as input, the ceiling on the fit is set by architecture and no amount of pairs
moves it.

**Supporting.** More pairs close the seen-to-unseen gap (0.54 to 0.40 cosine). Weight decay and a
stopping rule remove the late haze. The λ-window is a property of the problem, not the checkpoint.

**Unstated premises.**
1. That 0.54 and 0.40 measure fit. They do not. Run 1 in the idea map measured cos(r_hat, r_t)
   on the cached states themselves: 0.969 on train cells, 0.800 on held-out, with the late steps
   at 0.961 against 0.748. The 0.54 and 0.40 are read along the corrected trajectory, where the
   states differ from the cache. So the number that is flat from 10k is a divergence, and the
   number that measures learning is neither flat nor low.
2. The training states are on the uncorrected PoE trajectory, while at inference the adapter sees
   states on the corrected trajectory. Every checkpoint is evaluated off its training distribution,
   and premise 1 says this is where the 0.40 comes from.
3. Failure (a), the blend, is a sampling error that a correction can remove, rather than the
   product target itself.

## Part 3: where this sits in the literature

**The three failures by name.**

- (a) blend, one animal with mixed features: named in the attention-binding work as "semantic
  leakage" or "subject mixing" for visually similar subjects, in Bounded Attention, Dahary et al.
  2024, arXiv 2403.16990 (**verify** that the kitten-and-puppy example is theirs). It is also
  Bradley et al.'s failure A, "target wrong": Projective Composition, arXiv 2502.04549, §3.2
  argues the product p(x|cat) p(x|dog) / p(x) is zero wherever either factor is zero, so it cannot
  contain two-object images at all (**confident**, verified in the repo's own reading). Under an
  exact sampler of the product, the chimera is the answer. That makes (a) target-side, not
  sampler-side.
- (b) one clean animal: "catastrophic neglect" in Attend-and-Excite, Chefer et al. 2023, arXiv
  2301.13826 (**confident**); "mode collision" in CO3, arXiv 2509.25940 and the two-basin picture
  in PATHS, arXiv 2605.30991 (**verify**, both from the reading register).
- (c) late haze: not a composition failure. It belongs to the LoRA and optimizer literature, Part 5.

**Sampler-side versus model-side, by paper.**

| Work | Side | What it fixes | Touches (a)? | Touches (b)? |
|---|---|---|---|---|
| Liu et al. 2022, arXiv 2206.01714 (**confident**) | defines the target | nothing; assumes c_A ⊥ c_B given x | no | no |
| Du et al. 2023 Reduce Reuse Recycle, arXiv 2302.11552 (**confident**) | sampler (ULA, HMC, annealed) | error A: noising and multiplying do not commute | no, the target is still the product | partly: noise at each level escapes basins |
| SuperDiff, Skreta et al. 2024, arXiv 2412.17762 (**confident**) | sampler rule (AND by density matching, OR by logsumexp) | keeps both densities in play along the path | no | partly |
| Feynman-Kac Correctors, Skreta et al. 2025, arXiv 2503.02819 (**confident**) | sampler (SMC with reweighting, covers flows and diffusions) | exact product in the many-particle limit | no, and exactness makes the chimera more faithful | yes, resampling leaves basins |
| Attend-and-Excite, Bounded Attention, SynGen (**confident** family) | test-time latent optimization on the joint model | attention to each token | yes, but only with the joint prompt available | yes |
| CoInD, arXiv 2503.01145; TokenCompose, arXiv 2312.03626 (**verify** id) | model training | drives the interaction term to zero or fine-tunes the joint model | yes | yes |
| This LoRA | model, branch-blind | learns the interaction term from single-prompt branches | yes | only where the correction is large enough early |

**Learned corrections that generalize.** Guidance distillation, Meng et al. 2023, arXiv 2210.03142
(**confident**), and guidance-free training, arXiv 2501.15420 (**verify**), absorb an
inference-time combination into weights and generalize across prompts. Both give the network the
whole condition. No work known here learns a two-concept correction from branches that each see
one concept (**vague**: I cannot name a counterexample, and did not search).

**Deterministic trajectories and basins.** The initial noise fixes which subject survives: InitNO,
Guo et al. 2024, arXiv 2404.04650 (**confident**) optimizes x_T against neglect; Golden Noise, arXiv
2411.09502 (**verify**). ODE versus SDE with the same marginals: Nie et al. 2023, "The blessing of
randomness", arXiv 2311.01410 (**confident**) proves the SDE contracts a marginal mismatch and the
ODE carries it to t = 0; Restart Sampling, Xu et al. 2023, arXiv 2306.14878 (**confident**) and
EDM's churn, Karras et al. 2022, arXiv 2206.00364 (**confident**) use the same fact.

**Flow and flux matching.** The whole construction ports. For an interpolant x_t = α_t x_1 + σ_t ε
the velocity is affine in the score, v = (α̇/α) x_t + b_t ε with b_t = σ̇ - α̇ σ / α, so any
combination rule whose coefficients sum to one (PoE with w, CFG, SuperDiff's convex mixture)
leaves the affine part untouched and r_t is the same object in velocity units (**confident**,
derivation is two lines). Stochastic interpolants, Albergo et al. 2023, arXiv 2303.08797, and SiT,
Ma et al. 2024, arXiv 2401.08740 (**confident**) give the ODE-or-SDE choice at fixed marginals
that question 3 needs, so a rectified-flow model is where the sampler axis is cleanest to test.
Guided Flows, Zheng et al. 2023, arXiv 2311.13443 (**confident**) is CFG for flows. FKC and
SuperDiff are both written for flows as well as diffusions. FLUX.1 is a rectified-flow MM-DiT
with joint attention and no attn2 block, so the adapter's target modules do not exist there; the
analogue is a LoRA on the text-stream projections of the joint attention (**confident** on the
architecture, **vague** on whether FLUX blends similar animals at all: no measured result to hand).
Energy Matching and Adjoint Matching (arXiv 2409.08861, **verify**) train samplers toward an
explicit target density, which would let the product be a training target rather than a sampling
rule; that is the model-side sibling of FKC and is untested for this problem.

## Part 4: what is likely right

The correction is real and causal: 0 of 8 to 7 of 8 on a held-out pair, with a dose response, is
past what a sampler artefact produces. The observation that λ above 1 buys nothing is what a
partly-wrong direction predicts: at cosine c the fraction of injected energy along the target is
c², so 16% on held-out cells, and λ = 2 doubles the other 84%. The early-only window helping
only degraded checkpoints fits the record's own finding that the shared part of r_t lives in
steps 0 to 2 and the late part is chaotic.

## Part 5: what is likely wrong

**The 0.40 is not a fit gap, so more pairs cannot move it.** On cached states the adapter fits
0.97 train and 0.80 held-out. The render-time 0.54 and 0.40 are cos(Δ̂, Δ̄) against the pool-mean
correction over every train cell (`_inline_sampling.direction_metrics`), a shared-component share
that a state-specific target cannot push to 1. Run 2 in the idea map then measured the real
off-cache cost: on the adapter's own corrected path, where the state drifts 0.7 to 0.86 from the
cache by the last step and the live target is orthogonal to the cached one (cosine 0.015 late),
the fit is 0.82 against 0.86 on cache. Exposure bias, the compounding error of DAgger, Ross et
al. 2011, arXiv 1011.0686 (**confident**), named for diffusion by Ning et al. 2023, arXiv
2301.11706 (**confident**), is present and costs 0.04. On-policy targets are free here (one
joint-prompt forward at any state) and would recover at most that.

**The real generalization gap is 0.97 against 0.80, and it is largest late.** Early steps sit at
0.985 against 0.925, late steps at 0.961 against 0.748. The late correction is a state-specific
vector with no cross-seed structure, so generalizing it to an unseen pair asks the most of the
data. More pairs can close this gap. It is the on-cache held-out cosine, not the render-time
0.40, that the 5-pair and 20-pair checks should read.

**Branch blindness is a soft ceiling, not a hard one.** Each branch sees one concept, yet reaches
0.97 on train. The latent carries enough of the other concept for the adapter to read. It may
still be why the early held-out fit is 0.925 rather than 0.985, and the partner-embedding
variant in Part 7 tests exactly that, but it is no longer the first thing to fix.

**Fit does not predict the compose failure.** Seed 14 fits at 0.867, above the held-out mean, and
fails. Failure (b) is not a fit problem, which is the basin story below.

**Failure (a) is not the LoRA's to fix, and not the sampler's either.** Under Bradley's argument
the product target is the chimera. What the LoRA does is replace the product target with the
joint target, which is why it works where the exact samplers (FKC) cannot. This is the paper's
claim and it should be said as such, rather than as "the sampler was wrong".

**Failure (b) is a trajectory property, and a small correction cannot leave it.** With eta = 0 the
render is a deterministic function of x_T, and a correction with 16% on-target energy applied
after the basin is chosen (steps 0 to 10 per scope 05) moves nothing. Seed 9 flipping across
checkpoints with a flat direction is what a trajectory sitting on a basin boundary looks like:
tiny norm changes in r_hat push it either way. A stochastic sampler changes this per-seed
determinism into a per-seed probability, and raises the mean only if the drift is biased toward
two animals, which the LoRA supplies and the plain SDE does not.

**Failure (c) has a known shape.** AdamW with weight decay 0 and a constant learning rate on a
loss that stopped falling at 10k lets the LoRA factors grow in norm with no counter-force. With
alpha = rank the update scale is 1 at every rank, so rank 32 moves fastest and hazes first (70k
against 160k), which is what rsLoRA, arXiv 2312.03732 (**verify**), would predict. Add exposure
bias: training states come from the uncorrected trajectory, inference states from the corrected
one, so a larger adapter is evaluated further from where it was fit. The LoRA literature's
answers, in the order they are usually reached for: weight decay (0.01 to 0.1 with AdamW,
**confident** as practice, **vague** on the LoRA paper's own value), early stopping on a held-out
metric, EMA of the adapter weights (standard in diffusion fine-tuning), lower rank as the stronger
regularizer (LoRA Learns Less and Forgets Less, arXiv 2405.09673, **confident**), and a decaying
schedule. Prior preservation as in DreamBooth, arXiv 2208.12242 (**confident**) is the analogue of
keeping a plain-PoE term in the loss.

## Part 6: verdict

**Promising but needs rework.** The learned correction is real and fits the cached target well.
The number the pair-scaling plan chases, the render-time 0.40, is a trajectory-divergence
measurement that no pair count moves, because the adapter is trained only on states it never
visits at inference. Train on-policy first; read pair scaling on the on-cache held-out cosine.

## Part 7: what to do next

Three checks, all on the existing 11-pair cache, before any new cache:

1. **On-policy fit.** Roll out cat x dog with the rank 8 30k adapter, and at every step compute
   the true r_t at the corrected state with one joint forward. cos(r_hat, r_t) on those states,
   against step, beside the cached-state curve from run 1. Prediction: it starts at 0.9 and decays
   toward 0.4 as the trajectory leaves the cache. That is the 0.40 explained, in one render per
   seed.
2. **On-policy training.** Rerun the 11-pair training with half the batch drawn from states on the
   adapter's own corrected trajectory (recomputed every few hundred steps, DAgger-style), targets
   from a joint forward at those states. Read the same on-policy cosine. Run 2 already puts the
   ceiling on this at 0.04 of cosine (0.82 on-policy against 0.86 cached), so it is a
   second-order arm, not the first run. The partner-embedding
   variant (pooled embedding of the other prompt added to each branch's `text_embeds`) is the
   follow-up if the early held-out fit is still below train.
3. **Weight norm and off-trajectory norm per checkpoint.** ‖BA‖ of the adapter and ‖r_hat‖ / ‖r_t‖
   on cached and on corrected states against training step, one point per checkpoint. If both
   climb into the haze region, weight decay 0.01 plus EMA plus stopping on the on-policy cosine
   is the fix, and it costs one rerun.

For question 3, the sampler test already sits in scope 06. The literature's expectation to record
there before it runs: an eta > 0 or MCMC-corrected sampler with no adapter leaves some one-animal
basins and produces more chimeras, not fewer, because the target it samples better is the
product. The adapter plus a stochastic sampler is the arm that could raise cat x dog past 7 of 8.

## Answers to the four questions, in one line each

1. (a) and (b) are named: semantic leakage and catastrophic neglect. Du, SuperDiff and FKC are
   sampler-side and address (b) only; attention binding needs the joint prompt; CoInD and
   TokenCompose are model-side. (a) is the product target itself, which only a model-side change
   reaches. **Confident.**
2. The render-time 0.40 is neither a fit gap nor exposure bias: it is alignment against the
   pool-mean correction, and run 2 measured drift at 0.04 of cosine. The on-cache gap (0.97
   against 0.80, widest late) is the real one, and run 4 shows it is per pair, not per concept.
   No prior work learns a two-concept correction from one-concept branches, but the 0.97 says it
   is possible. **Confident on the mechanism, vague on the absence of a counterexample.**
3. Yes, the basin is the deterministic ODE map from x_T, and stochastic sampling is proven to
   contract marginal error where the ODE does not. It leaves basins on some seeds and does not
   raise the mean without a biased drift. **Confident.**
4. Late haze is the standard no-decay, constant-LR, saturated-loss signature, worse at high rank
   under alpha = rank, compounded by evaluating off the training trajectory. Weight decay, EMA,
   early stopping on held-out, lower rank. **Likely; the norm-growth mechanism is a prediction to
   check, not a citation.**
