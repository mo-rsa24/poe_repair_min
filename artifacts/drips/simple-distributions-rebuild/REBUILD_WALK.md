# 🔨 Rebuild walk: simple_distributions.ipynb in PyTorch, extended into the composition testbed

This file tracks the PyTorch rebuild of `notebooks/simple_distributions.ipynb` (JAX and Haiku) and its extension into the testbed of the [overlap-dial idea map](../../ideas/poe-overlap-dial/IDEA_MAP.md). It holds one row per brick, each with the check that compared the rebuilt brick to the original and what that check found. The results go back to the idea walk through [the route 3 return](../../ideas/poe-overlap-dial/routes/03-unbuild-walk.return.md).

## Pins

**Machine and interpreter**

Any mscluster GPU node, in the cluster checkout of poe_repair_min at `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain` (branch `exec/dog-and-rain`), on a card with under 1 GB in use when the run starts, per the GPU guard in `environment/hpc/execution-protocol.md`. Layers 1 to 5 ran on mscluster109's 48 GB cards. Python is `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python` (torch 2.5.1+cu118, NumPy 2.2.6). No JAX anywhere.

**Rebuild folder**

`toy/rebuild/` in this repo. `toy/anneal_samplers.py`, Du et al.'s sampler file, carries the MALA and CHA fix, and `toy/tests/test_anneal_samplers.py` checks it. The history up to the move is in the record repo `mo-rsa24/reduce-reuse-recycle-walk` (branch `walk`, commit f312a0f). Every phase is one command from toy/, `python -m rebuild.<script>`.

**Outputs**

Figures and numbers are in `rebuild/out/phase1` to `phase5`. The [unbuild-walk runs card](../../ideas/poe-overlap-dial/runs/unbuild-walk/README.md) says which of them the idea walk reads.

**Stand-ins**

None. Every brick is rebuilt, and nothing is imported from the JAX notebook.

## Layers

```
layer 12 scale ──────── the correction on Stable Diffusion and chest X-rays, tuned MALA or UHA   (not built)
layer 11 testbed ────── a 32 × 32 text-conditioned toy: encoder and cross-attention causes        (not built)
layer 10 latent space ─ change of variables: which compositions survive an encoder                (not built)
layer 9  correction ─── a network learned back on the interaction term, the fair test             (not built)
layer 8  interaction ── the term defined exactly, in log-density and in ε-space                   (not built)
layer 7  negation ───── A without B on the quadrant toy and the parts world                      (not built)
layer 6  guidance ───── classifier-free guidance as a tilted density, the weight w as a dial     (not built)
layer 5  overlap dial ── quadrant toy · parts world · fine-tune variant      phase5_dial.py, phase5_finetune.py
layer 4  parts world ─── Gaussian-mixture experts, exact products            parts_world.py, phase4_parts_world.py
layer 3  exact targets ─ ideal experts, the seven kinds, one dial each      ideal.py, kinds.py, phase3_kinds.py
layer 2  composition ─── product · mixture · negation, annealed samplers     compose.py, samplers.py, anneal_samplers.py (fixed)
layer 1  one expert ──── datasets · ResNet · energy wrapper · cosine schedule · training   datasets.py, model.py, train.py
```

## Bricks

| Brick | Layer | Files (reference) | File (rebuild) | Check and tolerance | Mark | Journey plans it linked |
|---|---|---|---|---|---|---|
| The ten 2D datasets | 1 | notebook cells 14 to 20 | `rebuild/datasets.py` | 1000 samples per panel beside `datasets-1000-samples-each.png` | matches by picture | [Gaussian densities](../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/03-densities-and-the-gaussian.md) |
| ResNet score network | 1 | cell 4 | `rebuild/model.py` | parameter count against the notebook's printed 565,606; noise prediction equals 2x at init (zero output layers) | matches: 565,506, the 100 missing are `out_logvar`, which the `beta_forward` variance never reads | [layer norm](../../../../../goal-setting/learning/deep-learning/transformers/plans/40-normalise-a-vector-by-hand-with-layer.md) |
| Energy parameterisation | 1 | cell 5 | `rebuild/model.py` | noise prediction is the gradient of E = \|net(x,t) − x\|² | matches (the init check above) | [a scalar energy whose gradient is the noise prediction](../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/34-a-scalar-energy-whose-gradient-is-the-score.md), [The Energy Parameterisation](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Energy%20Parameterisation.md) |
| Cosine schedule and DDPM | 1 | cells 10, 12 | `rebuild/model.py` | same formulas, float64 schedule | matches by code reading | [the schedule and one reverse step](../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/10-schedule-and-reverse-step.md), [noise prediction is the score](../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/11-noise-prediction-is-the-score.md) |
| Training (15,001 steps, Adam 1e-3, EMA 0.999) | 1 | cells 32, 33, 37, 38, 42, 43 | `rebuild/train.py`, `rebuild/phase1_check.py` | TV between 5000 model samples and 5000 data samples, 40 × 40 bins on [−1.5, 1.5]², within 0.10 of the TV between two data draws | 9 of 10 match; the ring (std 0.03) matches only without the last-step noise: TV 0.179, 0.075 without it, floor 0.049 | [the forward process by hand](../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md) |
| MALA and CHA fix | 2 | `anneal_samplers.py` | `anneal_samplers.py`, `tests/test_anneal_samplers.py` | 6 tests: proposal density against `torch.distributions`, per-sample acceptance, stationary variance of N(0, 1), the old divisor failing, the joint chain `inf_sample.py` uses | 6 pass | [Metropolis-Hastings](../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/12-metropolis-hastings.md), [the accept step needs the height](../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/36-the-accept-step-needs-the-height.md) |
| Product, mixture, negation wrappers | 2 | cells 6 to 8 | `rebuild/compose.py`, `rebuild/phase2_compose.py` | ring × bar, left + right, box(0.7) not box(0.3), beside the notebook's own three output pictures | matches by picture and measure | [Product, Mixture And Negation](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Product%2C%20Mixture%20And%20Negation.md), [the PoE rule](../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/12-the-poe-rule-assembled.md) |
| Annealed ULA, MALA, UHA, MUHA | 2 | cells 29, 30 and `anneal_samplers.py` | `rebuild/samplers.py` | MALA acceptance reported per sample | matches | [Langevin dynamics](../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/21-langevin-dynamics.md), [the Hamiltonian corrector](../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/23-the-hamiltonian-corrector.md), [annealed importance sampling](../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/25-annealed-importance-sampling.md) |
| Ideal experts and the seven kinds | 3 | the kinds notebook | `rebuild/ideal.py`, `rebuild/kinds.py`, `rebuild/phase3_kinds.py` | exact numbers to the notebook's printed precision; sampler numbers within 0.03 | matches: Figure 1 and Figure 2 to printed precision, runs 1 and 2 within 0.03 | [Kinds Of Composition](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Kinds%20Of%20Composition.md), [the exact score of a Gaussian mixture](../../../../../goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/plans/02-speciation/plans/02-exact-mixture-score.md) |
| Parts world | 4 | `example-9-parts-world/parts_world.py` | `rebuild/parts_world.py`, `rebuild/phase4_parts_world.py` | `numbers.json` to 1e-9 relative; each sampler against 100,000 exact draws | matches: largest relative difference 2.9e-16 | [Products And Quotients Of Diagonal Gaussians](../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Products%20And%20Quotients%20Of%20Diagonal%20Gaussians.md), [the PoE factorisation and normaliser](../../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/07-poe-factorisation-and-normaliser.md) |
| Overlap dial | 5 | new | `rebuild/phase5_dial.py` | curves of A∩B mass and TV against overlap, exact and per sampler | built | [When Neither Training Set Holds A And B](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/When%20Neither%20Training%20Set%20Holds%20A%20And%20B.md), [the MCMC correction](../../../../../goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/plans/05-composition-and-the-experiment/plans/03-mcmc-correction.md) |
| Fine-tune variant | 5 | new | `rebuild/phase5_finetune.py` | A∩B mass against fine-tune steps, exact and per sampler, with experts trained from scratch as the baseline | built: exact, tuned MALA and UHA agree from step 100 on (A∩B mass at least 0.15 inside the quadrant 0.000 to 0.005); at step 10,000 the grid density counts a spurious blob near (0.75, 0.9) that the samplers never reach | [What An Expert Learns From Its Training Set](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/What%20An%20Expert%20Learns%20From%20Its%20Training%20Set.md) |
| Guidance as a tilted density | 6 | new | `rebuild/guidance.py`, `rebuild/phase6_guidance.py` | the target p(x)·p(A\|x)^w·p(B\|x)^w computed exactly on the quadrant toy and the parts world for w = 1 to 7.5 (SDXL's hybrid appears at 7.5, and the 2D work so far measured only w = 1); A∩B share, TV and Z per w | not built | [classifier-free guidance as base plus omega times a difference](../../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/02-cfg-as-base-plus-omega-difference.md) |
| A without B (negation) | 7 | the notebook's `NegationEBMDiffusionModel` (cell 8), the kinds notebook's negation row | `rebuild/negation.py`, `rebuild/phase7_negation.py` | the pneumothorax target p(x\|PTX)^(1+w)/p(x\|drain)^w; exact mass in the A-without-B cell (quadrant: bottom-left; parts world: cat with no dog) against w, per sampler; reproduce first: in-bar mass `0.146` at w = 0, `0.037` at w = 0.1, `0.863` off every ring mode at the paper's w = 0.3 | not built | [Powers Of Gaussians](../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Powers%20Of%20Gaussians.md) |
| The interaction term, exact | 8 | new | `rebuild/interaction.py`, `rebuild/phase8_interaction.py` | in the parts world with a "wet" rule (the cat half shifts when a dog is present), r(x) = log p(x\|A,B) − [log p(x\|A) + log p(x\|B) − log p(x)] in closed form, constant with the rule off and varying with it on; the noised r_t and its gradient, written also in ε-space as ε_joint − ε_A − ε_B + ε_∅, the residual `poe_repair_min` already caches for 25 pairs at all 50 steps | not built | [What An Expert Learns From Its Training Set](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/What%20An%20Expert%20Learns%20From%20Its%20Training%20Set.md) |
| The correction, learned back | 9 | the residual a LoRA learns in `Playground/PhD/poe_repair_min` | `rebuild/correction.py`, `rebuild/phase9_correction.py` | the fair test from the parallel walk: a trained joint {A, B} condition against composition alone against composition plus a small correction fine-tuned on the same few A∩B samples, at overlap r = 0.05 and 0.001; the correction trained by regression on the exact ∇r_t and by denoising score matching with frozen experts; TV of tuned-MALA samples against the exact joint | not built | [the score function](../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/07-the-score-function.md) |
| Composing in a latent space | 10 | `../../ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/` | `rebuild/latent.py` | a composition whose exponents sum to 1 (construction (1), negation) is unchanged by an invertible warp (difference `4e-17`); the plain product is not (`0.035`, `0.080`, `0.126` at area stretches 2.6, 9 and 49) | not built | none yet |
| The text-conditioned testbed | 11 | `../../ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md` | to be designed in that walk | candidate causes of the hybrid switched on one at a time: single-object captions, the guidance weight, word similarity in a frozen embedding table, shared cross-attention (each attention reading paired with an intervention); pixel space has no exact density, so success is scored in the drawing parameters | not built | none yet |
| The correction at scale | 12 | `Playground/PhD/poe_repair_min` (the LoRA correction, its plans 04 and 09, the cached ε residuals) | the project's own training code | compose rate on held-out animal pairs, then pneumothorax without a drain, sampled with tuned MALA or UHA; caveats: samplers recover a third to four fifths of the exact mass at 2% to 10% overlap (layer 5), and the interaction term's lower bound did not track which of 25 pairs blend | not built | [classifier-free guidance as base plus omega times a difference](../../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/02-cfg-as-base-plus-omega-difference.md) |

## Instructions

These are for you to do by hand. Every check above ran in code and printed its numbers, but none has been looked at by a person yet.

### 1. 👀 Look at each phase's picture beside its original

- [ ] **Phase 1.** Open `rebuild/out/phase1/datasets-against-trained-models.png` next to `datasets-1000-samples-each.png` in toy/. Rows 2 and 4 are the trained models. Each orange panel should have its blue panel's shape. The ring's modes come out wider (sd 0.043 against 0.030), because the notebook's sampler adds noise at t = 0 as well.
- [ ] **Phase 2.** Open `rebuild/out/phase2/product-mixture-negation-every-sampler.png`, then the notebook's own pictures in `rebuild/out/phase2/reference/`. In `product-0.png`, green is the Metropolis-adjusted UHA (the last column of the rebuild figure) and blue is reverse diffusion (the fifth column). Green should sit on the two in-bar ring modes. Blue should also keep the six outer modes, pulled inward. Then open `mala-acceptance-per-sample.png`: at step 0.001 nearly every sample accepts every move, and tuned MALA spreads around 0.6.
- [ ] **Phase 3.** Run `python -m rebuild.phase3_kinds` from toy/. Every row should print `ok` or `matches`, and it takes about 2 minutes on one card.
- [ ] **Phase 4.** Run `python -m rebuild.phase4_parts_world`. The first line should read `largest relative difference 2.87e-16  matches`.
- [ ] **Phase 5.** Open `rebuild/out/phase5/overlap-dial-curves.png` and `overlap-dial-quadrant-pictures.png`, then `finetune-curves.png` and `finetune-pictures.png`.

### 2. 🌊 Walk the bricks yourself

The order, the skill for each step, what each step leaves you with, and which steps are critical are all in [the prompt guide](PROMPTS.md). Walk from there.

## Prompts, in order

Every prompt, with what it leaves you with, which layers are critical and which optional, and a scene for each critical concept, is in [the prompt guide](PROMPTS.md).

## Position

Layers 1 to 5 are built and returned to the idea walk. Layers 6 to 12 are planned, not built; they lead to the research goal: define the interaction term, show it, and learn a correction that scales to diffusion.
