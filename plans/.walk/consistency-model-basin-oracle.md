# Walk: consistency-model basin oracle

A drip-walkthrough under the diffusion-researcher role (`~/.claude/roles/diffusion-researcher.md`).
The target: judge the proposal to use LCM-SDXL as a basin oracle on the cached PoE trajectories,
with two claimed reads (committed-looking destination frames at high noise, and a per-step
commitment probe), and deliver a verdict per claim (a) to (e). Each verdict lands as the mark of
the layer that owns it; `compile` collects them into the decision ledger.

## Position

Compiled, and the scope is built. The ledger lives as
[the scope's decision file](../closing-the-compositional-gap/plans/when-does-the-outcome-lock-in/decisions-taken-here.md). The vocabulary table
below is the walk's record of what needed defining; where such tables should durably live is
still an open family question, so until then it lives here. Layer 1 is
walked after the pieces, or on `skip`.

## What is pinned

- epsilon prediction, DDIM scheduler, 50 steps, classifier-free guidance 7.5
  (`poe_repair/config.py:15-16`, `poe_repair/run.py:76`); each cached `latent_trajectory.pt`
  (one per pair-and-seed cell) is a deterministic discretization of the probability flow ODE
- variance-preserving process (SDXL's DDPM betas); the code runs high noise to low noise
- latent space: SDXL's VAE, 4x128x128 against 3x1024x1024 pixels
- SDXL's training set size and any intrinsic-dimension estimate for its data: both unknown.
  The role treats the absence as the finding; layer 4 is answered under that absence
- already measured: commitment by trajectory divergence at steps 18 to 36 per cell; the
  correction's effective window is steps 0 to 10 for all 8 tested pairs (EXP-04). The 8-to-26-step
  gap between those two numbers is what the diagnostic exists to explain

## The layers

| # | Layer | What it decides for this target | Mark |
|---|---|---|---|
| 1 | The structure the data actually has | whether "basin" names a real structure when the state is an off-manifold PoE composition (ask a, first half) | decided: yes, with the boundary caveat. The learned field is defined on the whole latent space, so a deterministic flow assigns every state exactly one endpoint, and endpoints cluster into modes; near the caustic set the label is genuinely unstable, and that instability is itself the commitment signal. Chimera states are pulled toward two modes at once, so they sit nearer the caustic set than typical states; every oracle read must carry a stability check (perturb the state, see if the endpoint flips). Whether LCM's approximation of the flow holds off-distribution is layer 4's question, deliberately not settled here |
| 2 | What the model transports, and between what endpoints | what map LCM-SDXL actually learned, and whose flow's endpoints it names (ask a, second half) | decided: the oracle is the endpoint map of the plain guided flow, so on a cached state it computes the counterfactual finish, through LCM's own parameterisation via one adapter. Prompt choice settled as both sweeps: the joint chimera prompt feeds the per-step commitment probe, the two expert prompts feed the which-mode destination frames, each figure labeled by which sweep produced it |
| 3 | The trajectory, and what it costs to walk it | whether "endpoint estimate stops changing" measures the same commitment event as trajectory divergence (ask b), and what the cache already answers without a new model (feeds ask e) | decided: they are different events, and the difference is the value. Endpoint stabilization reads the decision (basin entry; the field's word is speciation); divergence reads the visible separation, which lags the decision because shared noise masks a committed difference until it is removed. Pre-registered ordering: t_speciation <= t_divergence per cell; t_speciation near 10 explains the gap as decide-then-descend, t_speciation inside 18-36 kills that story. The stops-changing bar goes in code. To keep the comparison one-axis-clean, t_speciation is also computed by finish-the-run under the composed field on a subsample (the agreed layer-1 script), separating event-choice from flow-choice |
| 4 | Whether it generalised or memorised | whether a distilled endpoint map can be trusted on states it never trained on; where instrument-grade ends (feeds asks a, d) | decided: trust is purchased, never assumed. The teacher is in hand, so the student's error on exactly our states is computable: per-family agreement between the LCM endpoint and the teacher's finish-the-run ending (DINOv2 distance plus scorer-verdict match), on a subsample of roughly 240 states (3 families x 8 pairs x 2 seeds x 5 steps, about 1 to 3 GPU-hours). Bar in code before any figure ships, plus a comparability requirement: no family's agreement may sit far below another's, since a per-family bias is the confound that corrupts every cross-family figure. On failure the named fallback is downgrading scope (families and steps that pass) or finish-the-run everywhere, affordable at today's pool. Sharp frames are never evidence of trust. The theory route (N >= c*ID) is closed because both numbers are pinned unknown |
| 5 | Reading and steering what it learned | the cheaper probes that read commitment from the cache: the denoiser Jacobian's singular vectors, h-space (ask e) | open |
| 6 | Conditioning it on something you actually have | whether the guidance scale baked in by distillation corrupts the read at our 7.5 (ask c) | decided: no corruption by construction, and the proposal's caveat mis-names the mechanism. The checkpoint's unet config carries time_cond_proj_dim 256, a guidance-scale embedding, so w is an input rather than a baked constant; the card's own example runs w=8.0 through it, bracketing our 7.5, while the same card's use-1.0-to-2.0 line is stale advice belonging to the LoRA variant, which lacks the embedding. The trained w-range is undocumented (the card's Training section is TODO), so the residual risk is a range question, and the layer-4 calibration executed at exactly 7.5 measures it directly. Two conditions carried: the adapter passes w=7.5 through the embedding path and asserts no negative-prompt double pass runs on top (double-guiding would genuinely corrupt the read), and calibration at 7.5 must pass. Guidance is part of the flow's identity here: change w and the basins themselves move, so matching our 7.5 is what makes the oracle probe our flow |

No skips: all six have content for this target.

## Open decisions

- (a) is the endpoint of the flow from an off-manifold PoE state well-defined enough for "basin
  oracle" to mean something: CLOSED across layers 1, 2 and 4. Yes: the flow's endpoint is
  well-defined from any state (layer 1), the map is the base guided flow's endpoint map read as
  a counterfactual finish (layer 2), and off-distribution trust is establishable empirically per
  family against the teacher, never assumable (layer 4). "Basin oracle" is legitimate as a
  calibrated instrument; the proposal's instrument-grade caveat survives and now has a mechanism
- (b) answered at layer 3: a different event. Stabilization is the decision (speciation), divergence
  is the display; the lag between them is a measurable per-cell quantity and is the candidate
  explanation for the 8-to-26-step gap
- (c) closed at layer 6: no, by construction; w is a conditioned input (time_cond_proj_dim 256),
  7.5 sits beside the card's own demonstrated 8.0, and the undocumented trained range is covered
  by the layer-4 calibration at 7.5
- (d) half-answered at layer 3 by the scoped search: the measured quantity is established
  (speciation time: Biroli, Bonnaire, De Bortoli, Mezard, Dynamical regimes of diffusion models,
  Nat Comms 2024, arXiv 2402.18491; spontaneous symmetry breaking, Raya and Ambrogioni 2023;
  entropic signature of class speciation, arXiv 2602.09651; score shocks / Burgers caustics,
  arXiv 2604.07404). All measure commitment by theory or by regenerating with the full model.
  No published use of a distilled consistency model as the probe on cached states surfaced:
  "none found", never "none exists". Layer 4 adds: consistency distillation is documented to
  memorize (On the Memorization of Consistency Distillation, arXiv 2604.23552), which is exactly
  the failure the calibration detects. (d) is closed
- (e) closed at layer 5: yes for the per-step probe (posterior-mean drift, subsampled
  finish-the-run), yes at figure scale for the frames (finish-the-run filmstrips), no at grid
  scale, which is where the oracle earns its place, contingent on layer 4's calibration

## Vocabulary

Role terms used so far, quoted from the role file.

| Term | What it means | Source |
|---|---|---|
| probability flow ODE | the deterministic path whose distribution at every time matches the stochastic one exactly, so you can drop the randomness and keep the answer, and get exact likelihood and a usable latent in return | role `[7,5]` |
| Tweedie's formula | the identity that lets you move between three things that look different and are the same: what the denoiser outputs, the expected clean image given the noisy one, and the score | role `[29,4]` |
| projection caustic | the place where the nearest point on the data surface stops being unique, which is where a sample commits to one mode instead of another | role `[12]` |
| manifold hypothesis | the assumption that real data sits on or near a low-dimensional surface inside the high-dimensional space it is stored in | role `[14,23]` |
| mixture of low-rank Gaussians | the working stand-in for image data when you need something you can prove things about: several low-dimensional subspaces, each with a Gaussian on it | role `[29]` |
| classifier-free guidance | steering generation by mixing the model's conditioned and unconditioned predictions, so no separate classifier is needed | role `[6]` |
| consistency distillation | training a student so that every state along one teacher trajectory maps to the same endpoint, which turns the teacher's whole multi-step flow into a single jump-to-the-end function | walk-minted |
| counterfactual finish | what the oracle actually computes on a cached PoE state: the image the plain guided model would land on if composing stopped at this step | walk-minted |
| speciation time | the field's name for the step where a trajectory commits to a class or mode, before which regeneration from the state wanders and after which it is locked; what this walk had been calling basin entry | Biroli et al. 2024, walk-adopted |
| posterior-mean drift | how far Tweedie's x0-hat moves between consecutive steps; it settles when the posterior mass has concentrated on one basin, making it a free single-trajectory commitment probe | walk-minted |

## Deferred machinery

- The stability check layer 1 commits every oracle read to: perturb the cached state by a small
  epsilon, run the oracle on both, compare endpoints. Cost at today's size: roughly 8 pairs x 5
  seeds x 50 steps x 3 evaluations = about 6k LCM forward passes, under an hour on one GPU at a
  rough 0.1 to 0.2 s per pass. Measure it before trusting it.
- AGREED (layer 1's how): the three-ending perturbation check on one cell, steps 5, 25 and 40,
  finish-the-run with the repo's own DDIM code, no LCM needed. Goes into the compile's build
  order as the first slice.
- The role file has no source on consistency models, distillation, or commitment dynamics; the
  walk has now minted or imported three terms (consistency distillation, counterfactual finish,
  speciation time). Per the family rule that is the flag: the next /author-role --add should
  bring in one distillation source and one commitment-dynamics source (Biroli et al. 2024 is the
  natural pick).

## Question rounds

Asked at layer 2: does LCM-SDXL make manifold and latent-space analysis easier across the three
trajectory families (LoRA-corrected, PoE, joint prompting), grouping correction windows and
commitment demonstrably. Answer recorded: yes, as a common semantic readout. It turns any
mid-trajectory latent into a committed image, which the repo's validated image-space instruments
(the GroundingDINO instance-count scorer, the DINOv2 and CLIP embedding spaces) can then read;
the headline figure is the per-step counterfactual compose rate, one curve per family. Three
binds: (1) per-family instrument bias must be calibrated against finish-the-run on a subsample
before any cross-family claim, because PoE states sit furthest off the tube LCM trained on and
a bias that differs between compared groups is a confound; (2) 2D scatter embeddings are
illustration, never evidence (role traps, sources 27 and 9); (3) everything stays
instrument-grade until layers 4 and 6 clear. At today's pool finish-the-run is still feasible
(tens of GPU-hours for the full grid); the oracle turns the same grid into about one GPU-hour,
so it flips from luxury to enabling exactly at this three-family sweep.

Asked at layer 2: do we exclude PCA/LDA-style 2D/3D reduction of the latent states, and how is
the correction quantified. Answer recorded: linear reductions are in, with rules. PCA on
trajectories is earned by the boomerang result (source 11: trajectories occupy an extremely
low-dimensional subspace), so a 2D/3D projection with its explained-variance number printed can
be evidence, and that number decides evidence versus illustration. LDA only on PCA-reduced
coordinates (ambient 65k dims against tens of trajectories is degenerate). UMAP/t-SNE stay
excluded for distance claims (role trap, source 27). Every state-space picture is paired with an
outcome-side curve, because near in state space does not mean same destination, and the
project's claim is exactly that a small state difference flips the outcome. Three timestamps to
group: when the correction is applied (steps 0 to 10, EXP-04), when the outcome locks (basin
entry, the oracle's new number), when paths visibly separate (divergence, steps 18 to 36).

## Routes

Emitted at layer 3, on request, none yet run:

- /author-role --add diffusion-researcher: close the distillation and commitment-dynamics gaps
  (Consistency Models 2303.01469, Latent Consistency Models 2310.04378, guided distillation
  2210.03142; Dynamical regimes 2402.18491, symmetry breaking 2305.19693), folded into layers
  2, 3 and 4 rather than new layers.
- /submerge on the concepts: speciation time, symmetry breaking, caustics and score shocks,
  consistency distillation, aimed at deriving where speciation lands for a two-mode toy in 2D.
- /immerse on the system: how LCM-SDXL runs in diffusers, its scheduler, guidance entering as
  an embedding, and the adapter seams where cached DDIM states enter its conventions.
- /video-scout (emitted at layer 4, where the LCM adoption landed): "running LCM-SDXL few-step
  inference in diffusers, I need to wire cached SDXL latents into it myself".

## Next step

The route out is the plan-skill chain quoted at the end of the compile round: init-master-plan
adopts the decisions file, populate-plans breaks the five slices into plan files, verify-plan
checks them.
