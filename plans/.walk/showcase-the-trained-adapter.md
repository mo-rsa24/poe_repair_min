# Walk: showcase the trained adapter

Role: diffusion-researcher (`~/.claude/roles/diffusion-researcher.md`). Skill: drip-walkthrough.
Feature: turn the trained pooled LoRA adapter (phase1_r8_100k) into a coordinated set of paper
figures, with a shared standard for what each figure shows, its caption, its axes, and how a
reader traces the claim. Includes deciding whether to train longer, reproducing the seed-1
joint-prompt failure that the adapter repairs, reconciling the cross-seed W&B run
(poe-repair-cross-seed/pueuo7bl) against the animals-pairs run the paper uses, and comparing
the actual residual against the learned one from a loaded checkpoint.

A second thread joins the walk from the 2026-08-29 session: the basin-commitment idea. The
claim to test is that PoE commits to a mode (a basin of the learned distribution) early in the
50 steps, that an early nudge can move the sample into the basin the joint prompt would have
landed in, without the residual and without the joint prompt, and that a seed might carry a
pair-independent character (the same starting noise walking similar high-dimensional paths
across pairs and concepts). The repo already holds two numbers this thread needs: the best
injection window is steps 0-10 for all 8 tested pairs, and the commitment step ranges 18-36
(context/world/interaction-term.md). The role names the commitment event a projection caustic.
The concentric-contour intuition is walked under layer 1's rule that a distance in a 2D picture
must be earned, not assumed.

A third thread joins from the same session: the trajectory-embedding figure. The ask is to
project whole denoising trajectories (seeds x 50 steps x 4x128x128 latents) into a 2D picture,
overlay decoded thumbnails on the points, and read regions and axes off it: which region means
dog, which means cartoon, whether a cartoon axis exists to travel along, and whether seed 11's
cartoony character shows as a coherent path. The cache already holds x_t at every step for
every cell, so the figure runs on cached data with no new sampling. The role's relevant
results: every sampling trajectory lies in an extremely low-dimensional subspace with a shared
boomerang shape (source 11), which is what makes the projection possible at all, and the UMAP
trap (source 27) is what decides what the picture may claim. Which point maps to which concept
is layer 5's question (h-space directions, pullback metric), not something the scatter answers
by itself.

## Pinned before layer 1

- Prediction target: epsilon, SDXL base 1.0, variance-preserving, DDIM reverse direction.
  The adapter is LoRA rank 8, alpha 8, on cross-attention q/k/v only, trained to move the
  PoE epsilon toward the joint epsilon (target r_t) on cached trajectories.
- Space: SDXL VAE latents, 128x128x4 at 1024x1024 pixels. All residual claims live in
  epsilon-space over those latents.
- Sampler at eval: DDIM, 50 steps, guidance 7.5, lambda 1.0 injection. fp16.
- The training cache stores, per cell per step: x_t (1x4x128x128 fp16) and the four branch
  epsilons (a, b, j, uncond), 50 steps, timesteps 981 down to 1. Trajectory-level figures need
  no new GPU sampling.
- Training data: 88 cells = 11 training pairs x 8 seeds, 4,400 supervised timesteps total.
  100,000 optimizer steps = 2,000 epochs over that set. Held-out pairs include cat x dog,
  eagle x hawk, frog x toad, goose x swan (the F9 figure rows, all seed 9).
- No intrinsic-dimension estimate exists for this data. Layer 4's memorisation threshold
  (N >= c * ID) cannot be computed today; the role treats that absence as a finding.
- Run identity: the paper's adapter is phase1_r8_100k, W&B project
  prime_lab/poe-repair-animals-compose. The run the user cites (pueuo7bl) is in
  poe-repair-cross-seed, the shelved cross-seed pooling phase
  (plans/shelved/phases/08-cross-seed-lora-pooling.md). Two different adapters; the
  seed-1 cat-and-dog repair story currently belongs to the shelved one.

## Standing figure constraints (repo conventions, carried into every layer)

- Numbers live in figures with sidecar files, never inline in prose.
- Axes named, every number carries unit and meaning, no private labels.
- Few named examples over a band holding the rest; images beside the curve.
- Statistics to the appendix; simplification may never hide a real disagreement.

## Layers (role order, fixed)

| # | Layer | What it decides here | Mark |
|---|---|---|---|
| 1 | The structure the data actually has | What population of pairs, seeds and cells the figures claim over, whether the low-rank story is entitled to its assumption, what the basin/contour picture is allowed to mean in 65,536-dim latent space, and what the 2D trajectory embedding with thumbnails may claim | decided |
| 2 | What the model transports, and between what endpoints | The space and metric any actual-vs-learned residual comparison is made in | decided |
| 3 | The trajectory, and what it costs to walk it | Where along the 50 steps the correction acts, where the sample commits to a basin (steps 18-36 measured), and whether an early nudge before commitment is the mechanism the figures should claim | decided |
| 4 | Whether it generalised or memorised | Held-out claims, the cross-seed vs animals reconciliation, and whether training longer buys crispness or memorisation | decided |
| 5 | Reading and steering what it learned | What the learned residual is as an object, read against the actual one; whether PoE can be steered into the joint basin without the residual (Jacobian directions, h-space), whether seeds carry a pair-independent character, and where a cartoon or concept axis would actually live | decided |
| 6 | Conditioning it on something you actually have | The joint prompt's own failure (mono showing one animal) and how guidance frames it | decided |

## Vocabulary

| Term | What it means |
|---|---|
| the noise shell | in 65,536 dimensions a standard Gaussian puts nearly all its mass on a thin shell of radius ~sqrt(65536)=256; every seed starts at that radius to within about one percent, so no seed is nearer the "center" than another |
| basin, as this walk uses it | a region of starting directions on the noise shell that the sampler's flow carries to one mode of the data manifold; a partition of directions, not of radii |

## Example computed on the real cache (cat x dog, seed 9, heldout)

Per object, the % of that 50-step trajectory's variance its best 2D plane captures (PCA on the
centred 50 x 65,536 matrix, fp32, per-cell):

| object | top-2 var % | top-10 var % |
|---|---|---|
| x_t | 99.5 | 100.0 |
| eps_Mono | 83.1 | 97.1 |
| r_t | 57.3 | 86.3 |
| eps_PoE | 56.5 | 88.6 |
| x0hat_PoE | 55.2 | 89.7 |

r_t consecutive-step direction cosine: 0.564 mean over steps 0 to 9, -0.174 over steps 20 to 49.
Read: a 2D picture of one x_t trajectory is nearly lossless (the boomerang result reproduced on
this cache); a 2D scatter of r_t or eps_PoE hides roughly 45% of the variance and its caption
must say so. Per-cell planes are not a shared plane: embedding many cells together is a separate
computation layer 1 must decide.

## Layer 1's minted children (expand, 2026-08-29)

1. **What one shared embedding costs** — visited, built as a live prototype. Shared PCA over all
   3,200 step-points (8 held-out pairs x seeds 9-16, x_t object): the best shared 2D plane holds
   24.2% of variance (top-10: 87.3%), against 99.5% for a single trajectory's own plane. In that
   plane, same-seed trajectories start at one point (spread 0.00 by construction) and end a mean
   7.3 plane-units apart, while different seeds sit 171.6 apart: the seed owns the geography, the
   pair nudges. Interactive artifact (hover any step-point, card switches joint/PoE/adapter):
   https://claude.ai/code/artifact/c3901878-d91a-42d6-9cde-76f5f89f72a3
   Source + sidecar: artifacts/drips/showcase-the-trained-adapter/manifold/ (prototype html
   + manifold_data.json). Renders are final images; per-step decoded frames not built yet.
   Static figure built as F11, paper/iclr/figures/held-out-trajectories-in-one-shared-plane.pdf
   (scripts/plot_shared_plane_trajectories.py). Definition note: 7.3 here is the mean distance
   of a seed's ends to their centroid; the matched pairwise statistic is 11.2 (seed 9: 11.7).
   Route emitted (2026-08-29): /design-figure on promoting the prototype into the paper's
   manifold figure; prompt carries the measured numbers and the caption constraint.
2. **What stands in for intrinsic dimension here** — visited. Three dimensions get conflated and
   only two are known: ambient 65,536; trajectory-set effective dimension (2 axes hold 24.2%,
   10 hold 87.3% of pooled variance); the data manifold's intrinsic dimension, unmeasured. The
   stand-in today is the pooled spectrum; the afternoon-sized check is a two-NN estimate on the
   64 final latents (cache-only). Until it exists, layer 4's N >= c*ID threshold has no ID.
3. **What the thumbnail caption may legally say** — visited. May say: within-thread structure
   (per-cell plane 99.5%), seed-anchored starts (exact, by construction), in-plane end spreads
   with the "in this plane" qualifier, colour-is-step. May not say: between-thread distance or
   similarity, drift-toward claims, anything about the unshown 75.8%. Thumbnails on points are
   legal only if each is the decode of that step; final renders belong in a linked card, as the
   prototype does it.

## Figure candidates from the reader (each decided in its layer's round, not before)

| Concept figure | Layer that decides it | Seed of the design |
|---|---|---|
| manifold hypothesis: the thumbnail scatter of trajectories | 1 | what object, what shared embedding, what the caption may claim |
| pullback metric: what a distance means here | 2 | latent-space distance vs decoded-image distance, same pairs both ways |
| boomerang shape: each cell's x_t path in its own top-2 plane | 3 | 99.5% honest per cell; overlay across seeds and pairs |
| projection caustic: where the run commits | 3 | commitment steps 18-36 measured; divergence rise window 13-20 |
| the timing ladder: window, rise, commitment on one step axis | 3 | minted in layer 3's round; ASCII sketch delivered via draw; all numbers from existing files |
| h-space: where a dog or cartoon direction lives | 5 | needs new instrumentation (UNet bottleneck capture); repo does not run this today |

## Deferred machinery

- **A consistency-model endpoint probe (candidate instrument, not owed).** LCM-SDXL is a
  latent consistency model distilled from SDXL in the same VAE latent space; applied to the
  cached PoE states x_t it estimates the ODE *endpoint* from each step, where Tweedie's x0
  gives only the posterior mean (the blur in the layer-2 example's step-2 frame is that mean
  averaging over basins). Read per step, the endpoint estimate is a basin oracle: the step at
  which its prediction stops changing is a cheap per-step commitment probe. Starts mattering
  if the commitment figures need committed-looking destination frames at high noise, or a
  per-step "predicted destination" readout. Costs: one checkpoint download, one pass over
  cached states; caveats: distilled model (instrument-grade, never claim-grade), PoE states
  are off-distribution for it, guidance is baked into the distillation.
  Expanded 2026-08-29 into four children:
  1. *The checkpoint*: LCM-LoRA (latent-consistency/lcm-lora-sdxl, ~400MB on the already-loaded
     base UNet) over the full LCM-SDXL UNet; instrument-grade tolerates the lighter variant;
     switch only if frames come out too rough. Near-floor: a download-and-try settles it.
  2. *The condition fed to the probe* (needs a decision): the endpoint depends on the prompt the
     consistency function is given. (i) joint prompt = where the joint flow would carry this
     state, the sharp version of the layer-2 teacher-forced read; (ii) each expert prompt = a
     two-way basin-identity read, which single-concept destination the state is closer to
     (DINO distance between the two endpoint frames); (iii) uncond = prior destination, skip.
     Recommendation: (i) for the commitment probe, (ii) for basin identity.
  3. *The output panel*: per cell, endpoint frames every 5 steps under the joint prompt with the
     stabilisation step marked (first step where consecutive endpoint frames' DINO distance drops
     below a bar set in code), beside a scatter of probe commitment step vs divergence commitment
     step per cell with the 1:1 line. Claim if they agree: two independent instruments date
     commitment the same. Disagreement is a finding, not a failure.
  4. *Cost on this node* (RTX 3090, ~19GB free): 4 held-out pairs x 4 seeds x 50 steps x 2
     conditions ~ 3,200 UNet forwards at roughly 0.3s each ~ 16 min, plus VAE decodes at 0.36s
     per displayed frame. Rough figures; measure on the first cell.

## Routes emitted

- 2026-08-29, from layer 1: /deep-learning-scene and /picture-speak, both building the
  shell-and-basins correction (cartoon vs shell, flow, basins, caustic) with KaTeX for the
  concentration-of-measure math. Target grouping: artifacts/scenes/noise-shell-and-basins/.
  Emitted, not invoked.

## Decisions taken

- **Layer 2, the space split: SETTLED.** Measurement lives in guided-epsilon space (cosine,
  norm-ratio, per-step curves); illustration lives in predicted-x0 frames and decoded pixels;
  every figure caption states which space it is in. The rejected alternative, recorded: x0-space
  measurement throughout would read more naturally to non-diffusion reviewers but silently
  reweights the timesteps, multiplying every epsilon error by sigma_t/alpha_t (10.43 at step 2,
  0.53 at step 40 on the measured cell), so the same adapter would look early-dominated purely
  from the lens. The mode rule joins the settlement from this session's round: learned-vs-actual
  claims are computed teacher-forced on cached states; closed-loop curves claim behavior only
  (the divergence profile stays closed-loop by design); every comparison figure's sidecar names
  space AND mode. Reason: the lambda-0 canary reproduced the cached PoE render's mode but not
  its pixels with settings matched, so only teacher-forced comparisons are immune to fp16 drift.
- **Layer 2 example artifact, run on the real cache** (heldout a_cat__x__a_dog seed 9):
  artifacts/drips/showcase-the-trained-adapter/the-same-correction-in-epsilon-and-x0-view.png
  (+ .json sidecar, all 50 steps). Two findings: (1) the two lenses tell opposite stories,
  x0 view peaks 0.6-1.0 at steps 0-10 and dies by 50, epsilon view is 0.03-0.07 early and
  ~0.3 mid-late; (2) at step 40 even the joint epsilon evaluated at the PoE state pulls toward
  a single dog, not two animals: teacher-forced correction after commitment cannot recover
  composition on this cell. Feeds layer 3.

**Layer 2: decided** (settled 2026-08-29). The two-space rule, ratifying instruments that
already worked here rather than choosing fresh:

1. Mechanism claims (does the LoRA output point along r_t, how much does it supply) are made in
   epsilon space: per-step cosine for direction, the pre-registered norm ratio
   ||r_t||/||eps_PoE|| for size (report/normalization_preregistration.md), fp16 upcast to fp32.
2. Perceptual claims (when runs visibly separate, when a sample commits) are made in
   decoded-embedding space: Tweedie-decode x_t, embed, 1 - cosine on L2-normalised embeddings,
   the divergence instrument's proven metric. Backbone rule per figure: DINOv2 when the claim is
   about image content, CLIP when the prompt is part of the claim.
3. Raw latent L2 with no named metric is banned from every figure (the flat-space trap).
4. Every figure caption carries a "space and metric" field; no borrowed formula enters without
   checking it is written for variance-preserving epsilon-prediction.

**Layer 3: decided** (settled 2026-08-29). The timing story and its two boundaries:

1. The figures tell the measured ordering as one story: injection window (steps 0-10, window
   ablation, all 8 pairs) precedes visible divergence (13-20, divergence.json) precedes
   commitment (18-36 per pair), with the correction coherent early (consecutive cosine 0.564,
   steps 0-9) and jittery late (-0.174, steps 20-49). Four instruments, no contradiction.
2. Boundary one: "any early push works" is dead; the random-direction control at matched norm
   gives dose AUC 0.023 against 0.387 real, so the claim is early AND correctly aimed.
3. Boundary two: no sampler-vs-model attribution in any figure until the
   is-the-gap-the-samplers-or-the-models scope reports (its free-bound run is step 24).
4. Minted: the timing-ladder figure candidate (ASCII sketch above, route to design-figure
   emitted). The boomerang overlay stays per-cell planes per layer 1; cross-cell alignment is a
   design question for design-figure, not a science question.

Reason it settled in one round: every number already exists in a file, both boundary controls
have run, and the open attribution has a named owner elsewhere in the tree.

**Layer 4: decided by the reconciled ledger's evidence** (the sibling walk's session landed the
deciding fact this walk named: held-out compose rate saturates by step 50k, 0.812 at 10k and
0.961 at 50k and 60k, with F8b putting the adapter at or above the oracle ceiling per pair, so
training longer is a no; the softness is a quality dimension no current instrument measures,
and the ledger scopes late-checkpoint scoring plus the ceiling panel instead of a new run).
The original mark and its options, kept as the walk's record:

**Layer 4 as this walk left it: needs a decision** (settled as such 2026-08-29). The train-longer question has three
real options and two deciding facts, neither needing new GPU work:

Options: (a) train phase1_r8_100k longer on the same 88 cells (GPU-days; pays only if held-out
curves were still improving at 100k); (b) scale the data instead, more cached cells (the
principled fix if N=4,400 sits near the memorisation threshold); (c) capacity and inference
knobs first, rank and lambda (cheapest; may cure the F9 softness without touching
generalisation).

Deciding facts: (1) the W&B held-out curves at step 100k, one fetch away (and note the run
identity: pueuo7bl is the shelved cross-seed phase; the paper's adapter is phase1_r8_100k in
poe-repair-animals-compose; every figure names its adapter); (2) the two-NN
intrinsic-dimension estimate, already layer 1's open check.

Also on the table from the role: the reproducibility probe (train a twin adapter from a
different init, same data, same held-out noise; same image = rule, different = memorisation),
the one probe that answers the question directly, at the cost of one training run. Soft
held-out samples are not evidence of undertraining; the trap cuts both ways.

**Layer 5: split verdict** (settled 2026-08-29).

Decided: the learned-vs-actual residual figure (per-step direction cosine + norm ratio between
LoRA output and r_t, epsilon space fp32, per layer 2's rule; the ledger records this family as
owed, ~20 figures measure r_t and one ever measured the LoRA). The causal bar binds every
direction figure: a drawn axis must be intervened on, with the change matching the predicted
gain (the role's u_i / sigma_i line). Spectrum claims run SVD on target residuals, never LoRA
outputs. Minted: the seed-character panel (one seed across pairs, qual grid beside a
style-embedding variance split into seed part and pair part; cache + decodes only).

Not yet: the concept-axis machinery (h-space capture, Jacobian steering probes) for the paper.
Trigger named: it enters when a steering claim enters the paper, and not before; until then it
lives as an exploratory thread under artifacts/ideas/. No video route, because it is a not-yet.

**Layer 6: decided** (settled 2026-08-29). The conditioning frame and its honesty rules:

1. The frame sentence, stated plainly: the adapter never sees the joint prompt, so any
   composition it restores comes from the property it trained on.
2. The ceiling is measured, not assumed: figures report mono's own compose rate on the same 64
   cells with the validated scorer, and qualitative grids include a mono failure where one
   exists. The elephant x penguin default-composition question stays with its review-file owner.
3. Guidance 7.5 pinned in every caption's provenance; no cross-method figure varies it; softness
   is never cured by raising it (the off-manifold trap).
4. One cache-only sidecar number adopted: per-step norm of PoE's summed push vs mono's single
   push (how much harder PoE leans on guidance than the ceiling).
5. The three-way compose-rate comparison stays owned by the existing transfer-figures plan; this
   walk adds rules to it, not a rival figure.

## Open decisions

- **Layer 4 RECONCILED by the user, 2026-08-29: all three experiments run.** The twin
  session's evidence stands and re-scopes what each run can claim: held-out compose rate is
  saturated by 50k (0.961) and F8b puts the adapter at or above the oracle ceiling on the
  scored metric, so A (resume to 200k) and B (rank 16/32) cannot claim compose-rate gains;
  they chase the scorer-invisible softness and B's rank-ablation figure, and their
  falsification bars stay as written. C's lambda-window sweep probes the softness directly.
  The twin's two scoped tasks (score the unscored 70k-100k per-epoch samples; the qualitative
  oracle-ceiling panel) remain owed and complement the runs; the oracle panel doubles as B's
  interpretation key (oracle crisp with adapter soft is the one outcome that makes capacity
  the regime-correct lever).

- **Layer 6: SETTLED.** How the paper frames the joint prompt's own failure. The baseline is
  the target itself: on some held-out cells the joint prompt (the thing the correction was
  defined from) fails to show both animals, and the adapter, which never consumes the joint
  prompt at inference, restores them. Adopted:
  (i) score the cached mono.png renders across the held-out pool with the validated
  instance-count scorer (renders exist per cell; in-session, minutes), giving the
  joint-prompt baseline compose rate per pair;
  (ii) the counted figure: compose rate per pair, three bars (joint prompt, plain PoE,
  adapter), same cells and seeds, with the repair-cell strip beside it as the anecdote
  (evidence-ladder pairing);
  (iii) wording rule: "restores composition the joint prompt loses on this pool", never
  "outperforms SDXL" or "outperforms large models";
  (iv) guidance stays fixed at 7.5 and stated in every caption; no guidance sweep enters
  unless the spine ever claims WHY the joint prompt mode-drops (it does not today). The
  role's trap carried: raising guidance to buy fidelity pushes samples off-manifold, so a
  guidance sweep is not a free clarity knob; the CFG vector-sum reading stays a mental
  model, not a measured claim.
- **Layer 5 (RESOLVED, kept for the evidence pointers) — what enters this deliverable.** The object first: the
  repo's own measurement (same-pair cross-seed cosine 0.002) says r_t's direction is
  state-specific; the interaction term is a rule, not a vector, so no fixed injected direction
  can replace the adapter, and the adapter (a state-dependent map on cross-attention, the layer
  that carries object attribution) is the right *kind* of object. Options for the figure set:
  (a) show the learned rule against the actual one in epsilon space (per-step norm ratio and
  cosine of r_hat vs r on held-out cells, layer-2 settlement applies) plus the free
  seed-character panel, the per-seed marginal compose rate across pairs from already-scored
  cells (binomial noise as the bar); (b) add the Jacobian-alignment probe: does r_hat point
  along the denoiser's top Jacobian singular vectors at the same states (the role's steering
  machinery, predictable gain lambda*sigma_i); explanatory, a few JVPs per state on the 3090;
  (c) full residual-free steering (h-space or Jacobian intervention sweeps under the
  before-step-10 constraint, causal-tested on compose rate): real research, GPU-weeks, parks
  as its own idea thread, not this paper. Recommendation: (a) now, (c) parked, (b) only if the
  paper keeps a mechanism subsection. Deciding fact: whether the spine has a slot for
  "how the adapter does it" or only "what the adapter does".

- **Layer 4 (RESOLVED, kept for the evidence pointers) — the train-longer lever.** Reconciliation settled by the F9 sidecar: the figure's
  adapter column is phase1_r8_100k at checkpoint 100k (artifacts/results/does-the-fix-reach-
  unseen-pairs/pooled_lora/phase1_r8_100k), not the cited cross-seed run pueuo7bl; pueuo7bl is
  the shelved cross-seed adapter and now serves only as a free replication datum. Evidence from
  the run's own files: compose rate saturates by 50k (out_out 0.812 at 10k, 0.938 at 20k, 0.961
  at 50k, 0.961 at 60k; checkpoints 70k-100k were never scored, a gap); median train loss still
  creeps (0.00062 at 10k-20k to 0.00029 at 90k-100k, last decade -6%); 100k steps is 2,000
  epochs over 88 cells. Options: (a) do not train longer; score the existing 70k-100k per-epoch
  samples to close the gap, and route the crispness question to the oracle-ceiling test
  (adapter-corrected vs true-r_t-injected, same held-out cells, the owed ablation): if the
  oracle is also soft, no training run fixes it; (b) train to 200k: costs GPU-days, buys at
  most the loss's remaining few percent, risks memorising 88 cells; (c) change the lever: the
  cache holds ~58 pairs and up to 17 seeds while training used 11 pairs x 8 seeds, so more data
  (or rank) is the regime-correct move if the oracle is crisp and the adapter is not.
  Its (a) verdict is overridden by the user's reconciliation above (all three run); the evidence stays because it re-scopes the claims: F8b is built
  (paper/iclr/figures/compose-rate-by-pair-for-lora-against-the-joint-prompt-correction.json,
  adapter at 60k per pair 0.875-1.0 against oracle-at-lambda-1 0.75-1.0), so on the scored
  metric the adapter already sits at or above the oracle ceiling and training longer has
  nothing left to buy there. The joint-failure-repair artifact (cat x dog seed 1, step 100000)
  also exists. Remaining softness is a scorer-invisible quality dimension; not a training-
  duration question until an instrument measures it. Two tasks scoped into the plan tree
  rather than run ad hoc (route emitted 2026-08-29 to /refine-plan on
  figure-01-the-transfer-figures.md): (1) score the existing per-epoch samples for steps
  70k-100k and extend F8a to the full run; (2) the qualitative ceiling panel, adapter-corrected
  beside oracle-corrected renders on the same held-out cells, caption stating the comparison
  is qualitative.

- **Layer 3 (RESOLVED, kept for the evidence pointers) — which figure family makes the
  early-nudge mechanism a claim.** Evidence already
  on disk (outputs/interaction_term/window/window_curves.json, 8 pairs x 9 windows x 4 seeds,
  288 scored cells, width 10, fork_step 16): compose rate by window centre falls 0.656 (centre
  5) -> 0.250 (10) -> 0.094 (15) -> 0.031 (20) -> 0.0 (25 on). The intervention door closes at
  or before the visible commitment window (18-36), which is the caustic ordering: by the time
  the choice is visible, it is made. Options: (a) assemble-only two-panel, window-decay curve
  beside commitment frames (layer-2 example supplies the anecdote frames); (b) per-pair overlay,
  compose rate vs centre one curve per pair with each pair's commitment step marked, showing the
  door is fixed at centre 5 while commitment varies by pair, assembly from the existing 288
  cells (EXP-04's figure, refuted-hypothesis form); (c) a new sharp-edge sweep, width 1-2
  single-step nudges, only owed if the paper claims a door *location* rather than a door.
  SETTLED as (b): the per-pair overlay, compose rate vs window centre, one curve per pair,
  each pair's commitment step marked, assembled from the existing 288 cells. (c) is not owed
  unless the spine later claims a door location. Caveat owed: early-timing is also what a sampler
  artifact looks like; the sampler-vs-model split is owned by
  plans/is-the-gap-the-samplers-or-the-models/ and its falsification criterion is already
  written there. Constraint exported to layer 5: any residual-free steering must inject before
  ~step 10 to have a chance.

- **Layer 1, the population: SETTLED as a nested ladder, all three tiers.** Each caption
  claims exactly its own tier: (a) figures on the studied pool say "failing animal pairs,
  11 train + 4 held-out"; (b) held-out figures claim animal pairs as a class via the split;
  (c) SDXL composition generally is claimed only once the non-animal cells
  (a_dog x oil_painting_style, a_dolphin x an_ocean_wave, a_mailbox x a_snowfield,
  a_typewriter x a_cactus) are scored, which requires re-validating the instance-count scorer
  for non-two-animal scenes. Tier (c) enters compile as its own plan with the re-validation
  as its first task. Reason: the tiers cost nothing to keep separate and each figure stays
  honest about its reach.
- **Layer 1, the contour picture: SETTLED, repair accepted.** The concentric-contour intuition
  (seeds on an inner cartoony contour) does not survive the noise shell: all seeds share one
  radius, so seed character can only be directional. The basin map is a partition of the shell
  by flow destination. Any figure using the contour cartoon labels it illustrative; any 2D
  embedding panel (MDS/PCA of trajectories) claims layout only, never distance.

- **Layer 4, the longer-training question: SETTLED, all three causes run as separate
  experiments** (user's call: coverage over selection; the free 50k-vs-100k softness read no
  longer gates anything, though it stays a sanity check).
  - **Experiment A, length**: resume phase1_r8_100k from lora_step_100000.pt to 200k steps,
    config otherwise unchanged. One axis. ~6.6 GPU-hours for the added 100k steps at the
    measured 4.2 steps/s (measure it). Bar, written before launch: if held-out compose rate
    at 200k minus at 100k lies within the seed-noise band (spread over the 8 held-out seeds)
    and the frozen figure tracking set shows no crispness change, length is not the knob (null).
  - **Experiment B, capacity**: fresh runs at rank 16 and rank 32 (alpha = rank), everything
    else identical, 100k steps each. One axis. Deliverable: the rank-ablation figure, same
    cells, same lambda, same inference across r8/r16/r32 at matched steps. Bar: r16/r32
    held-out compose rate within r8's seed-noise band means capacity is not the knob.
  - **Experiment C, injection**: no training; lambda grid {0, 0.25, 0.5, 0.75, 1.0} crossed
    with the step window at inference on existing checkpoints. Bar: softness tracking lambda
    at fixed checkpoint means the softness is the injection, not undertraining.
  - **Shared instrument, frozen before launch, extends instrument-02 (no parallel twin)**:
    per checkpoint every 10k steps render the four F9 cells (held-out pairs, seed 9) plus the
    repair cell cat x dog seed 1; compose rate over the held-out pool (8 pairs x 8 seeds);
    direction-cosine and fraction-of-distance-reached (already wired); cosine(learned delta,
    cached r_t) per timestep bucket (norms logged already, cosine needs adding).
  - **Launch shape on biggpu** (4 idle nodes seen: mscluster 106, 108, 111-112; 3-day
    walltime; one Slurm job per user): A as the one sbatch job; B's two runs over the
    SSH-plus-nohup idle-node path per environment/hpc/execution-protocol.md step 3; C
    in-session. Roughly 20 GPU-hours across three nodes in parallel.
  - **Not launched from the walk**: per EXPERIMENT_CONVENTIONS each experiment gets a plan
    file with pre-registered review questions first; compile mints them.
- **Layer 4 harvest, the repair cell**: the repair REPRODUCES on the paper's adapter.
  Cat x dog seed 1, checkpoint 100k, lambda 1: a black dog and a large dark cat, clearly
  separated, plus a third smaller cream animal (a caption must say so). The lambda-0 canary
  matches the cached poe.png in mode (same blend creature, same collar tag) but not pixels;
  cache meta matches settings exactly, so the gap is known fp16 nondeterminism. Renders are
  mode-reproducible, not byte-reproducible, and figure provenance should say that. Filed
  with card at artifacts/results/does-the-fix-reach-unseen-pairs/joint-failure-repair/.
- **Layer 4 groundwork done in-session**: the joint-prompt failure precondition exists in the
  paper's own cache (heldout/a_cat__x__a_dog/seed_1: mono.png is a single cat, poe.png one
  blended creature); no adapter render existed for that cell (per-epoch renders cover seeds
  9-16 only); scripts/render_joint_failure_repair_cell.py renders it from a loaded
  checkpoint (lambda 1 plus a lambda-0 canary that must match the cached poe.png), output
  under artifacts/results/does-the-fix-reach-unseen-pairs/joint-failure-repair/.

- **Layer 5, round 1: the instrument's read of the learned object, NEEDS A DECISION.**
  Four candidate additions to the shared per-checkpoint tracking set, each computable from what
  the tracking set already renders or records (the fact that admits an addition; anything needing
  new forward passes inflates every run and waits):
  1. **Learned-vs-actual cosine per timestep bucket**: cos(learned delta, cached r_t) at the
     already-recorded steps 7, 15, 22 (record_delta_at_steps hook exists), held-out cells.
     y cosine, x training step, one curve per bucket. Claims which part of the trajectory is
     learned first; claims nothing causal.
  2. **Embedding drift of the rendered tracking set**: DINOv2 and CLIP embeddings (the scorer's
     two validated spaces) of each per-checkpoint render; y = distance-to-mono minus
     distance-to-poe, x = training step, per cell. Seconds per checkpoint. The instance-count
     scorer keeps ownership of any composition claim; any 2D scatter is layout-only per
     layer 1's rule.
  3. **Spectral share of the learned correction, diagnostic only**: stack learned deltas
     across held-out cells at a fixed bucket, top-k singular-value share vs training step.
     The paper's low-rank claim stays on targets (r_t), never on LoRA outputs; this curve is
     labelled diagnostic.
  4. **Divergence-step profile**: per checkpoint, norm of the learned delta across the 50
     sampling steps, read against the measured commitment steps 18-36 and best window 0-10.
     Ties the basin thread to training time.
  Not yet, with their numbers: h-space capture (repo does not run bottleneck instrumentation;
  starts mattering when a steering figure is wanted, about one plan of work), Jacobian
  singular directions (expensive per cell; starts mattering when an edit-magnitude claim is
  needed). Role traps carried: a direction found at one timestep need not hold at others;
  no causal language without an intervention (experiment C's lambda sweep is that
  intervention); the latent space is not flat, so no Euclidean 2D distances.
  The space each measurement reports in follows the other thread's layer 2 decision
  (guided-epsilon vs predicted-x0, per figure family), not this round.

- **Layer 5, round 2: would h-space and Jacobian reads help?** Verdict recorded: both make
  revealing qualitative over-training figures, neither belongs in the frozen tracking set, and
  each earns its place only through a named claim. h-space (UNet mid-block activations):
  capture is a forward hook, storage forces pooling or a cell subset; the figure "the
  correction in the model's own semantic space over training" is strong, and its causal
  upgrade is cheap (inject the found direction, score composition with the existing scorer),
  which is exactly the already-registered layer-5 question of steering PoE into the joint
  basin without the residual. Jacobian singular directions: power-iteration JVPs, minutes
  per (cell, timestep, direction), so showcase cells only; the figure "does the adapter open
  a new high-gain direction for the missing animal over training" is the mechanism read,
  correlational until intervened on. Proposed shape: one offline mechanism plan (few cells,
  about 3 checkpoints, both reads, each ending in its intervention), reading checkpoints
  after A/B exist, separate from the tracking set; the paper's figure ladder is deliberately
  oracle-lean, so this enters as a scope addition the compile flags for the user to weigh.
- **Layer 5: SETTLED.** Battery additions 1-4 (learned-vs-actual cosine per bucket,
  embedding drift in DINOv2 and CLIP, spectral share labelled diagnostic, divergence-step
  profile) enter the frozen per-checkpoint tracking set. The mechanism reads are ADOPTED, not
  merely flagged: h-space (the correction in the model's own semantic space) and Jacobian
  singular directions run DURING training as a follower process on a non-training device:
  it consumes broad-interval checkpoints as they are written (default first, middle, final;
  adjustable) and logs to the same W&B run, so the mechanism figures appear live beside the
  training curves. The training loop is never modified. A few showcase cells, each read
  ending in its intervention so its figure earns a causal caption.
  Reason: the reads answer the steering question this layer already carries, and broad
  intervals keep them off the training runs' critical path.

## Compile (2026-08-29)

Ledger written to
plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/decisions-taken-here.md.
Layers 3 and 6 compiled as open; layer 2's space decision stays with the other thread.

## Routes emitted

- design-figure on the manifold figure (under layer 1, child 1).
- submerge on layer 2's transport machinery (score, Tweedie, probability flow ODE), flagged to
  reconcile with the existing journeys poe-composition-diffusion and
  sampler-correctors-for-composition before minting anything new.
- design-figure on the timing ladder (layer 3): sketch + provenance carried in the prompt.
- immerse on rebuilding the trajectory-manifold pipeline by hand, carrying layer 1's decision
  ledger so the journey does not re-derive settled decisions.

## Next step

Walk complete. The ledger at plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/decisions-taken-here.md carries all six layers; the route out (init-master-plan, populate-plans, verify-plan) was emitted 2026-08-29 and is with the user. The vocabulary table lives here until a durable home exists. Other thread: layer 2 round 1 delivered: measure in guided-epsilon space vs show in predicted-x0/pixels,
per figure family. Settle or expand (teacher-forced vs closed-loop comparison is the child).
