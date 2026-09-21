# Decisions taken here

The ledger of the design walk that minted this scope (diffusion-researcher role, the walk at
`plans/.walk/consistency-model-basin-oracle.md`, compiled 2026-08-29). Every choice that could
have gone the other way, with its reason, so it can be reversed by someone who disagrees rather
than rediscovered by someone who is confused. The walk map holds the full record, question
rounds and vocabulary included.

> **Speciation** is the field's word for the step at which the outcome stops being undecided:
> before it the run could still finish as either animal, after it the ending is settled.

> A **basin** is the field's term for the set of states that all flow to the same ending. The
> boundary between two basins is where a small nudge switches which ending you get.

The **endpoint predictor** below is the distilled model that jumps a cached state straight to
the image that state would finish as, without stepping the rest of the way.

## The verdicts on the five asks

**(a) Is the endpoint from an off-manifold PoE state well-defined enough for the endpoint
predictor to mean something?** Yes. The learned field is defined on the whole latent space, so a
deterministic flow assigns every state exactly one endpoint, and endpoints cluster into modes.
Near [the fold where nearby states land in different endings](../../../../../goal-setting/learning/deep-learning/diffusion-models/speciation-before-divergence/plans/08-caustic-as-fold.md) the label is unstable, and that instability is itself the commitment
signal. What is being read is the base guided flow's map from a state to [where each state would finish](../../../../../goal-setting/learning/deep-learning/diffusion-models/speciation-before-divergence/plans/09-endpoint-map-and-consistency-condition.md), so the read is a counterfactual finish:
the image the plain guided model would land on if composing stopped at this step. Trust off
distribution is measurable against the teacher, never assumable.

**(b) Does "endpoint estimate stops changing" measure the same commitment event as trajectory
divergence?** No, and the difference is the value. Stabilization reads the decision (the
field's term: speciation); divergence reads the visible separation, which lags because shared
noise masks a committed difference until it is removed. The lag is measurable per pair-and-seed
run and is the candidate explanation for the 8-to-26-step gap between the correction window and
the divergence numbers.

**(c) Does the baked-in guidance scale corrupt the read at 7.5?** No, by construction, and the
caveat mis-names the mechanism. The checkpoint conditions on the scale (`time_cond_proj_dim:
256` in its unet config); the card's own example runs 8.0 through that embedding. The trained
range is undocumented, and the calibration at exactly 7.5 covers it.

**(d) Prior work?** The measured quantity is established: speciation time (Biroli, Bonnaire,
De Bortoli, Mezard, Nature Communications 2024, arXiv 2402.18491), symmetry breaking (Raya and
Ambrogioni, arXiv 2305.19693), entropic signature (arXiv 2602.09651), score shocks (arXiv
2604.07404). All regenerate with the full model or argue from theory. No published use of a
distilled consistency model to read cached states surfaced: none found, never none
exists. Consistency distillation is documented to memorize (arXiv 2604.23552), which the
calibration would catch.

**(e) A cheaper test from the cache?** Split. Per-step commitment: yes, posterior-mean drift
(free if per-step epsilons are cached, else 2 to 4 U-Net calls per state) and finish-the-run on
subsampled steps (exact, roughly 100 calls per state). Committed frames at high noise: yes at
figure scale only, by finish-the-run filmstrips (about an hour per pair-and-seed run). The
endpoint predictor's unique value is grid scale: three families, all pair-and-seed runs and all
steps, stability copies and calibration, about one GPU-hour against tens.

## The decisions

**The read is named the counterfactual finish, never the trajectory's own endpoint.** The
cached run's final frame is the last entry of `latent_trajectory.pt` and needs no predictor. The
endpoint predictor answers: stop composing here, let the plain guided model land it. Reverse
this only by finding a question the counterfactual finish does not serve.

**Both prompt passes run, each figure labeled by the pass it came from.** The joint
[chimera](../../context/world/chimera.md) prompt feeds the per-step commitment test; the two
expert prompts feed the which-mode destination frames. Cost at today's pool is a few thousand
forward passes for both.

**Speciation is the adopted term** for what the walk first called basin entry, so the figures
and paper join the field's vocabulary instead of minting a private one.

**Every endpoint-predictor read carries a stability check.** Perturb the state by 1% of its norm
in a random direction, twice; a flipped endpoint means on-the-fence, a held endpoint means
committed. Chimera states crowd basin boundaries by construction, so the pair-and-seed runs of
interest are exactly where reads are least stable; an unchecked committed-looking frame near a
boundary is a coin toss rendered in high fidelity.

**Calibration before use, thresholds in code.** Roughly 240 states (3 families x 8 pairs x 2
seeds x 5 steps), agreement scored as DINOv2 distance plus scorer-verdict match against the
teacher's finish-the-run ending, at guidance 7.5. Two thresholds, set in source before results
are seen (the `MIN_MEDIAN_RATIO` pattern): an absolute minimum, and a comparability requirement
that no family sits far below another, because a per-family bias corrupts every cross-family
figure. On failure: shrink scope to families and steps that pass, or run finish-the-run
everywhere.

**Guidance goes through the embedding path, and double-guiding is asserted absent.** The
adapter passes 7.5 as the conditioned scale and asserts no negative-prompt double pass runs on
top. Real CFG stacked on the embedded scale is the one configuration that genuinely corrupts
the read.

**Posterior-mean drift rides along as the free cross-check.** Two independent tests agreeing
on the speciation step is the corroboration figure; a systematic disagreement is reported as a
finding, because posterior-mean settling and flow commitment are cousins, not the same theorem.

**The pre-registered ordering, written before any run:** speciation lands at or before
divergence in every pair-and-seed run. Speciation clustering near step 10 explains the gap,
because the run decides early and then only descends; speciation inside 18 to 36 kills that
story and the diagnostic's premise.

**State-space pictures follow three rules.** PCA on trajectories is admitted with the explained
variance printed on the figure, and that number decides evidence against illustration; LDA only
on PCA coordinates; UMAP and t-SNE excluded from any distance claim. Every state-space picture
is paired with an outcome-side curve, because the project's central claim is that a small state
difference flips the outcome.

**The commitment tests are reads, never causal claims.** The causal side of the correction is
owned by the existing experiments showing that more correction gives more composition, and the
new figures do not re-claim it.

## Decisions behind the landing figures (plan 06)

**Endpoints are read in the compose scorer's image space, and the noisy-latent plane is kept
for timing only.** PCA over SDXL latents behaves like pixel PCA (the repo's shared plane keeps a
quarter of the variance and shows bundles, not animals), so "where does a run land" is measured
in DINOv2 space, where single animals and pairs separate, and "when do paths separate" stays in
the latent plane. The alternative, one plane for both questions, was rejected because the latent
plane cannot say what an endpoint is.

**Diffusion Explorer's idioms are borrowed; nothing of it is ported.** Helbling and Chau's tool
(arXiv 2507.01178) runs 2D toy models in the browser and cannot take SDXL; its precomputed
trajectories, single time slider, and arrows on the current sample are rebuilt over this repo's
cache instead. The forkable alternative was rejected because its state model is in-browser
training, none of which is ours.

**Axis pictures come from a representation autoencoder, and only after a chimera survives
reconstruction.** Walking a principal component in DINOv2 space and decoding through the
`nyu-visionx` DINOv2-B decoder (Zheng et al. 2025) labels an axis with pictures instead of a
legend. Every decoded frame is captioned as a reconstruction, and a known chimera render is
reconstructed first: if the decoder cleans it into one animal, decoded frames may label axes and
may never stand in for a PoE output.

**The nearest-centroid bar stays failed; both-ness is the candidate bar for the next pair.** The
first run on cat × dog failed the pre-registered nearest-centroid criterion because each cloud is
as wide as its gap to the joint cloud. The supervised "both-ness" projection separated the PoE
and corrected bands cleanly, but it was read after the result, so it is recorded as post-hoc in
the review file and becomes the pre-registered bar only for the unseen-pair render, not
retroactively for cat × dog.

**Eight seeds show where things land; a separation claim waits for about 32.** The figure is
presented as a look, with the full-space centroid distances in the sidecar, and no overlap claim
is made from it.

**Both-ness is rejected as a cross-pair measuring tool; the instance count stays the rule.** On
the eight held-out pairs the both-ness bar came back null (5 of 7 counted pairs overlap), while
the instance count on the same images read the correction composing on every pair. The axis is
built from each pair's own clouds, and when the two animals look alike a picture of two of them
embeds beside a picture of one, so the axis has nothing to measure. The plane stays as a look
for one pair at a time; any cross-pair verdict is judged on the instance count, and a
coordinate-valued replacement (the count axis decoded in the axis pictures, fixed once and
applied to every pair) would need its own pre-registered bar first.

## Build order, thin end-to-end slices

1. **Basins by hand.** The agreed perturbation check: one pair-and-seed run, steps 5, 25, 40,
   three endings each by finish-the-run with the repo's own DDIM code. No LCM. An afternoon;
   proves the valley picture or kills the proposal.
2. **The free test.** Posterior-mean drift for a handful of pair-and-seed runs from the cache;
   first speciation numbers set beside the divergence numbers; the ordering prediction gets its
   first test. Depends on whether per-step epsilons are cached.
3. **Wire the endpoint predictor.** LCM-SDXL download, the adapter (LCMScheduler, timestep
   mapping, w through the embedding, the no-double-guiding assert), one short check on a single
   state against a finish-the-run ending.
4. **Calibrate.** The 240-state pass, thresholds in code, verdict: adopt, shrink, or fall back.
5. **The grid and the figures.** Both prompt passes plus stability copies; the per-step
   counterfactual [compose-rate](../../context/world/compose-rate.md) curves per family; the
   three-timestamp aligned figure (correction window, speciation, divergence); filmstrips; the
   PCA overlay beside the outcome curves.

## Deliberately not built, with the number where each starts mattering

- **A custom consistency model distilled on the composed field.** Order of days on multiple
  A100s. Only if LCM fails calibration and the pool grows roughly tenfold, making
  finish-the-run unaffordable.
- **Finish-the-run at grid scale as the default.** Tens of GPU-hours per full pass over the
  grid; it stays the fallback and the calibration teacher.
- **An h-space read of commitment.** Needs labels and a trained classifier, and the role's
  linear-classifier trap applies. Only if the two adopted tests disagree and a third read is
  needed.
- **Any UMAP or t-SNE view.** No number makes it evidence for distances.

## Still open

- The checkpoint's trained guidance range (card's Training section is TODO); covered
  empirically by calibration, still worth one line if the authors ever document it.
- Whether per-step epsilons are cached, which prices slice 2 at zero or at 2 to 4 calls per
  state.
- The numeric values of the two calibration thresholds and the speciation threshold; set in code
  at slice time, before results are seen.
- Emitted routes not yet run: /author-role --add (distillation and commitment-dynamics
  sources), /submerge (the concepts), /immerse (the diffusers wiring), /video-scout (LCM
  few-step inference). All quoted in the walk map's routes section.
