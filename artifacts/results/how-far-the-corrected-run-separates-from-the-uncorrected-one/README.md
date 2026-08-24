# When the Mono and PoE paths part company, read in embedding space

The question: is the composition outcome decided in a narrow window around the
fork step (16 of 50, measured on raw latents), or does the correction steer the
image's meaning continuously? Raw-latent distance cannot answer this alone,
because early latents are noise-dominated and the fork could be an artefact of
where the noise floor sits.

## The read

For each cell, both arms of the run are decoded step by step: the correction-on
arm (`call__rall`, which reproduces the Mono prediction) and the correction-off
arm (`call__roff`, pure PoE). Every decoded per-step estimate of the finished
picture is embedded with CLIP and with DINOv2 ViT-S/14, and placed on the axis
from that cell's own PoE endpoint to its Mono endpoint, the same construction as
F5's manifold walk, extended over time. The curve reported per cell is the two
arms' separation along that axis: 0 means indistinguishable, 1 means the full
endpoint distance.

The naive alternative, raw pairwise embedding distance between the arms, is kept
in the lower panels of the eyeball figure as the instrument warning: two early
mush frames are far apart in any embedding space for texture reasons, so that
read shows distances at step 0 as large as at step 49 and cannot see timing.
Anchoring to the endpoint axis suppresses the mush, which is largely orthogonal
to it.

Produced by `scripts/trajectory_divergence.py` over the 9 cells with 50-step
cached trajectories (cat × dog, frog × toad, leopard × jaguar; seeds 9, 10, 11).
Pre-registered checks sit as constants in the script: near-flat separation
before step 10, steepest rise inside steps 13 to 20.

## What the curves show

The sharp-window story is falsified in embedding space. All 9 cells start near 0
(the arms begin indistinguishable, as predicted) but separate gradually from
about step 3 onward, in both spaces, with seeds agreeing. The steepest-rise
check passed for 3 of 9 cells in CLIP and 1 of 9 in DINOv2, and by the fork band
the separation has already covered roughly half the axis.

The reconciliation: the fork step is not when semantic separation begins, it is
roughly its midpoint. The raw-latent fork detects when accumulated drift clears
the noise floor; the embedding read shows the meaning of the image moving from
almost the first step. This does not contradict F4's causal claim (when the
correction must be present to flip the outcome), which only the window-sweep
grid answers. The 20-step population read over all 19 pairs shows the same
gradual climb with no plateau and no jump.

The difficulty correlation passes its pre-registered bar in both spaces:
Spearman ρ between early separation (timesteps 950 to 750) and per-pair
dose-curve AUC is −0.500 in CLIP, exactly at the 0.5 support bar, and −0.690
in DINOv2, over the 8 scored pairs. The sign, reported rather than
pre-committed, says pairs whose arms separate more early respond worse to the
correction. Eight points is support, not proof.

## Files

- `divergence.json`: per cell and per space, the separation curve, the raw
  distance curve, the steepest-rise step, and the early-separation mean, with
  the check constants recorded beside them.
- `divergence_eyeball.png`: top row the anchored separation (the read), bottom
  row the raw pairwise distance (the instrument warning), fork band 13 to 20
  shaded, step 16 dashed.
- `dose_divergence.json`, `dose_divergence_eyeball.png`: the same read over
  all 19 dose pairs at 20 steps, with x̂₀ recovered in closed form from
  consecutive noisy latents (validated against the cross cache's saved
  estimates, median relative error 2.6% against a 5% bar; step 0 alone is 14%
  and is the least trustworthy point), plus the Spearman correlation between
  early separation and per-pair dose-curve AUC, bars 0.5 to support and 0.3 to
  kill, from `scripts/dose_trajectory_divergence.py`. Two coverage facts bound
  it: 8 pairs saved trajectories for seeds 9 to 12 and the other 11 for seed 1
  only, so the band mixes seed counts; and the dose sweep scored only those
  same 8 pairs, so the correlation runs over 8 points, not 19.

Canonical copies live under
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/trajectory_divergence/`.
