# Showcase the trained adapter: the walk's working files

Working folder for the drip-walkthrough that designed the adapter-showcase figure set under
the diffusion-researcher role. The walk's live state is `plans/.walk/showcase-the-trained-adapter.md`.

## decisions-taken-here.md

This walk's compiled six-decision ledger plus the eight-line figure standard. Its content now
lives, merged with two parallel sessions' ledgers, in the adopted scope's
`plans/01-showcase-the-trained-lora/decisions-taken-here.md`,
which is the authoritative copy; this file is the walk's own record.

## reconciled-ledger-now-absorbed.md

The parallel sessions' reconciled ledger (experiments A/B/C, the tracking set, the mechanism
follower, the joint-prompt baseline, the repair-cell provenance rule). Fully absorbed into the
scope ledger above; kept because it was untracked and deletion has no undo.

## the-same-correction-in-epsilon-and-x0-view.png (+ .json sidecar)

**What it shows.** The identical correction Delta_t on one cached cell (held-out cat x dog,
seed 9, all 50 steps) read through two lenses: relative size in guided-epsilon view against
predicted-x0 view (log y), beside decoded predicted-x0 frames at steps 2 and 40 for plain PoE
and for the joint epsilon at the same state.

**How it was made.** Computed exactly from the training cache's raw eps arrays (no UNet
re-run), alphas from the repo's DDIM scheduler, frames decoded through the fp32-upcast SDXL
VAE on the session node's 3090. Script: the walk session's scratchpad `two_lenses_example.py`.

**What it is for.** The evidence behind ledger decision 2 (measure in epsilon, show in x0);
candidate methods/appendix figure justifying the space choice.

## manifold/ (manifold_data.json + trajectory-manifold-prototype.html)

**What it holds.** The shared-plane embedding of the 64 held-out PoE trajectories (8 animal
pairs x seeds 9-16, x_t flattened fp32, mean-centred): per-cell 50-step coordinates in the best
shared PCA plane, the variance shares (plane 24.2%, top ten axes 87.3%), and 256px final renders
(joint, PoE, adapter) embedded per cell. The html is the interactive prototype, hover any
step-point and the card switches between the three renders; it is the source for the paper's
interactive supplementary.

**How it was made.** Built in the walk's layer 1 round, shared PCA over all 3,200 pooled
step-points. Moved here from the walk session's scratchpad, which does not survive the session.

**What it is for.** The data sidecar behind figure F11,
`paper/iclr/figures/held-out-trajectories-in-one-shared-plane.pdf`, drawn by
`scripts/plot_shared_plane_trajectories.py`, which recomputes every drawn number from these
coordinates and asserts against the values recorded here. One definition trap it records: the
walk's 7.3 is the mean distance of a seed's eight ends to their centroid; the matched pairwise
statistic is 11.2 averaged over seeds, 11.7 for seed 9.
