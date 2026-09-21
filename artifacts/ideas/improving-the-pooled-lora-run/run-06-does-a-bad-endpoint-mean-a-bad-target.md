# Run 6: does a cell whose render shows the wrong picture carry a wrong correction?

The `break` on claim 7. The adapter is never trained on `mono.png`; it is trained to predict
`Delta_t = gs * (eps_j_raw - eps_a_raw - eps_b_raw + eps_uncond)` at each cached state
(`poe_repair/training_cache.py:13`), which is a property of the score fields under the joint
prompt rather than of where one deterministic trajectory landed. If a bad endpoint is a sampling
accident, dropping the cell throws away a usable target.

Script `scripts/showcase/residual_direction_by_target_quality.py`, cached tensors only, no model
forward, in-session 2026-09-08. Output
`/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/`. Ground truth is the eye
labels for lion x tiger from run 5: seeds 1, 3, 4, 6 good, seed 5 a single blended face, seeds
2, 7, 8 not callable either way.

## Read 1: cosine between corrections, bad cells against good cells

| Comparison | n | early | commit | late |
|---|---|---|---|---|
| ceiling, good against good, same pair | 6 | 0.211 | 0.004 | 0.001 |
| test, bad against good, same pair | 4 | 0.138 | 0.010 | -0.003 |
| ambiguous against good, same pair | 12 | 0.189 | 0.004 | 0.000 |
| floor, different pair, same seeds | 4 | 0.128 | 0.007 | 0.002 |

**Null, and confounded.** The ceiling sits on top of the floor: two good cells of the same pair
agree no better than cells of different pairs. Corrections at different states are orthogonal
whatever the target shows, which is the state-specific structure already recorded for this
project. A good cell and a bad cell never share a state, so no cross-cell cosine can separate
them. The read cannot answer the question and is recorded so it is not tried again.

## Read 2: does the joint prompt's own score lean to one concept?

Asked inside a single cell, where both experts are cached at the same state as the joint branch,
so no comparison and no extra forward is needed:
`lean_t = cos(eps_j - eps_uncond, eps_a - eps_uncond) - cos(eps_j - eps_uncond, eps_b - eps_uncond)`.
Zero means the joint prompt sits evenly between its two concepts at that state; a large magnitude
means it has committed to one, which is what a target rendering the same animal twice should look
like.

| seed | eye | early | commit | late |
|---|---|---|---|---|
| 1 | good | 0.038 | 0.297 | 0.044 |
| 2 | ambiguous | 0.076 | 0.355 | 0.059 |
| 3 | good | 0.409 | 0.539 | -0.149 |
| 4 | good | -0.161 | -0.321 | -0.117 |
| 5 | BAD | 0.058 | 0.457 | -0.138 |
| 6 | good | 0.092 | 0.161 | -0.275 |
| 7 | ambiguous | -0.058 | 0.179 | -0.194 |
| 8 | ambiguous | 0.096 | 0.096 | 0.047 |

**Null.** The one bad cell leans 0.457 in the commit window, below good seed 3 at 0.539, and good
seed 4 leans -0.321 in the other direction. The eye gave only one clearly bad cell on this pair,
so the read is underpowered as well as flat.

## What this leaves

Two cheap reads, both null, and neither refutes nor supports the training-target reading of
claim 7. Four objections from the `break` stand:

1. The adapter is trained on the per-step score difference, not on the picture.
2. The walk's own route-1 verdict explains a one-concept render as a basin of the deterministic
   trajectory, which makes a bad endpoint a sampling accident rather than a corrupt target.
3. This cache's step-0 noise is shared across pairs per seed, so dropping cells by endpoint
   removes particular seeds across the whole pool and trains on the noise draws where the joint
   trajectory happens to behave.
4. Run 1 puts train fit at 0.969 and held-out cat x dog composes on seven of eight seeds, so a
   pool whose targets mostly look wrong already yields an adapter that mostly works.

Only a training ablation settles it: the same pairs and steps, one adapter on all cells and one on
clean cells with the cell count matched by dropping random cells from the control, read on compose
count over unseen pairs. That costs a training run and is no longer a free change.
