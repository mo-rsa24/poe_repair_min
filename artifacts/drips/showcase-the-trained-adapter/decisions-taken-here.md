# Decisions taken here: showcasing the trained adapter

Every choice below could have gone another way. Each entry carries the decision, the reason,
and what would reverse it, so a reader who disagrees can reverse it rather than rediscover it.
Compiled from the drip-walkthrough under the diffusion-researcher role, 2026-08-29; the walk's
full evidence trail is in `plans/.walk/showcase-the-trained-adapter.md`.

## Words this uses

- **The adapter**: phase1_r8_100k, a rank-8 alpha-8 LoRA on SDXL cross-attention q/k/v, trained
  100k steps (2,000 epochs) on 88 cells (11 pairs x 8 seeds, 4,400 supervised timesteps).
- **The oracle**: the true cached interaction term r_t injected at lambda 1, the ceiling the
  adapter imitates.
- **The noise shell**: in the 65,536-dimensional latent space, a standard Gaussian concentrates
  at radius 256 with relative spread near 0.3%; a seed contributes a direction, never a radius.
- **A basin**: the set of starting directions the deterministic DDIM flow carries to one mode.

## 1. The population every figure claims over

A nested ladder, each caption claiming exactly its tier: (a) the studied pool, 11 training
pairs plus 4 held-out failing animal pairs; (b) animal pairs as a class, carried by the
held-out split; (c) SDXL composition generally, claimed only once the four existing non-animal
cells (dog x oil-painting-style, dolphin x ocean-wave, mailbox x snowfield, typewriter x
cactus) are scored, which first needs the instance-count scorer re-validated beyond two-animal
scenes. Reason: the tiers cost nothing to keep separate and each figure stays honest about its
reach. Reverses if: the paper's spine narrows to animal composition only, which drops tier (c).

## 2. The space measurements live in

Measurement in guided-epsilon space (per-step norm ratio and cosine); illustration in
predicted-x0 frames and decoded pixels; every caption states its space. Reason: epsilon is the
training target and scale-stable across the 50 steps. The rejected alternative, x0-space
measurement throughout, reads more naturally to non-diffusion reviewers but silently reweights
the timesteps by sigma_t/alpha_t, measured at 10.43 at step 2 against 0.53 at step 40, so the
same fit looks early-dominated purely from the lens. Evidence figure:
`the-same-correction-in-epsilon-and-x0-view.png` in this folder. Reverses if: never; the
distortion is arithmetic, not opinion.

## 3. The mechanism figure

The per-pair window overlay: compose rate against injection-window centre, one curve per pair,
each pair's commitment step marked, assembled from the 288 scored cells in
`outputs/interaction_term/window/window_curves.json`. It shows the intervention door fixed at
centre 5 (compose 0.656, falling to 0.0 by centre 25) for every pair while commitment varies
18 to 36, so the door is a property of the process, not the content. Caveat carried in the
caption: an early-acting correction is also what a sampler artifact looks like; the
sampler-versus-model split belongs to the is-the-gap-the-samplers-or-the-models scope, cited,
not re-argued. Reverses if: the spine claims a door location, which owes a width-1-to-2 sweep.

## 4. Training longer: no

The scored metric has nothing left to buy: held-out compose rate saturates by step 50k (0.812
at 10k, 0.961 at 50k and 60k) and F8b puts the adapter (0.875 to 1.0 per pair) at or above the
oracle ceiling (0.75 to 1.0 at lambda 1). Median train loss still creeps (0.00062 to 0.00029,
the last 10k steps buying 6%), and 100k steps is 2,000 epochs over 88 cells, deep in
memorisation territory. The visible softness in held-out renders is a quality dimension no
current instrument measures. Two tasks scoped into the transfer-figures plan instead of a new
run: score the existing per-epoch samples for steps 70k to 100k (extends F8a to the full run);
build the qualitative ceiling panel, adapter-corrected beside oracle-corrected renders on the
same held-out cells, caption stating the comparison is qualitative. Reverses if: the late
scores move, or the ceiling panel shows a crisp oracle against a soft adapter, which makes
more data (the cache holds ~58 pairs, up to 17 seeds) or more rank the lever, still not steps.

## 5. What enters the deliverable from reading and steering

Enters: the learned-versus-actual rule figures (per-step norm ratio and cosine of the
adapter's predicted correction against the cached true one, held-out cells, epsilon space) and
the seed-character panel (per-seed marginal compose rate across pairs from already-scored
cells, binomial noise as the bar, in code). Parked as its own idea thread: the residual-free
steering campaign, with its constraints pinned: the correction's direction is state-specific
(same-pair cross-seed cosine 0.002), so no fixed direction can replace the adapter; any
steering must act before roughly step 10; any found direction must pass a causal test on
compose rate. Conditional: the Jacobian-alignment probe, only if the paper keeps a mechanism
subsection. Reason: the steering question is a research campaign, not a figure.

## 6. Guidance

Every caption carries "at guidance 7.5" and the text states that the interaction term scales
linearly with the guidance scale (r_t = gs * (eps_j_raw - eps_a_raw - eps_b_raw + eps_uncond),
zero at gs 0). The appendix robustness strip (PoE and adapter compose rate at gs 3, 7.5, 12 on
a few pairs, roughly a hundred renders) is scoped; the full guidance axis is parked until
review demands it. Mono is a measured ceiling, never an assumed 1.0 (oracle rows sit at 0.75
on some pairs; the elephant x penguin default-compose question is open in the clean-pair-pool
review).

## The figure standard (cross-cutting, applies to every figure this scope produces)

1. The caption names the claim tier from decision 1.
2. The caption names the space from decision 2.
3. Axes named in words; every number carries its unit and meaning; data lands in a sidecar
   json beside the image; numbers never live in prose.
4. "At guidance 7.5" appears wherever a rate or a residual is shown.
5. The adapter is named with its checkpoint step; Mono appears as its measured rate.
6. Any 2D embedding panel claims layout, never distance; any cartoon (the shell-and-basins
   picture included) is labeled illustrative.
7. Thresholds live in code, not prose, per the repo's bars-in-code rule.
8. A mechanism claim cites the window and commitment numbers and points at the
   sampler-versus-model scope for the split it does not settle.
