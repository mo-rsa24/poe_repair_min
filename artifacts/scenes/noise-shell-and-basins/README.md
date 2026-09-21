# Scene: the noise shell and its basins

**ran under** any browser; `index.html` is one self-contained file with no build and no
dependencies. Also published as a private artifact:
[the noise shell and its basins](https://claude.ai/code/artifact/5cc908b7-59b9-4929-ad6b-3309bec51690).

**built from** a fixed claim list (the picture-speak dispatch of 2026-08-29), nothing beyond it:
the chi-square shell facts in d = 65,536, seed-as-direction, the 50-step DDIM flow at guidance
7.5 sorting directions into basins, and the measured nudge and commit windows. All drawing is
native Canvas plus MathML; no external libraries.

**why** the geometric backdrop for the sampler-vs-model argument: it shows where a seed lives
(a 0.3% band at radius 256), what a seed is (a direction), and when the flow's choice of mode
closes (measured, steps 18 to 36), so the "commit window" language in the paper has a picture a
reader can operate.

**depends on** nothing outside this folder.

See it from a laptop by serving the scenes folder in tmux and forwarding the port, per
[the runbook's showing-a-scene recipes](../../../runbook/looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md).

## What is on the page

A Guided walk of six frames, one claim each, plus an Explore mode. Arrow keys or the dots step
the walk.

1. **Every seed lands 256 from center.** The shell in a 2D slice, beside the distribution of
   seed lengths: 256 with a ±0.7 band (0.3%). Exact chi-square math, drawn positions illustrative.
2. **The dense center holds no mass.** The contour cartoon's shaded center against the radial
   mass curve, a needle at 256; the chance of landing nearer than 250 is below 10⁻¹⁵.
3. **A seed is only a direction.** Two same-length arrows; the angle is the seed's only choice.
4. **The flow sends directions to modes.** Animated trajectories from the shell to two modes,
   paths drawn for the eye; 50 DDIM steps and guidance 7.5 are the real pipeline settings.
5. **Basins tile the shell.** Arcs colored by destination; the boundary labeled as a projection
   caustic.
6. **Open at 10, sealed by 36.** The measured frame: a step timeline with the nudge window
   (steps 0 to 10, all 8 tested pairs) and the commit window (steps 18 to 36), hover to read a
   step, click to pin.

Explore has a dimension slider (d = 2 up to 65,536) that shows the Gaussian's mass migrate from
a blob to a shell, with live mean length and spread, and a toggle between the contour cartoon
and where the sampled seeds actually sit. At d = 2 the cartoon is honest; by d = 65,536 it is
dead.

## Provenance in one line each

The shell radius, the 0.3% band, and the empty-center bound are exact chi-square consequences,
badged "exact math" on the page. The nudge result (steps 0 to 10 change the outcome, 8 of 8
pairs) and the commit window (steps 18 to 36) are measured in poe_repair_min and badged so.
Discs, trajectories, modes, and basin boundaries are illustrative and badged so. The page's
closed disclosure repeats all of this beside the numbers.

**verified** `node --check` passed on the script block; no browser exists on this cluster, so
the layout was not eyeballed in one. Report collisions or overflow and they can be fixed and
republished to the same URL.

## Cross-references

- The mention of **level set** in [the scene map for the Gaussian score's base case](../../../../../../goal-setting/learning/deep-learning/diffusion-models/poe-composition-diffusion/artifacts/drips/gaussian-score-base-case/SCENE_MAP.md), piece 1a-4a-ii, "The ring's reach along u₁" (relevance match). Same primitive as this scene's contour cartoon, argued in the opposite regime: the scene map builds the low-dimension, anisotropic Σ-shaped ellipse; this scene shows that same contour-ring intuition failing once the dimension is high enough that mass concentrates on a shell instead.
