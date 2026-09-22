# 📋 Review: the same rule on four models

What the renders for [the figure plan](../plans/figures/12-the-same-rule-on-four-models.md)
showed. The expectations are in the plan and are not changed here.

## Prompts

Pinned before any render (task 1.1).

| Pair | Concept prompts | Joint prompt | Source |
|---|---|---|---|
| butterfly × flower meadow | "a butterfly", "a flower meadow" | "a butterfly and a flower meadow" | this project's existing pair `a_butterfly__x__a_flower_meadow` |
| camel × forest | "a camel", "a forest" | "a camel and a forest" | the GLIDE conjunction example in the README of Liu et al.'s released code (`image_sample_compose_glide.py --prompts "a camel" "a forest"`); not in the paper's text |
| cat × dog | "a cat", "a dog" | "a cat and a dog" | this project's existing pair `a_cat__x__a_dog` |

## Settings per model

| Model | Model id | Resolution | Sampler | Steps | Guidance |
|---|---|---|---|---|---|
| SD 1.4 | `CompVis/stable-diffusion-v1-4` | 512 | DDIM | 50 | 7.5 |
| SD 2.1 | `Manojb/stable-diffusion-2-1-base`, a mirror: `stabilityai/stable-diffusion-2-1-base` returns 404 on the hub (2026-09-22) | 512 | DDIM | 50 | 7.5 |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | 1024 | DDIM | 50 | 7.5 |
| SD 3.5 | `stabilityai/stable-diffusion-3.5-medium` | 1024 | flow-matching Euler, rule on velocities | 40 | 4.5 |

## Seed 42, cell by cell

Not yet rendered.

| Pair | Model | Joint prompt showed | PoE showed | Against the expectation |
|---|---|---|---|---|

## Still open

- The SD 2.1 column is a third-party mirror of the base weights, not Stability's own repository, which is gone.
- Whether a caption may call the camel row Liu et al.'s example: it is from their code's README, not their paper.
