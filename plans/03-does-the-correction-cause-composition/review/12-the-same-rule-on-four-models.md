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

Read by eye at full size, 2026-09-22. SD 1.4, SD 2.1 and SDXL are job 58176 on `co3` (diffusers 0.29.2). SD 3.5 is job 58190, on `co3` with diffusers 0.32.2 and sentencepiece placed ahead of it on `PYTHONPATH` from `/datasets/mmolefe/poe_repair_min/pylib/sd35`, because SD 3.5 medium's transformer does not load on 0.29.2.

| Pair | Model | Joint prompt showed | PoE showed | Against the expectation |
|---|---|---|---|---|
| butterfly × flower meadow | SD 1.4 | one small butterfly in a meadow | one large butterfly among flowers | as expected on both sides |
| butterfly × flower meadow | SD 2.1 | a butterfly in a meadow | a butterfly shape built out of flowers | PoE fuses the two rather than placing the butterfly in the meadow |
| camel × forest | SD 1.4 | a camel among forest trees | a camel standing in a forest | as expected on both sides |
| camel × forest | SD 2.1 | a camel at a forest edge | a camel with a clump of foliage on its back, trees faint in fog | PoE grafts the forest onto the camel |
| cat × dog | SD 1.4 | one cat, no dog | one white animal with a cat's head | the joint prompt fails too: one animal |
| cat × dog | SD 2.1 | two dogs, no cat | one cat with a collar and long legs | the joint prompt shows two animals but not a cat and a dog; PoE gives one animal |
| butterfly × flower meadow | SDXL | several butterflies over a meadow | several butterflies over a meadow | as expected on both sides |
| camel × forest | SDXL | a camel among bare trees | a camel with green mossy fur in a misty forest | both concepts present; the forest's colour bleeds into the camel |
| cat × dog | SDXL | a cat beside a dog | one white animal with a dog's body and a cat-like face | as expected; the same hybrid as the current Figure 2, so this column reproduces the paper's seed 42 |
| butterfly × flower meadow | SD 3.5 | one butterfly on a flower meadow | one butterfly on a flower meadow | as expected on both sides |
| camel × forest | SD 3.5 | a camel in a clearing with forest behind | a camel on a forest path | as expected on both sides |
| cat × dog | SD 3.5 | a tabby cat lying beside a puppy | one tabby cat with a dog's ear and a half-dog face | as expected |

## What the grid shows against the expectation

PoE gives one hybrid for cat × dog on all four models, which is the expectation. Two things were not
expected. The joint prompt fails on cat × dog on the two older models (SD 1.4 draws one cat, SD 2.1
draws two dogs), so joint prompting is a working reference only on SDXL and SD 3.5. And PoE fuses
the two concepts on two cells of the pairs expected to compose: SD 2.1 builds the butterfly out of
flowers and grafts foliage onto the camel. Seed 42 was fixed before rendering and is shown as it
came out.

## Still open

- The SD 2.1 column is a third-party mirror of the base weights, not Stability's own repository, which is gone.
- Whether a caption may call the camel row Liu et al.'s example: it is from their code's README, not their paper.
