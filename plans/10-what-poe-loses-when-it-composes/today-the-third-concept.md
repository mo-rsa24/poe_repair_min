# 🎯 Today: does a third concept bring the wet camel back?
<!-- template: exempt. Reason: a one-afternoon walk built for a presentation, not a plan -->

> [!NOTE]
> **You will hold:** for a camel and a dog, on the same seeds and the same starting noise, what SDXL
> draws for the joint prompt, the wet animal asked directly, PoE of animal and rain, PoE of animal and
> "wet", and PoE of all three; your call on every tile; and a count of the seeds where the third concept
> brought the effect back. **Time:** ~2.5 h, mostly rendering while you do the
> [caption walk](today-camel-and-dog-in-the-training-data.md) · **Stop point:** step 6's table

Walk it with:

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md
```

## Contents

- [Why this exists](#why-this-exists)
- [The test, written before any render](#the-test-written-before-any-render)
- [Where things land](#where-things-land)
- [Steps](#steps)
- [What it cannot claim](#what-it-cannot-claim)

## Why this exists

Navigation: 📋 [Contents](#contents) | [Next](#the-test-written-before-any-render) ➡️

**The idea.** Where the joint prompt shows rain wetting an animal and PoE of the animal and rain loses
it, adding "wet" as a third expert might bring it back. If it does, the lost causality is cheap to
recover: one more condition, no training.

**What the support claim predicts.** A third expert is one more reweighting of the animal's own
pictures (write-up [§7.6](what-poe-loses-when-it-composes.md#76-the-third-concept-can-poe-get-the-effect-back)):

$$
\varepsilon \;=\; \varepsilon_\varnothing
+ w\,({\color{teal}\varepsilon_{\text{animal}}} - \varepsilon_\varnothing)
+ w\,({\color{steelblue}\varepsilon_{\text{rain}}} - \varepsilon_\varnothing)
+ w\,({\color{purple}\varepsilon_{\text{wet}}} - \varepsilon_\varnothing).
$$

So it should bring the effect back for the dog, whose pictures include wet dogs, and not for the camel,
whose pictures rarely if ever show a wet camel. **A wet camel from the three-concept product on 3 or
more counted seeds falsifies the support claim.**

## The test, written before any render

Navigation: ⬅️ [Previous](#why-this-exists) | 📋 [Contents](#contents) | [Next](#where-things-land) ➡️

**The seven columns**, every one from the same noise on each seed:

| # | Column | What it is |
|---|---|---|
| 1 | "a camel" | the animal alone |
| 2 | joint "a camel and rain" | one prompt naming both |
| 3 | "a camel in the rain" | the scene asked directly |
| 4 | "a wet camel" | the named effect asked directly |
| 5 | PoE of "a camel" and "rain" | two-concept PoE |
| 6 | PoE of "a camel" and "wet" | PoE with the effect word (the walk map's second test) |
| 7 | PoE of "a camel", "rain" and "wet" | the third concept |

The same seven for the dog.

**Your calls on every tile:** `e` the effect is on the animal (wet fur, droplets on it, standing in
water), `n` not, `h` half, `x` the animal is lost. A half counts as not.

**A seed counts** for the third-concept test only when column 2 or column 4 is `e` (the effect is
reachable through language on that seed) and column 5 is not `e` (two-concept PoE lost it). Only
counted seeds are judged, and the counted seeds come from your calls before anyone looks at column 7's
totals.

**Pass marks, set now:**

| Result | Reading |
|---|---|
| dog recovered on at least half its counted seeds, camel on at most 1 | **supported**: the third concept brings back an effect only where the animal's pictures hold it |
| camel recovered on 3 or more counted seeds | **falsified**: PoE drew an effect the camel's pictures rarely hold |
| dog recovered on fewer than half its counted seeds | **the third concept is not a remedy**, even with support |
| either animal has fewer than 2 counted seeds | **inconclusive** for that animal: too few lost effects to test |

## Where things land

Navigation: ⬅️ [Previous](#the-test-written-before-any-render) | 📋 [Contents](#contents) | [Next](#steps) ➡️

```
/home/molef/micromamba/envs/sdxl-local/            the laptop's torch and diffusers (built in the background)
~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/    SDXL's fp16 weights, ~7 GB
artifacts/results/does-an-effect-need-support/laptop-third-concept/            gitignored, local only
  camel/seedNN/<column>.png and .json             every tile with its settings
  camel-sheet.jpg  dog-sheet.jpg                  seeds down, the seven columns across
  calls.json  tally.json                          your calls; the counted-seed table
scripts/local_sdxl_poe.py                         render, sheet, tally
```

## Steps

Navigation: ⬅️ [Previous](#where-things-land) | 📋 [Contents](#contents) | [Next](#what-it-cannot-claim) ➡️

Every command runs from `/home/molef/PhD/poe_repair_min-dog-and-rain` with
`PY=/home/molef/micromamba/envs/sdxl-local/bin/python`.

### 1. 🔧 Check the laptop can render (AI, ~2 min once the setup has finished)

- [x] ✅ **Read the setup log:** `tail -4 /home/molef/datasets/sdxl-local-setup.log` ✓ verified 2026-10-22: `ENV OK 2.5.1+cu121 0.31.0 True NVIDIA GeForce RTX 4070 Laptop GPU`, `WEIGHTS OK`, `SETUP EXIT 0`
  - ✅ `ENV OK 2.5.1+cu121 0.31.0 True NVIDIA GeForce RTX 4070 Laptop GPU` and `WEIGHTS OK`, then `SETUP EXIT 0`.
  - ❌ No `ENV OK` means the torch install failed; the log's last lines say why.
- [ ] **Check the card is free:** `nvidia-smi` should show under 1 GB in use before rendering.

### 2. 🔧 Time one seed, then choose how many (AI, ~10 min)

- [ ] **Run one seed for the camel:** `$PY scripts/local_sdxl_poe.py render --animals camel --seeds 1`
  - Each tile prints its seconds. Seven tiles run 18 UNet passes per step in total, so one seed of one animal is about 900 passes.
  - **Pick the seed count** so the rest fits the time left: total ≈ (seconds for this seed) × 2 animals × seeds. Aim for 8 seeds, take no fewer than 5.
- [ ] **Open the seven camel tiles** in `laptop-third-concept/camel/seed01/` and check they are photographs of a camel, not noise or black squares (a black square means the fp16 VAE overflowed; the script decodes in fp32 to avoid it).

### 3. 🔧 Render the rest in the background (AI starts it; ~1 to 1.5 h)

- [ ] **Run:** `$PY scripts/local_sdxl_poe.py render --animals camel dog --seeds 1 2 3 4 5 6 7 8` (or the count chosen in step 2), in the background.
  - It skips tiles already made, so a stopped run restarts where it left off.
- [ ] **Walk the [caption walk](today-camel-and-dog-in-the-training-data.md) meanwhile.** The GPU renders here while the network and the CPU work there.

### 4. 🔧 Build the two sheets (AI, ~1 min)

- [ ] **Run:** `$PY scripts/local_sdxl_poe.py sheet`
  - `camel-sheet.jpg` and `dog-sheet.jpg`: seeds down the page, the seven columns across, numbered 1 to 7.

### 5. 👀 Call every tile, one sheet per message (you, ~20 min)

- [ ] **The walk shows you the camel sheet one seed row at a time**, then the dog's.
- [ ] **Reply with the row's seven calls in column order**, for example `seed03: n n n e n n e`.
- [ ] **The walk writes them into `calls.json`** under the animal, seed and column names, and never proposes a call.
- [ ] **The call is about the animal.** Rain falling around a dry camel is `n`.

### 6. 🔧 The counted-seed table and the verdict (AI with you, ~10 min)

- [ ] **Run:** `$PY scripts/local_sdxl_poe.py tally`
  - Per animal: the effect count in each column, the counted seeds, and how many of them column 7 recovered.
- [ ] **Read it against [the pass marks](#the-test-written-before-any-render)** and write one verdict line, with its numbers.
- [ ] **For the slide:** the two sheets, the table, and the verdict line.

## What it cannot claim

Navigation: ⬅️ [Previous](#steps) | 📋 [Contents](#contents)

**Fresh noise.** The cluster's sheets started each seed from a cached latent, so these tiles do not
match them seed for seed. The seven columns here do match each other.

**A different scheduler setup.** These use diffusers' DDIM at 50 steps and guidance 7.5, the same
settings as the cluster's sheets; small differences in implementation could still change single tiles.

**Few seeds.** 5 to 8 seeds per animal; a borderline verdict reruns with more on the cluster, in step 8
of the [causal route](route-causal-does-an-effect-need-support.md#8--the-third-concept-can-poe-get-the-effect-back).

**The sampler.** Every PoE miss on SDXL could be the sampler rather than missing support; the toy is
where that is separated (write-up §8).
