# 🎯 Averaging in SDXL's latent: a ghost, or a fused animal?
<!-- template: exempt. Reason: a one-sitting laptop walk, not a plan -->

> [!NOTE]
> **You will hold:** for a cat and a dog on four seeds, the picture you get by averaging their SDXL
> latents and decoding, beside the plain pixel average and beside PoE's own picture. It shows whether the
> fused animal comes from the latent or from the denoiser. · **Time:** ~1.5 h on the laptop GPU ·
> **Stop point:** step 4's one line per seed

Walk it with:

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/today-averaging-in-the-sdxl-latent.md
```

## Contents

- [Why this exists](#why-this-exists)
- [The test, written before any picture](#the-test-written-before-any-picture)
- [Steps](#steps)
- [What it cannot claim](#what-it-cannot-claim)

## Why this exists

Navigation: 📋 [Contents](#contents) | [Next](#the-test-written-before-any-picture) ➡️

**On the toy, the product's peak is the pixel average: a purple overlay**
([the one-pixel derivation](what-poe-loses-when-it-composes.md#why-the-overlay-is-purple-one-pixel-worked-by-hand)).
On SDXL, PoE of "a cat" and "a dog" gives one fused animal, not a ghost. Something between the two must
pull the average onto a plausible picture.

**Two candidates.**

- **The latent:** if SDXL's VAE latent were organised by meaning, averaging a cat's and a dog's latent would already decode as one animal.
- **The denoiser:** if the latent is close to a shrunken picture (4 channels at an eighth of the size), averaging it decodes as a ghost, and the fused animal comes from the denoiser's learned density, which gives ghosted pictures almost no probability.

The [write-up](what-poe-loses-when-it-composes.md#what-this-means-at-sdxls-scale) expects the second.

## The test, written before any picture

Navigation: ⬅️ [Previous](#why-this-exists) | 📋 [Contents](#contents) | [Next](#steps) ➡️

**Per seed, one row:**
1. "a cat";
2. 25%, 50% and 75% latent averages toward the dog, decoded;
3. "a dog";
4. the 50% pixel average;
5. PoE of "a cat" and "a dog";
6. the joint prompt "a cat and a dog".

**Your call on the 50% latent average:** `ghost` (two half-transparent animals, or one animal's outline
over the other's), `fused` (one coherent animal with features of both), or `other`.

| Result | Reading |
|---|---|
| the latent average is a ghost on at least 3 of 4 seeds, and PoE shows one fused animal | **the denoiser makes the fused animal**: as expected |
| the latent average is one fused animal on at least 3 of 4 seeds | **surprise: the latent is more meaningful than assumed**; the write-up's account needs correcting |
| PoE does not fuse on these seeds | **inconclusive**: no hybrid to explain on these seeds |

## Steps

Navigation: ⬅️ [Previous](#the-test-written-before-any-picture) | 📋 [Contents](#contents) | [Next](#what-it-cannot-claim) ➡️

Every command runs from `/home/molef/PhD/poe_repair_min-dog-and-rain` with
`PY=/home/molef/micromamba/envs/sdxl-local/bin/python`. The environment and SDXL's weights were set up on
2026-10-22.

### 1. 🔧 Render the cat, the dog, PoE and the joint prompt (AI, ~30 min)

- [ ] **Extend** `scripts/local_sdxl_poe.py` with a `--pair "a cat|a dog"` mode that renders A alone, B
  alone, PoE of A and B, and the joint prompt "A and B" on the same seed, saving each tile's final latent
  beside its picture as `.pt`.
- [ ] **Run** on seeds 1 to 4, 50 DDIM steps, guidance 7.5.
- [ ] **📸 You should see** four tiles per seed. PoE should look like one fused animal on most seeds,
  as on the cluster's sheets.

### 2. 🔧 Average the latents and decode (AI, ~5 min)

- [ ] **For each seed,** mix the cat's and the dog's final latents at 25%, 50% and 75% and decode each
  with the VAE in fp32; also average the two pictures' pixels at 50%.
- [ ] **Build one strip per seed** in the order of [the test](#the-test-written-before-any-picture).
  Saved as `artifacts/results/averaging-in-the-sdxl-latent/strip-seedNN.jpg`.

### 3. 👀 Call the 50% latent average, one seed per message (you, ~10 min)

- [ ] **The walk shows one strip per message.** Reply `ghost`, `fused` or `other`, and a few words on
  what you see.
- [ ] **Compare it with the pixel average beside it.** If the two look alike, the latent behaves like
  pixels.

### 4. ✍️ One line per seed and the reading (you with the AI, ~10 min)

- [ ] **Write one line per seed:** latent average, pixel average, PoE, joint.
- [ ] **Read them against the table above** and write the one-line reading for the write-up's
  "What this means at SDXL's scale".

## What it cannot claim

Navigation: ⬅️ [Previous](#steps) | 📋 [Contents](#contents)

**Final latents only.** Averaging clean latents is not the same as PoE's averaging of noise predictions
at every step. This tests what the latent does to an average, not how PoE moves through it.

**Four seeds, one pair.** A first look. A different pair (eagle and hawk, a butterfly and a meadow)
would show whether the reading holds for concepts that compose.
