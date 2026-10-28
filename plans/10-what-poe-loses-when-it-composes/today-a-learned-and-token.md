# 🎯 One learned "<and>" token as a third expert: does it turn PoE's hybrids into a clear dog and a clear cat?
<!-- template: exempt. Reason: a one-run cluster test, not a plan -->

> [!NOTE]
> **You will hold:** the share of the gap between the [joint prompt](what-poe-loses-when-it-composes.md#21-the-objects) and plain PoE that one fitted token
> removes on held-out cat × dog (seeds 9 to 16, at steps 0, 10, 25, 40), and two sheets of
> PoE(animal A, animal B, "<and>") with a call per tile. · **Time:** three sittings of about 1 h, the
> training runs unattended on mscluster109 · **Stop point:** supported on both marks below, or under 10%
> removed at every step after 8,000 training steps

Walk it with:

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/today-a-learned-and-token.md
```

## Contents

- [Why this exists](#why-this-exists)
- [The test, written before any render](#the-test-written-before-any-render)
- [Steps](#steps)
- [What it cannot claim](#what-it-cannot-claim)

## Why this exists

Navigation: 📋 [Contents](#contents) | [Next](#the-test-written-before-any-render) ➡️

**Nine third experts have failed.** Text-embedding arithmetic and seven real phrases (the joint prompt
"dog and cat" included) draw a clear dog and a clear cat on 0 of 7 counted seeds, and two hybrids on 4 to 5
([claim 2 of the idea map](../../artifacts/ideas/the-and-vector-as-a-third-expert/IDEA_MAP.md)). The joint
prompt alone draws a clear pair on 7 of 8 seeds. The bar is a clear pair where the strongest third expert
there is gives two hybrids.

**This run fits the third expert instead of choosing it.** One new token `<and>` gets a trainable input
embedding row in each of SDXL's two text encoders, initialised from the word "and". Nothing else is
trained: not the UNet, not the rest of either text encoder, and the two animal prompts stay the plain
frozen prompts. The token is the same for every pair, and it is fitted over the curated pool of 182 cells
(48 pairs) in `artifacts/_shared/cross_pair_pool_configs/cells_v58.json`, read from the training cache. Cat
× dog is in no list, and seeds 9 to 16 are never trained on.

**The loss, per cached state.** ‖ε_<and>(x_t) − (ε_J + 2ε_∅ − ε_A − ε_B)‖², which makes the three-expert
product equal the guided joint at w = 7.5. Only ε_<and> carries gradient.

## The test, written before any render

Navigation: ⬅️ [Previous](#why-this-exists) | 📋 [Contents](#contents) | [Next](#steps) ➡️

**Two renders, both with the loop and noise of `scripts/and_vector_sheet.py`:**

1. PoE("dog", "cat", "<and>") on seeds 1 to 7.
2. PoE("a cat", "a dog", "<and>") on held-out seeds 9 to 16, beside plain PoE("a cat", "a dog") on the same
   seeds.

**Calls, from full-size tiles:** `2` a clear dog and a clear cat, `2h` two hybrids, `1` one animal, `x`
no animal. Two hybrids is a failure. On seeds 9 to 16 each render is judged directly, never by distance to
the joint render, because the joint reference is wrong on seeds 10, 13, 14 and 15.

| Result | Reading |
|---|---|
| `2` on at least 4 of the 7 seeds in render 1, and on more of seeds 9 to 16 than plain PoE in render 2 | **supported**: one fitted token carries what "and" adds |
| anything else | **not supported**, with the two counts reported as they fell |

**Stop rule.** At updates 0, 300, 1000, 2000, 4000 and 8000 the run logs the held-out share removed and a
sheet. If after 8,000 updates the share is under 10% at every one of steps 0, 10, 25 and 40, stop and
report: that is the result.

**Share removed** = 1 − ‖ε_<and> − (ε_J + 2ε_∅ − ε_A − ε_B)‖ / ‖ε_J + ε_∅ − ε_A − ε_B‖, one number per
cached step, on a_cat__x__a_dog seeds 9 to 16 at steps 0, 10, 25 and 40. The denominator is the gap
between the guided joint and plain PoE, divided by w.

## Steps

Navigation: ⬅️ [Previous](#the-test-written-before-any-render) | 📋 [Contents](#contents) | [Next](#what-it-cannot-claim) ➡️

Cluster runs use the worktree at `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
fast-forwarded to the committed branch. Every path on a launch line is absolute, outputs go to
`/datasets/mmolefe/poe_repair_min/outputs/learned_and_token/`, and the device is mscluster109 device 0
(RTX A6000, 48 GB) on the shared-device path, with a guard that aborts at more than 1 GB in use.

### 1. 🔧 The token (AI, done)

- [x] **Write** `scripts/learned_and_token.py` with `<and>` added to both tokenizers and a gradient-carrying
  encode path, pooled at the end-of-text position.
- [x] **Verify** that "<and>" encodes exactly as "and" does, and that the encode path matches the repo's.
  ✓ verified 2026-10-28 on mscluster109: difference 0 in embeds and pooled, both prompts.

### 2. 🔧 The pool loader and the starting row (AI, ~40 min)

- [x] **Load** the 182 cells from the training cache, asserting that none is cat × dog and none is a seed
  from 9 to 16. ✓ verified 2026-10-28: 182 cells over 48 pairs, all in the train split, assertions hold.
- [x] **Report** the held-out share removed at steps 0, 10, 25 and 40 with the untrained token.
  ✓ verified 2026-10-28, means over seeds 9 to 16: −0.142, −0.064, −0.086, −0.052. Log:
  `/datasets/mmolefe/poe_repair_min/outputs/learned_and_token/logs/baseline.log`.

### 3. 🔧 The trainer, 50 steps (AI, ~15 min)

- [x] **Run** 50 training steps with UNet gradient checkpointing and the 1 GB guard.
  ✓ ran 2026-10-28 on mscluster109 device 0, log `.../logs/smoke.log`.
- [x] **📸 You should see** the held-out share rise and the peak memory printed under 48 GB.
  ✓ confirmed by hand 2026-10-28 ("done", screenshot of the log): peak 7.4 GiB, held-out means rose at all
  four steps (−0.142, −0.064, −0.086, −0.052 to −0.116, −0.011, −0.024, −0.033); the per-update loss is
  noisy because each update draws a different pair and step.

### 4. 🔧 The first 2,000-update run (AI, done)

- [x] **Launch** under `nohup`, logging node, device and PID in the header, and report the held-out share
  removed every 500 steps. ✓ ran 2026-10-28, mscluster109 device 0, log `.../logs/run1.log`. Held-out mean
  share removed at steps 0, 10, 25, 40: −0.125, +0.012, −0.023, −0.037 at 500 updates; −0.070, +0.008,
  −0.018, −0.037 at 2,000.
- [x] **Apply** [the stop rule](#the-test-written-before-any-render). ✓ fired: every step's mean is under
  10% after 2,000 updates (best single step mean +0.012).

### 5. 🔧 The 8,000-update run with sheets (AI, about 65 min, unattended)

- [ ] **Launch** under `nohup` with W&B logging, sheets and held-out share at updates 0, 300, 1000, 2000,
  4000 and 8000. Each sheet has seeds 9 to 16 down the rows and, across, a dog alone, a cat alone, "a cat
  and a dog", PoE(a cat, a dog) and PoE(a cat, a dog, `<and>`).
- [ ] **Apply** [the stop rule](#the-test-written-before-any-render).

### 6. 🖼️ The card, the renders and the calls (AI, then you)

- [ ] **Copy** [the test](#the-test-written-before-any-render) into the run's results card before any tile
  is rendered.
- [ ] **Render** both sets and build the sheets as
  `artifacts/results/what-the-and-vector-is-made-of/learned-and-token-*.png`.
- [ ] **Call** each tile from full-size tiles, one seed per message.
- [ ] **List** the nearest real words to the fitted row in each encoder, by cosine over the vocabulary.

## What it cannot claim

Navigation: ⬅️ [Previous](#steps) | 📋 [Contents](#contents)

**One token, one pair held out.** A pass shows one fitted token composing cat and dog, not that it
transfers to another pair.

**Fitted on cached states, used on the states it visits.** The fit sees the cache's trajectory; the render
visits the states the three-expert product reaches.
