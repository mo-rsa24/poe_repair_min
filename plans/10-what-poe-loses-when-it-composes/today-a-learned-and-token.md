# 🎯 One learned "<and>" token as a third expert: does it turn PoE's hybrids into a clear dog and a clear cat?
<!-- template: exempt. Reason: a one-run cluster test, not a plan -->

> [!NOTE]
> **You will hold:** the share of the gap between the joint prompt and PoE's guided prediction that one
> fitted token removes (seeds 1 to 8, steps 0, 10, 25, 40), and a sheet of PoE("dog", "cat", "<and>") on
> seeds 1 to 8 with a call per tile. · **Time:** three sittings of about 1 h, the training itself runs
> unattended on mscluster109 · **Stop point:** supported on at least 4 of 7 counted seeds, or under 10%
> removed at every step after 2,000 training steps

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
prompt alone draws a clear pair on 7 of 8 seeds.

**This run fits the third expert instead of choosing it.** One new token `<and>` gets a trainable input
embedding row in each of SDXL's two text encoders, initialised from the word "and". Nothing else is
trained: not the UNet, not the rest of either text encoder, and "dog" and "cat" stay the plain frozen
prompts. The fit asks the three-expert product to match the guided joint prompt at states on the joint
prompt's own trajectory.

## The test, written before any render

Navigation: ⬅️ [Previous](#why-this-exists) | 📋 [Contents](#contents) | [Next](#steps) ➡️

**Render:** PoE("dog", "cat", "<and>") on seeds 1 to 8 with the loop and noise of
`scripts/and_vector_sheet.py`. Seeds 1 to 7 count; seed 8 is shown. Calls are made from full-size tiles:
`2` one clear dog and one clear cat, `2h` two animals with at least one a hybrid, `h` one hybrid, `1` one
plain animal, `x` no animal.

| Result | Reading |
|---|---|
| `2` on at least 4 of 7 counted seeds | **supported**: one fitted token carries what "and" adds |
| `2` on 0 to 3 of 7, including every `2h` | **not supported**: a fitted token also fails the strict mark |

**Stop rule.** After 2,000 training steps, if the share removed on seeds 1 to 8 is under 10% at every one
of steps 0, 10, 25 and 40, stop and report: that is the result and no render is made.

**Share removed** = 1 − ‖ε_joint_guided − ε_PoE3_guided‖ / ‖ε_joint_guided − ε_PoE2_guided‖, where
ε_joint_guided = ε_∅ + w(ε_joint − ε_∅), the PoE guided predictions sum the three (or two) experts'
guidance terms, w = 7.5, measured at the joint prompt's own DDIM states on seeds 1 to 8.

## Steps

Navigation: ⬅️ [Previous](#the-test-written-before-any-render) | 📋 [Contents](#contents) | [Next](#what-it-cannot-claim) ➡️

Cluster runs go through `scripts/cluster.sh`; the script path is absolute on the launch line, outputs go
to `/datasets/mmolefe/poe_repair_min/outputs/learned_and_token/`, and the device is mscluster109 device 0
(RTX A6000, 48 GB) on the shared-device path with a guard that aborts at more than 1 GB in use.

### 1. 🔧 The token (AI, ~20 min)

- [ ] **Write** `scripts/learned_and_token.py` with `<and>` added to both tokenizers and a gradient-carrying
  encode path, the two rows held in fp32 and written into the fp16 embedding tables.
- [ ] **Verify** that the prompt "<and>" encodes to exactly what "and" encodes to, in both encoders.
- [ ] **📸 You should see** the maximum absolute difference between the two encodings printed as 0.

### 2. 🔧 The state cache and the starting row (AI, ~40 min)

- [ ] **Run** the joint prompt's 50-step DDIM trajectory on the 29 seeds (the 21 training seeds and seeds 1
  to 8), keeping ε_∅, ε_dog, ε_cat and ε_joint at every state.
- [ ] **Report** the share removed on seeds 1 to 8 at steps 0, 10, 25 and 40 with the untrained token.

### 3. 🔧 The trainer, 50 steps (AI, ~15 min)

- [ ] **Run** 50 training steps with UNet gradient checkpointing and the 1 GB guard.
- [ ] **📸 You should see** the loss fall and the peak memory printed under 48 GB.

### 4. 🔧 The 2,000-step run (AI, hours, unattended)

- [ ] **Launch** under `nohup`, logging node, device and PID in the header, and report the share removed at
  the four steps at 500, 1,000 and 2,000 steps.
- [ ] **Apply** [the stop rule](#the-test-written-before-any-render).

### 5. 🖼️ The card, the render and the calls (AI, then you)

- [ ] **Copy** [the test](#the-test-written-before-any-render) into the run's results card before any tile
  is rendered.
- [ ] **Render** seeds 1 to 8 and build the sheet beside the earlier sheets in
  `artifacts/results/what-the-and-vector-is-made-of/`.
- [ ] **Call** each tile from the full-size contact sheet, one seed per message.
- [ ] **List** the nearest real words to the fitted row in each encoder, by cosine over the vocabulary.

## What it cannot claim

Navigation: ⬅️ [Previous](#steps) | 📋 [Contents](#contents)

**One token, one pair.** A pass shows one fitted token composing dog and cat on these seeds, not that it
transfers to another pair.

**Fitted on one set of states, used on another.** The fit sees the joint prompt's trajectory; the render
visits the states the three-expert product reaches.
