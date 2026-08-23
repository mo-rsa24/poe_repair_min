# 💡 The text embedding of the residual

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. the LoRA is trained on the mono-minus-PoE difference | holds | `poe_repair/embeddings/cache_dataset.py:274` `pmi_target`, w·(ε_J + ε_∅ − ε_A − ε_B) |
| 2. that difference is noise | skipped, still open | |
| 3. PoE has one text embedding you can look at | skipped, still open | |
| **4 (current)**: the residual has a text embedding | **wrong as stated** | **r_t is state-specific, cross-seed cosine +0.002, so there is no single vector to embed; repaired wording below** |
| 5. a CLIP image-to-text inverter is the tool that recovers it | open | |
| 6. the interaction term is a missing third expert in the product, and a prompt can supply it | open | appended mid-walk, not in the opening cut |

Load-bearing: claim 4. The idea dies without it.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

The LoRA trained on 11 animal pairs learns a correction that lives in noise-prediction space, and
the wish is to see that correction as a text embedding, so it can sit on the same plot as the mono
prompt's embedding and the two PoE prompt embeddings.

**Where the walk is**

Claim 4, the load-bearing one, round 1. Claims 2 and 3 were skipped past and are still open.

**What compile would produce today**

Destination A, thinly. The load-bearing claim has a repaired wording that is a real optimization,
but nothing has been checked yet and claims 2, 3 and 5 are unwalked.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

Three quantities are conditioned on text and two of them have embeddings you can already read: the
joint prompt "a cat and a dog" has one, and the two solo prompts "a cat" and "a dog" have one each.
The third quantity, the correction the LoRA learns, is a difference of two noise predictions at one
timestep and one image state. The wish is for a text embedding of that third quantity, recovered
the way a CLIP inverter recovers a caption from an image.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. ✅ The LoRA is trained on the mono-minus-PoE difference

Mark: holds. `pmi_target` in `poe_repair/embeddings/cache_dataset.py:274` returns
w·(ε_J + ε_∅ − ε_A − ε_B), which the docstring states is the guided residual ε_J − ε_PoE modulo the
guidance algebra.

### 2. ⬜ That difference is noise

Mark: open.

### 3. ⬜ PoE has one text embedding you can look at

Mark: open.

### 4. ✏️ The residual has a text embedding   ← current

Mark: wrong as stated. Load-bearing.

Two things block the wording. The correction is an output of the UNet and a text embedding is an
input to it, so nothing maps backwards in closed form. And the correction is not one vector: the
same pair under two seeds gives cross-seed cosine +0.002, measured on the heldout cache, so what is
shared across states is a rule, not a direction.

**The repaired claim.** There may be a token embedding whose classifier-free-guidance direction,
ε(x_t, v, t) − ε(x_t, ∅, t), reproduces the correction on average over sampled states and
timesteps. That is a fit with a reported error, not a recovery.

- [ ] 4.1 why nothing maps backwards from r_t to a text embedding   ← current
- [ ] 4.2 why there is no single r_t to embed: the +0.002 cross-seed cosine
- [ ] 4.3 the repaired claim, and the guidance direction as the type-correct target
- [ ] 4.4 the three ways round it, and what each costs
- [ ] 4.5 the pretrained route: decode ε to an x̂_0 image, caption that with an off-the-shelf
      captioner, no training anywhere. Needs a full ε, so it reads the corrected prediction
      ε_PoE + λ·r_t against ε_PoE, never r_t alone

### 5. ⬜ A CLIP image-to-text inverter is the tool that recovers it

Mark: open.

### 6. ⬜ The interaction term is a missing third expert, and a prompt can supply it

Mark: open. Appended in the round on 4.3, not part of the opening cut.

Guided PoE is a sum of guidance directions: ε_PoE − ε_∅ = (ε̃_A − ε_∅) + (ε̃_B − ε_∅). Adding a third
expert `v` adds one more term, so the corrected prediction is ε_PoE + (ε̃_v − ε_∅). Setting that
added term equal to r_t is the same equation as claim 4's repaired fit. The two claims are one
statement written two ways, and this wording is the one that can be generated from and looked at.

- [ ] 6.1 the free version: a real English word as the third expert, no optimization anywhere
- [ ] 6.2 whether a fixed embedding can produce a state-dependent direction (it can; the UNet reads
      x_t, so claim 4.2's +0.002 does not block this)
- [ ] 6.3 the three-way comparison: the LoRA, a fitted `v`, and the best literal word, same pairs
      and seeds, one compose rate each
- [ ] 6.4 whether the project's parked text-space intervention is this, or something else

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| getting a text embedding out of something that is not text | textual inversion | optimize a token embedding so the model's noise prediction under it matches a target | confident on the method |
| the leftover the LoRA learns | the interaction term, r_t | what a product of two experts drops when it assumes the two concepts are independent | confident, the project's own term |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|
| 4 | invert r_t directly, treating it as the target a text embedding is read off | r_t is a UNet output and a text embedding is a UNet input; there is no closed-form reverse map, and r_t is state-specific so there is no single target to invert |
| 4 | match ε(x_t, v, t) to r_t itself | type-wrong. ε under any conditioning predicts the whole noise, not a correction. The quantity with the same type as a correction is the guidance direction ε(x_t, v, t) − ε(x_t, ∅, t) |

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 6 | is r_t even shaped like a prompt's guidance direction? Take "a cat" alone, compute ε̃_A − ε_∅ at matched steps for two seeds of the same pair, and take the cosine. Compare it against r_t's own cross-seed cosine of +0.002 | a real prompt's direction also near zero means the +0.002 says nothing about whether r_t is prompt-like, and the objection to claim 6 evaporates. A real prompt's direction well above zero, say +0.4, means r_t behaves unlike any prompt direction and one fixed token has a low ceiling. Cache only, no sampling |
| 4 | the readback control ladder. Decode ε_A to x̂_0 at a mid timestep, caption it, and see whether the caption says "a cat". Then ε_J, which should say "a cat and a dog" | a caption that names the right animal proves the readback pipeline works on a noise prediction at all, which is the prerequisite for pointing it at a corrected prediction. A caption that does not means the readback fails on the easy case and nothing downstream is worth running |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 1 | claim 4.5, the x̂_0 readback route | `scripts/xhat0_readback.py --pair a_cat__x__a_dog --seed 1 --steps 10 25 40` | failed | `load_cell()` takes no `split` argument; it finds the split itself from the pair and seed |
| 2 | claim 4.5, the x̂_0 readback route | the same command after the signature fix, on mscluster85, one RTX 3090 | done | the written pass criterion is met and means nothing. The `a` image ranks "a cat" top and the `b` image ranks "a dog" top at steps 25 and 40, but the `j` row is within 0.002 of the `a` row at every step, and the ceiling row fails outright: on the finished `mono.png`, CLIP ranks "a dog" last of the four captions. Output at `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/xhat0_readback/a_cat__x__a_dog__seed1/` |

## Sources

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| Textual inversion, Gal et al., *An Image is Worth One Word* | the machinery for finding a token embedding whose noise prediction matches a target | confident on the method, unverified that anyone pointed it at a score difference |
| Null-text inversion, Mokady et al., CVPR 2023, arXiv 2211.09794 | optimizes an embedding per timestep against a target prediction, which matches r_t being time-dependent | confident |
| ELITE, arXiv 2302.13848 | a feed-forward encoder from image to token embedding, no per-concept optimization | confident it exists, unverified whether it helps here |
| PEZ / CLIP-Interrogator | recovers actual words from a CLIP image embedding, the "CLIP inverter" of the original description | likely, verify before citing |
| `scripts/language_probes.py` in this repo | L1 and L3 already compute a text-space difference between the joint prompt and the sum of the two solo prompts | confident, read the file |
| `scripts/caption_readback.py` in this repo | already scores generated images against a caption bank including a blend caption, and records that whole-image CLIP was nulled for telling a blend from a composition | confident, read the file |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

A `/frame-hypothesis` prompt was emitted from claim 4.5, the x̂_0 readback route, covering the
four-rung ladder from "readback works on a noise prediction at all" up to the controls. The walk is
parked at claim 4.5 with claims 2, 3 and 5 unwalked and claim 4's optimization route unchecked.
