# 💡 The "and" vector as a third expert

Mode: `--prompt`. The prompt under check is a `/skill-route` prompt for two pieces of work: render
PoE of "dog", "cat" and a text-space difference vector x, then train a LoRA so that three-expert
product matches the joint prompt, on a curated co-occurrence dataset.

## Position in the idea

| Claim | Kind | Mark | Settled by |
|---|---|---|---|
| **1 (current)**: x = E("cat and dog") − E("cat") − E("dog") is a usable conditioning | premise | **wrong as stated** | **run 1: x_raw carries the start token at −1× (norm 1190 against about 15 for a word); x_fix zeroes it, but its pads keep 0.77 (CLIP ViT-L) and 0.49 (bigG) of the joint prompt's pad norm against a pass mark of 0.1, and its content positions are larger than the joint prompt's own** |
| 2. PoE("dog", "cat", x) draws a cat and a dog together | aim | **null for arithmetic x** (Claude's calls, open to your veto) | run 2's sheet: 7 counted seeds; PoE(d, c, x_fix) shows one cat and one dog on at most 1 (seed 3), PoE(d, c, x_aff) on 0, PoE(d, c, x_raw) is a blank grey tile on all 8 |
| 3. the LoRA's target is f("dog and cat") | aim | **wrong as stated, repaired** | the target is f(joint) only on seeds whose joint render shows one of each, kept by eye; the project already did this for the pooled LoRA (`target_quality.json` and `--exclude-cells` in `train_pooled.py`); "dog and cat" keeps 7 of 8, "a cat and a dog" 6 of 17 |
| 4. a curated co-occurrence dataset supplies that target | means | needs a check | captions are plentiful (17,396 of 32,776,409 in the 2-file slice name both); whether the pictures show both is unknown: a sample of 200 pictures called by eye, or the repo's GroundingDINO instance count, settles it |
| 5. both are simple and need compute | limit | open | |
| 6. `/skill-route` is the right first stop, with the two as separate work | means | open | |

Load-bearing: claim 2. The second piece only earns x as an input if x moves the picture.

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

Build the text-space difference between the joint prompt and its two halves, feed it to SDXL as one
more expert in the product, and see whether the product then draws both animals; then train a LoRA
so the three-expert product matches the joint prompt.

**Where the walk is**

Claim 1, round 1.

**What compile would produce today**

A corrected `/skill-route` prompt with claims 2 to 6 stated as assumptions.

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
- [Routes](#routes)
- [Sources](#sources)
- [Next step](#next-step)

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

SDXL's joint prompt is classifier-free guidance on one prompt against the empty prompt. PoE sums one
guidance direction per concept, ε = ε_∅ + w Σᵢ (εᵢ − ε_∅), which is what
`scripts/concept_multi_poe.py` computes over any number of prompt strings. The idea adds a third
condition that is no prompt at all: an embedding built by arithmetic on SDXL's two text encoders'
outputs, in the spirit of word-vector analogies (king − man + woman ≈ queen). If that embedding
carries what "and" adds, the product of dog, cat and x draws two animals where the product of dog
and cat draws one hybrid. The second piece trains a LoRA to close whatever gap remains.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. ✅ x = E(m) − E(d) − E(c) is a usable conditioning

Mark: wrong as stated.

`encode_prompt_sdxl` (`poe_repair/_sdxl/runtime.py:176-211`) returns a 77 × 2048 token sequence and
a 1280 pooled vector. Every prompt's position 0 is the same start token, and CLIP's causal mask means
its hidden state is identical in every prompt, so x's position 0 is minus the start token. The
padding positions depend on the prompt before them, so they land near −1× only as far as they are
shared, which run 1 shows they are not. Cross-attention reads every one of the 77 positions, so most
of x_raw is a negated copy of the prompt scaffolding.

x_fix = E(m) − E(d) − E(c) + E("") zeroes the start token exactly and leaves the pads at 0.49 to 0.77
of a prompt's pad norm (run 1). Its coefficients sum to 0, so it is a difference with no origin, and
it works as a condition only if an all-zero condition behaved like the empty prompt.

Repaired wording, from the score side: **x_aff = E(m) − E(d) − E(c) + 2E("")**. Its coefficients sum
to 1. If the UNet were affine in its condition, f(Σ aᵢEᵢ) = Σ aᵢ f(Eᵢ) whenever Σ aᵢ = 1, so
ε(x_aff) = ε_J − ε_A − ε_B + 2ε_∅, which is exactly the third expert that turns PoE(d, c) into the
guided joint. x_aff also keeps the start token at +1× (h − h − h + 2h = h), the scaffolding of a
real prompt. The UNet is not affine, so this is the best guess the arithmetic offers rather than a
guarantee.

What the repair does not fix: content positions do not line up. "a cat and a dog" puts "dog" at a
different position from "a dog", so the subtraction there is word-by-different-word.
`scripts/language_probes.py` names the same alignment artefact in its docstring.

Position 1 does not cancel either: m, d and c put "a" there and "" puts `<eot>`, so
x_fix[1] = H_∅[1] − H_a[1].

**Ways round the misaligned positions, strongest first.**

a. Optimise x instead of computing it (textual inversion against the cache). PoE(d, c, x) equals the
guided joint prediction exactly when ε(x_t, x) = ε_J + 2ε_∅ − ε_A − ε_B, and every term on the right
is already cached per step. Gives the x that does the job if one exists. Costs a UNet backward pass
per step, about LoRA-training memory, an estimated 1 to 3 GPU-hours per pair (an estimate, not
measured), and x is no longer text arithmetic.

b. One content slot. Mean the content positions of each prompt, take m − d − c + ∅ of the means, and
write that one vector into position 1 of E(""). Minutes on the laptop. Loses word order, and the slot
holds a vector no prompt ever produced at that position.

c. Pooled vector only. Keep E("") as the sequence and use p_m − p_d − p_c + p_∅ as the pooled input.
Has no positions to misalign. Tests only whether SDXL's global channel carries "two animals", which
may be a weak channel.

Fix a has three forms, by where x lives and what it matches: x at the encoder output matched to the
cached joint scores (above); x at the encoder output matched to real co-occurrence photos with the
unguided product's prediction ε_A + ε_B + ε_x − 2ε_∅; x as a new input word passed through CLIP, which
keeps a nearest-word readout. The cluster LoRA trains the same target as the first form, with the
UNet's cross-attention weights trainable instead of one input.

**The break.** Against arithmetic x: the 2026-08-11 language probes found the leftover direction
differs between pairs and its size does not track how big a correction a pair needs. Against any
fixed x: the target ε_J + 2ε_∅ − ε_A − ε_B is a sum of four UNet outputs, and the UNet is nonlinear in
its condition, so no single condition is guaranteed to produce it. Neither kills the claim. Both turn
it into a measurement: the share of the correction a fitted x removes on held-out seeds, beside the
LoRA's share. The cross-seed cosine of +0.002 is not a break, since the joint prompt is one fixed
condition whose output also differs per seed.

- [ ] 1.1 the content-position mismatch: does it matter, or does cross-attention not care? The sheet's x_fix column answers this directly

### 2. ⬜ PoE("dog", "cat", x) draws a cat and a dog together

Mark: needs a check (load-bearing).

**The test, written 2026-10-28 before any tile rendered.** `scripts/and_vector_sheet.py`, prompts
"dog", "cat", "dog and cat", seeds 1 to 8, eight columns from the same noise per seed: dog alone,
cat alone, joint, PoE(d, c), x_raw alone, x_fix alone, PoE(d, c, x_raw), PoE(d, c, x_fix).

Calls on every tile: `2` one cat and one dog, `h` one hybrid animal, `1` a single plain animal, `x`
no animal. A seed counts when the joint is `2` and PoE(d, c) is not.

| Result | Reading |
|---|---|
| PoE(d, c, x_fix) is `2` on at least half the counted seeds | supported: the arithmetic vector carries what "and" adds |
| PoE(d, c, x_fix) is `2` on 0 or 1 counted seeds | null for arithmetic x; fix a (learn x) is the route |
| anything between | inconclusive at 8 seeds |

The prediction from run 1's norms: both "x alone" columns are `x` on most seeds, and the three-expert
columns look less like animals than PoE(d, c), because x is larger than a real prompt at every
position.

**Calls, made by Claude from the sheet on 2026-10-28, open to your veto.** Joint: one cat and one
dog on seeds 1 to 7, two dogs on seed 8. PoE(d, c): one hybrid on seeds 1 to 3 and 5 to 8, a
dog-print wallpaper on seed 4. Counted seeds: 1 to 7. x_raw alone and PoE(d, c, x_raw): a flat grey
tile on all 8. x_fix alone and x_aff alone: wallpaper or texture on all 8, no animal, as predicted.
PoE(d, c, x_fix): a kitten and a puppy on seed 3, wallpaper on seeds 1, 4 and 8, one hybrid on the
rest. PoE(d, c, x_aff): no seed with one of each. PoE(d, c, "single animal"): one animal on all 8,
matching the screenshot's claim that it changes nothing. PoE(d, c, "two animals"): a dog and a cat
side by side on seed 4, two animals on seeds 2 and 8, one on the rest. Reading: null for arithmetic
x, 1 of 7 at best against a support mark of 4.

**The phrase search, written 2026-10-28 before any of its tiles rendered.** `--set phrases`:
PoE of dog, cat and one of "a pair of animals", "side by side", "together", "a dog next to a cat",
"two different animals", the joint prompt itself, and x_fix at 0.5 and 0.25 of its guidance weight,
beside the references and the existing "two animals" column. Same 8 seeds, same noise, same device
model (Quadro RTX 8000). Counted seeds stay 1 to 7. Calls as before, plus `2h`: two animals, at
least one a hybrid (seed 3's x_fix tile is `2h`).

| Result | Reading |
|---|---|
| any phrase is `2` on at least 4 of 7 counted seeds | supported: a phrase supplies what "and" adds; that phrase leads the next round |
| the best phrase is `2` on 2 or 3 | a direction: more seeds and a second pair before claiming anything |
| every phrase is `2` on 0 or 1 | null for phrases at w = 7.5; the plurality on seeds 2, 3, 4 and 8 is noise until a larger sheet says otherwise |
| x_fix at 0.5 or 0.25 is `2` where x_fix at 1 was `2h` | turning x down lets the dog and cat experts keep their identities: the oversize reading of run 1 holds |

**Calls on runs 4 and 5.** Your calls, 2026-10-28, on the two columns that looked best:
"a dog next to a cat" draws one clear cat and one clear dog on **no** seed; two animals, each a
hybrid, on seeds 1, 3, 4, 6, 7 and 8 (5 of the 7 counted). The joint prompt "dog and cat" as the
third expert likewise draws a clear pair on no seed; two hybrid animals on seeds 1, 2, 3, 6 and 8
(4 of 7). Claude's first calls had both at 4 to 5 clear pairs from the downscaled sheet, which
overcalled; Claude's calls on the other columns (at most 1 clear pair each) are unchecked and lean
the same way. The weight sweep, Claude's calls: at λ 0.1 to 0.5 every seed returns PoE's own
hybrid; seed 3's two blended animals appear only at 0.75 and 1, the weights that turn seeds 1, 4
and 8 to wallpaper and print "CAT" on seed 7.

Reading: every third expert fails the strict mark, the joint prompt included. A third expert that
names both animals restores the count (two animals on 4 or 5 of 7) but never the identities: each
animal is itself a cat-dog. Since the joint prompt alone draws a clear pair on 7 of 8 seeds and
the same prompt inside the product does not, the fusion comes from the product of the "dog" and
"cat" experts, not from a missing "and" term that a third condition could supply.

**Separation steering (run 7), your calls from the sheet, 2026-10-28.** Pass mark written before
rendering: one cat and one dog on at least 4 of the 7 counted seeds at some step size. Result: 0 of 7
with no steering, 0 of 7 at each of three step sizes, plus one borderline tile (large step, seed 6,
two small animals on a background of saturated blocks). Small steps keep the photograph and give one
animal, a wallpaper of many animals or a dog-cat hybrid; medium and large steps turn every seed into
flat graphics or colour blocks. At steps 0 to 9 the two experts' attention maps are nearly flat
(largest value 0.0012 against 0.00098 for a flat map), and the two "1 minus largest value" terms
supply about 98% of each update (gradient norm 4e-5 to 2e-4 against 1e-6 to 8e-6 for the overlap
term), so the step sharpens each map and barely separates them. Reading: not supported for this
loss. It does not test steering on the overlap term alone, which the request fixed out.

**The break against a learned x.** A third expert adds a term and changes nothing else: ε_A and ε_B
are computed in their own passes, so if the two experts' cross-attention already fuses the animals
into one blob, x cannot move them; it has to cancel the fused signal by itself, and its error is
multiplied by w = 7.5. It would also be fitted on cached states and used on the states its own
product visits. Not fatal, because the LoRA faces the same mismatch and composes on 7 of 8 seeds, but
it turns the claim into a measurement: the share of r_t a fitted x removes on held-out seeds, beside
the LoRA's share on the same seeds.

### 3. ✅ The LoRA's target is f("dog and cat")

Mark: wrong as stated, repaired.

The joint prompt fails on some seeds (1 of 8 for "dog and cat", 11 of 17 for "a cat and a dog"),
and a LoRA trained to match it learns those failures too. Repaired wording: **the target is the
joint prompt's prediction on seeds whose joint render shows one cat and one dog, selected by eye
before training.** The project has done this curation once: the pooled LoRA's targets were read
into `/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json`
(field `both_present`, confirmed by eye) and dropped with `--exclude-cells` in
`poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`
([scope 08's master plan](../../../plans/08-improve-the-rank-32-pooled-adapter/MASTER_PLAN.md)).

What the repair commits to. Selection keeps the noises on which the joint prompt composes easily,
so the LoRA learns the correction only there; evaluation has to include held-out seeds of both
kinds and judge the render itself, not its distance to the joint render. At 7 of 8 kept, 32
rendered seeds give about 28 training seeds for "dog and cat".

### 4. ⬜ A curated co-occurrence dataset supplies that target

Mark: needs a check.

**The caption count, 2026-10-28.** In the laptop slice (2 of the mirror's 128 files, 32,776,409
captions), 167,778 captions name a cat, 233,938 a dog and 17,396 both, whole words, case-insensitive
(`artifacts/results/what-the-and-vector-is-made-of/cat-and-dog-captions.json`, with 25 random
examples). Scaled by 64 to the full mirror that is about 1.1 million, an estimate. The random
examples are mostly product listings ("flea spray for cats and dogs", "pet memorial urn small to
large dog cat"); two of the first twelve plausibly picture both animals. So the caption count is an
upper bound on the dataset, and the check is on the pictures: fetch 200 at random, call each one cat
and one dog, one animal, a product, or no animal, and scale.

What the claim commits to beyond the count. Training on photos means the denoising loss on
PoE(d, c, x)'s own prediction against real pictures, a different signal from the cluster LoRA's
match to SDXL's joint prediction; claim 3's curated joint target and this photo target are two
alternatives, not one.

### 5. ⬜ Both pieces are simple and need compute

Mark: open.

### 6. ⬜ `/skill-route` is the right first stop, with the two as separate work

Mark: open.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| Word | What it means | Confidence |
|---|---|---|
| word-vector analogy | king − man + woman ≈ queen in word2vec, which holds because that space was trained so co-occurrence ratios become vector differences | confident |
| textual inversion | optimizing a token embedding so the model's conditioned output matches a target; the fallback when arithmetic x fails, and the shape of "x such that PoE(d, c, x) = f(m)" | confident |
| additivity gap | how far E(joint) sits from E(a) + E(b); this repo measured it on 75 pairs (2026-08-11) | confident |
| binding direction | E(joint) minus the two solo embeddings, the same object as x | confident |
| negative / compositional prompt weighting | adding and subtracting conditions inside guidance (Composable Diffusion, Liu et al. 2022) | confident |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

None yet.

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

None yet.

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 1 | `scripts/and_vector_norms.py --joint "a cat and a dog" --a "a cat" --b "a dog"`, text encoders only, laptop CPU, fp16 weights upcast to fp32. Writes `artifacts/results/what-the-and-vector-is-made-of/a-cat-and-a-dog-norms.json` (77 norms per encoder slice for x_raw, x_fix and E(m), plus the three pooled norms) and a bar chart beside it: y is the norm, x is token position 0 to 76, one bar group per position for x_raw, x_fix and E(m) | x_fix[0] not 0.000 means the code is wrong. x_fix pad positions (7 to 76) under a tenth of E(m)'s pad norm means the repair works and x lives in positions 1 to 6. x_fix pads near E(m)'s pad norm means the pads carry the prompt, the repair fixes position 0 only, and fix a is the route |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 1 | claim 1, the norm check | `scripts/and_vector_norms.py --joint "a cat and a dog" --a "a cat" --b "a dog"` on mscluster84, fp32, sent through `cluster.sh sh` without a sync because the laptop worktree held other sessions' uncommitted edits | done | x_fix[0] = 0.000 on both encoders, so the code is right. Pad median x_fix over joint: 0.766 (CLIP ViT-L), 0.487 (bigG), so the repair fails its 0.1 mark and the pads carry the prompt. Content positions 1 to 6: x_fix 20 to 29 against the joint's 13 to 20 (ViT-L), 34 to 59 against 17 to 36 (bigG). Pooled: x_fix 44.6, joint 38.7. Files: [the chart](../../results/what-the-and-vector-is-made-of/a-cat-and-a-dog-norms.png) and its JSON beside it |
| 2 | claim 2, the sheet | `scripts/and_vector_sheet.py --render --a dog --b cat --joint "dog and cat" --seeds 1 2 3 4 5 6 7 8` via `scripts/and_vector_sheet.sh` on mscluster106 device 1, outside Slurm, launched 2026-10-27 23:52 cluster time (pid 943209); then the norm check on the same prompts | done | 64 tiles, eight columns; ended 00:17. Norm check on "dog and cat": x_fix pads at 0.572 (CLIP ViT-L) and 0.555 (bigG) of the joint's. Log `/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/and_vector/logs/sheet.log` |
| 3 | claim 1 angle 2 (x_aff) and the screenshot's phrases | the same launcher, queued to start when run 2's launcher exits (pid 945088); adds x_aff alone, PoE(d, c, x_aff), PoE(d, c, "single animal"), PoE(d, c, "two animals") | done | 32 more tiles; ended 00:34. Sheet: [dog-and-cat-sheet.png](../../results/what-the-and-vector-is-made-of/dog-and-cat-sheet.png). Log `.../and_vector/logs/sheet-pass2.log` |
| 4 | claim 2, the phrase search | `SET=phrases` through `scripts/and_vector_sheet.sh` on mscluster106 device 1, launched 2026-10-28 (pid 954505); 8 new columns × 8 seeds | done, 01:31; sheet `dog-and-cat-phrases-sheet.png` | log `/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/and_vector/logs/phrases.log` |
| 5 | claim 2, x_fix at rising weight | `SET=sweep`, queued on mscluster106 device 1 behind run 4 (pid 956073); PoE of dog, cat and λ·x_fix, λ = 0.1, 0.25, 0.5, 0.75, 1; 16 new tiles, the rest reused | queued | log `.../and_vector/logs/sweep.log`; read against run 4's last pass-mark row |
| 6 | claim 3, joint seeds to pass | `SET=curate`, seeds 1 to 32, mscluster109 device 0 (RTX A6000; seeds 9 to 32 new) | done; contact sheet `joint-seeds-to-pass.png`, calls `joint-seed-calls.json`: 28 pass, 2 fail (8, 14), 2 for you (13, 22) |
| 7 | claim 2, separation steering | `scripts/separation_steering.py` through `scripts/separation_steering.sh` on mscluster109 device 1, outside Slurm, 2026-10-28 01:22 to 01:32 cluster time; gradient step on the latent at steps 0 to 9, step sizes 2.5e4, 1.2e5, 5e5, seeds 1 to 8, with a no-steering column from the same script | done | [the sheet](../../results/what-the-and-vector-is-made-of/separation-steering-sheet.png), card entry in the [results README](../../results/what-the-and-vector-is-made-of/README.md). One cat and one dog on 0 of 7 counted seeds at every step size, plus 1 borderline tile; the mark was 4 of 7. 18.6 s per steered tile against 14.8 s plain. [Retrospective](../../notes/session-merges/2026-10-28-separation-steering-retro.md) |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Route | Why | State |
|---|---|---|---|
| 1 | `/submerge` journey on why text-embedding arithmetic fails to compose in SDXL | the walk measured the failure (runs 1 and 2) but the theory behind it (a causal text encoder's start token and padding, the UNet's non-affine response through cross-attention, the guided sum's amplification) is not built anywhere in the learning tree | emitted 2026-10-28, not run |
| 2 | `/drip-execute-plan --request`: separation steering on mscluster109 | the attention-side test of whether anything that names no animal pair separates the identities | done 2026-10-28, folded in as run 7 |
| 3 | `/drip-execute-plan --request`: a learned third condition, `<and>` | the condition-side test: can a token that names neither animal turn PoE's two hybrids into a clear pair, when the joint prompt itself as a third expert cannot | emitted 2026-10-28, not run |

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**The repo's own prior work on the same object.** The L1 additivity gap and L3 binding direction ran
on 75 pairs and both came back null: the gap does not predict how big a correction a pair needs, and
the leftover direction is not shared across pairs. Verdict in
[the three-sides review](../../../plans/03-does-the-correction-cause-composition/review/06-the-same-story-from-three-sides.md).
That tested x as a predictor. Nobody has generated from x.

**The earlier walk on the residual's text embedding.** Its claim 6, "the interaction term is a
missing third expert, and a prompt can supply it", is this idea's claim 2, left open.
[That walk's map](../text-embedding-of-the-residual/IDEA_MAP.md).

**The scoped search on claim 1 (2026-10-28).** No one found who learns a condition so that a product
of experts matches the joint prompt. Neighbours: Compositional Inversion (arXiv 2312.08048) finds
learned tokens drift out of distribution and fail to compose, and anchors them near real concepts;
Semantic Surgery (arXiv 2510.22851) subtracts concept offsets from prompt embeddings and reports that
summing several offsets subtracts the shared part twice, avoided by encoding the concepts together;
Compel's `.and()` is the prompt-level conjunction, separately diffused prompts summed, which is this
repo's PoE. Read from search summaries, not the papers: likely, verify before citing.

**The third-concept sheet.** The same three-expert PoE with "wet" as the third word, on the camel and
dog. [Its plan](../../../plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md).

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Settle claim 1, then walk claim 2.
