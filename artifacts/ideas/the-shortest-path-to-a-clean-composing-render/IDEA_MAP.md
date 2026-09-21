# 💡 The shortest path to a clean composing render

The walk was re-cut once. The first cut asked which checkpoint to use and why training degrades it.
That question is set aside: checkpoint 40,050 of `phase1_r32_100k` is the working adapter, fixed,
and the walk now asks what to stack on top of it.

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. 40k gets the arrangement right and the detail wrong | wrong as stated | the mushy render was the cheap tracking render the trainer makes as it goes. On the full fifty-step render at correction strength 1.2 the same seed is a crisp tabby cat beside a clean white labrador, close to the joint-prompt target |
| 2. the clean-estimate (x0) loss is better | needs a check: its own 30k and 40k renders, judged on the full sampler | at 20k the same seed composes but has left photography: a grey pencil sketch inside a drawn circular frame, and the two animals both read as dogs |
| 3. the remaining loss variations are worth trying as separate objectives | wrong as stated: they are switches on one trainer, and all of them already exist as flags | V1 and V0a fix an invisible-shift defect in the objective; V6 changes what the target picture is, which is the only one aimed at quality |
| 4. a cleaner pool, or training on certain steps only, fixes the softness | needs a check, and one half is already answered | the same sheet's second row applies the adapter on the first half of the run only, and the sketch look stays |
| 5. the composition is solved, and the detail can be won back by training, by selection, or by both | holds on the selection half | all eight nearby starting points drew a cat and a dog and several were clean; the picker chose one of the worst. The good picture is in the batch and nothing we own can find it | the adapter is the only thing that ever produced two animals; selection is the only thing that ever sharpened them, which says selection helps and not that training cannot |

| 6. **load-bearing.** every loss comparison so far was made on pairs that do not resemble the test pair, and on targets that are sometimes wrong | needs a check: the v57 run, already designed and unlaunched in scope 08 | all eleven training pairs are near-identical species; cat x dog is neither, and is held out. Appended in round 2 of claim 2, not part of the original cut |

Load-bearing: claim 6, which displaced claim 5. Both routes to the detail are live, training and
selection, and they are not exclusive. What decides between them is where the blur actually sits: in what the adapter learned,
or in which of the pictures it can draw we happened to keep.

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

## Quick context: where you are

**What the idea is**

Two objectives: an adapter that draws two co-occurring concepts, and an inference process that
draws them cleanly. The first is done. The walk is about the second, and about whether the pieces
already proven separately can be combined into one render as good as what the joint prompt gives.

**Where the walk is**

Claim 1 is current, settled by looking at the renders.

## The idea, as it stands

Checkpoint 40,050 of the rank-32 pooled adapter turns the product of "a cat" and "a dog" from one
fused animal into a cat and a dog. What it does not do is draw them well. The question is what to
add so the result matches what the single joint prompt "a cat and a dog" produces on its own.

## The claims

### 1. ✅ 40k gets the arrangement right and the detail wrong

Mark: holds.

**What it gets right.** Two animals rather than one. Both sitting, apart, facing the camera, on the
same grey studio backdrop the target uses. A cat on the left and a dog on the right, in the target's
own arrangement. It stays a photograph.

**What it gets wrong.** The cat's face has almost no features. The dog is a squat bulldog where the
target is a labrador. Everything is furrier and flatter than the target, and a stray object sits at
the bottom of the frame. It has the composition and not the picture.

### 2. ❌ The clean-estimate (x0) loss is better

Mark: needs a check. Two reasons the earlier reading was not a verdict. The run was two thirds
through, and the pictures being read were the cheap previews the trainer makes as it goes, which is
the same mistake that made the baseline look mushy in claim 1. What can be said is that at 10k and
20k the previews are grey and drawn rather than photographic. What cannot be said is whether that
survives to the end or survives a proper fifty-step render.

The named check: when it finishes, render its 30k and 40k checkpoints through the same fifty-step
path the baseline's good render came from, and compare like with like.

**The mechanism.** The clean-estimate weighting puts most of its weight on the earliest steps, which
is where the arrangement is decided. The detail is decided late. The correction itself also spends
most of its effort late. So the loss is weighted away from where both the correction and the
picture quality live.

### 3. ⚠️ The remaining loss variations are worth trying

Mark: needs a check, and they do not all aim at the same thing. V1 and V0a remove or price a shift
the objective cannot see but the sampler can. That is a correctness repair, not a quality one. V6,
training against renders that already composed instead of against the joint prompt's own output, is
the only variation pointed at picture quality.

### 4. ⚠️ A cleaner pool, or training only certain steps, fixes the softness

Mark: needs a check, half of it already answered. On the x0 run's own sheet, applying the adapter
only over the first half of the run leaves the sketch look untouched, so limiting where the adapter
acts does not by itself restore the photograph.

### 5. ⚠️ The composition is solved, and the detail can be won back by training, by selection, or by both

Mark: needs a check.

The named check: render the same seed from eight slightly different starting points with the
adapter attached, and look at all eight. If one is sharp, the adapter can already draw the picture
and the loss is in the picking. If none is, the blur is in what the adapter learned and the route
is a different training target. The two answers point at different work, which is why this runs
before either.

### 6. ⚠️ Every loss comparison so far was made on pairs that do not resemble the test pair

Mark: needs a check. Load-bearing. Appended mid-walk.

All eleven training pairs are two things that already look nearly the same: wolf and husky, lion and
tiger, crocodile and alligator, rabbit and hare, crow and raven. The adapter has never been trained
on two visibly different creatures. It is then asked for a cat and a dog. A blurred cat face beside
a dog of the wrong breed is what a model taught to draw near-copies would produce, and the same
signature is already on record elsewhere: the held-out seal and walrus renders as two seals, the
walrus having lost its tusks, and the scorer called it a perfect composition.

**Tier 0, free.** Six of the eight held-out pairs are look-alikes and two are genuinely different.
All are rendered at every checkpoint under `pooled_lora/phase1_r8_100k/samples/per_epoch/`. If the
distinct pairs are mushy where the look-alikes are crisp, the claim holds with no training at all.

**Tier 1, the real run.** The unchanged loss on the v57 pool, twenty-nine pairs of genuinely
different things, all already in the training cache. `--cells` replaces the training set and leaves
the held-out side alone.

## What the words are

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| plurality, co-occurrence | co-occurrence versus intersection | the product concentrates on one thing that is both concepts at once; the wanted picture has both present separately | confident, the project's own reframe |
| measuring the error between clean pictures | x0-parameterised loss | the same error rewritten so it is compared between denoised estimates instead of between noises, which reweights the steps | confident |
| draw several and keep the best | inference-time search with a verifier | change nothing in the model, sample several candidates, keep the one a scorer prefers | confident |

## Held claims

None.

## Dead ends

**Real photographs as the training target (V5).** Recorded in the
[correction-loss idea map](../designing-the-correction-loss/IDEA_MAP.md).

**Charging the adapter for the size of its correction.** It shrank the correction everywhere at
once and moved the render back toward the fused animal.

## Checks outstanding

- The x0 run's verdict, once it reaches the end. It was past two thirds when last read. It trains
  on every step, so the weight cap does not neuter it the way it would an arm restricted to the
  early window.
- Whether the contrast term survives being read off the frozen-null composition rather than the
  guided one.
- Whether V6's filtered render pool is large enough to train on.

## Runs

| Run | What it is | State |
|---|---|---|
| `phase1_r32_x0loss_40k`, W&B `b1xvsv29` | the clean-estimate loss | running, past two thirds; read mid-run at 20k |
| `zo_adapter_40k_seed9_lam1p2`, mscluster109 device 1 | eight nearby starting points on seed 9, adapter at full strength, 40k checkpoint | done. Every candidate composes, several are clean, the picker chose a poor one |
| `zo_adapter_40k_seed9_lam0p5`, mscluster107 device 1 | the same at half strength | done. Every column is one animal: half strength loses the composition on this seed |

Both anchored to claim 1. They answer where the blur lives: in what the adapter learned, or in
which of the pictures it can draw we happened to keep. Every candidate is saved separately, so the
read is by eye and does not depend on the scorer.

## What the first cohort settled

Six experiments, one switch each, on the curated pool over the first ten denoising steps, ten
thousand steps apiece.

**The clean-estimate loss with its cap lifted is the only one that works.** It renders
photographic, on the target's own backdrop, with several separate animals and no invented people.
Its remaining failure is identity: on one held-out seed it draws a cat and a dog, on the other it
draws three dogs. It is the parent from here on.

**The plain loss fails on this pool, and gets worse with training.** At three thousand steps it
draws several animals in a garden. By ten thousand it invents people, which is the catalogued
broad-pool failure arriving with run length.

**Adapting only the empty branch is dead.** At every checkpoint and on both held-out seeds it draws
the same flat clipart of one animal in a decorative frame. Against the plain loss at a matched step
and seed, one switch apart, the plain loss gives several animals and this gives one. So the
correction cannot be carried by a branch with no access to the words: plurality is not a spatial
rearrangement that can be applied blindly, which answers the where-or-what question with **what**.

**The useful checkpoint arrives early**, around three thousand steps, exactly where the
exposure-per-state arithmetic said it would. Ten thousand is past the peak, not short of it.

## Judged by eye

The one labelled batch this walk produced, kept because a picker has to be validated against
something and this is the start of it.

| Batch | Good, by eye | What the picker chose |
|---|---|---|
| `zo_adapter_40k_seed9_lam1p2/a_cat__x__a_dog/seed_9/sigma_0.1`, candidates c0 to c7 | c2, c6 | c4 |

The count could not separate the candidates, since all eight were read as two animals, so the
choice fell to the edge measure, which scores matted fur above clean fur.

## Routes

None emitted yet.

## Sources

- [What the project found, by question](../../../report/00-INDEX.md)
- [How the corrector is trained, and every way we could change it](../designing-the-correction-loss/maths/objectives-and-variations.tex)
- [Two legs, one loss](https://claude.ai/code/artifact/ae96d471-01cd-4764-b79d-41da5053230c), the interactive page. Its second lane carries this cohort: parent and five children, each one switch away, in both notations.

Every measured value behind the readings above lives in those findings and their figures, never in
this file.

## Next step

Claim 1 is current. `next` moves to claim 2.
