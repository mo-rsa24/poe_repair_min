# Designing the Correction Loss: illustrated map

Five pictures of the machinery every variation shares, and a capstone that puts them on one page.
The switch, where the nine objectives differ, is the one that carries the scope.

10 prompts · 10 rendered

Three pieces depict machinery that exists and are marked `[built]`. Two depict things this scope
creates and are marked `[planned]`, drawn hedged, with their prompts saying so in plain words rather
than leaning on a drawing convention. Every prompt is self-contained, so pasting any one of them
alone returns a picture that matches its siblings.

## Table of contents

- [Abstraction chain](#abstraction-chain)
- [Art direction](#art-direction)
- [Reference images](#reference-images)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Devices in play](#devices-in-play)
- [Subject lane](#subject-lane)
  - [Prompt 1 (Subject): What was written to disk once, and never again](#prompt-1-subject-what-was-written-to-disk-once-and-never-again)
  - [Prompt 2 (Subject): One pass, three prompts, three answers](#prompt-2-subject-one-pass-three-prompts-three-answers)
  - [Prompt 3 (Subject): The two legs of the loss, and the fine](#prompt-3-subject-the-two-legs-of-the-loss-and-the-fine)
  - [Prompt 4 (Subject): The four places the nine objectives differ](#prompt-4-subject-the-four-places-the-nine-objectives-differ)
  - [Prompt 5 (Subject): A checkpoint and a picture at the same moment](#prompt-5-subject-a-checkpoint-and-a-picture-at-the-same-moment)
  - [Subject capstone: From a frozen cache to one page per objective](#subject-capstone-from-a-frozen-cache-to-one-page-per-objective)
- [Process lane](#process-lane)
  - [Prompt 1 (Process): Fix the instruments before measuring anything](#prompt-1-process-fix-the-instruments-before-measuring-anything)
  - [Prompt 2 (Process): Read what is already on the disk](#prompt-2-process-read-what-is-already-on-the-disk)
  - [Prompt 3 (Process): Change one thing, and prove the two halves agree first](#prompt-3-process-change-one-thing-and-prove-the-two-halves-agree-first)
  - [Process capstone: One route from an unmeasured objective to one that is named](#process-capstone-one-route-from-an-unmeasured-objective-to-one-that-is-named)

## Abstraction chain

📋 [TOC](#table-of-contents) | [Next](#art-direction) ➡️

1. [Closing the Compositional Gap](../diagram-prompts.md): subject "The whole system on one page"
2. [Designing the Correction Loss](diagram-prompts.md): subject "From a frozen cache to one page
   per objective", process "One route from an unmeasured objective to one that is named"

This scope zooms into one glyph of the parent map's subject capstone, the adapter's training loop,
and opens it into the loss itself and the nine ways that loss can be written.

## Art direction

⬅️ [Previous](#abstraction-chain) | 📋 [TOC](#table-of-contents) | [Next](#reference-images) ➡️

**Empty, chosen 2026-09-16.** No style is prescribed. Each prompt's own detail does the work, which
suits a subject whose whole content is which tensor came from where.

**Zoom: level 2**, one mechanism opened into what actually happens inside it. The loss is the
mechanism, not the training loop around it.

**The known risk, and the mitigation.** Empty has split a technical set into two looks on this
project before: architecture pictures came back as flat vector diagrams while pictures carrying
generated animals came back photographic. So every prompt here carries a no-photography exclusion,
and **the first render accepted becomes a reference image for the other five**, recorded below once
it exists. If the set still splits, the fallback is Vendor tinted, which is what the flat pictures
already were.

## Reference images

⬅️ [Previous](#art-direction) | 📋 [TOC](#table-of-contents) | [Next](#meaning-palette) ➡️

`diagrams/corrloss-01-what-was-written-to-disk-once.png`, accepted 2026-09-16 and passed to every
prompt after it, downscaled to 1280 px on its longest side.

It is the reference because it is the simplest picture in the set and it settled what Empty does
with this subject: flat illustrated objects on a light ground, black text at two weights, one
colour per named thing with no colour carrying meaning on its own, and a shared label under a row
rather than repeated per item.

It also settled one correction the other nine inherit. The first attempt letterboxed the
illustration on a pure black canvas, which the style guide's light-theme rule forbids. Every
dispatch after this one carries the reference, so the correction does not have to be re-earned.

## Meaning palette

⬅️ [Previous](#reference-images) | 📋 [TOC](#table-of-contents) | [Next](#glyph-vocabulary) ➡️

Two meanings, and the whole set lives on the distinction between them. Under Empty the renderer
chooses how to draw each one; what is fixed is that the choice is consistent and carries a legend.

| Meaning | Where it applies |
|---|---|
| read from disk, frozen, no gradient | every tensor off the cache: the noisy latent, the three branch predictions, the joint-prompt prediction |
| computed now, adapter attached, carries a gradient | every tensor out of this step's forward pass |

No third meaning. A picture needing one is two pictures.

## Glyph vocabulary

⬅️ [Previous](#meaning-palette) | 📋 [TOC](#table-of-contents) | [Next](#reading-axes) ➡️

Described in words rather than assigned, because the direction is Empty.

| Tier | What it is |
|---|---|
| Primary | the storage holding the cache, the one U-Net, the guidance operation, the comparison |
| Secondary | the adapter sites on the cross-attention layers, the penalty term |
| Artifact | a single named tensor, shown with its real shape where the source has one |

The U-Net appears exactly once in any picture that contains it. There is no per-prompt adapter and
no second network.

## Reading axes

⬅️ [Previous](#glyph-vocabulary) | 📋 [TOC](#table-of-contents) | [Next](#devices-in-play) ➡️

**Dominant axis: left to right**, from what was recorded to what is computed to the number that
comes out.

**Secondary axis: top to bottom**, separating what is frozen (upper) from what learns (lower).

Pinned at scope birth. Every later regeneration of the process lane is compared against these.

## Devices in play

⬅️ [Previous](#reading-axes) | 📋 [TOC](#table-of-contents) | [Next](#subject-lane) ➡️

One legend per picture, naming the two meanings. Real tensor shapes shown where the source has
them. Nothing else: no numbered badges in the subject lane, because nothing here happens in an
order.

## Subject lane

⬅️ [Previous](#devices-in-play) | 📋 [TOC](#table-of-contents)

The machinery every variation shares, plus the one place they differ. Subject: the parts and how
they connect, no time in it.

### Prompt 1 (Subject): What was written to disk once, and never again
[built] 🖼️ rendered 2026-09-16 as `diagrams/corrloss-01-what-was-written-to-disk-once.png`
Save as: `diagrams/corrloss-01-what-was-written-to-disk-once.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One picture of what a training cache holds, drawn as a single storage device opened up to show
its contents. Nothing in this picture is computed: everything was produced once by a frozen
image-generation model months ago and written down.

CAST
A storage device. Inside it, many identical entries. One entry is drawn open, the rest are
implied.

The open entry holds five tensors, each one four channels at 128 by 128 numbers, stored as
32-bit floating point on the CPU. Each is labelled with what it is:
  the noisy latent at this denoising step
  what the frozen model predicted given the words "a cat"
  what it predicted given the words "a dog"
  what it predicted given an empty prompt
  what it predicted given the words "a cat and a dog"

Beside the open entry, three small labels saying what indexes an entry: which pair of animals,
which random seed, which denoising step.

Two real quantities to show as text: one sample is 65,536 numbers, and fifty denoising steps of
one pair and seed comes to about 250 megabytes.

WHAT IS CONFIRMED
All of it. This cache exists and these are its real contents and real shapes.

FLOWS
None. Nothing moves in this picture. It is an inventory of what is on the disk.

TEXT IN THE IMAGE, exactly as written
  "written once, frozen"
  "the noisy latent"
  "given a cat"
  "given a dog"
  "given an empty prompt"
  "given a cat and a dog"
  "4 x 128 x 128, float32"
  "one pair, one seed, one step"
  "65,536 numbers per tensor"

EXCLUSIONS
No arrows, because nothing flows here. No neural network: the model that made these is not in
this picture. No photographs and no photographic rendering of any kind. No cats or dogs drawn as
animals: these are text prompts and numeric tensors, not pictures. No logos. No components the
text above does not name. No placeholder gibberish text. No duplicated components, no orphan
arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative
circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible
pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon
without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts
the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare
connector line standing in for a real-world physical object the prompt names.
```

Visual thesis: every number the training compares against was decided before training started, so
the only thing that can change is the model's side.

Faithfulness note: all five tensors carry the same shape and the same precision, and none of them
is drawn as an output of anything. The joint-prompt prediction must not be visually privileged over
the other four: its role as the target belongs to prompt 3, not here.

### Prompt 2 (Subject): One pass, three prompts, three answers
[built] 🖼️ rendered 2026-09-19 as `diagrams/corrloss-02-one-pass-three-prompts.png`
Save as: `diagrams/corrloss-02-one-pass-three-prompts.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One picture of a single forward pass through one neural network, reading left to right. Three
text prompts on the left, one network in the middle, three separate answers on the right.

CAST
Three text prompts stacked vertically: "a cat", "a dog", and an empty prompt drawn as an empty
field rather than as something missing.

One U-Net, drawn once and only once. All three prompts pass through this same network in one
batched pass, 3K items wide, where K is how many cache entries are in the batch. Small trainable
adapter blocks sit on its cross-attention layers, 210 of them. Those blocks are the only part of
anything in this picture that learns: the network itself, both text encoders and the image
decoder are all frozen. The adapter is 9,912,320 numbers against the model's 2,577,376,004, or
0.38 percent.

Three outputs leaving that one network, one per prompt, each four channels at 128 by 128. Label
them the cat branch, the dog branch and the empty branch.

A note on the output: it comes out in 16-bit and is immediately converted to 32-bit, because at
16 bits the subtraction that happens later destroys the gradient.

WHAT IS CONFIRMED
All of it, from the training code.

FLOWS
Three prompt lines converging into the one network. Three answer lines leaving it. Every line in
this picture is computed now with the adapter attached and carries a gradient, and the legend
says so, because the sibling pictures in this set contain lines that do not.

The adapter sits on shared weights, so it changes all three answers at once. There is no separate
adapter per prompt, and the picture must not suggest one.

TEXT IN THE IMAGE, exactly as written
  "a cat"
  "a dog"
  "empty prompt"
  "one U-Net, one pass"
  "3K wide"
  "frozen"
  "adapter here, 210 sites"
  "0.38% of the weights"
  "the cat branch"
  "the dog branch"
  "the empty branch"
  "cast to 32-bit at once"
  "computed now, adapter attached"

EXCLUSIONS
No second copy of the network. No separate adapter per prompt. No cache and no storage device:
that is a different picture in this set. No photographs and no photographic rendering of any
kind. No cats or dogs drawn as animals: the prompts are text. No logos. No components the text
above does not name. No placeholder gibberish text. No duplicated components, no orphan arrows,
no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry
or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible
pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon
without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts
the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare
connector line standing in for a real-world physical object the prompt names.
```

Visual thesis: one adapter on one network produces all three answers at once, so nothing can be
changed for the cat alone.

Faithfulness note: exactly one network appears, all three prompt lines enter it, and the three
output lines leave the same body. The adapter blocks sit on the network, never on the prompt lines
or the output lines.

### Prompt 3 (Subject): The two legs of the loss, and the fine
[built] 🖼️ rendered 2026-09-19 as `diagrams/corrloss-03-two-legs-and-the-fine.png`
Save as: `diagrams/corrloss-03-two-legs-and-the-fine.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One wide landscape picture of what the training loss is made of, opened up far enough to show
every place the empty branch is used. Two horizontal tracks running left to right, meeting at one
point on the right.

The upper track is the target and it is entirely historical: computed months ago by the frozen
model and written to disk. The lower track is the composition and it is computed right now with
the adapter attached. The two tracks converge on a single subtraction on the right, which
produces one number.

Below the subtraction sits a third, much shorter element: the fine. It connects two things that
already appear in the picture and it never touches the subtraction.

CAST
Upper track, left to right:
  A text prompt reading "a cat and a dog".
  A storage device holding the cached tensors. It is read only. Nothing writes to it.
  A tensor coming off that storage, the model's prediction for that joint prompt.
  A second tensor off the same storage, the model's prediction for an empty prompt.
  A guidance operation combining those two cached tensors at a weight of 7.5.

Lower track, left to right:
  Three text prompts stacked vertically: "a cat", "a dog", and an empty prompt drawn as an empty
    field rather than as missing.
  One single U-Net, drawn once and only once, all three prompts through it in one batched pass
    3K items wide, with small trainable adapter blocks on its cross-attention layers, 210 of
    them, the only part that learns.
  Three outputs: the cat branch, the dog branch, the empty branch.
  A guidance operation combining those three at a weight of 7.5, in which the empty branch is
    added once inside the combination and subtracted once outside it at a weight of 6.5.

The meeting point:
  A subtraction of the upper track's result from the lower track's, then squared and averaged
  over 65,536 numbers, producing one number.

The fine, drawn as its own small element beneath everything:
  A short connection from the lower track's empty branch back to the upper track's cached
  empty-prompt tensor, carrying a weight of 10. Its result is added to the number, not to the
  subtraction.

WHAT IS CONFIRMED
All of it, from the training code and a saved run configuration. Nothing here is hypothetical.

FLOWS
Two kinds of line, and the difference between them is the most important thing in the picture:
lines carrying tensors read from storage, which carry no gradient, against lines carrying tensors
computed in this step with the adapter attached. The distinction must be consistent and explained
in a small legend inside the image.

The empty branch line splits three ways and all three destinations must be visible: into the
guidance combination, out to the subtraction beside it, and down to the fine.

TEXT IN THE IMAGE, exactly as written
  "a cat and a dog"
  "a cat"
  "a dog"
  "empty prompt"
  "read from disk"
  "computed now"
  "one U-Net, one pass"
  "adapter here, 210 sites"
  "guidance, w = 7.5"
  "read three times"
  "subtract once at 6.5"
  "subtract, square, average"
  "one number"
  "the fine, mu = 10"
  "held near the cached one"

EXCLUSIONS
No second copy of the network. No separate adapter per prompt. No arrow from the fine into the
subtraction. No photographs and no photographic rendering of any kind. No cats or dogs drawn as
animals: the prompts are text, not pictures. No logos. No components the text above does not
name. No placeholder gibberish text. No duplicated components, no orphan arrows, no arrow
terminating in whitespace, no color used without a legend entry, no decorative circuitry or
filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code,
no crossing arrows where routing could avoid it, no untitled containers, no icon without a label,
no perspective distortion applied to arrows, no visual metaphor that contradicts the system's
actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line
standing in for a real-world physical object the prompt names.
```

Visual thesis: the empty branch is read three times, only one of those readings carries a gradient,
and the fine is a short leash tying it back to a tensor sitting on the disk.

Faithfulness note: exactly one network appears and all three prompts enter it. The empty branch
leaves it on one line that splits three ways, and all three destinations are visible. The fine's
result joins the number and never enters the subtraction. Every line on the upper track is marked
as carrying no gradient; every line on the lower track and the fine is marked as carrying one.

### Prompt 4 (Subject): The four places the nine objectives differ
[planned] 🖼️ rendered 2026-09-19 as `diagrams/corrloss-04-the-four-places-they-differ.png`
Save as: `diagrams/corrloss-04-the-four-places-they-differ.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One picture showing that nine different training objectives are the same machine with four
settings changed. A simplified version of the loss runs across the picture: three branches
combining, a target, a comparison. Four settings are marked on it, each opening into the
objectives that use it.

CAST
The simplified loss, drawn small and unlabelled in detail, as the thing the settings sit on:
three branch outputs combining into one prediction, a target arriving from the side, and a
comparison producing one number.

Four settings, each marked at the place on that machine where it acts, and each listing the
objectives that change it:

  Which branches come from the model and which are read from the cache. Acts at the three branch
  outputs. Used by objective 01, which reads the empty branch from the cache, and objective 03,
  which reads both concept branches from the cache.

  What the target is. Acts where the target arrives. Used by objective 05, whose target becomes
  a real photograph with known noise added, and objective 06, whose target becomes the model's
  own pictures that passed a filter.

  The guidance weight. Acts on both the combination and the target. Used by objective 04, which
  sets it to one instead of 7.5.

  An extra penalty, and what it is measured against. Acts beside the comparison. Used by
  objective 00b, which holds the empty branch near its cached self; objective 02, which holds it
  near the prediction for the words "two animals"; and objective 07, which holds the concept
  branches near their frozen selves at states from single-prompt runs.

One more element, set apart: objective 00 changes nothing at all. It is the machine as drawn.

WHAT IS CONFIRMED AND WHAT IS NOT
Objectives 00 and 00b exist in the code and have been trained. The other seven do not exist yet:
01, 02, 03, 04, 05, 06 and 07 are all planned, and every one of them must be drawn visibly
unconfirmed, hedged in whatever way the composition allows, with the picture saying in plain words
that they are planned rather than built.

FLOWS
No flow of data between the four settings: they are four independent places on one machine, not a
sequence. Lines run only from each setting to the point on the machine where it acts.

TEXT IN THE IMAGE, exactly as written
  "same machine, four settings"
  "which branches are adapted"
  "what the target is"
  "the guidance weight"
  "an extra penalty, and on what"
  "00: changes nothing"
  "built"
  "planned"

EXCLUSIONS
No sequence, no numbered steps, no arrows between settings. No full detail of the loss: the
machine is deliberately simplified here, because another picture in this set draws it in full. No
photographs and no photographic rendering of any kind. No animals drawn as animals. No logos. No
components the text above does not name. No placeholder gibberish text. No duplicated components,
no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no
decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no
illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no
icon without a label, no perspective distortion applied to arrows, no visual metaphor that
contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or
bare connector line standing in for a real-world physical object the prompt names.
```

Visual thesis: the nine objectives are not nine systems, they are one system with four settings,
and seven of the nine settings have never been tried.

Faithfulness note: objectives 00 and 00b are drawn as confirmed and the other seven as planned,
with the difference visible without reading the labels. Each setting attaches to the one place on
the machine where it acts and nowhere else; the guidance weight is the only one attaching to two
places, the combination and the target.

### Prompt 5 (Subject): A checkpoint and a picture at the same moment
[planned] 🖼️ rendered 2026-09-19 as `diagrams/corrloss-05-checkpoint-and-picture-together.png`
Save as: `diagrams/corrloss-05-checkpoint-and-picture-together.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One picture of what a training run leaves behind, and it makes one point: the weights and the
picture are saved at the same moment, so every picture has the weights that made it.

A horizontal axis of training steps runs across the picture, from zero to fifty thousand. At
even intervals along it, twenty marks. At each mark, two things are produced together.

CAST
The training axis, marked every 2,500 steps, twenty marks in all, ending at 50,000.

At each mark, a pair:
  a checkpoint file holding the adapter's weights, 114 megabytes at this size of adapter
  a strip of four images side by side

The strip is drawn once in full, at one mark, and implied at the others. Its four panels, left
to right: the target the correction is defined from, the uncorrected combination of the two
concepts, the adapter's own render, and the same for a second animal pair. Two of the pairs the
strip covers were trained on and two were never seen.

To one side, the total: twenty checkpoints and twenty strips per run, 2.3 gigabytes, on a storage
volume with 233 terabytes free.

To the other side, the point of pairing them: any picture can be rescored later, and any point on
the curve can be resumed from.

WHAT IS CONFIRMED AND WHAT IS NOT
The strips are produced today, every 2,500 steps. The checkpoints are not: today they are saved
every 10,000 steps, so nine of every twelve pictures have no weights behind them. The pairing
this picture shows is planned, and must be drawn visibly unconfirmed with the picture saying so
in plain words.

FLOWS
Left to right along the training axis only. Each mark produces its pair downward. Nothing flows
between marks.

TEXT IN THE IMAGE, exactly as written
  "every 2,500 steps"
  "50,000 steps"
  "the weights, 114 MB"
  "the strip, four panels"
  "target"
  "uncorrected"
  "the adapter"
  "trained on"
  "never seen"
  "20 saves, 2.3 GB per run"
  "planned, not yet true"

EXCLUSIONS
No loss curve: this picture is about what is saved, not about what the loss did. No neural
network. No photographs and no photographic rendering of any kind, including inside the strip:
the four panels are placeholders for images, drawn as empty framed panels with their labels, and
must not contain generated pictures of animals. No logos. No components the text above does not
name. No placeholder gibberish text. No duplicated components, no orphan arrows, no arrow
terminating in whitespace, no color used without a legend entry, no decorative circuitry or
filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code,
no crossing arrows where routing could avoid it, no untitled containers, no icon without a label,
no perspective distortion applied to arrows, no visual metaphor that contradicts the system's
actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line
standing in for a real-world physical object the prompt names.
```

Visual thesis: the weights and the picture are written at the same moment, which is what makes any
point on a training run re-readable years later.

Faithfulness note: the checkpoint and the strip share one mark on the axis, never adjacent marks.
The whole pairing is drawn as planned rather than built, and the picture says in words that
checkpoints are currently saved four times less often than strips.

### Subject capstone: From a frozen cache to one page per objective
[planned] 🖼️ rendered 2026-09-19 as `diagrams/corrloss-capstone-cache-to-one-page-per-objective.png`
Save as: `diagrams/corrloss-capstone-cache-to-one-page-per-objective.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One page composing five parts into a single system, read left to right and top to bottom, with
frozen machinery above and learning machinery below. Every object in this picture appears in one
of the five sibling pictures of this set, and nothing new is introduced.

CAST, five parts

1. The cache, a storage device holding many entries, each with five tensors of four channels at
   128 by 128: a noisy latent and four predictions the frozen model made from four different
   prompts. Written once, read only.

2. One U-Net with small trainable adapter blocks on its cross-attention layers, 210 of them, the
   only part that learns. Three text prompts enter it in one batched pass: "a cat", "a dog", and
   an empty prompt. Three answers leave it: the cat branch, the dog branch, the empty branch.

3. The loss. Two tracks meeting: the target, built from cached tensors with a guidance weight of
   7.5, and the composition, built from the three answers with the same weight. They meet at a
   subtraction, squared and averaged to one number. The empty branch is read three times, and all
   three readings are visible.

4. The switch, marked on the loss at four places: which branches come from the model rather than
   the cache, what the target is, the guidance weight, and an extra penalty and what it is
   measured against. Nine objectives use these four settings. Two of the nine exist; seven are
   planned.

5. What comes out: every 2,500 steps, a checkpoint and a four-panel strip saved together, twenty
   of each per run, feeding one page per objective, all nine pages the same shape.

WHAT IS CONFIRMED AND WHAT IS NOT
Parts 1, 2 and 3 exist and are drawn confirmed. Part 4's seven unbuilt objectives and the whole
of part 5 are planned and must be drawn visibly unconfirmed, with the picture saying so in words.

FLOWS
The cache feeds both the target track and the network's input. The network feeds the composition
track. Both tracks meet at the loss. The switch attaches to the loss at four points and is not in
the flow. The loss leads to what comes out.

A legend inside the image, with exactly two entries: read from disk and carries no gradient,
against computed now with the adapter attached.

TEXT IN THE IMAGE, exactly as written
  "written once, frozen"
  "one U-Net, one pass"
  "adapter here, 210 sites"
  "guidance, w = 7.5"
  "read three times"
  "one number"
  "same machine, four settings"
  "nine objectives, two built"
  "every 2,500 steps"
  "one page per objective"
  "read from disk"
  "computed now"
  "built"
  "planned"

EXCLUSIONS
No object that does not appear in one of the five sibling pictures. No second copy of the network.
No numbered steps or sequence badges: this is the system, not the order of work. No photographs
and no photographic rendering of any kind; the strip's panels are empty labelled frames. No
animals drawn as animals. No logos. No placeholder gibberish text. No duplicated components, no
orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no
decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no
illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no
icon without a label, no perspective distortion applied to arrows, no visual metaphor that
contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or
bare connector line standing in for a real-world physical object the prompt names.
```

Visual thesis: everything except the switch is shared by all nine objectives, so choosing between
them is choosing four settings rather than building nine systems.

Faithfulness note: every object here appears in one of the five sibling prompts and none of them is
missing. The cache feeds both tracks. The switch touches the loss and never sits in the flow of
data. The frozen half stays above and the learning half below, matching the pinned reading axes.

## Process lane

⬅️ [Subject lane](#subject-lane) | 📋 [TOC](#table-of-contents)

The same cast with the order of work drawn over it. Four stages and a route map, regenerated
whenever the plan set changes, never patched. History in `diagrams/process-versions/`.

### Prompt 1 (Process): Fix the instruments before measuring anything
[planned] 🖼️ rendered 2026-09-19 as `diagrams/corrloss-p01-fix-the-instruments-first.png`
Save as: `diagrams/corrloss-p01-fix-the-instruments-first.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One picture of the first stage of work, drawn as three small changes made to a machine that is
already running, with a check after them. Order matters here and must be visible.

CAST
The machine: the training loop, shown small, as the loss with three branches feeding it. It is
the subject of the other pictures in this set and appears here only as the thing being changed.

Three changes, numbered, each attached to where it acts:
  1. A new number computed beside the loss: the same comparison with the guidance weight set to
     one instead of 7.5. It is read, never used to train.
  2. The save cadence: weights written at the same moment as pictures, where today pictures are
     written four times as often.
  3. The run's own record of what it was: three settings that change the objective and are
     currently missing from the file meant to identify it.

A gate after all three: a two-epoch run that must show the new number present and varying, a
checkpoint beside a picture, and the run's record naming its settings. The gate has two exits,
pass and fail, and the fail exit returns to the change that failed rather than to the start.

A result leaving the gate: one ratio, the new number against the loss. That ratio is what says
whether the rest of this work is about something real.

WHAT IS CONFIRMED AND WHAT IS NOT
All three changes are planned, not built. Draw the whole picture as planned and say so in words.

FLOWS
Numbered order through the three changes, then into the gate, then out to the ratio. The fail
exit returns upward to a specific change, not to the beginning.

TEXT IN THE IMAGE, exactly as written
  "1. the number nobody logged"
  "2. weights beside pictures"
  "3. the run records its own settings"
  "smoke: two epochs"
  "all three visible, or back"
  "the ratio"
  "planned, not built"

EXCLUSIONS
No full detail of the loss: it is drawn small here because another picture in this set draws it in
full. No photographs and no photographic rendering of any kind. No animals drawn as animals. No
logos. No components the text above does not name. No placeholder gibberish text. No duplicated
components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend
entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs
of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled
containers, no icon without a label, no perspective distortion applied to arrows, no visual
metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram,
circuit symbol, or bare connector line standing in for a real-world physical object the prompt
names.
```

Visual thesis: one cheap measurement comes before any expensive change, because it can make the
expensive change unnecessary.

Faithfulness note: the gate sits after all three changes, not after each one, and its fail exit
returns to the specific change rather than to the start. The ratio leaves the gate, never bypasses
it.

### Prompt 2 (Process): Read what is already on the disk
[planned] 🖼️ rendered 2026-09-19 as `diagrams/corrloss-p02-read-what-is-already-there.png`
Save as: `diagrams/corrloss-p02-read-what-is-already-there.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One picture of two stages of reading, running left to right, that produce two pages and train
nothing at all. The point of the picture is that everything it consumes already exists.

CAST
On the left, two piles of finished work:
  Five training runs holding 136 sets of weights, spread across two storage locations, with their
  numbers scattered across thirteen separate files and one paragraph of prose.
  Four more training runs that finished on one day four days ago and that no document mentions.

In the middle, the work:
  Collecting the scattered numbers into one file.
  Rebuilding the drawing program, because the one that made the existing pictures was never kept.
  Computing a spread that was written down as the threshold long ago and never calculated.
  Rendering and scoring the four unrecorded runs on held-out examples.
  A read by eye, with the names hidden and the order shuffled, and the key to that shuffle kept
  separately and opened only afterwards.

On the right, two pages, drawn the same shape as each other, each carrying a verdict line, a few
pictures and links out to the detail rather than the detail itself.

WHAT IS CONFIRMED AND WHAT IS NOT
The two piles on the left exist. Everything in the middle and on the right is planned. Draw the
left as confirmed and the rest as planned, and say so in words.

FLOWS
Left to right. Nothing loops back. The shuffle key runs alongside the blind read and joins it only
after the labels are recorded, and that ordering must be visible.

TEXT IN THE IMAGE, exactly as written
  "five runs, 136 checkpoints"
  "four runs, nothing written down"
  "collect the numbers"
  "rebuild the drawing program"
  "compute the spread"
  "render and score"
  "read by eye, names hidden"
  "key opened afterwards"
  "two pages, one shape"
  "nothing is trained here"

EXCLUSIONS
No training loop: this stage trains nothing and the picture must not imply it does. No
photographs and no photographic rendering of any kind; any rendered example is an empty labelled
frame. No animals drawn as animals. No logos. No components the text above does not name. No
placeholder gibberish text. No duplicated components, no orphan arrows, no arrow terminating in
whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented
or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows
where routing could avoid it, no untitled containers, no icon without a label, no perspective
distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics,
no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a
real-world physical object the prompt names.
```

Visual thesis: two of the nine objectives can be answered without training anything, because the
runs already happened and only the reading is missing.

Faithfulness note: no training appears anywhere in this picture. The shuffle key joins the blind
read after the labels, never before, and the picture shows that ordering rather than stating it.

### Prompt 3 (Process): Change one thing, and prove the two halves agree first
[planned] 🖼️ rendered 2026-09-19 as `diagrams/corrloss-p03-prove-they-agree-first.png`
Save as: `diagrams/corrloss-p03-prove-they-agree-first.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One picture of the stage that spends real machine time, and it is built around a gate that comes
before the spending rather than after it.

CAST
First, before anything else, a check with two renders in it: the same starting point drawn twice,
once with the adapter attached during the empty-prompt pass and once with it detached. The check
passes only if the two pictures differ. A check that cannot fail is worth nothing, and this one is
drawn as the first thing in the stage for that reason.

Then, only past that gate, three changes:
  reading the empty branch from storage instead of from the model
  narrowing the batch from three rows to two
  making the sampler read the empty branch the same way the training did

Then a run: fifty thousand steps, twenty sets of weights, twenty sets of pictures, roughly twenty
hours on one graphics card.

Then two reads: a scored comparison against the objective running today at the same step counts,
with a previously computed spread drawn on it, and a read by eye with names hidden.

A note attached to the gate: this exact failure happened on this project on the fifth of
September, when a sampler left the adapter attached while drawing what were supposed to be plain
references, and it cost a re-render.

WHAT IS CONFIRMED AND WHAT IS NOT
All of it is planned. The September failure is confirmed and should be drawn as the one
confirmed thing on the picture.

FLOWS
Strictly ordered: check, gate, changes, run, reads. The gate's fail exit stops the stage entirely
rather than routing onward, and that must be visible: nothing downstream of a failed gate is
reachable.

TEXT IN THE IMAGE, exactly as written
  "prove they disagree first"
  "adapter on"
  "adapter off"
  "must differ, or the check is empty"
  "read from storage, not the model"
  "three rows become two"
  "the sampler reads it the same way"
  "50,000 steps, about 20 hours"
  "inside the spread, or outside"
  "this failed here on 2026-09-05"

EXCLUSIONS
No path from a failed gate to the run. No photographs and no photographic rendering of any kind;
the two renders in the check are empty labelled frames. No animals drawn as animals. No logos. No
components the text above does not name. No placeholder gibberish text. No duplicated components,
no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no
decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no
illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no
icon without a label, no perspective distortion applied to arrows, no visual metaphor that
contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or
bare connector line standing in for a real-world physical object the prompt names.
```

Visual thesis: the check that costs one render comes before the run that costs twenty hours,
because this project has already paid for getting that order wrong.

Faithfulness note: nothing downstream of the gate is reachable from its fail exit. The two renders
in the check must differ, and the picture says that a check whose two renders match is an empty
check rather than a passing one.

### Process capstone: One route from an unmeasured objective to one that is named
[planned] 🖼️ rendered 2026-09-20 as `diagrams/corrloss-process-capstone-unmeasured-to-named.png`
Save as: `diagrams/corrloss-process-capstone-unmeasured-to-named.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
technique. Let the composition and visual treatment that best communicate this subject emerge
from the context below, rather than from an imposed convention.

SCENE
One route map across the whole scope, read left to right, with four stops and the gates between
them. Every object in this picture appears in one of the three sibling process pictures or in the
subject lane, and nothing new is introduced.

CAST, four stops in order

1. Fix the instruments. Three small changes and a two-epoch check. Out of it comes one ratio,
   saying whether the direction the training cannot see carries real size.
   Gate: if that ratio is small, the reason for stop 4 is written down as weakened, and stop 4
   still runs but its motivation is not the same.

2. Build the foundation. Five existing runs read into one page, with a drawing program rebuilt so
   the pictures can be redrawn, and a spread computed that was written down as a threshold long
   ago and never calculated.
   Gate: that page's shape is the shape every later page copies, so it is read by a person once
   before anything copies it.

3. Read what already ran. Four runs from one day, scored and read by eye with names hidden.
   Gate: a penalty that does not improve the picture is recorded as a null, which strengthens the
   reason for stop 4 rather than weakening the work.

4. Change one thing. A check that the two halves of the system agree, then the change, then fifty
   thousand steps, then a comparison against the objective running today at matched step counts.
   Gate: inside the spread confirms the derivation three unwritten objectives rest on. Outside it
   falsifies that derivation, and three unwritten objectives must be reconsidered before any of
   them is planned.

At the end, one objective named, and four others still carrying a single sentence each saying
what would promote them from a row in a table to work.

WHAT IS CONFIRMED AND WHAT IS NOT
Every stop is planned. The two piles of finished runs that stops 2 and 3 read are confirmed. Draw
the difference and say so in words.

FLOWS
One route, left to right, four stops. Each gate has a pass exit onward and a named consequence for
failing, and the consequence is written rather than drawn as a loop back, because none of these
failures repeats the stage: each one changes what the next stage means.

A legend inside the image with exactly two entries: already finished, and planned.

TEXT IN THE IMAGE, exactly as written
  "1. fix the instruments"
  "2. build the foundation"
  "3. read what already ran"
  "4. change one thing"
  "the ratio"
  "the page every page copies"
  "a null strengthens stop 4"
  "inside the spread, or outside"
  "one objective named"
  "four still waiting"
  "already finished"
  "planned"

EXCLUSIONS
No object that does not appear in a sibling picture. No loops back from a gate. No photographs and
no photographic rendering of any kind. No animals drawn as animals. No logos. No placeholder
gibberish text. No duplicated components, no orphan arrows, no arrow terminating in whitespace, no
color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied
brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing
could avoid it, no untitled containers, no icon without a label, no perspective distortion applied
to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no
geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical
object the prompt names.
```

Faithfulness note: the four stops are in the order the running order gives them, and each gate's
consequence is the one its plan's own pass-and-fail section states, never invented. No gate loops
back to its own stage, because none of these failures is repaired by repeating the work.
