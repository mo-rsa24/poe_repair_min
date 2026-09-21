# Improve the Rank-32 Pooled Adapter: illustrated map

Five pictures of the system this scope builds, one zoom into the training error, and a capstone
that puts them on one page. Then the same cast with the order of work drawn over it, in four
stages and a route map.

12 prompts · 6 rendered · 6 waiting

Everything here is `[planned]`: nothing in this scope exists yet, and every prompt says so in its
own words rather than leaning on a drawing convention. Every prompt is self-contained, so pasting
any one of them alone returns a picture that matches its siblings.

## Table of contents

- [Abstraction chain](#abstraction-chain)
- [Art direction](#art-direction)
- [Reference images](#reference-images)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Devices in play](#devices-in-play)
- [Subject lane](#subject-lane)
  - [Prompt 1 (Subject): Which cells the adapter is allowed to learn from](#prompt-1-subject-which-cells-the-adapter-is-allowed-to-learn-from)
  - [Prompt 2 (Subject): The five changes, and the four runs that each undo one](#prompt-2-subject-the-five-changes-and-the-four-runs-that-each-undo-one)
  - [Prompt 2a (Subject): What the loss weights up](#prompt-2a-subject-what-the-loss-weights-up)
  - [Prompt 3 (Subject): The two ways every checkpoint is rendered](#prompt-3-subject-the-two-ways-every-checkpoint-is-rendered)
  - [Prompt 4 (Subject): The strip, and the read that is done by eye](#prompt-4-subject-the-strip-and-the-read-that-is-done-by-eye)
  - [Prompt 5 (Subject): Five runs on one machine, overnight](#prompt-5-subject-five-runs-on-one-machine-overnight)
  - [Subject capstone: From cached cells to one comparison sheet](#subject-capstone-from-cached-cells-to-one-comparison-sheet)
- [Process lane](#process-lane)
  - [Prompt 1 (Process): Build the instruments, and prove each one can fail](#prompt-1-process-build-the-instruments-and-prove-each-one-can-fail)
  - [Prompt 2 (Process): Read the baseline before spending a night of training](#prompt-2-process-read-the-baseline-before-spending-a-night-of-training)
  - [Prompt 3 (Process): The night the chain runs, and the morning after](#prompt-3-process-the-night-the-chain-runs-and-the-morning-after)
  - [Prompt 4 (Process): File the sheet, and close the work](#prompt-4-process-file-the-sheet-and-close-the-work)
  - [Process capstone: One route from an unclean picture to a filed answer](#process-capstone-one-route-from-an-unclean-picture-to-a-filed-answer)

## Abstraction chain

📋 [TOC](#table-of-contents) | [Next](#art-direction) ➡️

1. [Closing the Compositional Gap](../diagram-prompts.md): subject "The whole system on one page",
   process "One route from measuring tools to submission"
2. [Improve the Rank-32 Pooled Adapter](diagram-prompts.md): subject "From cached cells to one
   comparison sheet", process "One route from an unclean picture to a filed answer"

This scope zooms into one glyph of the parent map's subject capstone, the adapter chip, and opens
it into the run set that trains it and the read that judges it.

## Art direction

⬅️ [Previous](#abstraction-chain) | 📋 [TOC](#table-of-contents) | [Next](#reference-images) ➡️

**Vendor tinted, chosen from a rendered example.** The set was first drawn with no direction
prescribed, which let each picture choose its own treatment: the architecture pictures came back as
flat vector diagrams and the ones carrying generated animal pictures came back photographic. Vendor
tinted is what the flat ones already were, so naming it makes the whole set match the one picture
that was right rather than each picture matching itself.

It suits this subject for the reason the guide gives for the tinted variant over plain Vendor:
nothing here carries a brand, and the set's central picture opens one container into five members
(a parent run and four children) that have to compare at a glance. Plain Vendor would leave all
five as grey boxes. The tint gives each its own identity colour, and that colour follows it into
every other picture it appears in.

Each prompt therefore opens with this paragraph verbatim:

> Style: clean corporate cloud-architecture diagram in the style of official vendor documentation.
> White background; containers are left white, outlined, titled in bold at the top, and never
> tinted, so colour never sinks a member into its container. Official product logos and service
> icons at consistent size. Large rounded-outline containers grouping phases. Within a container,
> each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its
> own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read
> apart from each other at a glance; the same member's colour carries into every picture that opens
> it further. A component that is not a member's own thing keeps a shared category colour instead,
> one hue per kind rather than per member: a light-brown tile for a named branded product with no
> logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or
> an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons
> (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A
> human figure standing for an end user or a named role is a fully coloured flat-illustration
> figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a
> photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no
> decoration that carries no meaning.

**No component here is a branded product**, so the paragraph's logo clause selects nothing and
every component gets a clean generic tile. Each prompt says so in its own exclusions rather than
relying on that being inferred.

**Zoom depth: level 0 plus one level-1 zoom.** Level 0 is the capstone and the five pieces that
compose it. The one level-1 zoom is prompt 2a, which opens the trainer's loss into the split
between the part of the correction the two experts can already supply and the part they cannot.
Nothing else opens further. A regeneration draws this same ladder.

**Held constant across the set:** the baseline checkpoint 30050 is always drawn as an existing,
finished thing in solid outline while the five new runs are always drawn grey dashed as not yet
existing, and the person doing the reading appears in exactly one subject picture and two process
pictures.

## Reference images

⬅️ [Previous](#art-direction) | 📋 [TOC](#table-of-contents) | [Next](#meaning-palette) ➡️

One, and it is a render from this set rather than a photograph of anything.

| File | What it is | How to use it |
|---|---|---|
| `diagrams/improve-r32-02-five-changes-four-runs-that-undo-one.png` | the first picture in this set drawn the way the whole set should look: white ground, white outlined containers titled in bold, one signature colour per run, pale tints, green and red state pills, small flat icons, no photography anywhere | attach it alongside every prompt below, as the look to match. It is the visual authority for this set; where it and the prose disagree about treatment, it wins |

Every subject in this scope is software and cached data, so no real-world physical object needs a
photograph to keep its identifying features honest. The generated pictures the comparison strips
carry (a cat and a dog, two dogs, one blended animal) are drawn as small flat illustrated
thumbnails, because what a strip is showing is its layout and the judgement made over it, never the
photographic quality of the picture inside a tile. A photographic tile also reads as a real
captured result, which none of these are.

## Meaning palette

⬅️ [Previous](#reference-images) | 📋 [TOC](#table-of-contents) | [Next](#glyph-vocabulary) ➡️

Under Vendor tinted the colour is per member, not per meaning, so this section pins which member
owns which colour. The same run keeps its colour in every picture it appears in.

| Carrier | Colour | Why it is that one |
|---|---|---|
| the parent run | blue | the run everything else is compared against, so it takes the calmest colour |
| C1, keep every cell | purple | |
| C2, train on all 50 steps | amber | |
| C3, plain error | teal | |
| C4, the eleven original pairs | green | |
| the baseline, checkpoint 30050 | solid green outline with a filled check | the only thing in the set that already exists |

Three shared category colours, one per kind rather than per member: light blue for a rule, a
threshold or a measurement; apricot for a person or anything a person does; grey dashed for
anything planned. A switch's state is a small pill, green reading ON and red reading OFF.

Everything in this set is planned except checkpoint 30050, so grey dashed is the default and the
member colours sit on top of it as the border and tint. A picture where nothing is dashed is
claiming the work is done.

## Glyph vocabulary

⬅️ [Previous](#meaning-palette) | 📋 [TOC](#table-of-contents) | [Next](#reading-axes) ➡️

Same glyph, same meaning, every picture. Tier says how much of the frame it may take.

| Glyph | Stands for | Tier |
|---|---|---|
| frozen model block | the large image model whose weights never change, marked with a snowflake | primary |
| adapter tile | the small trained component attached to it, marked with a flame while it trains | primary |
| run container | one training run: a white outlined box titled in bold, its member colour on the border | primary |
| setting row with a state pill | one of the five changes, ON or OFF | secondary |
| denoising track | the 50 generation steps as a long horizontal bar, noisy at the left, resolved at the right | primary |
| window bracket | the span of steps something is allowed to act over | secondary |
| comparison strip | four small flat picture thumbnails in a row, one seed | primary |
| label chip | one of clean, unclear, not two | secondary |
| measurement gear tile | an automatic read, light blue, always terminating at a table and never at a person | secondary |
| reader figure | the person judging, a flat coloured illustration, never a photograph and never a line icon | primary |
| gate marker | a point the work cannot pass until a check agrees, with both outcomes drawn | secondary |
| results table | where labels and measurements land | artifact |
| sheet | the filed comparison, the last thing in the set | artifact |

No product logos anywhere. Every component is bespoke research code, so the direction's logo clause
selects nothing.

## Reading axes

⬅️ [Previous](#glyph-vocabulary) | 📋 [TOC](#table-of-contents) | [Next](#devices-in-play) ➡️

Dominant axis, left to right: the path from cached data to a filed verdict. Secondary axis, top to
bottom: what is decided by a measurement above, what is decided by a person's eye below. Every
prompt states its own reading direction in its scene text.

## Devices in play

⬅️ [Previous](#reading-axes) | 📋 [TOC](#table-of-contents) | [Next](#subject-lane) ➡️

Subject lane: white outlined containers titled in bold, per-member border colour with a pale tint
fill, green and red state pills, thin single-accent arrows, small flat artifact icons beneath the
flow, grey dashed outlines on everything planned, an in-image legend where more than three colours
appear.

Process lane: the same cast with sequencing added, which under this direction means numbered stage
containers, gate markers drawn with both their outcomes, and the failing branch leaving the line
rather than being implied.

## Subject lane

⬅️ [Previous](#devices-in-play) | 📋 [TOC](#table-of-contents) | [Next](#prompt-1-subject-which-cells-the-adapter-is-allowed-to-learn-from) ➡️

The system this scope builds: which data the adapter learns from, what the five changes are, how
each checkpoint is rendered, how the renders are judged, and where the five runs execute.

### Prompt 1 (Subject): Which cells the adapter is allowed to learn from

⬅️ [Previous](#subject-lane) | 📋 [TOC](#table-of-contents) | [Next](#prompt-2-subject-the-five-changes-and-the-four-runs-that-each-undo-one) ➡️

[planned] ⏳ not rendered
Save as: `diagrams/improve-r32-01-which-cells-the-adapter-learns-from.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape image, 16:9, read left to right, showing how a pool of training examples is assembled and then filtered. On the left, a store of cached data. In the middle, that store expanded into a grid of small cells. On the right, a filter that removes some of those cells, and beyond it the smaller set that survives. None of this exists yet: it is the pool a planned training run will be built from, and the image should read as a plan rather than as a record of something already running.

Cast, described in full because none of it is a familiar object:
- The cache: a store of already-computed data on disk. It holds, for each pair of animals and each random seed, a 50-step sequence of what an image model predicted at every step, plus the picture that a single combined prompt produced on that seed.
- A cell: one pair of animals on one seed. The grid in the middle has 14 columns, one per animal pair, and up to 12 rows, one per seed. Eleven columns are the pairs already used for training, and they are, in this order: wolf and husky, turtle and tortoise, rabbit and hare, lion and tiger, horse and zebra, gorilla and chimpanzee, donkey and pony, dolphin and porpoise, crow and raven, crocodile and alligator, cheetah and cougar. Every one of those eleven is a pair of two look-alike animals, which is the single most important thing about this grid and must be visible at a glance: the two thumbnails in a cell should be hard to tell apart. Three further columns, marked as additions and set slightly apart, are lion and horse, wolf and horse, and bear and salmon. Those three are pairs of two obviously different animals, and none contains a cat or a dog. The three added columns hold 28 cells.
- The target picture: for each cell, the picture the combined prompt drew on that seed. This is what the adapter is trained to reproduce. Some of these target pictures are wrong: they show two dogs, or two cats, instead of one of each animal the pair names.
- The filter: an automatic check that crops each animal-shaped region out of a target picture and asks which of the pair's two animal names it matches. A cell passes when both of the pair's names win at least one region. A cell whose target picture shows the same animal twice fails.
- The surviving set: the cells the training run is allowed to learn from.

Flows and their direction: the cache feeds the grid. The grid feeds the filter. The filter has two outputs, one carrying the passing cells onward to the right, one carrying the failing cells away and out of the picture. Every cell reaching the right-hand side is one the training run will use.

Show clearly, since these are the points of the picture: failed cells must be visibly distinct from passing ones inside the grid, and it must be visible that failure is a property of the target picture rather than of the pair or of the seed, so failed cells are scattered across the grid rather than filling whole rows or columns. And the eleven original columns must read as look-alike pairs while the three added columns read as plainly different animals, because that contrast is what the picture is for: the check on the right is asking which of two animal names a region matches, and it is being asked hardest on eleven pairs where the two animals barely differ.

Text in the image: a title reading "which cells the adapter is allowed to learn from". Labels: "cached predictions and target pictures", "11 look-alike pairs", "up to 12 seeds each", "3 added pairs, plainly different animals, no cat, no dog, 28 cells", "does the target picture show both animals?", "kept", "dropped: the target shows the same animal twice", "the training set". Each of the fourteen columns is headed with its own pair's two animal names. One short note reading "planned, not yet built".

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no components other than those listed, no numbered step badges or progress indicators, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: the eleven original columns must carry the eleven pair names listed in the prompt and no invented ones, and each must show two animals that look alike. The three added columns must show two plainly different animals, so the contrast between the two groups is visible without reading the labels. The filter must sit between the grid and the surviving set, with two outputs, and the dropped cells must leave the picture rather than continue. The dropped cells must be scattered across the grid, never a whole row or column, because a wrong target picture is a property of one pair on one seed. The three added pairs must be readable as additions to eleven existing ones, not as a separate second pool.

### Prompt 2 (Subject): The five changes, and the four runs that each undo one

⬅️ [Previous](#prompt-1-subject-which-cells-the-adapter-is-allowed-to-learn-from) | 📋 [TOC](#table-of-contents) | [Next](#prompt-2a-subject-what-the-loss-weights-up) ➡️

[planned] 🖼️ rendered 2026-09-08 as `improve-r32-02-five-changes-four-runs-that-undo-one.png`
Save as: `diagrams/improve-r32-02-five-changes-four-runs-that-undo-one.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape image, 16:9, read top to bottom. The upper part shows one training run with five settings, all switched on. The lower part shows four further training runs, each identical to the one above except that exactly one of the five settings is switched back to how it used to be. Beside them, separate from all five, sits a sixth thing that is not a training run at all: an already-finished model checkpoint. Nothing in the upper or lower part exists yet; the sixth thing exists already.

Cast:
- The parent run, at the top: a planned training of a small adapter attached to a large frozen image model. It carries five settings, each drawn as its own labelled control, all in the on position:
  1. the pool includes three extra animal pairs beyond the original eleven,
  2. training cells whose target picture shows the wrong animals are excluded,
  3. training samples only the first 25 of the 50 denoising steps,
  4. the part of the training error that the two single-animal predictions cannot already account for is weighted up by a factor of three,
  5. the optimizer uses weight decay and keeps a slowly-updated average copy of the adapter's weights.
- Four child runs, in a row beneath: each one is a copy of the parent with a single control flipped back. The first keeps every cell instead of excluding any. The second trains on all 50 steps instead of the first 25. The third uses a plain training error with no weighting. The fourth uses only the original eleven pairs. The fifth setting, the optimizer one, is flipped in none of them and should be visibly shared by all five runs.
- The baseline, standing apart: a finished, already-trained checkpoint at training step 30,050. It is not being trained. It is what all five runs are compared against.

Flows and their direction: an arrow from the parent down to each of the four children, each arrow labelled with the single setting that child flips back. No arrow reaches the baseline, because the baseline is not derived from any of these runs.

Show clearly, since it is the point of the picture: exactly one control differs between the parent and each child, and it must be possible to see which one at a glance. The baseline must read as finished and existing while all five runs read as planned and not yet run.

Text in the image: a title reading "one run with everything on, four that each undo one thing". Labels: "parent", "keep every cell", "train on all 50 steps", "plain error, no weighting", "the original 11 pairs only", "baseline: checkpoint 30,050, already trained", "shared by all five: weight decay and an averaged copy of the weights". One short note reading "the five runs are planned; the baseline exists".

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no components other than those listed, no numbered step badges or progress indicators, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: each child must differ from the parent in exactly one visible control, and the label on its arrow must name that same control. The optimizer setting must be shown as shared rather than as a sixth flippable control, because no child flips it. The baseline must have no incoming arrow from the parent.

### Prompt 2a (Subject): What the loss weights up

⬅️ [Previous](#prompt-2-subject-the-five-changes-and-the-four-runs-that-each-undo-one) | 📋 [TOC](#table-of-contents) | [Next](#prompt-3-subject-the-two-ways-every-checkpoint-is-rendered) ➡️

[planned] 🖼️ rendered 2026-09-08 as `improve-r32-02a-what-the-loss-weights-up.png`
Save as: `diagrams/improve-r32-02a-what-the-loss-weights-up.png`
A level-1 zoom into the fourth setting of prompt 2, the weighted training error.

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape image, 16:9, read left to right, opening up one setting from a larger picture: how the training error is split into two parts and one of them is charged three times as much. A small locator in a corner shows this is a zoom into the training run's error term. Nothing here exists yet.

Cast:
- The target: at one denoising step, the difference between what the model predicts when given a single prompt naming both animals, and what it predicts when the two single-animal predictions are multiplied together. This difference is a direction in a high-dimensional space, drawn as one arrow.
- The plane of what the experts already say: the two single-animal predictions, each measured against what the model predicts with no prompt at all, span a flat plane. Drawn as a plane with those two directions marked on it.
- The split: the target arrow decomposed into two pieces, the piece lying in that plane and the piece sticking out of it. These two pieces add back to the target arrow.
- The adapter's prediction: a second arrow, its own split into the same two pieces, drawn beside the target's.
- The two error terms: the gap between target and prediction inside the plane, counted once; the gap between target and prediction outside the plane, counted three times.
- The step mask, along the bottom: the run of 50 denoising steps drawn as a long horizontal strip starting as noise on the left and ending as a finished picture on the right, with only the first 25 steps marked as the ones training samples from.

Flows and their direction: the target and the prediction both enter the split. The two gaps leave the split and meet at a summation that produces one number, the training loss. A line from the step mask into the summation shows that only steps in the first half contribute at all.

Show clearly, since it is the point of the picture: the out-of-plane piece is the part of the correction the two single-animal predictions cannot supply no matter how they are re-weighted, and it is the piece charged three times. The in-plane piece is charged once.

Text in the image: a title reading "the error the run charges three times". Labels: "what the joint prompt adds", "what the two single prompts already span", "in the plane, charged once", "out of the plane, charged three times", "the adapter's prediction", "training loss", "only the first 25 of 50 steps". One short note reading "planned, not yet built".

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no equations or mathematical notation, no components other than those listed, no numbered step badges, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: the two split pieces must visibly add back to the whole arrow, or the picture teaches that the correction is one of them rather than both. The factor of three must attach to the out-of-plane piece and to nothing else. The step mask must cover the first half of the strip, the noisy end, not the finished end.

### Prompt 3 (Subject): The two ways every checkpoint is rendered

⬅️ [Previous](#prompt-2a-subject-what-the-loss-weights-up) | 📋 [TOC](#table-of-contents) | [Next](#prompt-4-subject-the-strip-and-the-read-that-is-done-by-eye) ➡️

[planned] 🖼️ rendered 2026-09-08 as `improve-r32-03-two-ways-every-checkpoint-is-rendered.png`
Save as: `diagrams/improve-r32-03-two-ways-every-checkpoint-is-rendered.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape image, 16:9, showing two ways of running the same 50-step image generation, one above the other so they can be compared step for step. Both start from the same trained adapter on the left and end in a generated picture on the right. The upper way exists and is in use today; the lower way is planned and needs one new capability before it can run.

Cast:
- The adapter: a small trained component attached to a large frozen image model, feeding a correction into the generation as it runs.
- Each way of running is a long horizontal strip divided into 50 equal segments, starting as visual noise at its left end and resolving into a clear picture at its right end.
- Upper strip, the way in use today: the adapter's correction is fed into every one of the 50 segments. The generation is fully deterministic, adding no fresh randomness at any step. A single guidance strength of 7.5 applies across the whole strip.
- Lower strip, the planned way: the adapter's correction is fed into the first 25 segments only, and the frozen model finishes the remaining 25 alone. Fresh randomness is added at every step. Guidance strength is 7.5 only from step 5 to step 35, and 1.0 before and after that span, so it must be visible as a raised band over the middle of the strip with lower ends.
- At the right end of each strip, one finished picture.

Flows and their direction: from the adapter into the upper strip along its whole length. From the adapter into the lower strip's first half only. From the frozen model into the lower strip's second half. Both strips run left to right.

Show clearly, since it is the point of the picture: the two strips are the same length and the same 50 steps, and differ in exactly three ways, each of which must be readable without the caption: where the adapter acts, whether fresh randomness is added, and where guidance is strong.

Text in the image: a title reading "the same checkpoint, rendered two ways". Labels: "trained adapter", "frozen model", "50 steps", "noise", "finished picture", "correction on every step", "correction on the first 25 steps, frozen model finishes", "no fresh randomness", "fresh randomness every step", "guidance 7.5 throughout", "guidance 7.5 from step 5 to 35, 1.0 outside", "in use today", "planned: needs the guidance span added". 

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no components other than those listed, no numbered step badges beyond the step-number labels described, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: on the lower strip the adapter's line must stop at the halfway point and the frozen model's must continue to the end, and the guidance band must be raised only over the middle span with visibly lower ends. Both strips must be equal in length and divided into the same number of segments, since the two ways differ in what happens during the run and never in how long it is.

### Prompt 4 (Subject): The strip, and the read that is done by eye

⬅️ [Previous](#prompt-3-subject-the-two-ways-every-checkpoint-is-rendered) | 📋 [TOC](#table-of-contents) | [Next](#prompt-5-subject-five-runs-on-one-machine-overnight) ➡️

[planned] ⏳ not rendered
Save as: `diagrams/improve-r32-04-the-strip-and-the-read-by-eye.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape image, 16:9, read left to right, showing how generated pictures are judged. On the left, a stack of comparison rows. In the middle, the step that hides which is which. On the right, a person assigning one of three labels to each picture. Beneath and to the side, two automatic measurements that feed into the ordering of the work but never into the judgement itself. Nothing here exists yet.

Cast:
- A comparison row: four picture tiles side by side, all generated from the same random seed. One tile is the picture a single prompt naming both animals produced, one is the picture produced with no correction at all, one is the picture the existing checkpoint produced, and one is the picture a new run produced. There are 25 such rows in total, seventeen for cat-and-dog seeds and eight for elephant-and-penguin seeds. The four tiles are drawn identically: same size, same plain neutral border, no colour coding, no per-column heading, and no fixed left-to-right order. This is not a stylistic choice and it is the whole subject of the picture, so any treatment that lets a reader tell the four apart defeats it.
- What the tiles actually show, which must not be uniformly good: most rows are failures. Draw a mixture across the visible rows. Some rows show two clearly different animals. Some show two animals that are hard to tell apart, or one animal with features of both blended into a single creature. Some show the same animal twice. At least one row's combined-prompt tile is itself wrong, showing the same animal twice, since on some seeds the reference the run was trained toward is faulty.
- The shuffle: a step that hides the four tiles' names and reorders them within each row, so the judgement cannot know which condition it is looking at.
- The person: one human figure, looking at one shuffled row and choosing among three labels. This is the only place in the whole system where a person decides anything.
- The three labels, drawn as three distinct choices: "clean", meaning both named animals are present and each is clearly itself with nothing unnatural on inspection; "unclear", meaning two animals but not clearly the two named ones, or clearly both but with unnatural details; "not two", meaning one animal, a single blended creature, or the same animal twice.
- Two automatic measurements, off to one side: a detector that counts how many animal-shaped regions a picture contains, and a check that crops each region and asks which of the two names it matches. Both feed a table that orders the rows for viewing.

Flows and their direction: rows enter the shuffle, leave the shuffle with their labels hidden, and reach the person. The person's chosen label leaves toward a results table on the far right. The two automatic measurements feed the same table from the side, on a connection that must read as ordering the rows rather than as deciding anything, and they must not touch the person.

Show clearly, since these are the points of the picture: the automatic measurements never reach the label, arriving at the table alongside it and never before it; the four tiles in every row are visually indistinguishable from one another, with no colour, no heading and no fixed position that would identify which condition made which; and the rows show mostly failures rather than mostly successes, because a picture whose every row already looks right gives the reader no reason the judgement is needed at all.

Text in the image: a title reading "the labels are assigned by eye, and only by eye". Labels: "both animals in one prompt", "no correction", "existing checkpoint", "this run", "25 seeds", "names hidden, order shuffled", "clean", "unclear", "not two", "counts animal-shaped regions", "which name does each region match", "orders the rows, never decides". One short note reading "planned, not yet built".

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no components other than those listed, no numbered step badges or progress indicators, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: the four tiles in a row must be indistinguishable, carrying no per-column colour, no per-column heading and no fixed order, because the picture is about a judgement made without knowing which condition produced which tile and any coding of the columns contradicts it. The rows must show a mixture including blended single creatures and the same animal twice, not four clean pairs on every row. The two automatic measurements must connect to the results table and never to the person or to the label, because the whole design rests on the count being unable to see the defect. The shuffle must sit between the rows and the person, not after the labelling. Exactly one human figure appears.

### Prompt 5 (Subject): Five runs on one machine, overnight

⬅️ [Previous](#prompt-4-subject-the-strip-and-the-read-that-is-done-by-eye) | 📋 [TOC](#table-of-contents) | [Next](#subject-capstone-from-cached-cells-to-one-comparison-sheet) ➡️

[planned] 🖼️ rendered 2026-09-08 as `improve-r32-05-five-runs-on-one-machine-overnight.png`
Save as: `diagrams/improve-r32-05-five-runs-on-one-machine-overnight.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape image, 16:9, read left to right, showing where five planned training runs will execute and what has to be checked before each one starts. On the left, a set of machines of which only one is usable. In the middle, that machine running five jobs one after another. On the right, the results arriving at an experiment tracker. Nothing here has run yet.

Cast:
- Three compute machines drawn side by side, all of the same powerful kind. The first is busy with someone else's work. The second is broken: its hardware reports no readable state at all. The third is free and is the one everything below runs on. Only the third may carry any job in this picture.
- A launcher: a single script, stored on the large shared data disk rather than on the machine itself, which starts the five runs one after another and continues to the next one if any of them dies.
- Three checks the launcher performs before starting anything, drawn as a short checklist: is the data disk under 90 percent full, is the correct machine-specific version of the language runtime being used, and is the target device genuinely idle rather than holding another process.
- Five jobs in a queue on the free machine, run strictly in order: the parent first, then the child that trains on all 50 steps, then the child that keeps every cell, then the child with the plain error, then the child with the original eleven pairs. Each takes roughly two hours.
- The experiment tracker on the right: a service receiving, from each run separately, its own record with sample pictures and running measurements.

Flows and their direction: the launcher reaches only the free machine. The checklist gates the start of the queue. Each of the five jobs sends its own record to the tracker. The busy machine and the broken machine have no connections at all.

Show clearly, since it is the point of the picture: the five jobs run one after another on a single device rather than side by side on several, and the order they run in is fixed and readable.

Text in the image: a title reading "five runs, one machine, one night". Labels: "busy with another job", "hardware fault, unusable", "free", "launcher on the shared data disk", "disk under 90 percent", "machine-specific runtime", "device genuinely idle", "parent", "all 50 steps", "every cell", "plain error", "11 pairs only", "about 2 hours each", "one record per run". One short note reading "planned; nothing has been launched".

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no real machine names or hostnames, no components other than those listed, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: the busy and broken machines must carry no job and no incoming connection, since the point is that one device is available. The five jobs must be sequential on one machine, never parallel across machines. The three checks must gate the start of the queue rather than sit beside it decoratively.

### Subject capstone: From cached cells to one comparison sheet

⬅️ [Previous](#prompt-5-subject-five-runs-on-one-machine-overnight) | 📋 [TOC](#table-of-contents) | [Next](#process-lane) ➡️

[planned] ⏳ not rendered
Save as: `diagrams/improve-r32-06-capstone-cells-to-comparison-sheet.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a wide landscape image, 16:9, putting the whole of one research effort on one page. It reads left to right along the main path, and top to bottom in what decides things: measurements sit above, a person's judgement sits below. Every part of this is planned except one already-finished checkpoint, which must read as existing.

The main path, left to right:
1. A store of cached data on disk, holding for each animal pair and each random seed a 50-step record of model predictions plus the picture a combined prompt drew.
2. A filter that removes any training example whose target picture shows the wrong animals, for instance two dogs where the pair names a cat and a dog. Removed examples leave the path.
3. A training stage producing five separate runs of a small adapter attached to a large frozen image model. One is the parent, carrying five settings at once. Four are children, each identical to the parent except for one setting switched back.
4. A rendering stage where every finished adapter, and separately the already-existing checkpoint, is run in two different ways: once with the correction applied across all 50 generation steps with no fresh randomness, and once with the correction applied only to the first 25 steps, the frozen model finishing, fresh randomness added at every step, and strong guidance confined to a middle span.
5. Comparison rows, one per random seed: four picture tiles side by side, the combined-prompt picture, the uncorrected picture, the existing checkpoint's picture, and this run's picture. 25 rows in total.
6. A person examining the rows with the tile names hidden and the order shuffled, assigning each picture exactly one of three labels. The three labels are named with these exact words and no paraphrase, because they are fixed vocabulary used unchanged across the whole project: "clean" (both animals present, each clearly itself, nothing unnatural), "unclear" (two animals but not clearly the right two, or the right two with unnatural details) and "not two" (one animal, a single blended creature, or the same animal twice).
7. A final sheet on the far right, filed as a lasting result: the comparison rows for whichever run came out best, shown under both ways of rendering.

Also on the page, entering the path from one side rather than sitting in it:
- The already-existing checkpoint, at training step 30,050. It joins the path at the rendering stage and never passes through the filter or the training stage. It is the thing all five runs must beat.
- Two automatic measurements taken on every rendered picture, a count of animal-shaped regions and a check of which name each region matches. Both connect to the results table beside the labels, and neither connects to the person. They order the work; they never judge it.

Show clearly, since these are the claims the picture has to carry: exactly one setting differs between the parent and each child; the existing checkpoint bypasses training and enters at rendering; the automatic measurements never reach the judgement; and the judgement is made by a person on shuffled, unnamed pictures.

Text in the image: a title reading "from cached cells to one comparison sheet". Section labels along the path: "cached predictions and target pictures", "drop the examples with wrong target pictures", "one parent, four children, one setting each", "render two ways", "four pictures per seed, 25 seeds", "judged by eye, names hidden", "the sheet". The three label chips read exactly "clean", "unclear" and "not two", with their one-line definitions beneath them. Other labels: "already trained: checkpoint 30,050", "counts regions", "matches names to regions", "orders the work, never judges it". A small in-image legend explaining that everything is planned except the existing checkpoint.

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no real machine names or hostnames, no equations or mathematical notation, no components other than those listed, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: the three label chips must read the exact words "clean", "unclear" and "not two", since those are the project's fixed vocabulary and a paraphrase in the overview picture contradicts the five documents that use them. The existing checkpoint must enter at the rendering stage with no line reaching back into the filter or the training stage, because it is a fixed baseline rather than a run. The two automatic measurements must terminate at the results table and never touch the person or the labels. Exactly one human figure appears, at the judging step. Each child must differ from the parent in one readable setting. The legend must mark everything as planned except the checkpoint, which is the only part that already exists.

## Process lane

⬅️ [Previous](#subject-capstone-from-cached-cells-to-one-comparison-sheet) | 📋 [TOC](#table-of-contents) | [Next](#prompt-1-process-build-the-instruments-and-prove-each-one-can-fail) ➡️

The same cast as the subject lane, with the order of work drawn over it. History lives in
`diagrams/process-versions/`. This lane is regenerated whole and never patched.

Five plans are drawn as four stages: plans 01 and 02 are grouped as one stage because both build
an instrument and neither runs an experiment, and their gates are checked together before anything
is rendered. The other three plans are one stage each.

### Prompt 1 (Process): Build the instruments, and prove each one can fail

⬅️ [Previous](#process-lane) | 📋 [TOC](#table-of-contents) | [Next](#prompt-2-process-read-the-baseline-before-spending-a-night-of-training) ➡️

[planned] 🖼️ rendered 2026-09-08 as `improve-r32-p01-build-the-instruments-and-prove-they-fail.png`
Save as: `diagrams/improve-r32-p01-build-the-instruments-and-prove-they-fail.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape image, 16:9, read left to right, showing the first stage of a research project: building the tools it will measure with, and testing each tool by pointing it at something whose answer is already known. Nothing here has been done yet, and the image should read as a plan.

Cast, in the order the work happens:
- Four switches being added to existing software. Three go into the program that trains a small adapter: one that skips chosen training examples, one that restricts training to the first half of a 50-step generation, one that charges part of the training error three times. The fourth goes into the program that generates pictures: it holds strong guidance over a middle span of steps only.
- A check attached to each of the three trainer switches: each one prints how many things it actually selected. A switch that selects nothing is the failure this check exists to catch, because the program would still finish and still report plausible numbers.
- A check attached to the fourth switch: the same picture generated with the new switch set to cover everything, held beside a picture generated before the switch existed, with a tolerance of 6 units of brightness between them.
- A second tool being built beside the first: the reading apparatus. Rows of four pictures, a step that hides which picture came from where, a person choosing among three labels, and a results table.
- A test of that second tool: a small set of pictures whose correct labels were written down in advance, run through the whole apparatus, with each label compared against what was expected.
- Two gates, drawn as points the work cannot pass until the checks agree: one after the switches, one after the reading apparatus.

Flows and their direction: the switches are built, then checked, then reach the first gate. The reading apparatus is built, then tested on the known set, then reaches the second gate. Both gates lead onward, off the right edge of the image, toward work not shown here.

Show clearly, since it is the point of the picture: each check has two outcomes and both are drawn, the pass continuing right and the failure returning to the thing it tested. The work does not proceed past a gate whose check failed.

Text in the image: a title reading "build the instruments, and prove each one can fail". Labels: "skip chosen examples", "train on the first half only", "charge part of the error three times", "strong guidance in the middle only", "how many did it actually select?", "does it still draw the old picture?", "hide which is which", "clean, unclear, not two", "examples whose answers we already know", "gate: every switch selects something", "gate: the read separates the known cases". One short note reading "planned; nothing has been built".

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no real filenames or code, no components other than those listed, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: every check must have both outcomes drawn, with the failing one returning to what it tested rather than trailing off. The two gates must sit between the building and everything downstream, not beside them, because the whole point is that nothing proceeds past a failed check.

### Prompt 2 (Process): Read the baseline before spending a night of training

⬅️ [Previous](#prompt-1-process-build-the-instruments-and-prove-each-one-can-fail) | 📋 [TOC](#table-of-contents) | [Next](#prompt-3-process-the-night-the-chain-runs-and-the-morning-after) ➡️

[planned] 🖼️ rendered 2026-09-08 as `improve-r32-p02-read-the-baseline-first.png`
Save as: `diagrams/improve-r32-p02-read-the-baseline-first.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape image, 16:9, read left to right, showing a cheap measurement taken deliberately before an expensive one, because its answer can change what the expensive one is compared against. Nothing here has been done yet.

Cast, in the order the work happens:
- One already-trained model checkpoint on the left. It is the only thing in this image that already exists; everything else is planned.
- A step where a rule is written down and fixed in the software before any picture is made: at least one more good result, and none of the existing good results lost. Drawn as a small sealed note attached to the software rather than to a person, with a line saying it cannot be changed later without the change being visible.
- Two generation paths from that one checkpoint, drawn as two horizontal 50-step runs, the upper one the way the checkpoint is run today and the lower one the alternative way. Both start from the same starting randomness, drawn as a single source feeding both.
- Rows of four pictures, 25 rows, produced from each path.
- The hide-which-is-which step, then a person assigning one of three labels, then a count of good results for each path.
- A fork at the far right with two labelled outcomes: the alternative way wins, in which case it becomes the thing everything later is compared against; or it does not, in which case the original way stays.

Flows and their direction: one checkpoint into two paths, both from the same starting randomness; both into rows of pictures; both through the hiding step to one person; the two counts meeting at the fork.

Show clearly, since it is the point of the picture: both paths draw from a single source of starting randomness, because two paths started from different randomness would not be comparable. And the rule is fixed before the first picture exists, not after the counts are known.

Text in the image: a title reading "read the baseline before spending a night of training". Labels: "already trained", "the rule, fixed in software first", "the way it is run today", "the alternative way", "same starting randomness", "25 seeds", "names hidden", "good results, today's way", "good results, the alternative", "the alternative wins: it becomes the baseline", "it does not: today's way stays". One short note reading "about one hour of computing; it can change what twelve hours are compared against".

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no real machine names, no components other than those listed, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: a single source of starting randomness must visibly feed both generation paths. The rule must attach to the software and sit before the first picture in reading order, since a rule written after the counts is not a rule. Both fork outcomes must be drawn and labelled; neither is the expected one.

### Prompt 3 (Process): The night the chain runs, and the morning after

⬅️ [Previous](#prompt-2-process-read-the-baseline-before-spending-a-night-of-training) | 📋 [TOC](#table-of-contents) | [Next](#prompt-4-process-file-the-sheet-and-close-the-work) ➡️

[planned] ⏳ not rendered
Save as: `diagrams/improve-r32-p03-the-night-the-chain-runs.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a wide landscape image, 16:9, read left to right, showing the largest single stage of a research project: five training runs executed one after another overnight on one machine, then read the next morning. Nothing here has been done yet.

Cast, in the order the work happens:
- A pre-flight check comparing five configuration files: one parent and four children, where each child must differ from the parent by exactly one line. Drawn as five short documents with the differing line marked on each child, and a gate that stops the work if any child differs by more than one line.
- A second pre-flight check printing what each configuration actually builds: how many training examples per animal pair, which generation steps it will train on, how heavily it charges part of the error. A gate that stops the work if a child's example count differs for any reason other than the one line it changed.
- A short trial run of a few hundred steps on the chosen machine, drawn small, followed by it being stopped. Four things are read off it: the experiment tracker received the run, a saved model file reached the shared disk, the time per step is about a quarter of a second, and the machine identifier in the log is the one that was chosen.
- The five full runs in a queue on one machine, running strictly one after another, roughly two hours each, through the night. If one dies the queue continues to the next and records that it died.
- The next morning: five saved models, each fed through the reading apparatus, producing rows of four pictures under two different generation settings.
- A person assigning labels with the names hidden, ten separate sittings.
- Two readings at the end: the parent against the baseline, which asks whether anything helped at all; and the parent against each of the four children, which asks which single change was responsible.

Flows and their direction: the two pre-flight gates, then the trial run, then the queue, then the reading, then the labelling, then the two readings. Time runs left to right throughout, with the overnight portion visibly the longest.

Show clearly, since it is the point of the picture: the five runs are sequential on one machine and not parallel across several; the two gates sit before the queue and not after it; and the final two readings are different questions rather than one, the first asking whether anything helped and the second asking which change did it.

Text in the image: a title reading "the night the chain runs, and the morning after". Labels: "one line different, and only one", "what does each configuration actually build?", "a few hundred steps, then stopped", "tracker received it", "model file on the shared disk", "about a quarter second per step", "parent", "all 50 steps", "every example kept", "plain error", "fewer pairs", "about two hours each", "five saved models", "two generation settings", "names hidden, ten sittings", "did anything help?", "which change did it?". One short note reading "about twelve hours of one machine".

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no real machine names or hostnames, no components other than those listed, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: the five runs must be sequential on one machine, never parallel across machines, and the overnight span must be visibly the longest part of the image. Both pre-flight gates must sit before the queue. The two final readings must be drawn as two distinct questions, since collapsing them into one is exactly the mistake the four children exist to prevent.

### Prompt 4 (Process): File the sheet, and close the work

⬅️ [Previous](#prompt-3-process-the-night-the-chain-runs-and-the-morning-after) | 📋 [TOC](#table-of-contents) | [Next](#process-capstone-one-route-from-an-unclean-picture-to-a-filed-answer) ➡️

[planned] ⏳ not rendered
Save as: `diagrams/improve-r32-p04-file-the-sheet-and-close-the-work.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape image, 16:9, read left to right, showing the last stage of a research project: turning scattered working files into the one thing somebody who was not involved can look at. Nothing here has been done yet.

Cast, in the order the work happens:
- On the left, the scattered state: ten result tables and five saved models, drawn as a loose pile, with a note saying these live in places nobody will reopen.
- An assembly step producing two large sheets, one per generation setting. Each sheet is rows of four pictures with a short written judgement under every single picture. Every result that was judged appears, including the failures.
- A separate small measurement taken alongside: the size of the correction the model produces, one number per animal pair, drawn as a short table.
- A person comparing that table against actual examples before writing any sentence about it, drawn as the table and three example pictures held side by side, with two outcomes: a pattern survives the comparison and one sentence is written, or it does not and the honest statement is that the measurement explains nothing here.
- A filing step: the two sheets, the table, and a written card describing what each file shows and where it came from, all placed together in one labelled folder.
- A final check reading the sheet against the written judgements it came from, confirming the sheet claims no more than they do.

Flows and their direction: the scattered pile into the assembly, the assembly into the sheets, the sheets and the measurement table into the filing step, the filing step into the folder. The person's comparison sits above the path and feeds one sentence into the card.

Show clearly, since it is the point of the picture: the failures stay on the sheet, drawn explicitly rather than implied; and both outcomes of the comparison are drawn, including the one where the measurement explains nothing.

Text in the image: a title reading "file the sheet, and close the work". Labels: "ten tables, five models, nobody reopens these", "one sheet per generation setting", "a judgement under every picture", "the failures stay on the sheet", "size of the correction, per pair", "lay the numbers against real examples first", "a pattern survives: write one sentence", "none survives: say it explains nothing", "the card: what it shows and where it came from", "does the sheet claim more than the evidence?". One short note reading "planned; the sheet is the only thing this work files".

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no real filenames or paths, no components other than those listed, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: the failures must be visibly present on the sheet rather than implied, because a sheet showing only successes is the omission this stage exists to prevent. Both outcomes of the person's comparison must be drawn, since the one where the measurement explains nothing is the more likely of the two and is a real result.

### Process capstone: One route from an unclean picture to a filed answer

⬅️ [Previous](#prompt-4-process-file-the-sheet-and-close-the-work) | 📋 [TOC](#table-of-contents)

[planned] ⏳ not rendered
Save as: `diagrams/improve-r32-p05-capstone-one-route-to-a-filed-answer.png`

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a wide landscape image, 16:9, showing one continuous route through a piece of research, from the problem that started it to the single filed answer at the end. It reads as one main line with a few branches leaving it, in the manner of a route map. Every stop is planned; nothing on this route has happened yet.

The main line, in order, each stop named and its gate drawn where it has one:
1. The problem: a trained model produces pictures that are supposed to show a cat and a dog, and mostly do not. An automatic counter says they are fine. The counter is drawn beside the problem with a line showing it disagrees with what a person sees.
2. Build the instruments: four switches added to existing software, and the reading apparatus. Gate: every switch prints how many things it selected, and the reading apparatus returns the right answer on examples whose answers were already known.
3. Read the baseline: the existing model run two different ways, judged with the names hidden. A branch leaves the line here, labelled with the case where simply running it differently is already better, in which case the thing everything later is compared against moves.
4. The night: five training runs one after another on one machine. Gate before it: each of the four variant runs differs from the main one by exactly one line, and a short trial run confirms the machine and the speed. A branch leaves the line, labelled with the case where a run dies and the queue continues without it.
5. The morning: every saved model rendered two ways, and a person labelling with the names hidden, in ten separate sittings.
6. The two readings: did anything help, and which single change did it. A branch leaves the line here, labelled with the case where nothing helped, which closes this route and names the two more expensive routes that remain.
7. The filed answer: two sheets with a judgement under every picture, a card describing them, and a folder they live in.

Also on the map, entering the line rather than sitting on it: the existing model checkpoint, joining at stop 3 and again at stop 5 as the thing to beat, never passing through the training stop.

Show clearly, since these are the claims the picture has to carry: every gate is a place the route can stop, and its branch is drawn and labelled rather than implied; the person appears at exactly two stops, 3 and 5, and nowhere else; the automatic counter appears only at stop 1, where it is shown disagreeing with a person, and never again on the main line.

Text in the image: a title reading "one route from an unclean picture to a filed answer". Stop names: "the problem", "build the instruments", "read the baseline", "the night", "the morning", "the two readings", "the filed answer". Branch labels: "running it differently is already better", "a run dies, the queue continues", "nothing helped: two costlier routes remain". A small in-image legend saying that every stop is planned and that only the existing checkpoint already exists.

Exclusions: no photographs or photorealistic rendering anywhere, any picture shown inside a tile or frame being a small flat illustrated thumbnail, no company or product logos, no brand marks, no real machine names, no components other than those listed, no placeholder or lorem-ipsum text, no watermark.
```

Faithfulness note: every gate must show its branch leaving the line, labelled, because a route drawn with only its happy path claims the work cannot fail. The person must appear at exactly two stops and the automatic counter only at the first, where the disagreement between them is the reason the whole route exists. The existing checkpoint must join at stops 3 and 5 without ever passing through the training stop.
