# When Does the Outcome Lock In: illustrated map

Five pictures of the measuring tool this scope builds, plus a capstone. Everything here is
`[planned]` and drawn dashed, because nothing exists yet and the map says so truthfully. Every
prompt is self-contained: paste any one into ChatGPT on its own and it comes back matching its
siblings and the parent scope's map.

## Table of contents
- [Abstraction chain](#abstraction-chain)
- [Art direction](#art-direction)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Devices in play](#devices-in-play)
- [Subject lane](#subject-lane)
- [Process lane](#process-lane)

## Abstraction chain
This scope: subject capstone "The whole measuring tool on one page", process capstone "Five stops
to a verdict". The parent chain is [the parent map](../diagram-prompts.md), whose art
direction, palette and glyphs this file inherits; the new pieces here zoom into the reading
side of the parent's cache stack.

## Art direction
**Vivid circuit, inherited from the parent map.** Inheritance is the rule for a child scope, so
no new direction is picked: the same flow-colour language must carry a reader from the parent
capstone into this one without relearning anything. The direction's style paragraph is embedded
verbatim at the head of every prompt below.

## Meaning palette
Inherited unchanged. Blue the uncorrected path, amber the correction and anything carrying it,
green measured-and-passed, red measured-and-failed or a null. A control is a hollow amber
outline, never a fifth colour.

## Glyph vocabulary
Reused from the parent: cache stack, denoising track, window bracket, scorer lens, outcome tile,
researcher persona. Two new glyphs join the family:

| Glyph | Stands for |
|---|---|
| Endpoint spyglass | the distilled jump-to-the-end model (LCM-SDXL) that reads a mid-run state ahead to its settled image; kin to the scorer lens, both are reading tools |
| Calibration gauge | the agreement dial between the spyglass's ending and the teacher's true finish-the-run ending, with its two thresholds |

## Reading axes
Dominant axis, left to right: data flow, from a cached state to a filed figure. Secondary axis,
top to bottom: trust, measuring tools above, checked verdicts and figures below.

## Devices in play
Subject lane: flow colour coding with an in-image legend, truth anchors where a real frame
exists, dashed outlines on every platform while pieces are `[planned]`.
Process lane: the same cast under operation, plus numbered step badges, phase containers, and
status chips ("Completed", "In progress", "Not started", "Blocked"); every chip reads "Not
started" in version 01.

Piece-to-plan mapping: prompt 1 belongs to [plan 01](plans/01-basins-by-hand.md), prompt 3 to
[plans 01 and 02](plans/02-the-free-probe.md), prompt 2 to
[plan 03](plans/03-wire-the-oracle.md), prompt 4 to
[plan 04](plans/04-calibrate-the-instrument.md), prompt 5 to
[plan 05](plans/05-the-grid-and-the-figures.md).

## Subject lane

### Prompt 1 (Subject): The valley fork

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: a single wide landscape cross-section drawn as two soft valleys separated by a narrow
ridge, all platforms and outlines dashed to mean planned. In the left valley floor an outcome
tile of one blended animal bordered blue; in the right valley floor an outcome tile of two
separate animals bordered amber. On the ridge crest a small glowing state marker with a blue
dashed arrow pulling it left and an amber dashed arrow pulling it right. Two thin gray test
arrows nudge the marker either side, each continuing as a faint line into a different valley.

Cast: two basins (the endings a trajectory can commit to), the ridge (where the nearest ending
stops being unique), the state marker (one cached mid-run latent), the two outcome tiles.

Flows: blue dashed pull toward the blended ending, amber dashed pull toward the composed
ending, thin solid gray for the perturbation nudges. Legend lower left.

Text in the image: title banner "which ending wins from here". Labels: "blended ending",
"composed ending", "the ridge", "one cached state", "a small nudge decides". Legend: "blue:
uncorrected pull", "amber: corrected pull", "gray: test nudge".
```

### Prompt 2 (Subject): The endpoint spyglass

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: left, the cache stack with three thin trays labeled by family; one state card slides out.
Centre, a large spyglass glyph on a dashed platform receives the state card plus two small
prompt cards stacked beside it. Right, two settled outcome tiles, one per prompt card, each
connected by a dashed arrow from the spyglass, with a small tag on each arrow naming its series.

Cast: cache stack (per-step latents saved for later analysis), the endpoint spyglass (the
distilled model that jumps a state to its settled image), two prompt cards ("a cat and a dog";
"a cat" / "a dog" as a pair), two outcome tiles.

Flows: amber dashed from state to spyglass to tiles, the reading path. Thin gray from the
prompt cards into the spyglass, conditioning. Legend lower left.

Text in the image: title banner "reading the ending early". Labels: "cached state, any step",
"endpoint spyglass (distilled)", "joint series: does it compose from here", "expert series: which
animal wins", "settled frame, not a blur".
```

### Prompt 3 (Subject): The free tests

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: two side-by-side dashed panels. Left panel "three endings by hand": one state card forks
into three copies (one nudged up, one untouched, one nudged down), each walking a short denoising
track to its own small outcome tile; a bracket compares the three tiles. Right panel "the
posterior mean settles": a denoising track with a small blurred thumbnail above each early step
sharpening toward the late steps, and a curve beneath that descends and flattens, with a marker
where it flattens.

Cast: the state card and its two nudged copies, three short denoising tracks (finishing the run with
the base sampler), outcome tiles, the drift curve (how far the running estimate of the final
image moves per step), the settling marker.

Flows: blue dashed for the three finish-the-run walks, thin gray for the nudges, amber marker
at the settling step. Legend lower left.

Text in the image: title banner "two tests that need no new model". Labels: "nudge, finish,
compare", "same ending: committed", "different endings: on the fence", "estimate stops moving
here".
```

### Prompt 4 (Subject): The calibration gauge

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: centre, a large dashed gauge glyph with two marked thresholds on its dial. Feeding it from the
left, paired cards: the spyglass's settled frame above, the teacher's true finish-the-run frame
below, repeated as three thin stacks labeled by family. Right of the gauge, three small status
chips, one per family, all drawn hollow because no verdict exists yet.

Cast: the calibration gauge (agreement between student and teacher endings), the paired frame
cards, the two thresholds (an absolute minimum agreement, and no family far below another), per-family status
chips.

Flows: amber dashed from spyglass frames, blue dashed from teacher frames, meeting at the
gauge. Legend lower left.

Text in the image: title banner "trust is purchased, never assumed". Labels: "student ending",
"teacher ending", "threshold one: minimum agreement", "threshold two: families comparable", "verdict: adopt,
shrink, or fall back".
```

### Prompt 5 (Subject): The three-timestamp strip

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one long denoising track, noise at the left, finished image at the right, on a dashed
platform. Above it three annotations in order: a window bracket over steps 0 to 10, a hollow
amber marker labeled with a question mark somewhere between 10 and 18, and a shaded band over
steps 18 to 36. Beneath the strip, three thin curves, one per family, rising and flattening at
different steps, drawn faint to mean expected shape, not data.

Cast: the denoising track, the window bracket (when the correction is allowed to act), the
speciation marker (when the ending is decided; position unknown, hence hollow and questioned),
the divergence band (when paths visibly separate), the per-family compose-rate curves.

Flows: none between components; this piece is a timeline. Legend lower left.

Text in the image: title banner "three timestamps, one strip". Labels: "correction acts here,
0 to 10", "decided here?", "paths visibly separate, 18 to 36", "one curve per family, faint:
expected, not measured".
```

### Subject capstone: The whole measuring tool on one page

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: left to right on dashed platforms. The cache stack with its three family trays releases a
state card. It forks: up to the endpoint spyglass, down to a finish-the-run teacher panel (a
short denoising track ending in a tile). Both endings meet at the calibration gauge with its two thresholds. Past the gauge, a scorer lens reads the settled frames, feeding a wide results panel on
the right containing a miniature of the three-timestamp strip and three faint per-family
curves. A small valley-fork vignette sits above the spyglass as its meaning. Bottom right, a
figure board receives one dashed line from the results panel.

Cast: cache stack, state card, endpoint spyglass, teacher denoising track, calibration gauge, scorer
lens, three-timestamp strip, per-family curves, valley fork vignette, figure board.

Flows: amber dashed for the spyglass reading path, blue dashed for the teacher path, both into
the gauge; green solid only on the gauge's pass arrow toward the lens, meaning measured and
passed is the only way through; thin gray from results to figure board. Legend lower left.

Text in the image: title banner "when does the outcome lock in". Labels: "cached steps, three
families", "read the ending early", "true ending, walked", "agreement, two thresholds", "count the
animals", "three timestamps", "to the paper's reserved figure places".
```

## Process lane

Version 01, authored with the plan set. The checks and forks below come from the plans' engagement checks, never invented. Regenerations go whole into
`diagrams/process-versions/` per DIAGRAM_PROMPTS_FORMAT.md.

### Prompt 1 (Process): Prove the premise for free

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: a phase container titled "prove it cheap" holding two numbered stops. Stop 1, badge
"44": the valley-fork vignette beside a small three-endings triplet, status chip "Not started";
below it a red branch arrow labeled "endings differ late: STOP, premise dead" leaving the
container downward, and a green arrow onward. Stop 2, badge "45": a drift curve settling, a
small scatter with an identity line, chip "Not started"; a red branch "settles after diverging:
reassess the story" loops back to the container edge.

Cast: the two stops as the same glyphs the subject lane uses (valley fork, endings triplet,
drift curve), step badges, status chips, check branches.

Flows: green solid arrows for the pass path, red dashed for each check's fail branch. Legend
lower left.

Text in the image: title banner "prove the premise for free". Labels: "44: basins by hand",
"45: the free test", "STOP: picture wrong", "reassess: ordering broke".
```

### Prompt 2 (Process): Build and buy trust

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: a phase container titled "build and buy trust" holding two numbered stops. Stop 3,
badge "46": the endpoint spyglass on its platform with three small assert seals beside it and
a tiny side-by-side first-short-run card, chip "Not started"; a red branch "an assert fires or the first short run fails: fix before calibrating" loops back into the stop. Stop 4, badge "47": the calibration gauge with its two thresholds; from it THREE outgoing labeled arrows, a green "adopt", an amber "shrink:
only families and steps that pass", and a blue "fall back: teacher everywhere". All three
arrows leave the container rightward; none is a stop.

Cast: spyglass, assert seals, first-short-run card, calibration gauge, step badges, status chips.

Flows: green solid pass path; red dashed fail loop on stop 3; the three verdict arrows in
green, amber, blue. Legend lower left.

Text in the image: title banner "trust is purchased". Labels: "46: wire the endpoint predictor",
"47: calibrate", "adopt", "shrink", "fall back".
```

### Prompt 3 (Process): Deliver the grid and the figures

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: a phase container titled "deliver" holding one wide stop, badge "48": the denoising track
with its three annotations (window bracket 0 to 10, hollow speciation marker, divergence band
18 to 36), beside a stack of five small figure cards and a filmstrip, chip "Not started". Two
outgoing arrows: green "speciation at or before 10: gap explained", red "speciation inside 18
to 36: story killed, said plainly"; both continue rightward to the figure board, because the
figures ship either way.

Cast: denoising track, figure cards, filmstrip, figure board, step badge, status chip.

Flows: both verdict arrows reach the figure board; the red one is dashed but arrives. Legend
lower left.

Text in the image: title banner "the figures ship either way". Labels: "48: the grid and the
figures", "gap explained", "story killed, captioned honestly", "to the register".
```

### Process capstone: Five stops to a verdict

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: a metro-style journey left to right, five stations with numbered badges 44 to 48, each
drawn as its stop's key glyph in miniature (valley fork, drift curve, spyglass, gauge, denoising
track), all chips "Not started". Red branch lines leave stations 44 and 46 (the two hard
stops); station 47 fans into the three verdict tracks that all rejoin before station 48; after
48 the line terminates at the figure board. Beneath the track, one thin amber line runs from
station 45 directly to station 48, labeled as the free test's cross-check riding along.

Cast: five stations, the branch lines, the three verdict tracks, the figure board terminus.

Flows: the main line in green, hard-stop branches in red dashed, the verdict fan in green,
amber and blue, the cross-check line in amber. Legend lower left.

Text in the image: title banner "five stops to a verdict". Labels: "44 basins", "45 free test", "46 wire", "47 calibrate", "48 grid and figures", "cross-check rides along".
```
