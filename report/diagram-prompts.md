# The report's illustrated map

The eight-rung reading arc in [the report index](00-INDEX.md#read-it-in-this-order) drawn as
pictures: four of what a finding is made of, eight of the arc itself one rung at a time, and a
capstone closing each lane.

15 prompts · 0 rendered · 14 waiting · 1 rendering

Every prompt below is self-contained: paste any one into ChatGPT on its own and it comes back
matching its siblings. This map inherits its direction, palette and glyphs from
[the project's illustrated map](../plans/diagram-prompts.md), so a render from here sits beside a
render from there without a seam.

## Table of contents

- [Abstraction chain](#abstraction-chain)
- [Art direction](#art-direction)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Devices in play](#devices-in-play)
- [Subject lane](#subject-lane)
  - [Prompt 1 (Subject): What a finding is made of](#prompt-1-subject-what-a-finding-is-made-of)
  - [Prompt 2 (Subject): The four shelves and what each holds](#prompt-2-subject-the-four-shelves-and-what-each-holds)
  - [Prompt 3 (Subject): The four ways out of a finding](#prompt-3-subject-the-four-ways-out-of-a-finding)
  - [Prompt 4 (Subject): What one corrector step is, inside one noise level](#prompt-4-subject-what-one-corrector-step-is-inside-one-noise-level)
  - [Subject capstone: The findings folder on one page](#subject-capstone-the-findings-folder-on-one-page)
- [Process lane](#process-lane)
  - [Prompt 1 (Process): Rung 1, when the product works](#prompt-1-process-rung-1-when-the-product-works)
  - [Prompt 2 (Process): Rung 2, where it breaks and how we know](#prompt-2-process-rung-2-where-it-breaks-and-how-we-know)
  - [Prompt 3 (Process): Rung 3, the four checks that make it causal](#prompt-3-process-rung-3-the-four-checks-that-make-it-causal)
  - [Prompt 4 (Process): Rung 4, what the correction is made of](#prompt-4-process-rung-4-what-the-correction-is-made-of)
  - [Prompt 5 (Process): Rung 5, the adapter and the checkpoint to show](#prompt-5-process-rung-5-the-adapter-and-the-checkpoint-to-show)
  - [Prompt 6 (Process): Rung 6, the pairs it never trained on](#prompt-6-process-rung-6-the-pairs-it-never-trained-on)
  - [Prompt 7 (Process): Rung 7, the ten cheaper fixes that did not work](#prompt-7-process-rung-7-the-ten-cheaper-fixes-that-did-not-work)
  - [Prompt 8 (Process): Rung 8, what is still open](#prompt-8-process-rung-8-what-is-still-open)
  - [Prompt 9 (Process): How the corrector's number was made](#prompt-9-process-how-the-correctors-number-was-made)
  - [Process capstone: The whole arc as one line](#process-capstone-the-whole-arc-as-one-line)

## Abstraction chain

Navigation: 📋 [TOC](#table-of-contents) | [Next](#art-direction) ➡️

1. [Closing the Compositional Gap](../plans/diagram-prompts.md): subject "The whole system on one page", process "One route from measuring tools to submission"
2. [The report's illustrated map](diagram-prompts.md): subject "The findings folder on one page", process "The whole arc as one line"

The parent map draws the machine. This one draws what the machine returned, which is why its
process lane is ordered by the arc a reader follows rather than by the order the runs happened.

## Art direction

Navigation: ⬅️ [Previous](#abstraction-chain) | 📋 [TOC](#table-of-contents) | [Next](#meaning-palette) ➡️

**Vivid circuit**, inherited from the parent map rather than chosen again. The parent's reason
holds here too: three kinds of traffic (an uncorrected path, a correction added on top, matched
substitutes that should do nothing) need colour to be told apart at a glance, and the same
direction carries the process lane once the sequencing devices are added.

Zoom depth: one level. Each arc rung is one picture and does not open into children. Where a rung
needs more room, the parent map already holds the zoom (its prompts 2a, 4a, 4b, 4c).

The direction's style paragraph is embedded verbatim at the head of every prompt below.

## Meaning palette

Navigation: ⬅️ [Previous](#art-direction) | 📋 [TOC](#table-of-contents) | [Next](#glyph-vocabulary) ➡️

Four colours, each meaning exactly one thing, in every image of both lanes. Identical to the
parent map.

| Colour | Means |
|---|---|
| Blue | the uncorrected path: what plain product-of-experts does on its own |
| Amber | the correction `r_t`, and anything carrying or learning it |
| Green | measured and passed: a bar met, a verdict of support |
| Red | measured and failed, or a null: a bar not met, a control reading what luck would give |

> A null here means the numbers came out the same with the thing as without it.

A finding with no file written yet is drawn as a **dashed outline** at full size, never as a fifth
colour and never faded. It is a result that exists and is unwritten, not a weak one.

## Glyph vocabulary

Navigation: ⬅️ [Previous](#meaning-palette) | 📋 [TOC](#table-of-contents) | [Next](#reading-axes) ➡️

Inherited from the parent map, plus three glyphs this lane needs because it draws findings rather
than components.

| Glyph | Stands for |
|---|---|
| Outcome tile | a generated picture, either one fused animal or two separate animals |
| Correction coil | `r_t`, the step-by-step gap between the joined prompt and plain product-of-experts |
| Denoising track | the 50 denoising steps, noise at the left, finished image at the right |
| Multiplier dial | λ, how much of the correction is added back |
| Adapter chip | the low-rank cross-attention adapter that learns the correction |
| Scorer lens | the detector that counts animals in a picture |
| Finding card | one answered question: a claim strip, a picture well, a number strip, a verdict chip |
| Verdict chip | a small pill on a finding card reading support, null or inconclusive |
| Shelf | one of the four question-group folders, holding its finding cards upright |

No product logos anywhere in this set. Every component is bespoke research code.

## Reading axes

Navigation: ⬅️ [Previous](#glyph-vocabulary) | 📋 [TOC](#table-of-contents) | [Next](#devices-in-play) ➡️

Dominant axis is left to right, and in the process lane it is the arc's order, rung 1 at the left
edge to rung 8 at the right. It is deliberately not time: the runs did not happen in this order,
and any picture implying they did is wrong.

The secondary axis, top to bottom, encodes the verdict: what passed sits above the mainline, what
came back null or inconclusive sits below it.

## Devices in play

Navigation: ⬅️ [Previous](#reading-axes) | 📋 [TOC](#table-of-contents) | [Next](#subject-lane) ➡️

**Subject lane:** flow colour coding, an in-image legend, and truth anchors (a real generated
picture drawn at the edge where one exists). No sequencing devices.

**Process lane:** the same cast, plus numbered rung badges 1 to 8, phase containers grouping the
rungs into the three acts (the failure, the fix, the alternatives), and verdict chips.

## Subject lane

Navigation: ⬅️ [Previous](#devices-in-play) | 📋 [TOC](#table-of-contents) | [Next](#prompt-1-subject-what-a-finding-is-made-of) ➡️

What this folder is made of: the shape of one finding, the four shelves they sit on, and the ways
a reader leaves a finding for the rest of the repository.

### Prompt 1 (Subject): What a finding is made of

[built] ⚙️ rendering
Save as: `diagrams/report-01-what-a-finding-is-made-of.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, one reading direction, left to right. Centre of the frame holds one large rounded card standing upright on a platform, divided into four stacked horizontal bands. The top band is a wide text strip. The second band is a picture well holding a small image of two separate animals, a cat and a dog, side by side in one frame. The third band is a narrow strip holding a numeral and a small ruler glyph beside it. The fourth band holds a small green pill at its left edge. To the left of the card, outside it, a small clipboard glyph on its own platform with an arrow entering the card's fourth band. To the right of the card, three small document glyphs stacked vertically, each receiving a thin line from the card's lower edge.

Cast: the finding card (one answered question), its claim strip, its picture well (the figure that shows the claim), its number strip (the statistic with its units), its verdict chip, the clipboard (the bar written before the run), three document glyphs (the plan, the runbook recipe, the context entry the finding links out to).

Flows: an amber dashed arrow from the clipboard into the verdict chip, meaning the bar was set before the number existed and decides the chip. Thin solid gray lines from the card down to the three documents, meaning reference, not computation. Legend inside the image, lower left.

Text in the image: title banner "one finding, four parts". Labels: "the claim, in one sentence", "the figure that shows it", "the number, with its units", "verdict", "support", "the bar, written first", "the plan that ran it", "the recipe that redoes it", "what the words mean". Legend: "amber: the bar decides the verdict", "gray: a link out".

Exclusions: no product logos, no numbered step badges, no components other than those listed, no charts or curves, no placeholder gibberish text, no watermark.
```

Faithfulness note: the arrow must run from the clipboard into the verdict chip and in no other direction, because the bar is set before the number and the number never sets the bar. The picture well must contain two separate animals, not one fused animal, or the card illustrates a failure while its chip reads support.

### Prompt 2 (Subject): The four shelves and what each holds

Navigation: ⬅️ [Previous](#prompt-1-subject-what-a-finding-is-made-of) | 📋 [TOC](#table-of-contents) | [Next](#prompt-3-subject-the-four-ways-out-of-a-finding) ➡️

[built] ⏳ not rendered
Save as: `diagrams/report-02-the-four-shelves.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, four rounded soft-tinted panels of unequal width standing side by side across the frame, each a shelf holding small upright cards. The first panel is narrow and holds two cards. The second is narrow and holds two cards. The third is narrow and holds two cards. The fourth is very wide, taking roughly half the frame, and holds ten cards in two rows of five. Each card carries a small colored pill at its foot: in the first panel one green and one red pill, in the second one green and one red, in the third one red and one gray, and in the fourth panel eight gray pills, one red and one green. Above the four panels, floating clear of them, three cards drawn with dashed outlines and no pill at all.

Cast: four shelves (the four question-group folders), sixteen finding cards, verdict chips (green support, red null or failed bar, gray inconclusive), three dashed cards (results with a verdict on disk and no finding file written).

Flows: none. This picture is an inventory, not a route. Legend inside the image, lower left.

Text in the image: title banner "sixteen findings on four shelves". Panel titles: "when does the outcome lock in", "how long to train the adapter", "does the fix reach unseen pairs", "is the gap the samplers or the models". Labels: "2", "2", "2", "10", "written up", "verdict on disk, not written up". Legend: "green: support", "red: null or bar not met", "gray: inconclusive", "dashed: no finding file yet".

Exclusions: no arrows of any kind, no product logos, no numbered step badges, no components other than those listed, no charts or curves, no placeholder gibberish text, no watermark.
```

Faithfulness note: the fourth panel must be visibly the widest and hold ten cards against two on each of the others, because ten of the sixteen findings are ways of trying to avoid the correction and that imbalance is the substance of the folder. The three dashed cards must sit outside every panel, since they belong to no shelf yet. No arrows: a shelf does not feed another shelf.

### Prompt 3 (Subject): The four ways out of a finding

Navigation: ⬅️ [Previous](#prompt-2-subject-the-four-shelves-and-what-each-holds) | 📋 [TOC](#table-of-contents) | [Next](#subject-capstone-the-findings-folder-on-one-page) ➡️

[built] ⏳ not rendered
Save as: `diagrams/report-03-the-four-ways-out.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, radial composition. One finding card sits at the centre on a raised platform, drawn as a rounded card with four stacked bands. Four rounded soft-tinted panels sit at the four corners of the frame, each holding one distinct glyph: upper left a checklist glyph, upper right a dictionary glyph with a small open book, lower left a wrench-and-terminal glyph, lower right a server rack glyph with a small clock beside it. A thin solid gray line runs from the centre card out to each of the four panels. A small cartoon researcher figure stands at the lower centre edge, facing the card.

Cast: the finding card at the centre, four destination panels (the plan that ran the experiment, the context entry that defines the words, the runbook recipe that regenerates the number, the environment fact the run depended on), the researcher persona (the reader arriving).

Flows: four thin solid gray lines from the card to the four panels, each bidirectional with an arrowhead at both ends, meaning the finding names its source and the source names the finding back. Legend inside the image, lower left.

Text in the image: title banner "four ways out of one finding". Labels: "the finding", "the plan that ran it, and the bar it pre-registered", "the words: what a chimera and a compose rate are", "the recipe that regenerates this number", "the cluster fact the run depended on", "the reader". Legend: "gray: a link, readable from either end".

Exclusions: no product logos, no colored flow lines, no numbered step badges, no components other than those listed, no charts or curves, no placeholder gibberish text, no watermark.
```

Faithfulness note: every one of the four lines must carry an arrowhead at both ends, because each finding links out and each target links back; a one-way arrow would teach that the report is a leaf, which is the thing this picture exists to deny. Exactly four destinations, no fifth.

### Prompt 4 (Subject): What one corrector step is, inside one noise level

Navigation: ⬅️ [Previous](#prompt-3-subject-the-four-ways-out-of-a-finding) | 📋 [TOC](#table-of-contents) | [Next](#subject-capstone-the-findings-folder-on-one-page) ➡️

[built] ⏳ not rendered
Save as: `diagrams/report-04-what-one-corrector-step-is.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, one wide rounded soft-tinted panel filling the centre, titled as one noise level, with a thin vertical dashed divider at its right edge separating it from a narrow strip on the far right. Inside the panel, a single latent glyph drawn as a small square tile of noise sits at the left. From it, three identical short hops run left to right along a blue rail, each hop drawn as a pair of stacked arrows merging into the next tile: a long straight arrow labelled as the score push and a short curled arrow labelled as the fresh noise. After the third hop a fourth tile sits on a slightly raised platform, drawn larger than the others. Below the rail and clear of it, a small ellipsis glyph and the text for k of them. Above the fourth tile, two network glyphs side by side on their own small platforms, one blue and one amber, each sending a thin line down into that tile and no line onward. To the right of the dashed divider, in the narrow strip, one downward arrow leaves the fourth tile and ends at a fresh noise tile at the strip's foot.

Cast: the latent at the start of the level, three drawn hops standing for k Langevin steps, the settled latent on its raised platform, the two network calls made at that settled point (the product-of-experts prediction in blue and the joint-prompt prediction in amber), the ordinary reverse step leaving for the next level.

Flows: a blue rail carrying the latent through the hops and into the settled tile. Two thin gray lines from the network glyphs down into the settled tile, meaning read, not move. One blue arrow crossing the dashed divider, meaning the reverse step to the next noise level. Legend inside the image, lower left.

Text in the image: title banner "one noise level, k corrector steps, then one reverse step". Panel title: "inside one of the 50 levels". Labels: "the latent, as it arrives", "push along the score", "add a little fresh noise", "the settled latent", "k of them", "what the product predicts here", "what the joint prompt predicts here", "on to the next level". Legend: "blue: the product path", "amber: the joint prompt", "gray: a reading that changes nothing".

Exclusions: no arrows from either network glyph back into the rail, no numbered step badges, no charts or curves, no product logos, no components other than those listed, no placeholder gibberish text, no watermark.
```

Faithfulness note: neither network glyph may send a line back into the rail. Both calls are observers made at the settled point, and drawing either one feeding the chain would turn the picture into a guided sampler, which is exactly what this corrector is not. The hops must all sit inside the one panel and the reverse step must be the only arrow crossing the divider, because the whole point is that the chain moves the latent without advancing the denoising clock.

### Subject capstone: The findings folder on one page

Navigation: ⬅️ [Previous](#prompt-3-subject-the-four-ways-out-of-a-finding) | 📋 [TOC](#table-of-contents) | [Next](#process-lane) ➡️

[built] ⏳ not rendered
Save as: `diagrams/report-04-the-folder-on-one-page.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, three zones left to right inside one wide rounded panel. The left zone holds four shelves of unequal width standing side by side, the rightmost shelf twice the width of the others, cards upright on each with small green, red and gray pills at their feet, and three dashed cards floating above them. The centre zone holds one enlarged finding card on a raised platform, divided into four bands: a text strip, a picture well showing a cat and a dog as two separate animals, a number strip with a small ruler, and a verdict chip. A magnifier bracket connects one small card on the shelves to this enlarged card. The right zone holds four small panels stacked vertically, each with a distinct glyph: a checklist, an open book, a wrench-and-terminal, a server rack. Thin gray double-headed lines run from the enlarged card to all four. At the lower edge, a small cartoon researcher figure walks left to right beneath the three zones.

Cast: the four shelves and their sixteen cards, the three unwritten results as dashed cards, one finding card enlarged, its four bands, the four destinations out of it, the researcher persona.

Flows: a magnifier bracket from a shelf card to the enlarged card, meaning zoom and not flow. Thin gray double-headed lines from the enlarged card to each of the four destinations. No colored flow lines anywhere. Legend inside the image, lower left.

Text in the image: title banner "what this folder holds and how to leave it". Labels: "16 findings, 4 shelves", "3 results not written up yet", "one finding", "the claim", "the figure", "the number", "verdict", "the plan", "the words", "the recipe", "the cluster". Legend: "green: support", "red: null", "gray: inconclusive", "dashed: no finding file yet".

Exclusions: no product logos, no numbered step badges, no arrows between shelves, no charts or curves, no components other than those listed, no placeholder gibberish text, no watermark.
```

Faithfulness note: this scene must reassemble prompts 1, 2 and 3 without adding a component none of them held, and the shelf widths, the pill colours and the four destinations must match those three exactly. The magnifier bracket must read as zoom rather than as a flow arrow, or the picture claims a shelf produces a finding.

## Process lane

Navigation: ⬅️ [Previous](#subject-capstone-the-findings-folder-on-one-page) | 📋 [TOC](#table-of-contents) | [Next](#prompt-1-process-rung-1-when-the-product-works) ➡️

The eight rungs of the reading arc, one picture each, closed by the whole arc as one line. History:
`diagrams/process-versions/`. This lane is regenerated, never patched: it is keyed to the arc in
[the report index](00-INDEX.md#read-it-in-this-order), so any change to the arc rewrites the lane.

A rung is marked `[built]` when a written finding file backs every claim in its picture, and
`[planned]` when the rung's evidence is on disk but no finding file has been written yet. Rungs 1,
2, 3 and 8 are the `[planned]` ones, which is the same hole the arc marks with ⬜.

### Prompt 1 (Process): Rung 1, when the product works

[planned] ⏳ not rendered
Save as: `diagrams/report-p01-when-the-product-works.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, left to right, with a numbered round badge reading "1" in the upper left corner. On the left, two small prompt cards stacked vertically, each feeding its own prediction glyph; the two predictions meet at a multiply node and leave it as a single blue line. Above them, one wider prompt card feeds one prediction glyph directly and leaves as an amber line. Both lines travel right into a shared scorer lens glyph. To the right of the lens, two rows of outcome tiles: the upper row eight tiles each showing a butterfly above a meadow, every tile bordered green; the lower row eight tiles each showing a cat and a dog as two separate animals, every tile bordered green. Beneath the lower row a wide green counter strip. The whole scene sits inside one rounded panel.

Cast: prompt cards, prediction glyphs, the multiply node, the scorer lens (the detector that counts animals), sixteen outcome tiles, a counter strip.

Flows: blue dashed line for the product path from the two separate prompts through the multiply node to the lens. Amber dashed line for the joined-prompt path to the same lens. Legend inside the image, lower left.

Text in the image: title banner "rung 1: this is what working looks like". Panel title: "the two columns everything else is measured against". Labels: "a butterfly", "a meadow", "multiply", "plain product of experts", "a cat and a dog", "joined prompt", "counts two", "8 of 8 seeds", "8 of 8 seeds". Legend: "blue: the product path", "amber: the joined prompt", "green: composed".

Exclusions: no red anywhere in this image, no fused or hybrid animal, no adapter chip, no correction coil, no product logos, no charts or curves, no placeholder gibberish text, no watermark.
```

Faithfulness note: every tile must be bordered green and no fused animal may appear, because this rung's entire job is to show the successful case before any failure is introduced. The blue line must carry the butterfly pair and the amber line the cat-and-dog pair, because plain product-of-experts composes the control and only the joined prompt composes cat and dog.

### Prompt 2 (Process): Rung 2, where it breaks and how we know

Navigation: ⬅️ [Previous](#prompt-1-process-rung-1-when-the-product-works) | 📋 [TOC](#table-of-contents) | [Next](#prompt-3-process-rung-3-the-four-checks-that-make-it-causal) ➡️

[planned] ⏳ not rendered
Save as: `diagrams/report-p02-where-it-breaks.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, left to right, with a numbered round badge reading "2" in the upper left corner. On the left, two prompt cards feeding two prediction glyphs into a multiply node, leaving as a blue line. The blue line runs into a large grid occupying the centre of the frame: seventeen rows by eight columns of small square tiles, every tile bordered red, each tile showing one single animal whose left half carries one species' features and whose right half carries another's. On the right, a scorer lens glyph on a raised platform with a small numeral "1" displayed inside it, and beneath the lens a small panel holding a stack of labelled thumbnails with green ticks and red crosses beside them. A thin amber line runs from that panel up into the lens.

Cast: prompt cards, prediction glyphs, the multiply node, the blend gallery grid (17 animal pairs by 8 seeds), the scorer lens, the labelled validation set that the lens was checked against.

Flows: blue dashed line from the multiply node into the grid, meaning every tile in the grid was made this one way. Amber dashed line from the labelled set into the lens, meaning the instrument was validated before it was trusted. Legend inside the image, lower left.

Text in the image: title banner "rung 2: one animal where two were asked for". Labels: "a cat", "a dog", "multiply", "17 pairs x 8 seeds = 136 renders", "counts one", "the instrument, checked against labelled images first". Legend: "blue: the product path", "red: counted one animal", "amber: what validated the counter".

Exclusions: no correction coil, no adapter chip, no multiplier dial, no green tiles in the grid, no product logos, no charts or curves, no placeholder gibberish text, no watermark.
```

Faithfulness note: every tile in the grid must show a single fused body rather than two animals sharing a frame, because a blend and a miss are different failures and only the blend is what this project is about. The validation panel must feed the lens and not the grid, or the picture claims the labelled set produced the renders.

### Prompt 3 (Process): Rung 3, the four checks that make it causal

Navigation: ⬅️ [Previous](#prompt-2-process-rung-2-where-it-breaks-and-how-we-know) | 📋 [TOC](#table-of-contents) | [Next](#prompt-4-process-rung-4-what-the-correction-is-made-of) ➡️

[planned] ⏳ not rendered
Save as: `diagrams/report-p03-four-checks.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, with a numbered round badge reading "3" in the upper left corner. Across the top of the frame, a subtract node outputs a thick amber coil which fans out into four rounded soft-tinted panels arranged left to right below it. Panel one holds a circular dial with a pointer and four outcome tiles beneath it in a row, the leftmost bordered red and each successive tile more clearly showing two separate animals until the rightmost is bordered green. Panel two holds one solid amber coil above four hollow amber outlines of identical size, each of the five feeding one outcome tile: the tile under the solid coil is bordered green, the four under the hollow outlines are bordered red. Panel three holds a long horizontal denoising strip, dark and grainy at its left end and a clear picture at its right, with a rounded bracket over its leftmost fifth, that bracket alone tinted green while a second bracket drawn over the strip's right end is tinted red. Panel four holds three small chart frames side by side, each drawn as a plain empty axis pair, joined beneath by a single green tick.

Cast: the subtract node and the correction coil, the multiplier dial (how much correction is added back), four hollow substitutes of matched size, the denoising track and two window brackets, three chart frames standing for three independent reads.

Flows: amber dashed lines from the coil into all four panels, meaning all four checks act on the same object. Legend inside the image, lower left.

Text in the image: title banner "rung 3: four ways of asking whether the correction is the cause". Panel titles: "more of it", "the same size, pointing elsewhere", "only early", "three reads agree". Labels: "r_t: the gap", "multiplier", "own correction", "other pair", "other seed", "steps shuffled", "random vector", "noise", "finished image", "steps 0 to 10", "steps 40 to 50". Legend: "amber: the correction", "green: composed", "red: did not".

Exclusions: no adapter chip anywhere in this image, no plotted data points or curves inside the chart frames, no product logos, no numbers on the dial, no placeholder gibberish text, no watermark.
```

Faithfulness note: the four substitutes must be drawn at the same size as the real correction and differ only in being hollow, because they are matched in size and differ only in direction; drawing them smaller would teach that the control failed for lack of magnitude. Panel three's green bracket must sit at the strip's left end and the red one at its right, because the correction works early and not late. No adapter appears at this rung: the correction here is injected from a stored joint-prompt run, not predicted.

### Prompt 4 (Process): Rung 4, what the correction is made of

Navigation: ⬅️ [Previous](#prompt-3-process-rung-3-the-four-checks-that-make-it-causal) | 📋 [TOC](#table-of-contents) | [Next](#prompt-5-process-rung-5-the-adapter-and-the-checkpoint-to-show) ➡️

[built] ⏳ not rendered
Save as: `diagrams/report-p04-what-its-made-of.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, two rounded soft-tinted panels side by side, with a numbered round badge reading "4" in the upper left corner. The left panel holds a shallow tilted plane drawn in blue, with three blue arrows lying flat inside it, and one amber arrow rising out of the plane at a modest angle; a small pie glyph beside them is filled roughly one third amber and two thirds blue. The right panel holds a wide oval field with three labelled clusters of small dots: a blue cluster at the lower left, an amber cluster in the middle, and a green cluster at the upper right, with a short arrow running from the blue cluster to the amber cluster and a longer dashed arrow continuing from the amber cluster toward the green one. Two outcome tiles sit at the field's edge, one bordered blue showing a fused animal and one bordered amber showing two separate animals, each joined by a thin line to its own cluster.

Cast: the span of the three predictions the product already has (the blue plane), the correction (the amber arrow), the share of it lying outside that span (the pie), the scorer's image space (the oval field), three condition clusters, two outcome tiles.

Flows: a solid amber arrow from the blue cluster to the amber cluster, meaning the correction moves the render. A dashed amber arrow from the amber cluster toward the green cluster, meaning the remaining distance was not closed. Legend inside the image, lower left.

Text in the image: title banner "rung 4: what the correction is made of, and where it lands". Panel titles: "outside what the product already has", "in the scorer's image space". Labels: "the three predictions the product has", "the correction", "0.374 outside the span", "plain product of experts", "corrected", "joint prompt", "one fused animal", "two separate animals", "distance not closed". Legend: "blue: uncorrected", "amber: corrected", "green: the joint prompt".

Exclusions: no adapter chip, no denoising track, no multiplier dial, no product logos, no axis ticks or gridlines on the oval field, no placeholder gibberish text, no watermark.
```

Faithfulness note: the amber arrow must leave the blue plane at a modest angle rather than perpendicular to it, and the pie must read as roughly a third amber, because 0.374 of the correction's squared norm lies outside the span and most of it does not. The arrow from corrected to joint must be dashed and must not reach the green cluster, because the correction moves the render toward the joint prompt without arriving.

### Prompt 5 (Process): Rung 5, the adapter and the checkpoint to show

Navigation: ⬅️ [Previous](#prompt-4-process-rung-4-what-the-correction-is-made-of) | 📋 [TOC](#table-of-contents) | [Next](#prompt-6-process-rung-6-the-pairs-it-never-trained-on) ➡️

[built] ⏳ not rendered
Save as: `diagrams/report-p05-which-checkpoint.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, with a numbered round badge reading "5" in the upper left corner. Across the full width runs a long horizontal rail marked with evenly spaced tick posts, left to right. An adapter chip glyph sits on the rail and is drawn three times along it: near the left at a tick, in the middle at a tick, and near the far right at a tick. Above the middle chip sits a green outcome tile showing a cat and a dog as two separate animals, with a small green flag planted at that tick. Above the leftmost chip, a red outcome tile showing one fused animal. Above the rightmost chip, an outcome tile showing two separate animals but visibly hazy and washed out, bordered red. Below the rail, a thick amber wedge grows steadily thicker from left to right along its whole length. To the left of the rail, outside it, a stack of stored prediction discs feeds the first chip with a thin gray line.

Cast: the training rail (steps of training, left to right), three adapter chips at three checkpoints, three outcome tiles, the green flag (the checkpoint chosen for the paper), the widening amber wedge (the adapter's weight norm), the cache stack (the stored targets it trains on).

Flows: a thin gray line from the cache stack into the first chip, meaning the targets are read, not computed. No other flows. Legend inside the image, lower left.

Text in the image: title banner "rung 5: further is not better". Labels: "training steps", "early: not yet composing", "the checkpoint we show: 7 of 8 seeds", "late: composes but drifts", "weight norm, growing without bound", "stored targets". Legend: "green: composed and sharp", "red: failed or drifted", "amber: how large the adapter's weights have grown".

Exclusions: no plotted curve or axis pair, no product logos, no denoising track, no scorer lens, no numbered step badges other than the rung badge, no placeholder gibberish text, no watermark.
```

Faithfulness note: the amber wedge must widen monotonically across the entire rail including past the green flag, because the weights keep growing after the useful checkpoint and that divergence between "still growing" and "no longer improving" is the finding. The rightmost tile must show two animals and still be bordered red, since the late failure is loss of fidelity, not loss of composition.

### Prompt 6 (Process): Rung 6, the pairs it never trained on

Navigation: ⬅️ [Previous](#prompt-5-process-rung-5-the-adapter-and-the-checkpoint-to-show) | 📋 [TOC](#table-of-contents) | [Next](#prompt-7-process-rung-7-the-ten-cheaper-fixes-that-did-not-work) ➡️

[built] ⏳ not rendered
Save as: `diagrams/report-p06-unseen-pairs.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, with a numbered round badge reading "6" in the upper left corner. On the left, a rounded soft-tinted panel holding a grid of small pair cards, each showing two animal silhouettes; one adapter chip sits beneath the panel with an amber line rising into it. From the chip, a single amber line travels right and splits into three branches. The top branch ends at a rounded panel of outcome tiles all bordered green, labelled with a large numeral. The middle branch ends at a panel of tiles mixed green and red. The bottom branch ends at a panel of tiles all bordered red. Beside the three panels, a vertical ruler glyph runs top to bottom with three marks on it, the top mark aligned with the green panel and the bottom mark with the red one. At the far right, a small pair card drawn with a dashed outline and a question mark beside a second ruler that reads the same at both ends.

Cast: the training pair pool, the adapter chip, three destinations (pairs sharing both words with training, pairs sharing one word, pairs sharing none), a fit ruler, and a second ruler standing for the interaction-strength read that failed to separate pairs.

Flows: one amber line from the pool into the chip and out of it in three branches, meaning one adapter meets three kinds of unseen pair. Legend inside the image, lower left.

Text in the image: title banner "rung 6: it carries to some pairs and not others". Labels: "the pairs it trained on", "one pooled adapter", "seen words, seen pairing", "seen words, new pairing", "no seen word", "0.97", "0.84", "0.80", "how well it fits", "does interaction strength predict which pairs blend? no". Legend: "green: composed", "red: did not", "dashed: the question the measure could not answer".

Exclusions: no product logos, no denoising track, no multiplier dial, no plotted curve or axis pair, no placeholder gibberish text, no watermark.
```

Faithfulness note: the three fit numbers must descend left to right as 0.97, 0.84, 0.80 against their three branches, and the second ruler must visibly read the same at both of its ends, because the interaction-strength measure spans a wide range across pairs that all fail identically. Drawing that second ruler as separating the pairs would invert the null.

### Prompt 7 (Process): Rung 7, the ten cheaper fixes that did not work

Navigation: ⬅️ [Previous](#prompt-6-process-rung-6-the-pairs-it-never-trained-on) | 📋 [TOC](#table-of-contents) | [Next](#prompt-8-process-rung-8-what-is-still-open) ➡️

[built] ⏳ not rendered
Save as: `diagrams/report-p07-ten-cheaper-fixes.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, with a numbered round badge reading "7" in the upper left corner. A single blue rail enters from the left edge carrying a small outcome tile with one fused animal. The rail passes through three rounded soft-tinted panels arranged left to right, each panel a gateway with a barrier arm across the rail. The first panel holds a shaking-lattice glyph repeated three times. The second holds two prediction glyphs joined by a small blending node, repeated three times. The third holds a fan of four candidate tiles with a scorer lens above them, repeated four times. Each of the ten repeated glyphs carries its own small pill: nine gray or red, one green. All three barrier arms are down and the blue rail continues past all of them unchanged, still carrying a fused animal at the right edge. Above the rail and clear of it, a separate short amber rail carries a tile with two separate animals and ends in a green pill.

Cast: the blue product path, three gateway panels (correctors, another composition method, search and selection), ten attempt glyphs with verdict pills, the amber path standing for the learned correction that does work.

Flows: one blue dashed line running the full width, unchanged at every gateway. One amber dashed line running above it, short and separate, never touching the blue. Legend inside the image, lower left.

Text in the image: title banner "rung 7: ten ways of avoiding the correction". Panel titles: "correctors: re-sample without changing the score", "another method: SuperDiff", "search and selection: spend compute at inference". Labels: "one fused animal", "still one fused animal", "the learned correction", "two separate animals", "9 of 10 null or inconclusive", "1 of 10 sharpens what already composed". Legend: "blue: the product path", "amber: the correction", "green: support", "red: null", "gray: inconclusive".

Exclusions: no product logos, no adapter chip inside any gateway panel, no plotted curve or axis pair, no green pill on more than one of the ten attempt glyphs, no placeholder gibberish text, no watermark.
```

Faithfulness note: the blue rail must arrive and depart carrying the same fused animal, because none of the ten alternatives turned one animal into two; the single green pill must sit on a glyph in the third panel, because the one support verdict is noise search applied on top of an adapter that was already composing, which is a sharpening result and not a composing one. Exactly one green pill, or the picture overstates the rung.

### Prompt 8 (Process): Rung 8, what is still open

Navigation: ⬅️ [Previous](#prompt-7-process-rung-7-the-ten-cheaper-fixes-that-did-not-work) | 📋 [TOC](#table-of-contents) | [Next](#process-capstone-the-whole-arc-as-one-line) ➡️

[planned] ⏳ not rendered
Save as: `diagrams/report-p08-still-open.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, with a numbered round badge reading "8" in the upper left corner. The frame is one wide rounded panel holding two rows. The upper row holds four dashed-outline finding cards standing upright, each with an empty verdict slot at its foot and a small full picture well and full number strip above it, so each card is visibly complete except for its verdict slot. The lower row holds three small glyphs: a wrench-and-terminal glyph with a dashed outline, a scale glyph weighing an adapter chip against a small coil, and a stopped progress bar drawn as a track filled two thirds of the way with a small halt marker at its head. A small cartoon researcher figure stands at the right edge with an open notebook.

Cast: four unwritten results as dashed finding cards, a missing runbook recipe, the energy-penalty result as a scale weighing the adapter against the true correction, three trainings stopped before their planned end, the researcher persona.

Flows: none. This picture is a to-do inventory, not a route. Legend inside the image, lower left.

Text in the image: title banner "rung 8: verdicts on disk with nowhere to read them". Labels: "how much correction is needed", "the same story from three sides", "does one pooled fix transfer at all", "charging the adapter for its energy", "no recipe regenerates this number", "the adapter spends half what the correction spends", "stopped at 61k of 100k". Legend: "dashed: the result exists, the write-up does not".

Exclusions: no verdict pills of any colour on the four cards, no arrows, no product logos, no charts or curves, no placeholder gibberish text, no watermark.
```

Faithfulness note: each of the four cards must be visibly complete apart from an empty verdict slot, because the numbers and figures exist and only the finding file does not; drawing them as blank cards would claim the experiments were never run. No coloured pill may appear anywhere, since assigning a verdict is exactly the work outstanding.

### Prompt 9 (Process): How the corrector's number was made

Navigation: ⬅️ [Previous](#prompt-8-process-rung-8-what-is-still-open) | 📋 [TOC](#table-of-contents) | [Next](#process-capstone-the-whole-arc-as-one-line) ➡️

[built] ⏳ not rendered
Save as: `diagrams/report-p09-how-the-correctors-number-was-made.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, read left to right in four rounded soft-tinted panels of increasing width. Panel one holds a single noise tile on a platform. Panel two is drawn as a long horizontal track of fifty small identical slots; six of the slots are widened into small cells, each cell holding a tiny stack of hops, and the six cells carry small count labels 0, 1, 5, 20, 100, 200 stacked vertically at the panel's left edge to show they are six separate runs rather than six moments of one run. Panel three holds a tall thin bar chart glyph of fifty bars with the rightmost five bars tinted green and the leftmost ten tinted gray. Panel four holds a rounded verdict card divided into three horizontal slots, the top two slots empty and outlined, the third slot holding an amber question-mark pill. A thin gray bracket spans panels two and three from below, carrying a small GPU glyph with a warning triangle on it.

Cast: the pinned starting noise shared by every run, six separate corrector runs at six counts, the per-step ratio each run produces, the read zone as the five green bars, the un-attributable window as the ten gray bars, the three-way verdict card with its two unused branches, and the one-device bracket.

Flows: a blue dashed arrow from panel one into every one of the six cells, meaning all six runs start from the same noise. Six thin amber lines leaving the cells and converging into the bar chart, meaning each run contributes one curve. One gray arrow from the green bars into the verdict card. Legend inside the image, lower left.

Text in the image: title banner "one seed, six runs, one number that could not be read". Panel titles: "the pinned noise", "six runs, not six moments", "the size at every step", "the three-way bar". Labels: "corrector count", "the last five steps: the model's share", "the first ten: the two errors cannot be told apart", "support", "null", "inconclusive", "one device for all six, or the numbers do not compare". Legend: "blue: the shared start", "amber: one run's curve", "green: where the reading is licensed", "gray: where it is not".

Exclusions: no arrow between any two of the six cells, no verdict pill in the support or null slot, no product logos, no components other than those listed, no placeholder gibberish text, no watermark.
```

Faithfulness note: no arrow may run between the six cells. They are six independent trajectories that separate at the first noise level and never meet, and a chain of arrows would draw them as one run measured six times, which is the single most misleading thing this figure could say. The support and null slots must stay empty: at every step size tried the bar returned inconclusive. The GPU bracket must span panels two and three, because the comparison is only fair when all six runs sat on one card.

### Process capstone: The whole arc as one line

Navigation: ⬅️ [Previous](#prompt-8-process-rung-8-what-is-still-open) | 📋 [TOC](#table-of-contents)

[planned] ⏳ not rendered
Save as: `diagrams/report-p09-the-whole-arc.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: landscape 16:9, a metro-map journey read left to right across the full width, with eight numbered round station badges 1 to 8 spaced along one mainline. The mainline enters blue at station 1 and stays blue through station 2, changes to amber between stations 3 and 4, and remains amber through stations 5 and 6. Between stations 6 and 7 a wide loop branches off the mainline below it, runs through ten small unlabelled stops and rejoins the mainline without having changed its colour; the mainline continues amber into station 8, where it ends at an open terminus drawn as a dashed arc rather than a solid cap. Three rounded soft-tinted phase containers sit behind the mainline, the first spanning stations 1 and 2, the second spanning 3 to 6, the third spanning 7 and 8. Above each station badge sits one small outcome tile: station 1 green with two separate animals, station 2 red with one fused animal, station 3 green with two separate animals, station 4 amber, station 5 green, station 6 mixed green and red, station 7 red with one fused animal, station 8 empty and dashed.

Cast: the mainline (the reading arc), eight stations (the eight rungs), ten unlabelled stops on the loop (the ten alternatives), three phase containers, eight outcome tiles, one open terminus.

Flows: one mainline changing colour once, from blue to amber, between stations 3 and 4. One loop leaving and rejoining below the mainline between stations 6 and 7. Legend inside the image, lower left.

Text in the image: title banner "read the results in this order". Phase titles: "the failure", "the fix", "the alternatives and what is left". Station labels: "1 when it works", "2 where it breaks", "3 the correction causes it", "4 what it is made of", "5 the adapter", "6 unseen pairs", "7 ten cheaper fixes", "8 still open". Loop label: "none of these changed the line". Legend: "blue: uncorrected", "amber: corrected", "green: composed", "red: did not", "dashed: unfinished".

Exclusions: no dates or timestamps anywhere, no arrows implying the experiments happened in this order, no product logos, no charts or curves, no components other than those listed, no placeholder gibberish text, no watermark.
```

Faithfulness note: the loop must rejoin the mainline the same colour it left, because none of the ten alternatives changed the outcome, and the terminus at station 8 must be open rather than capped, because the arc ends in unfinished work. The colour change must fall between stations 3 and 4 and nowhere else, since rung 3 is where the correction first enters the story. No date may appear: the left-to-right axis is reading order, not time, and a timestamp would turn it into a chronology it is not.
