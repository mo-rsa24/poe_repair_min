# Closing the Compositional Gap: illustrated map

Six pictures of the system this scope builds, six of the process that builds it, and a capstone
closing each lane. Where one plan needs its own pair, it is filed as a lettered zoom into the piece
it belongs to (2a zooms into piece 2) rather than as a seventh piece. Every prompt below is self-contained: paste any one into ChatGPT on its own and
it comes back matching its siblings.

## Art direction

**Vivid circuit.** The subject is a topology with genuinely different kinds of traffic moving
through it (an uncorrected prediction path, a correction added back on top of it, and matched
substitutes that should do nothing), and colour-coded flow lines are the fastest way to tell those
three apart at a glance. It also carries the process lane once the sequencing devices are added,
so both lanes read as one system.

The direction's style paragraph is embedded verbatim at the head of every prompt below.

## Meaning palette

Four colours, each meaning exactly one thing, in every image of both lanes.

| Colour | Means |
|---|---|
| Blue | the uncorrected path: what plain Product-of-Experts does on its own |
| Amber | the correction `r_t`, and anything carrying or learning it |
| Green | measured and passed: a bar met, a figure built, a component confirmed in the repo |
| Red | measured and failed, or a null: a control at the floor, a bar not met, a known limitation |

A **control** is drawn as a hollow amber outline at the same size as the solid amber correction,
never as a fifth colour. That is faithful to what a control is here: the same size as the real
correction, differing only in direction.

## Glyph vocabulary

The same glyph means the same thing in every image. This is what both capstones reassemble.

| Glyph | Stands for |
|---|---|
| Cache drum | the store of cached per-step predictions every analysis reads |
| Correction coil | `r_t`, the step-by-step gap between the joined prompt and plain PoE |
| Step strip | the 50 denoising steps, noise at the left, finished image at the right |
| Dose dial | λ, how much of the correction is added back |
| Window bracket | a stretch of steps during which the correction is allowed to act |
| Adapter chip | the rank-8 cross-attention adapter that learns the correction |
| Scorer lens | the detector that counts animals in a picture |
| Register board | the figure register: one card per slot the paper reserves |
| Outcome tile | a generated picture, either one fused animal or two separate animals |
| Researcher persona | the human, appearing only where a judgement is made by eye |

No product logos anywhere in this set. Every component here is bespoke research code, so the logo
rule gives all of them generic glyphs.

## Devices in play

**Subject lane:** flow colour coding and an in-image legend, truth anchors (a real generated
picture drawn at the edge where one exists). No sequencing devices.

**Process lane:** the same cast, plus numbered step badges, phase containers, and status chips
("Completed", "In progress", "Not started", "Blocked").

## Subject lane

The system as it stands: what fails, what fixes it, what measures the fix, and what the paper is
built from.

### Prompt 1 (Subject): Where the blend comes from

[built]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one reading direction, left to right. On the left, two small prompt cards stacked vertically. Each feeds its own prediction glyph. The two prediction glyphs meet at a multiply node in the lower middle, which outputs a blue prediction. Above them, separated by clear white space, a single wider prompt card feeds one prediction glyph directly, outputting an amber prediction. On the right, the two predictions meet at a subtract node, and the difference leaves it as a thick amber coil glyph sitting on its own platform. Far right, two outcome tiles stacked: the upper tile shows one animal with a cat ear and whiskers on its left half and a dog ear and muzzle on its right half, bordered blue; the lower tile shows a cat and a dog as two separate animals, bordered amber. Beneath the whole flow, a cylinder drum glyph receives a thin gray line from both prediction glyphs.

Cast: prompt cards (a text prompt entering the model), prediction glyphs (what the model predicts at one step), a multiply node, a subtract node, the correction coil (r_t, the gap between the two readings), two outcome tiles (generated pictures), the cache drum (per-step predictions saved for later analysis).

Flows: blue dashed line for the uncorrected Product-of-Experts path, from the two separate prompts through the multiply node to the blended outcome tile. Amber dashed line for the joined-prompt path and for the correction leaving the subtract node toward the two-animal tile. Thin solid gray lines from both prediction glyphs down into the cache drum, meaning saved, not computed. Legend inside the image, lower left.

Text in the image: title banner "two readings of one sentence". Labels: "a cat", "a dog", "a cat and a dog", "multiply", "plain PoE", "joined prompt", "subtract", "r_t: the gap", "one blended animal", "two separate animals", "cached every step". Legend: "blue: uncorrected path", "amber: the correction".

Exclusions: no product logos, no brand marks, no components other than those listed, no numbered badges or status pills in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the blue path must reach the blended tile and the amber path the two-animal tile, never crossed. The correction must leave the subtract node and nowhere else, because `r_t` is defined as that difference and drawing it as an independent input would teach the opposite.

### Prompt 2 (Subject): The harness that adds the correction back

[built]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one reading direction, left to right. Across the full width of the lower two thirds runs a long horizontal strip divided into many small equal segments, dark and grainy at its left end and resolving into a clear picture at its right end. Above the strip's left quarter sits a rounded bracket spanning about a fifth of the strip's length, with a small arrow showing it can slide along. Above the bracket, a circular dial glyph with a pointer. On the far left, five small vector glyphs stacked vertically inside a rounded soft-tinted panel, each with an arrow feeding into the dial: the topmost drawn as a solid amber coil, the four beneath it drawn as hollow amber outlines of the same size. On the far right, two outcome tiles stacked, the upper bordered amber showing two separate animals, the lower bordered blue showing one blended animal.

Cast: the step strip (the 50 denoising steps), the window bracket (the stretch of steps during which the correction may act), the dose dial (λ, how much correction is added), the correction coil solid (the pair's own real correction), four hollow outlines of identical size (the matched substitutes), two outcome tiles.

Flows: blue dashed line running the full length of the step strip underneath it, meaning the uncorrected base runs at every step regardless. Amber dashed line from the dial down into only the segments the bracket covers, meaning the correction acts only inside the window. Legend inside the image, lower left.

Text in the image: title banner "adding the correction back". Labels: "noise", "finished image", "50 steps", "window gate", "dose dial", "own correction", "other pair", "other seed", "steps shuffled", "random vector", "same size, wrong direction", "two separate animals", "one blended animal". Legend: "blue: base runs always", "amber: correction injected".

Exclusions: no product logos, no numbers on the dial, no components other than those listed, no numbered step badges or status pills in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the four substitutes must be drawn the same size as the real correction and differ only in being hollow, because they are matched in size and differ only in direction. The amber line must enter only the segments under the bracket, and the blue line must run the strip's full length, or the picture claims the prompt is switched off outside the window, which it is not.

### Prompt 2a (Subject): One thing changes, which is when the correction acts

[planned] Zoom into prompt 2's window bracket, for
`plans/does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md`.

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one tall composition, read top to bottom. Nine identical horizontal strips stacked with even spacing, each divided into many small equal segments, each dark and grainy at its left end and resolving into a clear picture at its right end. Above each strip sits a rounded bracket covering one fifth of that strip's length; the bracket sits at the far left on the topmost strip and shifts one notch further right on each strip below, reaching the far right on the bottom strip. A single thin vertical guide line runs down through all nine strips at a position just under a third from the left, crossing inside the bracket on only two of the strips. At the right edge, level with the topmost strip, one outcome tile bordered amber showing two separate animals; level with the bottom strip, one outcome tile bordered blue showing one blended animal. On the far left, a small rounded soft-tinted panel holding a grid glyph labelled with its three dimensions.

Cast: nine step strips (the 50 denoising steps, one strip per window placement), nine window brackets (the stretch of steps during which the correction may act), the vertical guide line (the fork step, step 16, measured independently from cached runs), two outcome tiles, one grid glyph.

Flows: a blue dashed line running the full length of every one of the nine strips, underneath it, meaning the uncorrected base and the prompt run at every step in every row. An amber dashed line entering each strip only in the segments its own bracket covers. Legend inside the image, lower left.

Text in the image: title banner "one thing changes: when the correction acts". Labels: "noise", "finished image", "50 steps", "steps 0-10", "steps 40-50", "window gate", "fork step, measured elsewhere", "9 windows x 8 pairs x 4 seeds = 288 cells", "two separate animals", "one blended animal". Legend: "blue: base and prompt run always", "amber: correction injected".

Exclusions: no product logos, no numbered step badges, no status pills, no arrows implying the nine rows happen in sequence, no curve or chart, no components other than those listed, no placeholder gibberish text, no watermark.
```

Faithfulness note: the blue line must run the full length of all nine strips. If it stops outside a bracket the picture claims conditioning is gated too, which is the confound the whole design exists to avoid. The fork-step guide must cross inside only two of the nine brackets, because that is what lets the sweep disagree with it rather than being built to agree. The nine rows are placements, not stages, so no arrows between them.

### Prompt 3 (Subject): The instruments that decide what happened

[built]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: three rounded soft-tinted panels side by side, each titled at its top, one reading direction left to right. Left panel holds a magnifying lens glyph over a small picture with two rectangular boxes drawn on the animals in it, and beneath the lens a small square swatch showing a minimum box size. Middle panel holds a bench with three small chart glyphs on it: a descending bar series, a rising-then-forking pair of lines, and a single curve against a shaded band. Right panel holds two glyphs: a speech-bubble over a picture, and a small track with a dot sliding along it between two endpoint thumbnails. Beneath the left panel, outside it, one small red warning badge attached by a thin leader line to the lens.

Cast: the scorer lens (counts how many animals a picture holds), the cache analysis bench (measurements taken off the saved predictions with no new generation), the second-opinion probes (instruments that share none of the scorer's machinery), the researcher persona standing beside the right panel.

Flows: green solid arrows from each panel toward a shared strip at the bottom of the image, meaning a measured verdict leaves each instrument. One red solid arrow from the warning badge pointing back at the lens, meaning a known limitation of that instrument. Legend inside the image, lower right.

Text in the image: title banner "three ways to check". Panel titles: "count the animals", "read the cache", "ask another instrument". Labels: "at least two boxes", "minimum box size", "how few directions", "where paths fork", "size against noise", "caption readback", "manifold slide", "cannot tell which animals". Legend: "green: a verdict", "red: a known limit".

Exclusions: no product logos, no components other than those listed, no numbered step badges or status pills in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the red warning must attach to the counting lens and to nothing else. That instrument counts animals without identifying them, so two dogs scores the same as a cat and a dog, and attaching the limitation to the cache bench or the second-opinion probes would misplace the one weakness the paper has to declare.

### Prompt 4 (Subject): The adapter that learns the correction

[built]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one reading direction, left to right. On the left, a cylinder drum glyph with eleven small amber coil glyphs arranged in a column beside it inside a rounded soft-tinted panel. An amber arrow leads right into a small rectangular chip glyph seated inside a larger transparent block, the chip noticeably smaller than the block. From the chip, amber arrows fan right to eight outcome tiles arranged in two rows of four, each tile showing two separate animals and bordered green. Below that fan, drawn entirely in dashed gray outline with no fill, a second fan of fifteen tiny chip glyphs, visibly hedged and faded against the solid work above.

Cast: the cache drum (saved corrections), eleven correction coils (the training pairs), the adapter chip (rank-8, seated inside a cross-attention block), eight outcome tiles (pairs the adapter never trained on), the dashed fifteen-chip fan (the leave-one-pair-out sweep).

Flows: amber dashed lines from the cached corrections into the chip, meaning it learns from them. Amber dashed lines from the chip out to the eight held-out tiles, meaning it is applied to pairs it never saw. The lower fan carries no flow lines at all, only its dashed outline. Legend inside the image, lower left.

Text in the image: title banner "learned once, applied to new pairs". Labels: "cached corrections", "eleven training pairs", "rank-8 adapter", "cross-attention block", "eight held-out pairs", "never sees the joined prompt", "leave-one-pair-out", "not yet run". Legend: "amber: the correction", "dashed: not yet run".

Exclusions: no product logos, no components other than those listed, no numbered step badges or status pills in this image, no solid fill or colour on the lower fan, no placeholder gibberish text, no watermark.
```

Faithfulness note: the lower fan of fifteen must be visibly hedged and carry no flow line, because that sweep has not run. The chip must be drawn smaller than the block it sits in, and no arrow may run from the joined prompt into the chip, since the whole claim is that the adapter works without ever seeing the joined prompt.

### Prompt 5 (Subject): Where the evidence is kept

[built]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: two rounded soft-tinted panels side by side, one reading direction left to right. The left panel is a board holding eight small framed picture cards in a grid, most bordered green, one bordered red, two drawn as empty dashed frames. The right panel is a shelf of upright document glyphs, each with a small tag clipped to its top edge. A green arrow runs from the shelf to the board, entering the board from its left side. A researcher persona stands between the two panels.

Cast: the register board (one card per figure slot the paper reserves), the framed cards (figures), the document shelf (the review files that judge each run), the tags (the bar each run was judged against, written before the run), the researcher persona.

Flows: green solid arrow from the shelf to the board, meaning a figure may only be built from an answered review question. Legend inside the image, lower right.

Text in the image: title banner "nothing prints without its verdict". Panel titles: "figure register", "review files". Labels: "built", "reserved", "needs a decision", "bar written first", "answered", "one card, one claim". Legend: "green: answered and built", "red: argument does not stand".

Exclusions: no product logos, no components other than those listed, no numbered step badges in this image, no readable chart content inside the framed cards, no placeholder gibberish text, no watermark.
```

Faithfulness note: the arrow must run from the review shelf into the register board and never the reverse, because a figure is licensed by an answered question rather than the other way round. One card must be red: the low-rank slot's argument does not currently stand.

### Prompt 6 (Subject): The manuscript, which produces nothing of its own

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one reading direction, left to right. On the left, the register board glyph and the document shelf glyph from the previous image, drawn smaller and solid. Both feed right into a single large document glyph drawn entirely in dashed outline with a pale unfilled interior, sitting on its platform. The document shows a title line drawn as a gray placeholder bar rather than real text, and several section blocks beneath it, two of which are filled green and the rest left empty dashed. No arrow leaves the document to the right. A researcher persona sits at the document's left, working on it.

Cast: the register board and the document shelf (both carried over unchanged in meaning), the manuscript document drawn hedged because it is not yet written, the researcher persona.

Flows: green dashed arrows from both the board and the shelf into the manuscript, meaning it consumes them. No outgoing flow of any kind. Legend inside the image, lower left.

Text in the image: title banner "the paper consumes, never produces". Labels: "figure register", "review files", "reads both", "produces no figures", "title still a stub", "build works", "sections owed". Legend: "green: consumed evidence", "dashed: not yet written".

Exclusions: no product logos, no readable title text on the document, no outgoing arrow from the manuscript, no components other than those listed, no numbered step badges in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: nothing may leave the manuscript. This scope's rule is that the writing consumes the register and the review files and produces no figures of its own, and an outgoing arrow would invert it. The document stays hedged: the build works but the title is still the stock template's.

### Subject capstone: The whole system on one page

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one wide landscape composition, reading left to right across three rounded soft-tinted panels, with a fourth smaller panel at the far right. Panel one, titled at its top, holds the two prompt cards, the multiply node, the subtract node and the correction coil, with the cache drum beneath them. Panel two holds the step strip with its sliding window bracket and dose dial above it, the five vector glyphs feeding the dial (one solid amber, four hollow amber outlines), and two outcome tiles at its right edge. Panel three holds the scorer lens, the cache analysis bench, the second-opinion probes, and beneath them the adapter chip with its dashed fifteen-chip fan. Panel four, at the far right and narrower, holds the register board above the document shelf, and beneath both, the manuscript document drawn in dashed outline. A researcher persona stands once, between panel three and panel four.

Cast: every glyph from prompts 1 to 6, each keeping the meaning it has there and nothing added: prompt cards, prediction glyphs, multiply and subtract nodes, correction coil, cache drum, step strip, window bracket, dose dial, four hollow control outlines, outcome tiles, scorer lens, cache analysis bench, second-opinion probes, adapter chip, dashed fifteen-chip fan, register board, document shelf, manuscript document, researcher persona.

Flows: blue dashed for the uncorrected path, running from the multiply node in panel one along the full step strip in panel two to the blended outcome tile. Amber dashed for the correction, running from the subtract node in panel one to the dose dial in panel two, and separately from the cache drum to the adapter chip in panel three. Green solid for a measured verdict, running from the instruments in panel three to the register board and the document shelf in panel four, then into the manuscript. The dashed fifteen-chip fan carries no flow. A single in-image legend sits in the lower left, spanning the width of panel one.

Text in the image: title banner "closing the compositional gap". Panel titles: "where the blend comes from", "adding it back", "measuring and learning", "the paper". Labels: "r_t: the gap", "50 steps", "window gate", "dose dial", "same size, wrong direction", "count the animals", "rank-8 adapter", "not yet run", "figure register", "review files", "produces no figures". Legend: "blue: uncorrected path", "amber: the correction", "green: a measured verdict", "dashed: not yet run".

Exclusions: no product logos, no object that did not appear in prompts 1 to 6, no object from those prompts left out, no numbered step badges or status pills anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the four panels must read left to right as one chain, and the green verdict flow must be the only thing entering the paper. The correction has two distinct amber destinations and both must be drawn: injected by hand at the dose dial, and learned from the cache by the adapter. Drawing only one of them collapses the scope's two claims into one.

## Process lane

History: `diagrams/process-versions/`. This lane is regenerated whole from the current plan files
whenever the plans change shape, never patched in place.

### Prompt 1 (Process): Fix the instruments before any result exists

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding three stage cards in a horizontal row connected by directional arrows. Each card carries a small numbered circular badge at its upper left and two or three tiny bullet lines inside it. Card one holds the scorer lens glyph, card two holds a padlock over a small written note, card three holds the correction coil beside a small pool of paired animal thumbnails. All three cards carry a green "Completed" pill at their lower right. A small legend row runs along the bottom edge.

Cast: the scorer lens, a padlock over a written choice (a decision committed in writing before any result could be seen), the correction coil, a pool of animal-pair thumbnails, all keeping their meanings from the subject lane.

Flows: green solid arrows left to right between the three cards, meaning each is finished and the next may start. Legend inside the image, bottom edge.

Text in the image: title banner "instruments first". Phase container title: "committed before any result". Card titles: "build the measuring scripts", "fix the size measure", "curate the pair pool". Bullets: "thirteen scripts", "smoked end to end", "chosen before results", "cannot follow the answer", "eleven pairs", "all blend by default". Pills: "Completed". Legend: "green: completed".

Exclusions: no product logos, no stage other than the three listed, no red or gray status pills in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: all three stages carry a completed pill. Each of these instruments was fixed in writing before any result was read, which is the only reason the later numbers count as evidence, so none may be drawn as in progress.

### Prompt 2 (Process): The causal runs

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding five stage cards. Four sit in a horizontal row connected by directional arrows; the fifth sits below the row, connected upward into the third card. Each card carries a numbered circular badge at its upper left and two or three tiny bullet lines. Card one holds the dose dial, card two holds the sliding window bracket over a short step strip, card three holds the cache analysis bench, card four holds three small instrument glyphs side by side, and the lower card holds the scorer lens over a probe map. All five carry a green "Completed" pill.

Cast: the dose dial, the window bracket and step strip, the cache analysis bench, the three second-opinion probes, the scorer lens over a probe map, all carrying their subject-lane meanings.

Flows: green solid arrows between the cards, meaning each finished. Legend inside the image, bottom edge.

Text in the image: title banner "the causal runs". Phase container title: "all five answered". Card titles: "more correction, more composition", "when in the run it matters", "read the cached runs", "the same story three ways", "what changes inside the model". Bullets: "three rows compared", "controls stayed flat", "early window only", "cliff at the start", "no new generation", "two nulls kept", "held-out pairs". Pills: "Completed". Legend: "green: completed".

Exclusions: no product logos, no stage other than the five listed, no gray skipped pills, no red callout or warning badge anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: all five stages are completed and none carries an outstanding cleanup. The dose stage's output now sits on `/datasets` where it belongs, so a red callout on card one would report a problem that no longer exists.

### Prompt 2a (Process): The stages that produced the timing answer

[planned] Zoom into prompt 2's card two, for
`plans/does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md`.

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning most of the image, titled at its top, holding six stage cards in a horizontal row connected by directional arrows. Each card carries a numbered circular badge at its upper left and two or three tiny bullet lines. Card one holds a caliper glyph measuring a short step strip. Card two holds a window bracket sitting entirely off the end of a step strip, with a small equals sign beside it. Card three holds a step strip with three brackets at left, middle and right. Card four holds a grid glyph. Card five holds two step strips stacked with a dose dial between them. Card six holds a register board. Cards one to five carry a green "Completed" pill; card six carries a green "Completed" pill and a red warning badge with a dashed callout box hanging below it. Below the container and to the right, a seventh card sits on its own, connected upward into card six, holding a researcher persona beside a slider glyph and carrying an amber "In progress" pill. A red dashed branch leaves card two downward into a small muted card holding a crossed-out step strip.

Cast: the caliper over a step strip, the window bracket and step strip, the grid glyph, the dose dial, the register board, the researcher persona with a slider, all carrying their subject-lane meanings.

Flows: green solid arrows between cards one through six, meaning each finished. A red dashed branch from card two into the muted card, meaning the run that stops everything if it fires. A green solid arrow from card six down into the seventh card. Legend inside the image, bottom edge.

Text in the image: title banner "when in the run it matters". Phase container title: "answered: the cliff is at the start". Card titles: "fix the width in source", "prove the gate does not leak", "smoke three windows", "run 288 cells and score", "untie timing from dose", "eight figures built". Bullets: "width 10, stride 5", "chosen before any run", "all-off equals plain PoE", "early, middle, late", "9 x 8 x 4", "no missing windows", "same total, later", "cliff survives", "F4a to F4h". Seventh card: "drive the timing tab by hand", bullets "slider moves picture and curve", "the check numbers cannot make". Pills: "Completed", "In progress". Muted card: "the gating leaks, stop". Callout: "one figure unregistered, two rows still say reserved". Legend: "green: completed", "amber: yours to do", "red: stops the plan".

Exclusions: no product logos, no stage other than the seven listed, no gray "Skipped" pills, no chart or curve, no placeholder gibberish text, no watermark.
```

Faithfulness note: card six is completed and its red callout is a register gap, not a failed result. Drawing the stage as failed would misreport eight figures that exist. The seventh card is the only work not done and it is the human's, which is why it carries the researcher persona and sits outside the container.

### Prompt 3 (Process): The transfer runs

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding four stage cards in a horizontal row connected by directional arrows. Each carries a numbered circular badge and two or three tiny bullet lines. Card one holds three small rising curves side by side. Card two holds the adapter chip with eight outcome tiles beside it. Card three holds the dashed fifteen-chip fan. Card four holds two pools of thumbnails balanced on a beam. Card one carries an amber "In progress" pill, card two carries an amber "In progress" pill, and cards three and four are drawn muted and grayed with "Not started" pills. A small legend row runs along the bottom edge.

Cast: three live training curves, the adapter chip, eight outcome tiles, the dashed fifteen-chip fan, two thumbnail pools on a balance beam (a pool of animal pairs against a size-matched mixed pool), all carrying their subject-lane meanings.

Flows: green solid arrow from card one to card two. Gray dashed arrows from card two onward to cards three and four, meaning downstream work not yet begun. Legend inside the image, bottom edge.

Text in the image: title banner "does the fix reach new pairs". Phase container title: "started, not finished". Card titles: "three live curves", "does one adapter transfer", "transfer as a rate", "the size-matched pool". Bullets: "wired, smoke owed", "held-out pairs compose", "later checkpoints unscored", "fifteen adapters", "one pair held out each", "kills the more-data excuse". Pills: "In progress", "Not started". Legend: "green: completed", "amber: in progress", "gray: not started".

Exclusions: no product logos, no stage other than the four listed, no red failure styling anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the second stage must read as in progress rather than completed. Its held-out pairs do compose, which is a real positive, but the later checkpoints are unscored and the go-ahead note is unwritten, so the number cannot yet be quoted. Nothing here has failed, so no red may appear.

### Prompt 4 (Process): Can the compose rate be trusted

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding four stage cards in a horizontal row connected by directional arrows. Every card is drawn muted and grayed with a "Not started" pill, and every card carries a numbered circular badge and two or three tiny bullet lines. Card one holds a stack of paper glyphs with a magnifier over them. Card two holds the researcher persona at a small screen showing one picture and four choice buttons. Card three holds two scorer lens glyphs side by side with a small comparison bracket between them. Card four holds a signpost with two arms pointing in different directions. Attached to the whole container's left edge, outside it, sits the scorer lens glyph from the subject lane with a red warning badge on it.

Cast: a stack of published papers with a magnifier (checking whether this hole is already known), the researcher persona at a labelling screen, two scorer lenses under comparison, a two-armed signpost (the promote-or-close decision), the scorer lens with its red warning badge.

Flows: gray dashed arrows left to right between all four cards, meaning nothing has begun. One red solid arrow from the warning-badged lens into card one, meaning this whole phase exists because of that instrument's known limitation. Legend inside the image, bottom edge.

Text in the image: title banner "how far above the truth". Phase container title: "not started". Card titles: "is this hole already known", "build the labelled set", "score the candidates", "promote or close". Bullets: "one literature verdict", "judgeable pairs first", "rule written before labels", "against the same labels", "keep the better detector", "wording only, or a real run". Pills: "Not started". Legend: "gray: not started", "red: the limitation".

Exclusions: no product logos, no stage other than the four listed, no green completed pills anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: every stage is grayed. Not one of this phase's four plans has begun, and its four review files carry no answered question, so any green here would be false. The red arrow must originate at the counting lens, because the phase exists to measure how far that instrument's rate sits above the truth.

### Prompt 5 (Process): The figures and the two print gates

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding three stage cards in a horizontal row connected by directional arrows. Each carries a numbered circular badge and two or three tiny bullet lines. Card one holds the register board with most of its framed cards bordered green, one bordered red, and two left as empty dashed frames; it carries an amber "In progress" pill. Card two holds the adapter chip beside a leaderboard glyph and is drawn muted and grayed with a "Not started" pill. Card three holds two upright gate glyphs side by side with a stack of paper behind them, and is drawn muted and grayed with a "Not started" pill. A dashed red callout box hangs below card one, connected to its single red-bordered frame.

Cast: the register board and its framed cards, the adapter chip, a leaderboard, two upright print gates, a stack of published papers, all carrying their subject-lane meanings.

Flows: amber solid arrow from card one onward, meaning work underway continuing. Gray dashed arrows into cards two and three. Red dashed line from the red-bordered frame in card one down into the callout box. Legend inside the image, bottom edge.

Text in the image: title banner "what reaches the page". Phase container title: "most built, one undecided". Card titles: "the causal figures", "the transfer figures", "two checks before print". Bullets: "fourteen slots built", "captions capped by review", "waits on the sweep", "leaderboard and curve", "novelty of the timing", "the span sentence". Pills: "In progress", "Not started". Callout: "low-rank argument fails". Legend: "amber: in progress", "gray: not started", "red: needs a decision".

Exclusions: no product logos, no stage other than the three listed, no readable chart content inside the framed cards, no placeholder gibberish text, no watermark.
```

Faithfulness note: the causal figure card is in progress with most frames green, because fourteen register slots are built. The one red frame is the low-rank slot, whose argument does not stand against a floor that controls for the right thing, and it must be a single frame rather than the whole card, since the rest of the set is fine.

### Prompt 6 (Process): The manuscript

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding four stage cards in a horizontal row connected by directional arrows, with the researcher persona seated at the left end. Each card carries a numbered circular badge and two or three tiny bullet lines. Card one holds a document glyph with a green tick on its corner and a gray placeholder bar where its title would be, and carries an amber "In progress" pill. Cards two, three and four are drawn muted and grayed with "Not started" pills: card two holds a spine of stacked section blocks, card three holds a page with empty framed slots and small placeholder marks in the text, card four holds a finished page with a small padlock on it.

Cast: the researcher persona, the manuscript document, a section spine, a page with owed figure slots, a finished anonymised page, all carrying their subject-lane meanings.

Flows: amber solid arrow from card one to card two, then gray dashed arrows onward through cards three and four. Legend inside the image, bottom edge.

Text in the image: title banner "writing it up". Phase container title: "build works, prose owed". Card titles: "make the template build", "title and section order", "results skeleton", "anonymise and submit". Bullets: "builds to a pdf", "title still a stub", "one claim per section", "placeholders, not prose", "every slot named", "page limit met". Pills: "In progress", "Not started". Legend: "amber: in progress", "gray: not started".

Exclusions: no product logos, no stage other than the four listed, no readable title text on any document, no green completed pills anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the first stage is in progress and not completed. The template does build to a PDF and the figure-path rule is written, but the title is still the stock template's, so a completed pill would overstate it. No stage after it has begun.

### Process capstone: One route from instruments to submission

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one wide landscape composition. A single thick continuous route line runs left to right across the image like a metro line, with circular icon nodes sitting on it. Segments of the route are enclosed in rounded soft-tinted phase containers, each titled above the route. From left: a container holding three nodes, then a container holding five nodes, then a container holding four nodes, then a container holding three nodes, then a container holding four nodes at the right end. The route is drawn solid green through the first two containers, solid amber through the third, and gray dashed through the last two. One branch leaves the route beneath the third container, loops down through a small separate container holding four gray nodes, and rejoins the mainline before the fourth container. The researcher persona stands at the route's left start. Small status chips sit beside the nodes. A small legend row runs along the bottom edge.

Cast: the route line (the order the work is actually done in), circular nodes carrying the subject-lane glyphs at reduced size (scorer lens, dose dial, window bracket, cache bench, adapter chip, register board, print gates, manuscript document), the researcher persona, the looping branch (the phase that may or may not change a claim).

Flows: the route itself is the only flow, coloured by state: green where finished, amber where underway, gray dashed where not begun. The loop is drawn gray dashed and physically rejoins the mainline. Legend inside the image, bottom edge.

Text in the image: title banner "instruments to submission". Phase container titles: "instruments first", "the causal runs", "transfer and figures", "before print", "the manuscript". Loop container title: "can the rate be trusted". Node labels: "measuring scripts", "size measure fixed", "pair pool", "dose", "timing", "cached runs", "three ways", "inside the model", "does it transfer", "the figures", "two checks", "build", "sections", "submit". Legend: "green: completed", "amber: in progress", "gray: not started".

Exclusions: no product logos, no node that did not appear in process prompts 1 to 6, no node from those prompts left out, no red failure styling anywhere on the mainline, no placeholder gibberish text, no watermark.
```

Faithfulness note: the route must be green only through the instruments and the causal runs, because those are the two phases actually finished. The trust-the-rate phase must be drawn as a loop off the mainline rather than a node on it: it may change a claim or may change nothing, and putting it inline would say the paper waits on it, which it does not.
