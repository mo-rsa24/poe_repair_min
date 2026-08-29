# poe_repair_min learning map: what teaches this work

How the repo's working documents connect to the learning journeys under
`/home-mscluster/mmolefe/goal-setting/learning`, and what to learn next for the work the running
order says is coming. Maintained by `learning-pulse`; rendered by `/render-diagrams`.

7 prompts · 0 rendered · 7 waiting

## Table of contents

- [Art direction](#art-direction)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Subject lane](#subject-lane)
- [Process lane](#process-lane)

## Art direction

Navigation: 📋 [TOC](#table-of-contents) | [Meaning palette](#meaning-palette) ➡️

**Vivid circuit**, inherited from the project map at `plans/diagram-prompts.md` so the two read as
one system. The style paragraph is embedded verbatim at the head of every prompt below.

## Meaning palette

Navigation: ⬅️ [Art direction](#art-direction) | 📋 [TOC](#table-of-contents) | [Glyph vocabulary](#glyph-vocabulary) ➡️

| Colour | Means |
|---|---|
| Blue | the doing side: a working folder or plan in this repo |
| Amber | the teaching side: a journey, its plans, its textbook |
| Green | a woven thread: links that exist on disk, labelled with the count |
| Red | unused or missing: a journey with zero inbound links, or a named gap |

## Glyph vocabulary

Navigation: ⬅️ [Meaning palette](#meaning-palette) | 📋 [TOC](#table-of-contents) | [Reading axes](#reading-axes) ➡️

| Glyph | Stands for | Tier |
|---|---|---|
| folder tile | one working folder (`plans/`, `context/`, `environment/`, `runbook/`, `report/`) | primary |
| journey ladder | one journey's plan tree, drawn as a vertical ladder, rung count labelled | primary |
| small book | a textbook beside its journey's ladder | secondary |
| green thread | the woven links from one folder to one journey, count on the thread | primary |
| dim shelf | the side platform holding journeys nothing links to | secondary |
| learning ramp | numbered amber rungs laid in front of a blue work stage (process lane only) | primary |

Every journey ladder is a pointer to that journey's own illustrated map or master plan, never a
redrawing of its contents.

## Reading axes

Navigation: ⬅️ [Glyph vocabulary](#glyph-vocabulary) | 📋 [TOC](#table-of-contents) | [Subject lane](#subject-lane) ➡️

Dominant axis: left to right, from the doing side (blue, the repo) to the teaching side (amber,
the learning root). In the process lane, left to right is the order things are learned and then
done; the secondary axis separates the two work tracks.

## Subject lane

Navigation: ⬅️ [Reading axes](#reading-axes) | 📋 [TOC](#table-of-contents) | [Process lane](#process-lane) ➡️

### Prompt 1 (Subject): The doing side

[observed] ⏳ not rendered
Save as: `learning-diagrams/learning-map-01-the-doing-side.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: a rounded blue-tinted panel titled "the repo: poe_repair_min" holding five folder tiles in
one row: "plans/", "context/", "environment/", "runbook/", "report/". The plans/ tile is larger
and carries two small stage chips inside it: "sampler vs model scope" and "cause-composition
review". Green thread stubs leave the tiles that carry links, each with a count pill: "5 links"
from the sampler-vs-model chip, "1 link" from the review chip, "1 link" from runbook/, "2 links"
from report/. The context/ and environment/ tiles have no stubs. Title banner: "where the work
happens, and which documents reach for teaching".
Exclusions: no journey ladders in this image, no logos, no red.
```

Faithfulness note: thread stubs leave exactly four places with counts 5, 1, 1, 2; `context/` and
`environment/` carry none, because on disk they hold no learning links.

### Prompt 2 (Subject): The woven journeys

[observed] ⏳ not rendered
Save as: `learning-diagrams/learning-map-02-the-woven-journeys.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: an amber-tinted panel titled "the learning root" holding two journey ladders. Left ladder
labelled "sampler-correctors-for-composition" with a count pill "28 plans"; five green threads
arrive bundled from off-canvas left, bundle pill "5 links, all from the sampler-vs-model scope".
Right ladder labelled "spectral-structure-of-the-correction" with pills "14 plans" and a small
book glyph "3 chapters"; four green threads arrive with pills "1 review", "1 runbook", "2 report".
Each ladder has a small signpost icon at its base: "opens the journey's own map".
Exclusions: no rung-by-rung content of either journey, no red, no logos.
```

Faithfulness note: five threads into the left ladder and four into the right, nine woven links in
total; the tenth link in the repo (a paper-scout selection in `paper/`) is deliberately absent
because it points at a journey this map shelves.

### Prompt 3 (Subject): The unused shelf

[observed] ⏳ not rendered
Save as: `learning-diagrams/learning-map-03-the-unused-shelf.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: a dimmed side platform titled "built, and nothing points here yet". Two journey ladders
drawn at half opacity: "poe-composition-diffusion" with pill "57 leaves" and a red tag "0 links";
"trajectory-manifold-by-hand" with pill "flow map" and two red tags "0 links" and "no master
plan". No threads touch either ladder. A small caption strip at the bottom: "unfindable from the
work is the same as lost".
Exclusions: no green anywhere, no connections to anything.
```

Faithfulness note: both ladders must be visibly disconnected; the "no master plan" tag sits only
on trajectory-manifold-by-hand, because that gap is real on disk and blocks any link to it.

### Subject capstone: The whole weave

[observed] ⏳ not rendered
Save as: `learning-diagrams/learning-map-04-subject-capstone-the-whole-weave.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: the full picture composed from prompts 1 to 3. Blue repo panel on the left (five folder
tiles, two stage chips inside plans/), amber learning panel on the right (the two woven ladders),
the dimmed shelf below the amber panel (the two unused ladders). Nine green threads cross the gap
with their count pills: 5, 1, 1, 2. In-image legend for the four colours. Title banner: "what
teaches this work".
Exclusions: nothing crosses into the shelf; no thread without a count pill.
```

Faithfulness note: the thread counts sum to nine, they land on exactly two ladders, and the shelf
stays disconnected; a viewer should be able to answer "which journeys does the work actually
reach" from this one image.

## Process lane

Navigation: ⬅️ [Subject lane](#subject-lane) | 📋 [TOC](#table-of-contents)

History: `learning-diagrams/process-versions/`. This lane is regenerated whole, never patched.

### Prompt 1 (Process): The sampler-vs-model ramp

⏳ not rendered
Save as: `learning-diagrams/learning-map-05-process-sampler-vs-model-ramp.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: a single left-to-right track. First a learning ramp of five numbered amber rungs, badges 1
to 5, labelled "15 the two gaps", "16 which gap a corrector removes", "21 langevin dynamics",
"22 predictor-corrector sampling", "27 a corrector on the cached trajectories". The ramp feeds a
blue stage block labelled "repo steps 24 to 26: the free bound, the corrector build, the
chain-settles gate". A small flag on the stage: "then run". Banner: "learn first, then run the
sampler-vs-model scope".
Exclusions: no spectral material, no shelf, no red.
```

Faithfulness note: the five rungs are these five journey plans in this order, and the ramp sits
before the stage, never after; the picture claims a learning order, not a result.

### Prompt 2 (Process): The spectrum-figure ramp

⏳ not rendered
Save as: `learning-diagrams/learning-map-06-process-spectrum-figure-ramp.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: a second left-to-right track. A learning ramp of three numbered amber rungs, badges 1 to
3, labelled "06 svd of the real matrix", "09 energy at k and the floors", "10 energy at k on the
real matrix". The ramp feeds a blue stage block labelled "defend the spectrum figure while
writing". A small open-question chip beside the stage: "what does the spectrum add beyond D1 and
D3?". Banner: "learn first, then judge the spectrum".
Exclusions: no sampler material, no red.
```

Faithfulness note: the open-question chip quotes the review file's still-open question; it is
drawn beside the stage, not answered by the picture.

### Process capstone: This week's learning order

⏳ not rendered
Save as: `learning-diagrams/learning-map-07-process-capstone-this-weeks-order.png`

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one metro-style mainline running left to right through both tracks: the sampler ramp
(five amber rungs) into its blue stage, then a transfer dot, then the spectrum ramp (three amber
rungs) into its writing stage. Station dots on every rung, larger dots on the two blue stages.
In-image legend: amber station "learn", blue station "do". Banner: "the learning order the
running order implies".
Exclusions: no shelf, no counts, no third track.
```

Faithfulness note: eight learning stations and two doing stations, in the order prompts 1 and 2
fixed; every learning station precedes its doing station on the mainline.
