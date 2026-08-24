# poe_repair_min context: illustrated map

What the project's real-world objects are (subject lane) and how one cell's data moves from a
prompt pair to a scored verdict (process lane). 8 prompts · 0 rendered · 8 waiting.

## Table of contents

- [Art direction](#art-direction)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Devices in play](#devices-in-play)
- [Subject lane](#subject-lane)
- [Process lane](#process-lane)

## Art direction

Navigation: 📋 [TOC](#table-of-contents) | [Next](#meaning-palette) ➡️

**Layered cutaway.** Per `~/.claude/CONTEXT_FORMAT.md`'s own guidance ("Layered cutaway where the
data visibly changes shape from input to output... and the reader is technical enough to want
it"), picked over the outline-cartoon default because this project's central object, the
interaction term, only means anything as a per-step transformation of a noise prediction, and the
reader (this project's own researcher, a supervisor, a reviewer) is technical.

> Style: technical cutaway diagram, light background. The subject drawn as labeled 3D layered
> volumes or stacked stages left to right, with the actual data visibly transforming between
> stages. Real sample artifacts at the edges: an example input drawn at the left, the concrete
> output (bars, a table row, a labeled result) at the right. Horizontal labeled brackets beneath
> the image grouping stages into named phases. Thin leader lines from labels to parts. Restrained
> color: neutral grays and one warm plus one cool accent. Precise, textbook-quality, no cartoon
> elements.

## Meaning palette

Navigation: ⬅️ [Art direction](#art-direction) | 📋 [TOC](#table-of-contents) | [Next](#glyph-vocabulary) ➡️

| Color | Means |
|---|---|
| warm accent (amber) | the interaction term / the correction being added |
| cool accent (blue) | the plain PoE prediction, uncorrected |
| green | a cell the scorer calls `compose` |
| gray (neutral) | everything not carrying meaning: containers, labels, backgrounds |

## Glyph vocabulary

Navigation: ⬅️ [Meaning palette](#meaning-palette) | 📋 [TOC](#table-of-contents) | [Next](#reading-axes) ➡️

| Glyph | Tier | Stands for |
|---|---|---|
| two small labeled boxes ("cat expert", "dog expert") feeding one merge arrow | primary | PoE composition, see [world/poe-composition.md](world/poe-composition.md) |
| one box labeled "joint prompt" feeding the same merge point, drawn separately | primary | Mono, the ceiling, see [world/poe-composition.md § Mono](world/poe-composition.md#what-poe-composition-is) |
| a thin amber vector arrow between the PoE and Mono predictions | primary | the interaction term, see [world/interaction-term.md](world/interaction-term.md) |
| a small stacked-layer icon reading "rank-8, attn2" | secondary | the LoRA corrector, see [world/lora-corrector.md](world/lora-corrector.md) |
| a magnifier over two bounding boxes | secondary | the compose-rate scorer, see [world/compose-rate.md](world/compose-rate.md) |
| a fused single creature icon (cat/dog blend outline) | artifact | a chimera render, see [world/chimera.md](world/chimera.md) |
| a two-separate-animals icon | artifact | a compose render |

## Reading axes

Navigation: ⬅️ [Glyph vocabulary](#glyph-vocabulary) | 📋 [TOC](#table-of-contents) | [Next](#devices-in-play) ➡️

Dominant axis: left to right, noise to finished image (the 50-step denoising schedule). Secondary
axis: top to bottom, one render method (PoE, Mono, corrected) per band, so the same step column
can be compared across methods.

## Devices in play

Navigation: ⬅️ [Reading axes](#reading-axes) | 📋 [TOC](#table-of-contents) | [Next](#subject-lane) ➡️

Subject lane: none of the sequencing devices, per the guide. Process lane: numbered step badges
matching [data/01-where-it-comes-from.md](data/01-where-it-comes-from.md)'s six stages (①-⑥), plus
a status chip (green check / red cross) on the scorer's verdict stage.

## Subject lane

Navigation: ⬅️ [Devices in play](#devices-in-play) | 📋 [TOC](#table-of-contents) | [Next](#process-lane) ➡️

### Prompt 1 (Subject): PoE composition and Mono, side by side

[verified] 🖼️ ⏳ not rendered
Save as: `diagrams/context-01-poe-and-mono.png`

```
Style: technical cutaway diagram, light background. The subject drawn as labeled 3D layered
volumes or stacked stages left to right, with the actual data visibly transforming between
stages. Real sample artifacts at the edges: an example input drawn at the left, the concrete
output (bars, a table row, a labeled result) at the right. Horizontal labeled brackets beneath
the image grouping stages into named phases. Thin leader lines from labels to parts. Restrained
color: neutral grays and one warm plus one cool accent. Precise, textbook-quality, no cartoon
elements.

Scene: two side-by-side lanes sharing one input. Top lane: a box labeled "expert: a cat" and a
box labeled "expert: a dog" each feed a merge stage labeled "combine (PoE)", producing an output
panel showing a fused single-creature silhouette. Bottom lane: one box labeled "joint prompt: a
cat and a dog" feeds the model directly, producing an output panel showing two separate, clearly
distinct animal silhouettes.

Cast: two "expert" glyphs (top lane), one "joint prompt" glyph (bottom lane), one shared
model-icon block each lane passes through, two output-panel artifacts (fused silhouette,
separated-pair silhouette).

Flows: cool-blue arrows from each expert box into the merge stage (top lane). A single gray
arrow from the joint-prompt box into the model (bottom lane). No amber in this piece; the
interaction term is introduced in Prompt 2.

Text in image: "expert: a cat", "expert: a dog", "combine (PoE)", "joint prompt: a cat and a
dog", "PoE render", "Mono render".

Exclusions: no logos, no components beyond the two lanes and their labeled boxes, no watermark.
```

Faithfulness note: the top lane must show two separate expert boxes merging into one fused
output; the bottom lane must show one box (the joint prompt) with no merge stage. If a viewer
cannot tell which lane is PoE and which is Mono from the shape alone (merge vs no merge), the
image has failed.

### Prompt 2 (Subject): The interaction term as a gap

[verified] 🖼️ ⏳ not rendered
Save as: `diagrams/context-02-interaction-term.png`

```
[style paragraph identical to Prompt 1, verbatim]

Scene: one horizontal band per denoising step (5 representative steps shown, labeled "step 0" to
"step 49" left to right), each step showing two overlapping prediction vectors: a blue "PoE
prediction" arrow and a gray "Mono prediction" arrow, with the amber gap between their tips
highlighted and labeled "interaction term". The amber gap visibly grows from step 0 (near zero)
to step 49 (largest).

Cast: the blue PoE-prediction glyph, the gray Mono-prediction glyph, the amber
interaction-term-gap glyph (repeated per step column, growing).

Flows: no directional arrows between steps; each step is an independent snapshot. A single
horizontal leader line beneath the bands reading "step 0 → step 49" ties them into one sequence.

Text in image: "PoE prediction", "Mono prediction", "interaction term", "step 0", "step 49".

Exclusions: no numeric axis values beyond the step labels named above (the real per-step values
belong in a figure, not this diagram), no watermark.
```

Faithfulness note: the amber gap must visibly grow left to right and must never exceed the
distance between the blue and gray arrows it sits between. This draws the *shape* of growth
described in [world/interaction-term.md](world/interaction-term.md#what-the-interaction-term-is),
not the literal measured numbers, which live in a figure instead.

### Prompt 3 (Subject): The LoRA corrector, Mono-free

[stated] 🖼️ ⏳ not rendered
Save as: `diagrams/context-03-lora-corrector.png`

```
[style paragraph identical to Prompt 1, verbatim]

Scene: a cutaway of the model's cross-attention layer, drawn as one thick horizontal slab, with
a thin stacked-layer insert labeled "LoRA, rank-8" attached to it. An arrow labeled "predicted
interaction term" leaves the LoRA insert in amber and rejoins the PoE prediction arrow (labeled
"corrected PoE"), producing an output panel showing two separated animal silhouettes closer to
the Mono output of Prompt 1. A crossed-out "joint prompt" box sits beside the LoRA insert with a
"never seen at inference" label, drawn hedged (dashed outline).

Cast: the cross-attention slab, the rank-8 LoRA insert, the amber predicted-interaction-term
arrow, the crossed-out joint-prompt box, the corrected-output artifact.

Flows: amber arrow from the LoRA insert joining the blue PoE arrow, producing one corrected
arrow into the output panel.

Text in image: "cross-attention (attn2)", "LoRA, rank-8", "predicted interaction term",
"corrected PoE", "never seen at inference".

Exclusions: no training-time joint-prompt box drawn as active (it is crossed out and dashed,
inference-time only), no watermark.
```

Faithfulness note: the joint-prompt box must be visibly crossed out or dashed, never solid,
because [world/lora-corrector.md](world/lora-corrector.md#what-the-lora-corrector-is)'s whole
point is that this box is absent at inference. This piece is `[stated]`, not `[verified]`: this
build read the LoRA's rank and attachment point from `MASTER_PLAN.md`'s Mission line, not from
running the model.

### Subject capstone: One system, prompt pair to verdict

[stated] 🖼️ ⏳ not rendered
Save as: `diagrams/context-00-subject-capstone.png`

```
[style paragraph identical to Prompt 1, verbatim]

Scene: one wide cutaway composing Prompts 1 to 3 left to right: the two-lane PoE/Mono split, the
amber interaction-term gap it produces, and the LoRA insert consuming that gap as its training
target while never seeing the joint prompt at inference. A small in-image legend box in one
corner naming the four glyph colors (amber = interaction term, blue = PoE, gray = Mono/neutral,
green = scored compose).

Cast: every glyph from Prompts 1 to 3, reused without restyling.

Flows: the same flows as each individual prompt, now connected: PoE and Mono outputs feed the
interaction-term measurement; the interaction-term measurement feeds the LoRA training arrow
(dashed, since training happens offline); the LoRA's inference-time prediction feeds the
corrected-PoE output.

Text in image: same labels as Prompts 1-3, no new text.

Exclusions: no process-lane elements (no step badges, no scorer verdict chip); this is subject
only, no time in it.
```

Faithfulness note: this capstone must contain no numbered stage badges and no pass/fail chip;
those belong only to the process lane below. Every relationship drawn here must trace to a
sentence in `world/poe-composition.md`, `world/interaction-term.md`, or `world/lora-corrector.md`.

## Process lane

Navigation: ⬅️ [Subject lane](#subject-lane) | 📋 [TOC](#table-of-contents)

History: `diagrams/process-versions/`. Regenerated whole, never patched, whenever
[data/01-where-it-comes-from.md](data/01-where-it-comes-from.md)'s stages change.

### Prompt 1 (Process): Pair and seed to cached trajectory

[verified] 🖼️ ⏳ not rendered
Save as: `diagrams/context-04-process-sampling.png`

```
[style paragraph identical to Prompt 1 above, verbatim]

Scene: left to right, badge ① a pair-and-seed input card ("a cat" + "a dog", seed 42), feeding
badge ② the SDXL model block running under one of several render-method lanes stacked vertically
(PoE, Mono, corrected-with-interaction-term, corrected-with-LoRA), feeding badge ③ a per-step
cache stack (a thin repeating layer motif, 50 layers compressed with "..." after the first few
and last few).

Cast: the pair-and-seed input card, the SDXL model block, the four render-method lane labels, the
per-step cache stack.

Flows: one gray arrow from the input card into the model block; four parallel arrows (one per
lane) from the model block into the cache stack, each tinted per its method (blue for PoE, gray
for Mono, amber for the two corrected lanes).

Text in image: "pair + seed", "SDXL, 50 steps", "PoE", "Mono", "corrected (interaction term)",
"corrected (LoRA)", "cached per step".

Exclusions: no numeric step count beyond "50 steps" already named, no watermark.
```

Faithfulness note: exactly four render-method lanes, matching
[data/01-where-it-comes-from.md § The journey, stage ②](data/01-where-it-comes-from.md#the-journey).
Adding or dropping a lane without editing that stage's text first is the failure this note exists
to catch.

### Prompt 2 (Process): Cache to images and the interaction term's size

[verified] 🖼️ ⏳ not rendered
Save as: `diagrams/context-05-process-summary.png`

```
[style paragraph identical to Prompt 1 above, verbatim]

Scene: badge ③'s cache stack (carried over from Prompt 1) feeding badge ④, a split output: on
one side a row of small image thumbnails (PoE render, Mono render, solo-a, solo-b), on the other
side a small labeled data card reading "summary.json: d_T = 0.246" with a tiny amber sparkline
beneath it rising left to right.

Cast: the cache stack, four image thumbnails, one data card with its sparkline.

Flows: one gray arrow from the cache stack splitting into two: one to the thumbnail row, one to
the data card.

Text in image: "cached trajectory", "poe.png", "monolithic.png", "solo_a.png", "solo_b.png",
"summary.json", "d_T = 0.246".

Exclusions: no full 50-point sparkline data (illustrative rise only, not a real plotted curve;
the real curve is a figure, not this diagram), no watermark.
```

Faithfulness note: `d_T = 0.246` is copied verbatim from
`data/pilot/seed_42/a_cat__x__a_dog/summary.json`, read directly this session
([data/02-dictionary.md § d_T / d_t](data/02-dictionary.md#d_t-d_t)); using a different number
without checking that file first is the mistake this note exists to catch.

### Prompt 3 (Process): Scorer verdict to compose rate

[verified] 🖼️ ⏳ not rendered
Save as: `diagrams/context-06-process-scorer.png`

```
[style paragraph identical to Prompt 1 above, verbatim]

Scene: badge ⑤, a magnifier-over-bounding-boxes glyph reading "count: animal instances", taking
one image thumbnail as input and producing a status chip ("compose", green check, "n=2") or
("blend", red cross, "n=1"). Badge ⑥, a small bar reading "fail_rate: 1.00 (8/8)" aggregating
several per-seed chips into one pair-level number, feeding a final labeled arrow into an
off-panel "figure / plan verdict" box drawn faint and cut off at the frame edge (signalling it is
out of scope for this diagram).

Cast: the magnifier/detector glyph, the compose (green) and blend (red) status chips, the
fail-rate aggregation bar, the faint off-panel "figure / verdict" box.

Flows: one arrow from the image thumbnail into the detector; two conditional arrows from the
detector into either status chip; multiple per-seed chips feeding one aggregation arrow into the
fail-rate bar; one final gray arrow toward the faint off-panel box.

Text in image: "count: animal instances", "compose", "blend", "n=2", "n=1",
"fail_rate: 1.00 (8/8)", "figure / plan verdict".

Exclusions: the off-panel box must stay faint and visibly cut at the frame edge, never a full
detailed panel, since what a figure or a plan verdict contains is out of this folder's scope per
[00-INDEX.md](00-INDEX.md)'s "The question | The file that owns it" table.
```

Faithfulness note: the compose rule ("≥2 distinct animal instances") and the example numbers
(`n=2`, `n=1`, `1.00 (8/8)`) must match
[world/compose-rate.md](world/compose-rate.md#what-a-compose-rate-is) and
`artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md` exactly; this is a `[verified]` piece because both
source numbers were read directly this session.

### Process capstone: One journey, prompt to verdict

[verified] 🖼️ ⏳ not rendered
Save as: `diagrams/context-07-process-capstone.png`

```
[style paragraph identical to Prompt 1 above, verbatim]

Scene: one continuous left-to-right journey composing Prompts 1 to 3 of this lane: badge ① input
through badge ⑥ aggregated fail/compose rate, all six badges visible on one mainline, with the
four render-method lanes from Prompt 1 shown merging back into a single mainline at badge ④
(since every lane produces the same downstream shape of output: images plus a summary).

Cast: every glyph from process Prompts 1 to 3, reused without restyling, plus the six numbered
step badges in one continuous sequence.

Flows: one mainline arrow left to right, with the four-lane split from Prompt 1 shown as a
widening and re-narrowing of the mainline (a visual "swelling" between badges ② and ④), not a
separate branch that never rejoins.

Text in image: badges "①" through "⑥" only, plus the stage names already used in Prompts 1-3
("pair + seed", "SDXL, 50 steps", "cached per step", "summary.json", "count: animal instances",
"fail_rate").

Exclusions: no subject-lane content (no expert/joint-prompt boxes, no LoRA insert); this is
process only, time is the point.
```

Faithfulness note: the six badges must appear in the same order and carry the same names as
[data/01-where-it-comes-from.md § The journey](data/01-where-it-comes-from.md#the-journey)'s six
numbered stages. A regeneration that adds a seventh stage without that file changing first is the
drift this note exists to catch.
