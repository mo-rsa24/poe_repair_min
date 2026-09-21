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
| Green | measured and passed: a threshold met, a figure built, a component confirmed in the repo |
| Red | measured and failed, or a null: a control that reads what you would get by luck, a threshold not met, a known limitation |

> A null here means the numbers came out the same with the correction as without it.

A **control** is drawn as a hollow amber outline at the same size as the solid amber correction,
never as a fifth colour. That is faithful to what a control is here: the same size as the real
correction, differing only in direction.

## Glyph vocabulary

The same glyph means the same thing in every image. This is what both capstones reassemble.

| Glyph | Stands for |
|---|---|
| Cache stack | the store of cached per-step predictions every analysis reads |
| Correction coil | `r_t`, the step-by-step gap between the joined prompt and plain PoE |
| Denoising track | the 50 denoising steps, noise at the left, finished image at the right |
| Multiplier dial | λ, how much of the correction is added back |
| Window bracket | a stretch of steps during which the correction is allowed to act |
| Adapter chip | the rank-8 cross-attention adapter that learns the correction |
| Scorer lens | the detector that counts animals in a picture |
| Figure board | the figure register: one card per place the paper reserves |
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

Scene: one reading direction, left to right. On the left, two small prompt cards stacked vertically. Each feeds its own prediction glyph. The two prediction glyphs meet at a multiply node in the lower middle, which outputs a blue prediction. Above them, separated by clear white space, a single wider prompt card feeds one prediction glyph directly, outputting an amber prediction. On the right, the two predictions meet at a subtract node, and the difference leaves it as a thick amber coil glyph sitting on its own platform. Far right, two outcome tiles stacked: the upper tile shows one animal with a cat ear and whiskers on its left half and a dog ear and muzzle on its right half, bordered blue; the lower tile shows a cat and a dog as two separate animals, bordered amber. Beneath the whole flow, a stacked-discs cache glyph receives a thin gray line from both prediction glyphs.

Cast: prompt cards (a text prompt entering the model), prediction glyphs (what the model predicts at one step), a multiply node, a subtract node, the correction coil (r_t, the gap between the two readings), two outcome tiles (generated pictures), the cache stack (per-step predictions saved for later analysis).

Flows: blue dashed line for the uncorrected Product-of-Experts path, from the two separate prompts through the multiply node to the blended outcome tile. Amber dashed line for the joined-prompt path and for the correction leaving the subtract node toward the two-animal tile. Thin solid gray lines from both prediction glyphs down into the cache stack, meaning saved, not computed. Legend inside the image, lower left.

Text in the image: title banner "two readings of one sentence". Labels: "a cat", "a dog", "a cat and a dog", "multiply", "plain PoE", "joined prompt", "subtract", "r_t: the gap", "one blended animal", "two separate animals", "cached every step". Legend: "blue: uncorrected path", "amber: the correction".

Exclusions: no product logos, no brand marks, no components other than those listed, no numbered badges or status pills in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the blue path must reach the blended tile and the amber path the two-animal tile, never crossed. The correction must leave the subtract node and nowhere else, because `r_t` is defined as that difference and drawing it as an independent input would teach the opposite.

### Prompt 2 (Subject): The runner that adds the correction back

[built]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one reading direction, left to right. Across the full width of the lower two thirds runs a long horizontal strip divided into many small equal segments, dark and grainy at its left end and resolving into a clear picture at its right end. Above the strip's left quarter sits a rounded bracket spanning about a fifth of the strip's length, with a small arrow showing it can slide along. Above the bracket, a circular dial glyph with a pointer. On the far left, five small vector glyphs stacked vertically inside a rounded soft-tinted panel, each with an arrow feeding into the dial: the topmost drawn as a solid amber coil, the four beneath it drawn as hollow amber outlines of the same size. On the far right, two outcome tiles stacked, the upper bordered amber showing two separate animals, the lower bordered blue showing one blended animal.

Cast: the denoising track (the 50 denoising steps), the window bracket (the stretch of steps during which the correction may act), the multiplier dial (λ, how much correction is added), the correction coil solid (the pair's own real correction), four hollow outlines of identical size (the matched substitutes), two outcome tiles.

Flows: blue dashed line running the full length of the denoising track underneath it, meaning the uncorrected base runs at every step regardless. Amber dashed line from the dial down into only the segments the bracket covers, meaning the correction acts only inside the window. Legend inside the image, lower left.

Text in the image: title banner "adding the correction back". Labels: "noise", "finished image", "50 steps", "window switch", "multiplier dial", "own correction", "other pair", "other seed", "steps shuffled", "random vector", "same size, wrong direction", "two separate animals", "one blended animal". Legend: "blue: base runs always", "amber: correction injected".

Exclusions: no product logos, no numbers on the dial, no components other than those listed, no numbered step badges or status pills in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the four substitutes must be drawn the same size as the real correction and differ only in being hollow, because they are matched in size and differ only in direction. The amber line must enter only the segments under the bracket, and the blue line must run the strip's full length, or the picture claims the prompt is switched off outside the window, which it is not.

### Prompt 2a (Subject): One thing changes, which is when the correction acts

[built] `diagrams/when-the-correction-acts.png`. Zoom into
prompt 2's window bracket, for
`plans/03-does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md`.

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one tall composition, read top to bottom. Nine identical horizontal strips stacked with even spacing, each divided into many small equal segments, each dark and grainy at its left end and resolving into a clear picture at its right end. Above each strip sits a rounded bracket covering one fifth of that strip's length; the bracket sits at the far left on the topmost strip and shifts one notch further right on each strip below, reaching the far right on the bottom strip. A single thin vertical guide line runs down through all nine strips at a position just under a third from the left, crossing inside the bracket on only two of the strips. At the right edge, level with the topmost strip, one outcome tile bordered amber showing two separate animals; level with the bottom strip, one outcome tile bordered blue showing one blended animal. On the far left, a small rounded soft-tinted panel holding a grid glyph labelled with its three dimensions.

Cast: nine denoising tracks (the 50 denoising steps, one strip per window placement), nine window brackets (the stretch of steps during which the correction may act), the vertical guide line (the fork step, step 16, measured independently from cached runs), two outcome tiles, one grid glyph.

Flows: a blue dashed line running the full length of every one of the nine strips, underneath it, meaning the uncorrected base and the prompt run at every step in every row. An amber dashed line entering each strip only in the segments its own bracket covers. Legend inside the image, lower left.

Text in the image: title banner "one thing changes: when the correction acts". Labels: "noise", "finished image", "50 steps", "steps 0-10", "steps 40-50", "window switch", "fork step, measured elsewhere", "9 windows x 8 pairs x 4 seeds = 288 runs", "two separate animals", "one blended animal". Legend: "blue: base and prompt run always", "amber: correction injected".

Exclusions: no product logos, no numbered step badges, no status pills, no arrows implying the nine rows happen in sequence, no curve or chart, no components other than those listed, no placeholder gibberish text, no watermark.
```

Faithfulness note: the blue line must run the full length of all nine strips. If it stops outside a bracket the picture claims conditioning is switched off too, which is the confound the whole design exists to avoid. The fork-step guide must cross inside only two of the nine brackets, because that is what lets the series disagree with it rather than being built to agree. The nine rows are placements, not stages, so no arrows between them.

### Prompt 3 (Subject): The measuring tools that decide what happened

[built]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: three rounded soft-tinted panels side by side, each titled at its top, one reading direction left to right. Left panel holds a magnifying lens glyph over a small picture with two rectangular boxes drawn on the animals in it, and beneath the lens a small square swatch showing a minimum box size. Middle panel holds a bench with three small chart glyphs on it: a descending bar series, a rising-then-forking pair of lines, and a single curve against a shaded band. Right panel holds two glyphs: a speech-bubble over a picture, and a small track with a dot sliding along it between two endpoint thumbnails. Beneath the left panel, outside it, one small red warning badge attached by a thin leader line to the lens.

Cast: the scorer lens (counts how many animals a picture holds), the cache analysis bench (measurements taken off the saved predictions with no new generation), the second-opinion tests (measuring tools that share none of the scorer's machinery), the researcher persona standing beside the right panel.

Flows: green solid arrows from each panel toward a shared strip at the bottom of the image, meaning a measured verdict leaves each measuring tool. One red solid arrow from the warning badge pointing back at the lens, meaning a known limitation of that measuring tool. Legend inside the image, lower right.

Text in the image: title banner "three ways to check". Panel titles: "count the animals", "read the cache", "ask another measuring tool". Labels: "at least two boxes", "minimum box size", "how few directions", "where paths fork", "size against noise", "caption readback", "manifold slide", "cannot tell which animals". Legend: "green: a verdict", "red: a known limit".

Exclusions: no product logos, no components other than those listed, no numbered step badges or status pills in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the red warning must attach to the counting lens and to nothing else. That measuring tool counts animals without identifying them, so two dogs scores the same as a cat and a dog, and attaching the limitation to the cache bench or the second-opinion tests would misplace the one weakness the paper has to declare.

### Prompt 4 (Subject): The adapter that learns the correction

[built]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one reading direction, left to right. On the left, a stacked-discs cache glyph with eleven small amber coil glyphs arranged in a column beside it inside a rounded soft-tinted panel. An amber arrow leads right into a small rectangular chip glyph seated inside a larger transparent block, the chip noticeably smaller than the block. From the chip, amber arrows fan right to eight outcome tiles arranged in two rows of four, each tile showing two separate animals and bordered green. Below that fan, drawn entirely in dashed gray outline with no fill, a second fan of fifteen tiny chip glyphs, visibly dashed and faded against the solid work above.

Cast: the cache stack (saved corrections), eleven correction coils (the training pairs), the adapter chip (rank-8, seated inside a cross-attention block), eight outcome tiles (pairs the adapter never trained on), the dashed fifteen-chip fan (the leave-one-pair-out series).

Flows: amber dashed lines from the cached corrections into the chip, meaning it learns from them. Amber dashed lines from the chip out to the eight held-out tiles, meaning it is applied to pairs it never saw. The lower fan carries no flow lines at all, only its dashed outline. Legend inside the image, lower left.

Text in the image: title banner "learned once, applied to new pairs". Labels: "cached corrections", "eleven training pairs", "rank-8 adapter", "cross-attention block", "eight held-out pairs", "never sees the joined prompt", "leave-one-pair-out", "not yet run". Legend: "amber: the correction", "dashed: not yet run".

Exclusions: no product logos, no components other than those listed, no numbered step badges or status pills in this image, no solid fill or colour on the lower fan, no placeholder gibberish text, no watermark.
```

Faithfulness note: the lower fan of fifteen must be visibly dashed and carry no flow line, because that series has not run. The chip must be drawn smaller than the block it sits in, and no arrow may run from the joined prompt into the chip, since the whole claim is that the adapter works without ever seeing the joined prompt.

### Prompt 4a (Subject): The joined prompt is a stored target, never an input

[planned] Zoom into prompt 4's adapter chip, for paragraph 5.2 of
[the manuscript](../paper/iclr/iclr2027_conference.tex), walked in
[the draft map](../paper/iclr/DRAFT_MAP.md).

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: two stacked horizontal bands of equal width, each inside its own rounded soft-tinted panel with a short title at its top left, separated by clear white space. Both bands read left to right and share the same glyph language. In the upper band: on the left a small square state token, then three small prompt cards stacked vertically, all four feeding one wide transparent block that has a small rectangular chip seated inside it, the chip clearly smaller than the block and drawn glossy and lit. Three prediction glyphs leave the block and meet at a multiply node, whose output is an amber prediction glyph. Above and to the right, a stacked-discs cache glyph sits with a single flat artifact card beside it, and the card carries a thin solid gray line down to a compare node on the far right. The amber prediction also enters that compare node. In the lower band: the same square state token on the left feeds two separate transparent blocks stacked vertically, the upper block holding the same chip drawn flat, gray and unlit, the lower block holding the same chip drawn glossy and lit. The same three prompt cards sit between the state token and the two blocks, with lines fanning to both. Each block's three predictions meet their own multiply node inside the block's rounded panel. The upper path leaves as a blue prediction glyph, the lower as an amber prediction glyph. Both enter a subtract node in the middle right, from which a single thick amber coil glyph leaves and passes through a circular dial glyph with a pointer, then into an add node. The blue prediction also enters the add node by a second line routed beneath the coil. From the add node, one outcome tile on the far right, bordered amber, showing a cat and a dog as two separate animals. There is no prompt card anywhere in the lower band other than the three already named.

Cast: the state token (the same noisy image both passes read), three prompt cards (the first concept, the second concept, and no prompt), the cross-attention block (the frozen network's cross-attention), the adapter chip (rank-8, lit when active and gray when switched off), prediction glyphs, multiply nodes (the fixed combination rule), the cache stack (per-step predictions saved earlier), the stored target card (the joined-prompt prediction, read from the cache stack), the compare node (squared error), the subtract node, the correction coil (the predicted correction), the multiplier dial (lambda), the add node, one outcome tile.

Flows: amber dashed lines for anything carrying or learning the correction, which is the adapted pass in both bands, the coil leaving the subtract node, and the line into the outcome tile. Blue dashed lines for the uncorrected path, which is the unlit block's output in the lower band and its line into the add node. One thin solid gray line from the stored target card into the compare node, meaning saved, not computed. The three prompt cards connect to the blocks with plain thin lines carrying no colour. Legend inside the image, lower left.

Text in the image: title banner "the joined prompt is only a target". Panel titles: "training", "inference". Labels: "same noisy state", "a cat", "a dog", "no prompt", "cross-attention block", "rank-8 adapter", "adapter off", "adapter on", "combine", "stored target", "squared error", "subtract", "predicted correction", "multiplier dial", "add", "two separate animals". Legend: "blue: uncorrected path", "amber: the correction", "gray: saved, not computed".

Exclusions: no prompt card reading "a cat and a dog" anywhere in the image, no line of any colour running from the cache stack or the stored target card into either band's blocks, no numbered step badges, no status pills, no arrows between the two bands, no components other than those listed, no product logos, no placeholder gibberish text, no watermark.
```

Faithfulness note: the joined prompt must appear once, as the flat stored target card on a gray line into the compare node in the upper band, and nowhere else. Any card or arrow putting it into a block would draw the opposite of the section's claim. The two blocks in the lower band must share one state token and differ only in whether the chip is lit, since that difference is the entire predicted correction. The coil must leave the subtract node and nowhere else, matching prompt 1's rule for `r_t`. The blue line must reach the add node, because the corrected prediction is the uncorrected one plus a scaled correction, not the adapted pass used on its own.

### Prompt 4b (Subject): Where the adapter hooks, and what flows through it

[built] Zoom into prompt 4's adapter chip at the level of one cross-attention block. Every
placement below is read from code, not stated: the adapter targets `attn2.to_q`, `attn2.to_k`,
`attn2.to_v` at rank 8 (`poe_repair/experiments/cross_pair_lora_pooling/sample_crossbar.py`,
`_attach_lora_from_ckpt`), the two text encoders' penultimate states are concatenated into the
cross-attention sequence and the pooled embedding enters with the time ids
(`poe_repair/_sdxl/runtime.py`, `encode_prompt_sdxl`), and one step is one three-branch forward
on the concept prompts and the null prompt (`poe_repair/methods/_sampling.py`, `run_cfg_poe`).
The manuscript's own sentence is at `paper/iclr/iclr2027_conference.tex` line 261. 🖼️ rendered 2026-09-03 `diagrams/where-the-adapter-hooks.png` (attempt 2: labels and guide routing corrected by raster edit in the same Codex session; both checks passed).
Save as: `diagrams/where-the-adapter-hooks.png`

Visual thesis: the adapter touches exactly three projections, the ones where the prompt meets the
latent inside every cross-attention block, and the same frozen network is read three times per
step, once per prompt card, with no joined prompt anywhere.

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one reading direction, left to right, three rounded soft-tinted panels of increasing width, in the manner of an architecture figure with region boxes. Left panel: three small prompt cards stacked vertically. Each card's line passes through a pair of small encoder glyphs drawn side by side, then into one concat node, which emits a long thin token strip. A thinner second line leaves only the right-hand encoder glyph and runs beneath the strip toward the middle panel, entering it low, at a small clock-shaped time node. Middle panel: a U-shaped stack of layered blocks, the whole stack drawn matte and gray with a small lock glyph at its corner. A square state token sits at the stack's left entrance. One block of the stack is pulled out into a rounded callout drawn larger: inside the callout, three small projection boxes in a row, and a fourth box to their right drawn flat gray. A small glossy amber chip is seated on each of the three boxes; no chip sits on the fourth box. The state token's line enters the first box; the token strip's line splits and enters the second and third boxes. Beside the callout, a second small block of the stack labeled as self-attention is drawn flat gray with no chip. Three prediction glyphs leave the right edge of the stack, one per prompt card, each drawn amber. Right panel: the three predictions pass two small guidance nodes and meet a combine node; one amber prediction leaves the combine node into a step glyph drawn as a short slanted arrow; a loop line returns from the step glyph back to the state token along the top of the panel. At the far right, one outcome tile showing a cat and a dog as two separate animals, bordered green. Legend inside the image, lower left.

Cast: three prompt cards (the first concept, the second concept, no prompt), two encoder glyphs (the two text encoders), the concat node, the token strip (the prompt as a sequence of vectors), the time node (where the pooled prompt vector enters, away from the cross-attention), the state token (the noisy latent), the frozen network stack with its lock, the cross-attention callout, three projection boxes (query, key, value), the fourth flat box (the output projection, untouched), the flat self-attention block (untouched), three adapter chips (the rank-8 LoRA, one per projection), three prediction glyphs, two guidance nodes, the combine node (the fixed rule), the step glyph (one denoising step), the loop line (fifty steps), one outcome tile.

Flows: thin solid gray lines for all wiring that carries no correction, which is the prompt cards into the encoders, the strip into key and value, the pooled line into the time node, and the loop back to the state token. Amber dashed lines for the three predictions and the combined prediction, meaning the adapter was on when they were computed. No blue line in this image. Legend inside the image, lower left.

Text in the image: title banner "where the adapter hooks". Panel titles: "conditioning", "frozen UNet ε_θ", "one step". Labels: "a cat", "a dog", "no prompt", "text encoder 1", "text encoder 2", "concat", "77 × 2048 tokens", "pooled, with time", "x_t", "frozen", "cross-attention", "Q", "K", "V", "out: untouched", "self-attention: untouched", "rank-8 LoRA", "ε(x_t | a cat)", "ε(x_t | a dog)", "ε(x_t | ∅)", "guide: ε_∅ + w(ε − ε_∅)", "combine: ε̃_a + ε̃_b − ε_∅", "ε̂_PoE", "DDIM step", "× 50", "two separate animals", "no joined prompt". Legend: "amber: adapter on", "gray: frozen wiring".

Exclusions: no prompt card reading "a cat and a dog", no chip on the output projection, on the self-attention block, on any encoder, or anywhere outside the three projection boxes, no line from the pooled path into the cross-attention callout, no fourth prompt card, no correction coil or subtract node in this image, no numbered step badges, no status pills, no product logos, no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no watermark.
```

Faithfulness note: the three chips must sit on the query, key and value boxes of the cross-attention callout and nowhere else, because `peft` targets exactly `attn2.to_q`, `attn2.to_k` and `attn2.to_v`; the output projection and the self-attention block must be visibly unlit. The token strip must enter key and value and the state token must enter query, since that is where the prompt meets the latent. The pooled line must bypass the callout and enter the time node, because SDXL's pooled embedding is added with the time ids and never passes through cross-attention. Three prompt cards, one network, three predictions: one three-branch forward at the same state, never three networks.

### Prompt 4c (Subject): The objective, and why matching the joined prompt is matching the residual

[built] Zoom into prompt 4a's compare node. The two bands of 4a (training, inference) are not
redrawn here; this piece shows only the algebra of the loss and the return path. Read from
`poe_repair/experiments/one_pair_one_seed/trainer.py` (`_train_one_step`: adapter-on
three-branch pass, guided, combined by the PoE rule, mean squared error against the cached
joined-prompt prediction guided the same way), `poe_repair/_sdxl/metrics.py` (`guided_eps`,
`poe_eps`), and `poe_repair/methods/_sampling.py` (`run_lora_residual_inject`: the corrected
prediction is the frozen combination plus λ times the adapter's shift, and λ = 1 is the adapted
combination itself; the joined prompt is deleted from the inference signature). 🖼️ rendered 2026-09-03 `diagrams/what-the-adapter-is-trained-to-match.png` (attempt 2: subtract-node labels swapped to match their inputs and the sibling's light style restored, in the same Codex session; both checks passed).
Save as: `diagrams/what-the-adapter-is-trained-to-match.png`

Visual thesis: the adapter is never handed the residual as an input; it is trained so that plain
PoE through the adapted network lands on the joined-prompt prediction, and by one line of
algebra that is the same as making its own shift equal the residual, which is why the corrected
sampler at inference is just the adapted network run as plain PoE.

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one reading direction, left to right, in three rounded soft-tinted panels. Left panel, narrow: a stacked-discs cache glyph with one flat card pulled out to its right, and beneath that card a second flat card drawn amber-bordered. Middle panel, widest, two rows sharing the pulled-out card as their common input. Upper row: a square state token, then one transparent block with a small glossy lit chip seated inside it, then three prediction glyphs, then a combine node, then an amber prediction glyph. Lower row, directly beneath: the same state token, the same block with the same chip drawn flat, gray and unlit, three prediction glyphs, a combine node, a blue prediction glyph. Between the rows on the right, two small compare nodes stacked and joined by a short bracket carrying an equals sign: the upper compare node takes the amber prediction and a thin gray line from the amber-bordered stored card; the lower compare node takes two short coil glyphs, one amber coil that leaves a subtract node fed by the amber and blue predictions, and one amber coil that leaves a second subtract node fed by the stored card's gray line and the blue prediction. From the upper compare node, one thin amber dashed line runs back leftward along the top of the panel and terminates on the lit chip, and only on the chip; the block around the chip carries a small lock glyph. Right panel, narrow, titled as the return: the lit chip drawn once more inside its block, one row of three prediction glyphs into a combine node, an amber prediction glyph, a small dial glyph set to its end stop, and one outcome tile showing two separate animals, bordered green. Legend inside the image, lower left.

Cast: the cache stack (per-step predictions saved earlier), the pulled-out card (one cached step: the noisy state and its three stored predictions), the stored target card (the joined-prompt prediction, saved, drawn amber-bordered), the state token, the cross-attention block with its lock (the frozen network), the adapter chip (lit when on, gray when off), prediction glyphs, two combine nodes (the fixed rule), two subtract nodes, two correction coils (the true residual, and the adapter's shift), two compare nodes joined by an equals bracket (the same loss written two ways), the gradient line, the multiplier dial at its end stop (λ = 1), one outcome tile.

Flows: amber dashed lines for anything carrying or learning the correction: the adapted row's predictions, both coils, the gradient line, and the return panel's prediction. Blue dashed lines for the uncorrected row's predictions and its line into both subtract nodes. Thin solid gray lines from the cache stack and the stored target card, meaning saved, not computed. The gradient line is the only line that runs right to left, and it ends on the chip. Legend inside the image, lower left.

Text in the image: title banner "what the adapter is trained to match". Panel titles: "cached step", "the objective", "the return". Labels: "x_t, t", "stored ε_a, ε_b, ε_∅", "stored ε_J", "a cat", "a dog", "no prompt", "adapter on", "adapter off", "frozen", "combine: ε̃_a + ε̃_b − ε_∅", "ε̂_PoE lora", "ε̂_PoE frozen", "target ε̃_J", "MSE", "r_t = ε̃_J − ε̂_PoE", "Δ̂_t = lora − frozen", "same loss", "gradient: LoRA only", "λ = 1", "plain PoE, adapter on", "two separate animals". Legend: "blue: uncorrected", "amber: the correction", "gray: saved, not computed".

Exclusions: no prompt card reading "a cat and a dog" anywhere, no line from the stored target card into either block, no gradient line reaching anything but the chip, no arrow between the two rows other than into the subtract and compare nodes, no third row, no numbered step badges, no status pills, no product logos, no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no watermark.
```

Faithfulness note: the joined prompt must appear only as the stored target card on a gray line, never as an input to a block, matching prompt 4a's rule. The two compare nodes must be joined by the equals bracket, because the loss on predictions and the loss on residuals are the same quantity exactly (the adapted prediction minus the target equals the adapter's shift minus the residual). The gradient line must end on the chip alone, since the base network is frozen. The return panel's dial must sit at its end stop, because at λ = 1 the corrected sampler is the adapted network run as plain PoE, with nothing added afterwards.

### Prompt 5 (Subject): Where the evidence is kept

[built]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: two rounded soft-tinted panels side by side, one reading direction left to right. The left panel is a board holding eight small framed picture cards in a grid, most bordered green, one bordered red, two drawn as empty dashed frames. The right panel is a shelf of upright document glyphs, each with a small tag clipped to its top edge. A green arrow runs from the shelf to the board, entering the board from its left side. A researcher persona stands between the two panels.

Cast: the figure board (one card per figure place the paper reserves), the framed cards (figures), the document shelf (the review files that judge each run), the tags (the threshold each run was judged against, written before the run), the researcher persona.

Flows: green solid arrow from the shelf to the board, meaning a figure may only be built from an answered review question. Legend inside the image, lower right.

Text in the image: title banner "nothing prints without its verdict". Panel titles: "figure register", "review files". Labels: "built", "reserved", "needs a decision", "threshold written first", "answered", "one card, one claim". Legend: "green: answered and built", "red: argument does not stand".

Exclusions: no product logos, no components other than those listed, no numbered step badges in this image, no readable chart content inside the framed cards, no placeholder gibberish text, no watermark.
```

Faithfulness note: the arrow must run from the review shelf into the figure board and never the reverse, because a figure is licensed by an answered question rather than the other way round. One card must be red: the low-rank card's argument does not currently stand.

### Prompt 6 (Subject): The manuscript, which produces nothing of its own

[planned]

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one reading direction, left to right. On the left, the figure board glyph and the document shelf glyph from the previous image, drawn smaller and solid. Both feed right into a single large document glyph drawn entirely in dashed outline with a pale unfilled interior, sitting on its platform. The document shows a title line drawn as a gray placeholder bar rather than real text, and several section blocks beneath it, two of which are filled green and the rest left empty dashed. No arrow leaves the document to the right. A researcher persona sits at the document's left, working on it.

Cast: the figure board and the document shelf (both carried over unchanged in meaning), the manuscript document drawn dashed because it is not yet written, the researcher persona.

Flows: green dashed arrows from both the board and the shelf into the manuscript, meaning it consumes them. No outgoing flow of any kind. Legend inside the image, lower left.

Text in the image: title banner "the paper consumes, never produces". Labels: "figure register", "review files", "reads both", "produces no figures", "title still a stub", "build works", "sections owed". Legend: "green: consumed evidence", "dashed: not yet written".

Exclusions: no product logos, no readable title text on the document, no outgoing arrow from the manuscript, no components other than those listed, no numbered step badges in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: nothing may leave the manuscript. This scope's rule is that the writing consumes the register and the review files and produces no figures of its own, and an outgoing arrow would invert it. The document stays dashed: the build works but the title is still the stock template's.

### Subject capstone: The whole system on one page

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one wide landscape composition, reading left to right across three rounded soft-tinted panels, with a fourth smaller panel at the far right. Panel one, titled at its top, holds the two prompt cards, the multiply node, the subtract node and the correction coil, with the cache stack beneath them. Panel two holds the denoising track with its sliding window bracket and multiplier dial above it, the five vector glyphs feeding the dial (one solid amber, four hollow amber outlines), and two outcome tiles at its right edge. Panel three holds the scorer lens, the cache analysis bench, the second-opinion tests, and beneath them the adapter chip with its dashed fifteen-chip fan. Panel four, at the far right and narrower, holds the figure board above the document shelf, and beneath both, the manuscript document drawn in dashed outline. A researcher persona stands once, between panel three and panel four.

Cast: every glyph from prompts 1 to 6, each keeping the meaning it has there and nothing added: prompt cards, prediction glyphs, multiply and subtract nodes, correction coil, cache stack, denoising track, window bracket, multiplier dial, four hollow control outlines, outcome tiles, scorer lens, cache analysis bench, second-opinion tests, adapter chip, dashed fifteen-chip fan, figure board, document shelf, manuscript document, researcher persona.

Flows: blue dashed for the uncorrected path, running from the multiply node in panel one along the full denoising track in panel two to the blended outcome tile. Amber dashed for the correction, running from the subtract node in panel one to the multiplier dial in panel two, and separately from the cache stack to the adapter chip in panel three. Green solid for a measured verdict, running from the measuring tools in panel three to the figure board and the document shelf in panel four, then into the manuscript. The dashed fifteen-chip fan carries no flow. A single in-image legend sits in the lower left, spanning the width of panel one.

Text in the image: title banner "closing the compositional gap". Panel titles: "where the blend comes from", "adding it back", "measuring and learning", "the paper". Labels: "r_t: the gap", "50 steps", "window switch", "multiplier dial", "same size, wrong direction", "count the animals", "rank-8 adapter", "not yet run", "figure register", "review files", "produces no figures". Legend: "blue: uncorrected path", "amber: the correction", "green: a measured verdict", "dashed: not yet run".

Exclusions: no product logos, no object that did not appear in prompts 1 to 6, no object from those prompts left out, no numbered step badges or status pills anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the four panels must read left to right as one chain, and the green verdict flow must be the only thing entering the paper. The correction has two distinct amber destinations and both must be drawn: injected by hand at the multiplier dial, and learned from the cache by the adapter. Drawing only one of them collapses the scope's two claims into one.

## Process lane

History: `diagrams/process-versions/`. This lane is regenerated whole from the current plan files
whenever the plans change shape, never patched in place.

### Prompt 1 (Process): Fix the measuring tools before any result exists

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding three stage cards in a horizontal row connected by directional arrows. Each card carries a small numbered circular badge at its upper left and two or three tiny bullet lines inside it. Card one holds the scorer lens glyph, card two holds a padlock over a small written note, card three holds the correction coil beside a small pool of paired animal thumbnails. All three cards carry a green "Completed" pill at their lower right. A small legend row runs along the bottom edge.

Cast: the scorer lens, a padlock over a written choice (a decision committed in writing before any result could be seen), the correction coil, a pool of animal-pair thumbnails, all keeping their meanings from the subject lane.

Flows: green solid arrows left to right between the three cards, meaning each is finished and the next may start. Legend inside the image, bottom edge.

Text in the image: title banner "measuring tools first". Phase container title: "committed before any result". Card titles: "build the measuring scripts", "fix the size measure", "curate the pair pool". Bullets: "thirteen scripts", "run end to end once", "chosen before results", "cannot follow the answer", "eleven pairs", "all blend by default". Pills: "Completed". Legend: "green: completed".

Exclusions: no product logos, no stage other than the three listed, no red or gray status pills in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: all three stages carry a completed pill. Each of these measuring tools was fixed in writing before any result was read, which is the only reason the later numbers count as evidence, so none may be drawn as in progress.

### Prompt 2 (Process): The causal runs

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding five stage cards. Four sit in a horizontal row connected by directional arrows; the fifth sits below the row, connected upward into the third card. Each card carries a numbered circular badge at its upper left and two or three tiny bullet lines. Card one holds the multiplier dial, card two holds the sliding window bracket over a short denoising track, card three holds the cache analysis bench, card four holds three small measuring-tool glyphs side by side, and the lower card holds the scorer lens over a map of tests. All five carry a green "Completed" pill.

Cast: the multiplier dial, the window bracket and denoising track, the cache analysis bench, the three second-opinion tests, the scorer lens over a map of tests, all carrying their subject-lane meanings.

Flows: green solid arrows between the cards, meaning each finished. Legend inside the image, bottom edge.

Text in the image: title banner "the causal runs". Phase container title: "all five answered". Card titles: "more correction, more composition", "when in the run it matters", "read the cached runs", "the same story three ways", "what changes inside the model". Bullets: "three rows compared", "controls stayed flat", "early window only", "cliff at the start", "no new generation", "two nulls kept", "held-out pairs". Pills: "Completed". Legend: "green: completed".

Exclusions: no product logos, no stage other than the five listed, no gray skipped pills, no red callout or warning badge anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: all five stages are completed and none carries an outstanding cleanup. The multiplier stage's output now sits on `/datasets` where it belongs, so a red callout on card one would report a problem that no longer exists.

### Prompt 2a (Process): The stages that produced the timing answer

[planned] Zoom into prompt 2's card two, for
`plans/03-does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md`.

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning most of the image, titled at its top, holding six stage cards in a horizontal row connected by directional arrows. Each card carries a numbered circular badge at its upper left and two or three tiny bullet lines. Card one holds a caliper glyph measuring a short denoising track. Card two holds a window bracket sitting entirely off the end of a denoising track, with a small equals sign beside it. Card three holds a denoising track with three brackets at left, middle and right. Card four holds a grid glyph. Card five holds two denoising tracks stacked with a multiplier dial between them. Card six holds a figure board. Cards one to five carry a green "Completed" pill; card six carries a green "Completed" pill and a red warning badge with a dashed callout box hanging below it. Below the container and to the right, a seventh card sits on its own, connected upward into card six, holding a researcher persona beside a slider glyph and carrying an amber "In progress" pill. A red dashed branch leaves card two downward into a small muted card holding a crossed-out denoising track.

Cast: the caliper over a denoising track, the window bracket and denoising track, the grid glyph, the multiplier dial, the figure board, the researcher persona with a slider, all carrying their subject-lane meanings.

Flows: green solid arrows between cards one through six, meaning each finished. A red dashed branch from card two into the muted card, meaning the run that stops everything if it fires. A green solid arrow from card six down into the seventh card. Legend inside the image, bottom edge.

Text in the image: title banner "when in the run it matters". Phase container title: "answered: the cliff is at the start". Card titles: "fix the width in source", "prove the switch does not leak", "a short run on three windows", "generate and score 288 runs", "untie timing from the multiplier", "eight figures built". Bullets: "width 10, stride 5", "chosen before any run", "all-off equals plain PoE", "early, middle, late", "9 x 8 x 4", "no missing windows", "same total, later", "cliff survives", "F4a to F4h". Seventh card: "drive the timing tab by hand", bullets "slider moves picture and curve", "the check numbers cannot make". Pills: "Completed", "In progress". Muted card: "the switch leaks, stop". Callout: "one figure unregistered, two rows still say reserved". Legend: "green: completed", "amber: yours to do", "red: stops the plan".

Exclusions: no product logos, no stage other than the seven listed, no gray "Skipped" pills, no chart or curve, no placeholder gibberish text, no watermark.
```

Faithfulness note: card six is completed and its red callout is a register gap, not a failed result. Drawing the stage as failed would misreport eight figures that exist. The seventh card is the only work not done and it is the human's, which is why it carries the researcher persona and sits outside the container.

### Prompt 3 (Process): The transfer runs

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding four stage cards in a horizontal row connected by directional arrows. Each carries a numbered circular badge and two or three tiny bullet lines. Card one holds three small rising curves side by side. Card two holds the adapter chip with eight outcome tiles beside it. Card three holds the dashed fifteen-chip fan. Card four holds two pools of thumbnails balanced on a beam. Card one carries an amber "In progress" pill, card two carries an amber "In progress" pill, and cards three and four are drawn muted and grayed with "Not started" pills. A small legend row runs along the bottom edge.

Cast: three live training curves, the adapter chip, eight outcome tiles, the dashed fifteen-chip fan, two thumbnail pools on a balance beam (a pool of animal pairs against a size-matched mixed pool), all carrying their subject-lane meanings.

Flows: green solid arrow from card one to card two. Gray dashed arrows from card two onward to cards three and four, meaning downstream work not yet begun. Legend inside the image, bottom edge.

Text in the image: title banner "does the fix reach new pairs". Phase container title: "started, not finished". Card titles: "three live curves", "does one adapter transfer", "transfer as a rate", "the size-matched pool". Bullets: "wired, first short run owed", "held-out pairs compose", "later checkpoints unscored", "fifteen adapters", "one pair held out each", "kills the more-data excuse". Pills: "In progress", "Not started". Legend: "green: completed", "amber: in progress", "gray: not started".

Exclusions: no product logos, no stage other than the four listed, no red failure styling anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the second stage must read as in progress rather than completed. Its held-out pairs do compose, which is a real positive, but the later checkpoints are unscored and the go-ahead note is unwritten, so the number cannot yet be quoted. Nothing here has failed, so no red may appear.

### Prompt 4 (Process): Can the compose rate be trusted

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding four stage cards in a horizontal row connected by directional arrows. Every card is drawn muted and grayed with a "Not started" pill, and every card carries a numbered circular badge and two or three tiny bullet lines. Card one holds a stack of paper glyphs with a magnifier over them. Card two holds the researcher persona at a small screen showing one picture and four choice buttons. Card three holds two scorer lens glyphs side by side with a small comparison bracket between them. Card four holds a signpost with two branches pointing in different directions. Attached to the whole container's left edge, outside it, sits the scorer lens glyph from the subject lane with a red warning badge on it.

Cast: a stack of published papers with a magnifier (checking whether this hole is already known), the researcher persona at a labelling screen, two scorer lenses under comparison, a two-armed signpost (the promote-or-close decision), the scorer lens with its red warning badge.

Flows: gray dashed arrows left to right between all four cards, meaning nothing has begun. One red solid arrow from the warning-badged lens into card one, meaning this whole phase exists because of that measuring tool's known limitation. Legend inside the image, bottom edge.

Text in the image: title banner "how far above the truth". Phase container title: "not started". Card titles: "is this hole already known", "build the labelled set", "score the candidates", "promote or close". Bullets: "one literature verdict", "judgeable pairs first", "rule written before labels", "against the same labels", "keep the better detector", "wording only, or a real run". Pills: "Not started". Legend: "gray: not started", "red: the limitation".

Exclusions: no product logos, no stage other than the four listed, no green completed pills anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: every stage is grayed. Not one of this phase's four plans has begun, and its four review files carry no answered question, so any green here would be false. The red arrow must originate at the counting lens, because the phase exists to measure how far that measuring tool's rate sits above the truth.

### Prompt 5 (Process): The figures and the two checks before print

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding three stage cards in a horizontal row connected by directional arrows. Each carries a numbered circular badge and two or three tiny bullet lines. Card one holds the figure board with most of its framed cards bordered green, one bordered red, and two left as empty dashed frames; it carries an amber "In progress" pill. Card two holds the adapter chip beside a leaderboard glyph and is drawn muted and grayed with a "Not started" pill. Card three holds two upright barrier glyphs side by side with a stack of paper behind them, and is drawn muted and grayed with a "Not started" pill. A dashed red callout box hangs below card one, connected to its single red-bordered frame.

Cast: the figure board and its framed cards, the adapter chip, a leaderboard, two upright barriers (the two checks before print), a stack of published papers, all carrying their subject-lane meanings.

Flows: amber solid arrow from card one onward, meaning work underway continuing. Gray dashed arrows into cards two and three. Red dashed line from the red-bordered frame in card one down into the callout box. Legend inside the image, bottom edge.

Text in the image: title banner "what reaches the page". Phase container title: "most built, one undecided". Card titles: "the causal figures", "the transfer figures", "two checks before print". Bullets: "fourteen reserved places built", "captions capped by review", "waits on the series", "leaderboard and curve", "novelty of the timing", "the span sentence". Pills: "In progress", "Not started". Callout: "low-rank argument fails". Legend: "amber: in progress", "gray: not started", "red: needs a decision".

Exclusions: no product logos, no stage other than the three listed, no readable chart content inside the framed cards, no placeholder gibberish text, no watermark.
```

Faithfulness note: the causal figure card is in progress with most frames green, because fourteen reserved places on the register are built. The one red frame is the low-rank card, whose argument does not stand against what you would get by luck once that is worked out the right way, and it must be a single frame rather than the whole card, since the rest of the set is fine.

### Prompt 6 (Process): The manuscript

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one rounded soft-tinted phase container spanning the image, titled at its top, holding four stage cards in a horizontal row connected by directional arrows, with the researcher persona seated at the left end. Each card carries a numbered circular badge and two or three tiny bullet lines. Card one holds a document glyph with a green tick on its corner and a gray placeholder bar where its title would be, and carries an amber "In progress" pill. Cards two, three and four are drawn muted and grayed with "Not started" pills: card two holds a stack of section blocks, card three holds a page with empty framed places and small placeholder marks in the text, card four holds a finished page with a small padlock on it.

Cast: the researcher persona, the manuscript document, a stack of section blocks, a page with owed figure places, a finished anonymised page, all carrying their subject-lane meanings.

Flows: amber solid arrow from card one to card two, then gray dashed arrows onward through cards three and four. Legend inside the image, bottom edge.

Text in the image: title banner "writing it up". Phase container title: "build works, prose owed". Card titles: "make the template build", "title and section order", "results skeleton", "anonymise and submit". Bullets: "builds to a pdf", "title still a stub", "one claim per section", "placeholders, not prose", "every reserved place named", "page limit met". Pills: "In progress", "Not started". Legend: "amber: in progress", "gray: not started".

Exclusions: no product logos, no stage other than the four listed, no readable title text on any document, no green completed pills anywhere in this image, no placeholder gibberish text, no watermark.
```

Faithfulness note: the first stage is in progress and not completed. The template does build to a PDF and the figure-path rule is written, but the title is still the stock template's, so a completed pill would overstate it. No stage after it has begun.

### Process capstone: One route from measuring tools to submission

```
Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded dashed arrow lines, each color meaning exactly one kind of flow, with a small legend inside the image. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Official product logos only on components that ARE that product; every other component gets a clean generic glyph. Small cartoon figures or vehicles for external actors (users, clients). Clean sans-serif labels under every icon, short and lowercase-friendly. Generous spacing, no clutter, no watermark.

Scene: one wide landscape composition. A single thick continuous route line runs left to right across the image like a metro line, with circular icon nodes sitting on it. Segments of the route are enclosed in rounded soft-tinted phase containers, each titled above the route. From left: a container holding three nodes, then a container holding five nodes, then a container holding four nodes, then a container holding three nodes, then a container holding four nodes at the right end. The route is drawn solid green through the first two containers, solid amber through the third, and gray dashed through the last two. One branch leaves the route beneath the third container, loops down through a small separate container holding four gray nodes, and rejoins the mainline before the fourth container. The researcher persona stands at the route's left start. Small status chips sit beside the nodes. A small legend row runs along the bottom edge.

Cast: the route line (the order the work is actually done in), circular nodes carrying the subject-lane glyphs at reduced size (scorer lens, multiplier dial, window bracket, cache bench, adapter chip, figure board, the two barriers before print, manuscript document), the researcher persona, the looping branch (the phase that may or may not change a claim).

Flows: the route itself is the only flow, coloured by state: green where finished, amber where underway, gray dashed where not begun. The loop is drawn gray dashed and physically rejoins the mainline. Legend inside the image, bottom edge.

Text in the image: title banner "measuring tools to submission". Phase container titles: "measuring tools first", "the causal runs", "transfer and figures", "before print", "the manuscript". Loop container title: "can the rate be trusted". Node labels: "measuring scripts", "size measure fixed", "pair pool", "multiplier", "timing", "cached runs", "three ways", "inside the model", "does it transfer", "the figures", "two checks", "build", "sections", "submit". Legend: "green: completed", "amber: in progress", "gray: not started".

Exclusions: no product logos, no node that did not appear in process prompts 1 to 6, no node from those prompts left out, no red failure styling anywhere on the mainline, no placeholder gibberish text, no watermark.
```

Faithfulness note: the route must be green only through the measuring tools and the causal runs, because those are the two phases actually finished. The trust-the-rate phase must be drawn as a loop off the mainline rather than a node on it: it may change a claim or may change nothing, and putting it inline would say the paper waits on it, which it does not.
