# Showcase the trained adapter: illustrated map

The scope's pictures. Everything here is marked [planned] and drawn hedged: nothing is built yet,
and the map says so truthfully.

**9 prompts · 2 rendered · 1 rendering · 6 waiting**

## Abstraction chain

1. [Closing the compositional gap](../diagram-prompts.md): the parent map; this scope inherits
   its art direction, palette, glyphs and reading axes.
2. [Showcase the trained adapter](diagram-prompts.md): subject capstone "the adapter on trial", process capstone "the scope closed".

## Art direction

**Vivid circuit**, inherited from the parent map, verbatim style paragraph at the head of every
prompt. Palette inherited: blue the uncorrected PoE path, amber the correction and anything
carrying it, green a bar met, red a null or a failure; a control is a hollow amber outline.
Glyphs inherited: cache drum, correction coil, step strip, dose dial, window bracket, adapter
chip, scorer lens, register board, outcome tile, researcher persona.

## Subject lane

### Prompt 1 (Subject): the plateau read [planned]

🖼️ rendered 2026-08-30 `diagrams/the-plateau-read.png` (one revision: the curves arrived solid, which claimed a measured plateau)
Save as: `diagrams/the-plateau-read.png`

> Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded directional arrows, each color meaning exactly one kind of traffic, with a small legend inside the image. Solid lines are actual transfers; dashed lines are reserved for control, acknowledgement, retry, and optional paths. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Clean sans-serif labels under every icon. Generous spacing, no clutter, no watermark.
>
> A monitor showing two rising-then-flat curves, drawn hedged in dashed outline: an amber curve labeled "fraction of distance reached" flattening at a dashed line labeled "0.4 plateau", and a gray loss curve beside it split into three bands labeled "early", "commit", "late". A researcher persona reads the monitor; two dashed exit arrows leave it, one green labeled "still rising: train longer", one red labeled "flat: ceiling, spend nothing". Legend inside the image.

Faithfulness note: Both curves are dashed: no plateau has been read yet. The 0.4 line is the pre-registered bar, not a measurement, and the two exits are equally weighted because either answer is a real result.

### Prompt 2 (Subject): the null-input probe [planned]

🖼️ rendered 2026-08-30 `diagrams/the-null-input-probe.png` (no revision needed)
Save as: `diagrams/the-null-input-probe.png`

> Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded directional arrows, each color meaning exactly one kind of traffic, with a small legend inside the image. Solid lines are actual transfers; dashed lines are reserved for control, acknowledgement, retry, and optional paths. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Clean sans-serif labels under every icon. Generous spacing, no clutter, no watermark.
>
> Two expert cards both reading "a dog" feed blue arrows into a PoE junction; the junction's outcome tile shows one dog. An adapter chip attaches from below with an amber arrow labeled "r-hat, should be near zero", drawn hedged. Two dashed outcome tiles to the right: green "still one dog: the rule held" and red "two dogs: a plurality prior, the caption dies". A small window bracket labeled "steps 0 to 10" sits on the step strip beneath.

Faithfulness note: Both expert cards must read the same word, and the amber arrow says 'should be near zero' rather than 'is': the probe has not run. The red exit is the one that kills the plurality caption, so it is drawn no smaller than the green.

### Prompt 3 (Subject): the dose dial moves to the adapter [planned]

⚙️ rendering
Save as: `diagrams/the-dose-dial-moves-to-the-adapter.png`

> Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded directional arrows, each color meaning exactly one kind of traffic, with a small legend inside the image. Solid lines are actual transfers; dashed lines are reserved for control, acknowledgement, retry, and optional paths. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Clean sans-serif labels under every icon. Generous spacing, no clutter, no watermark.
>
> Left panel faded: a cache drum feeding a dose dial and a correction coil into the step strip, labeled "the oracle sweep, measured". Right panel hedged and prominent: the adapter chip feeding the same dose dial and step strip, labeled "the same sweep, on r-hat". Four hollow amber control outlines queue beside the dial. A register board at the far right holds one empty card labeled "AUC, to compare against 0.387 vs 0.023".

Faithfulness note: The left panel is faded because that sweep is measured and done; the right is hedged because it has not run. The two AUC numbers on the register card, 0.387 for the real correction against 0.023 for the random one, are the oracle result this sweep is compared against, and belong to the left panel's work, never to the right.

### Prompt 4 (Subject): the transfer matrix [planned]

⏳ not rendered
Save as: `diagrams/the-transfer-matrix.png`

> Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded directional arrows, each color meaning exactly one kind of traffic, with a small legend inside the image. Solid lines are actual transfers; dashed lines are reserved for control, acknowledgement, retry, and optional paths. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Clean sans-serif labels under every icon. Generous spacing, no clutter, no watermark.
>
> A grid of outcome tiles, rows labeled "trained on group", columns labeled "evaluated on disjoint pair", drawn hedged. A scorer lens hovers over one tile. Green and red corner chips mark composed and blended cells. A side caption card: "group-pooled, concept-disjoint: the tier a reviewer believes".

Faithfulness note: Every tile is hedged: no cell is scored. Rows are training groups and columns are disjoint evaluation pairs, so no cell on the diagonal may be drawn as a claim.

### Prompt 5 (Subject): the grid and the meter [planned]

⏳ not rendered
Save as: `diagrams/the-grid-and-the-meter.png`

> Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded directional arrows, each color meaning exactly one kind of traffic, with a small legend inside the image. Solid lines are actual transfers; dashed lines are reserved for control, acknowledgement, retry, and optional paths. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Clean sans-serif labels under every icon. Generous spacing, no clutter, no watermark.
>
> A two-by-three grid of platforms, columns labeled "100k steps" and "200k steps", rows labeled "rank 8", "rank 16", "rank 32", drawn hedged. The rank-8-at-100k platform is solid (it exists); an amber arrow labeled "A: resume" runs right along the rank-8 row; two amber arrows labeled "B: fresh runs" rise into the rank-16 and rank-32 platforms of the 100k column; the 200k platforms of ranks 16 and 32 are faded with a dashed arrow labeled "optional later". A small meter panel attaches to every solid platform: four miniature gauges labeled "cosine to the true correction", "embedding drift", "spectral share (diagnostic)", "where in the 50 steps it acts". A caption card: "claims read one line of the grid, never a diagonal".

Faithfulness note: Exactly one platform is solid, rank 8 at 100k, because that is the only trained artifact that exists. The caption is load-bearing: a claim reads one row or one column, never a diagonal across both.

### Prompt 6 (Subject): the follower and its intervention [planned]

⏳ not rendered
Save as: `diagrams/the-follower-and-its-intervention.png`

> Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded directional arrows, each color meaning exactly one kind of traffic, with a small legend inside the image. Solid lines are actual transfers; dashed lines are reserved for control, acknowledgement, retry, and optional paths. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Clean sans-serif labels under every icon. Generous spacing, no clutter, no watermark.
>
> A checkpoint shelf (three highlighted slots labeled "first", "middle", "final") feeding a small watcher glyph on its own platform labeled "off the training device", drawn hedged. Two amber read-out arrows leave the watcher: one into a bottleneck lens labeled "the correction in the model's own space", one into a fan of thin arrows labeled "highest-gain directions". Both converge on a syringe-like injector over the step strip, its window bracket pinned at steps 0 to 10, ending at a scorer lens with green and red outcome chips. A red caption chip: "no intervention, no main-text figure".

Faithfulness note: The watcher sits on its own platform, off the training device, which is a real constraint and not decoration. The window bracket is pinned at steps 0 to 10, the measured injection window, and the red caption states the condition under which none of this reaches the main text.

### Subject capstone: the adapter on trial [planned]

⏳ not rendered
Save as: `diagrams/subject-capstone-the-adapter-on-trial.png`

> Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded directional arrows, each color meaning exactly one kind of traffic, with a small legend inside the image. Solid lines are actual transfers; dashed lines are reserved for control, acknowledgement, retry, and optional paths. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Clean sans-serif labels under every icon. Generous spacing, no clutter, no watermark.
>
> One canvas composing six stations around the adapter chip at center: the monitor with the plateau curves (top left), the two agreeing expert cards with their null-probe outcome tiles (top center), the dose dial with its hollow controls (top right), the two-by-three length-and-rank grid with its miniature meter panels (bottom left), the checkpoint watcher with its injector and scorer lens (bottom center), and the transfer grid under the scorer lens (bottom right). Blue arrows carry the uncorrected path, amber arrows everything the adapter emits, all stations drawn hedged with dashed platforms. Legend inside the image; title banner "the adapter on trial".

Faithfulness note: Six stations, each recognisably the same glyph as its own prompt, all hedged. The adapter chip is the only thing at centre; nothing in this image asserts a result.

## Process lane

Version 02, regenerated by populate-plans from the thirteen plan files; snapshot in
[diagrams/process-versions/02-2026-08-29.md](diagrams/process-versions/02-2026-08-29.md).
Gates and failure paths come from the plans' engagement gates, none invented.

### Prompt 1 (Process): thirteen stations in five phases [planned]

⏳ not rendered
Save as: `diagrams/process-thirteen-stations-in-five-phases.png`

> Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded directional arrows, each color meaning exactly one kind of traffic, with a small legend inside the image. Solid lines are actual transfers; dashed lines are reserved for control, acknowledgement, retry, and optional paths. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Clean sans-serif labels under every icon. Generous spacing, no clutter, no watermark.
>
> One route left to right through five rounded phase panels, thirteen numbered stations, status chips all "Not started". Panel "read first" holds station 1 "the plateau read" (monitor glyph; exit chips green "waypoint" and red "ceiling", both continuing, chip caption "re-scopes A and B, gates nothing"). Panel "instruments" holds station 2 "the tracking set extended" (four miniature gauges; red gate chip "a curve missing from the smoke run: no launches") and station 3 "the scorer revalidated off animals" (scorer lens over non-animal tiles; chip "gates tier-three captions only"). Panel "in-session probes" holds station 4 "the null probe" (red gate "identity preflight fails: stop"), station 5 "the lambda-and-window sweep" (red gate "lambda-0 canary breaks mode: fix the harness"), station 6 "the dose sweep on the adapter" (red gate "a control off the floor: instrument first"), and station 7 "the counted joint-prompt figure" (three bars with a strip beside). Panel "the training runs" holds station 8 "A: resume to 200k" (red gate "dry resume fails: no sbatch"), station 9 "B: rank 16 and 32" (red gate "device carrying a foreign process: do not start"), and station 10 "the follower" (watcher glyph; red chip "intervention moves nothing: figures stay out of main text"). Panel "figures and close" holds station 11 "close F8a and the oracle panel" (chip "reconcile with the transfer-figures plan first"), station 12 "the transfer matrix" (red chip "shared token found: cell voided"), and station 13 "assembly" (register board; gate chip "no sidecar, no ship"). The route ends at a green banner "the showcase set, under one standard".

Faithfulness note: Thirteen stations, five phases, every status chip reading 'Not started', which is true of the whole scope. Each red gate chip quotes a real engagement gate from the plan of that number; none is invented.

### Process capstone: the scope closed [planned]

⏳ not rendered
Save as: `diagrams/process-capstone-the-scope-closed.png`

> Style: premium system-design infographic. Clean white or very light gray background. Glossy semi-3D icons with soft drop shadows, one icon per component, sitting on subtle rounded platforms. Flows drawn as vivid color-coded directional arrows, each color meaning exactly one kind of traffic, with a small legend inside the image. Solid lines are actual transfers; dashed lines are reserved for control, acknowledgement, retry, and optional paths. Related components grouped inside rounded soft-tinted panels with a short title on the panel. A bold title banner across the top. Clean sans-serif labels under every icon. Generous spacing, no clutter, no watermark.
>
> The thirteen stations shrunk along the bottom edge, all green; above them the recall-gallery page glyph labeled "/recap-plan-tree, URL recorded" and the register board holding its filled cards. Title banner "showcase the trained adapter: closed".

Faithfulness note: This is the only image in the map drawn green, and it depicts a future state: the scope closed. It must be readable as an end state, never as a status report.
