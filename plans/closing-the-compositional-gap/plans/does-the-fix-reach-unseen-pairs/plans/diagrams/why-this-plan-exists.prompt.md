# Diagram Prompt: Why this plan exists (the cost of finding out late)

## What to draw

One picture contrasting the GPU hours burned before anyone can tell a sweep is broken, with and without live logging. This is the same argument as `before-after-logging.prompt.md`, but that one draws the mechanism (curves appearing during training rather than after); this one draws the bill.

## Left side: "Blind" (no live curves)

- A single wide bar spanning the full width, labelled "15 runs × 6h = 90 GPU hours".
- The whole bar is one flat gray: nothing is readable while it fills.
- Only at the far right, a small panel opens showing three curves, one of them flat at zero, with a red "the fix never arrived" label.
- Callout under the bar: "90 hours spent. Then you learn it was broken from the start."

## Right side: "Instrumented" (three live curves)

- A short bar, roughly one-fifteenth the width of the left one, labelled "1 run × ~2h".
- Three small curves drawn *inside* the bar, appearing as it fills: compose-rate, direction-cosine, fraction-of-distance-reached.
- A decision fork at the end of the short bar: green arrow "curves alive → run the other 14", red arrow "compose-rate flat → fix the delivery first".
- Callout under the bar: "2 hours spent. You know which branch you are on."

## The number to make felt

The two bars must be drawn to scale against each other: the left bar is 45× the area of the right one. The saving is the picture; no text should have to state the ratio.

## Color and style

Left: flat grays, one red accent at the very end. Right: greens for the live curves, one red for the kill branch. No gradients, no drop shadows, thin lines, plenty of white space.

## In-image labels, spelled exactly

"15 runs × 6h = 90 GPU hours", "1 run × ~2h", "compose-rate", "direction-cosine", "distance-reached", "curves alive → run the other 14", "compose-rate flat → fix the delivery first"

## Exclusions

No cluster hardware, no wandb logos, no cartoon people, no clock icons.

## Audience

Someone deciding whether this instrumentation plan is worth doing before the sweep. The picture answers that in one look.

## Save instructions

**Output file:** `diagrams/figures/why-this-plan-exists.png`
