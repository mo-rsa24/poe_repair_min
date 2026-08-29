# Diagram Prompt: The three metrics at a glance

## What to draw

Three separate panels, one per metric. Each panel shows:
- What the metric measures (in words)
- An example curve or visualization
- What high vs low looks like
- What it tells you

## Panel 1: Compose-rate

**What it measures:**
Fraction of generated images classified as "PoE-blend" (vs "Mono") by the scorer.

**Visual:**
- Y-axis: compose-rate (0 to 1, or 0–100%).
- X-axis: training epoch or time.
- **Good case:** A curve that climbs from 0 to 0.3–0.5 by epoch 1 (shows the fix is generating PoE-like outputs).
- **Bad case:** A flat line at 0 (fix never reached the model, or scoring is broken).

**Labels:**
- Top of panel: "✅ High: Fix is generating intended outputs"
- Bottom of panel: "⚠️ Low/zero: Fix didn't arrive, or scorer is broken"

**What it tells you:** Is the correction even being applied to the eval outputs? Dead-simple yes/no.

## Panel 2: Direction-cosine

**What it measures:**
Cosine similarity between the current run's correction and the pool-mean correction direction (Task D metric).

**Visual:**
- Y-axis: cosine value (-1 to +1, or visually 0–100% similarity).
- X-axis: training epoch or time.
- **Good case:** A curve starting at ~0.3–0.5 and climbing to 0.7–1.0 (the fix is learning in the ensemble direction).
- **Bad case:** A flat line near 0, or negative (fix is diverging, learning a different direction).

**Labels:**
- Top: "✅ High (0.7–1.0): Aligned with pool-mean, transferable"
- Bottom: "⚠️ Low (~0): Diverged, not following ensemble direction"

**What it tells you:** Is this run's fix moving in the same direction as the ensemble? High = transferable, Low = stuck in local direction.

## Panel 3: Fraction-of-distance-reached

**What it measures:**
How far the correction has moved toward the PoE→Mono target, as a fraction of the full distance (0–1).

**Visual:**
- Y-axis: fraction reached (0 to 1).
- X-axis: training epoch or time.
- **Typical case:** A curve that climbs to ~0.4 by mid-training, then plateaus (the known "40% plateau").

**Labels:**
- Middle: "Typical: Plateaus around 40% (known limitation)"
- Callout: "The plateau is expected. We're measuring whether it reaches its plateau or stalls earlier."

**What it tells you:** Is the correction reaching its "natural" magnitude, or is it stuck at a smaller value?

## Bottom panel (optional): Decision tree

**If all three are high/good:** "✅ Fix is working, transferable. Proceed."

**If compose-rate is zero, others don't matter:** "⚠️ Fix never arrived. Debug delivery."

**If compose-rate is high but direction-cosine is low:** "⚠️ Fix arrived but diverged. Debug the fix itself."

## Color and style

- Panel 1 (Compose-rate): teal or green (delivery signal).
- Panel 2 (Direction-cosine): purple or orange (direction/alignment).
- Panel 3 (Fraction-of-distance): gray or blue (magnitude).
- Good curves: bright colors. Bad curves: muted or red.

## Key message

Three independent signals that together tell you whether the fix is working and transferable. Each one can fail independently, so you need all three to diagnose.

## Audience

Someone reading the plan and wondering "What do these metrics actually mean?" They should see three curves and immediately understand: "Ah, one measures if it's happening, one measures if it's in the right direction, one measures how much it's happening."

## Save instructions

**Output file:** `diagrams/figures/three-metrics-explained.png`

After generating this diagram, save it to the path above. The plan file will reference it.
