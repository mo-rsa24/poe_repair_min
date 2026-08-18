# Diagram Prompt: Context diagram for plan step 9

## What to draw

A single diagram showing the training loop, the eval hook, and where the three metrics are computed.

## Layout and labels

**Main elements:**

1. **Training loop box** (center-left): shows epochs, one arrow pointing right.
2. **Eval hook box** (center-right, connected by arrow): shows "eval on held-out pairs".
3. **Three parallel processes below the eval box:**
   - Compose-scorer → `eval/compose_rate`
   - Direction-cosine calculator → `eval/direction_cosine`
   - Distance-reached calculator → `eval/frac_distance_reached`
4. **W&B logging box** (far right): shows the three curves being logged live to W&B.

**Arrows:**
- Training → Eval hook → (three processes in parallel) → W&B.

**Labels on boxes:**
- Training loop: "Epoch N / 1", "LoRA correction r_t"
- Eval hook: "Held-out eval set (cat×dog, eagle×hawk)"
- Compose-scorer: "Is output PoE-blend or Mono?"
- Direction-cosine: "Alignment to pool-mean r_t"
- Distance-reached: "Correction magnitude as % of PoE→Mono"
- W&B: "Live curves, not post-run"

## Color and style

- Training loop: neutral (gray or light blue).
- Eval hook: emphasis color (green or accent).
- Three processes: each a distinct color or icon (checkmark, alignment, bar chart).
- W&B logging: bright (e.g., teal for "live").

## Key message

Show that the three metrics are computed *inside the eval hook, live during training*, not after the run finishes. Emphasize parallelism: all three compute independently and log to W&B as separate curves.

## Audience

Someone who hasn't seen this repo in a while. They should look at this and immediately understand: "Oh, we're adding measurements to the eval hook so we see what's happening while training runs."
