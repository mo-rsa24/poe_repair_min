# Diagram Prompt: Before/after logging (the problem this plan solves)

## What to draw

A side-by-side comparison showing the difference between post-run analysis and live logging during a 15-run sweep.

## Left panel: "Before" (post-run only)

**Timeline:**
- Run 1: 6 hours → finished → (nothing visible during)
- Run 2: 6 hours → finished → (nothing visible during)
- Run 3: 6 hours → finished → (nothing visible during)
- **Then:** All three runs are analyzed. Curves appear. Dead runs are identified too late.

**Visual:**
- Three horizontal bars, each labeled "6h training", stacked vertically.
- At the end of all three: a question mark or "?" or a red X (confusion).
- Time axis: 18+ hours total.
- Callout: "You wasted 18 GPU hours before realizing the fix doesn't work."

## Right panel: "After" (live logging)

**Timeline:**
- Run 1: 1h into epoch 1 → compose_rate: 0.23 ✅, direction_cos: 0.87 ✅ → Continue run.
- Run 2: 1h into epoch 1 → compose_rate: 0.0 ⚠️ → Kill this run and runs 3, 4, 5. Debug now.
- (Runs 3–5 not started)

**Visual:**
- Two horizontal bars, each labeled "~1h epoch".
- Live curves appear *during* epoch 1: checkmarks (✅) for good curves, red X or ⚠️ for dead ones.
- Time axis: 1–2 hours total to diagnose.
- Callout: "You know within 1 hour. Stop wasting GPU."

## Color and style

- Before: grays, question marks, waste.
- After: greens (good) and reds (stop), emphasis on early termination.
- Curves in both panels should be simple line graphs.

## Key message

The 15-run sweep takes hours per run. Without live logging, you're blind. With live logging, you kill bad runs early and save GPU time.

## Audience

Someone about to run a big unattended sweep. They need to feel the urgency of this plan: "If I don't do this first, I'll waste a lot of GPU time flying blind."

## Save instructions

**Output file:** `diagrams/figures/before-after-logging.png`

After generating this diagram, save it to the path above. The plan file will reference it.
