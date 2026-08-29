# Diagram Prompt: Why this plan exists (the diagnostic dilemma)

## What to draw

An illustration showing the **information asymmetry problem** this plan solves: running a 15-run sweep without live logging leaves you blind until it's done.

## Scene 1 (left): The problem — unattended sweep, post-run analysis

**Visual setup:**
- A timeline showing 15 runs stacked vertically, each labeled "6h training".
- A clock/timer showing 90 hours passing (15 × 6).
- At the end of all 15 runs: a question mark or a person scratching their head.
- A thought bubble from the person: "Did this work? I have no idea until now."

**What's happening:**
- The user starts the 15-run sweep at 9am Monday.
- Each run takes 6 hours. No metrics are visible during training.
- By Thursday morning, all 15 runs finish.
- Only then can the user analyze whether the fix transferred.
- If 10 of the 15 runs failed silently, the user has wasted 60 GPU hours.

**Emotional tone:**
- Waiting, uncertainty, waste.

## Scene 2 (right): The solution — live logging, early termination

**Visual setup:**
- A timeline showing 5–6 runs, each ~1 hour into epoch 1.
- During run 1 (epoch 1): compose-rate and direction-cosine curves appear on screen. Both look good (✅).
- During run 2 (epoch 1): compose-rate is flat at 0 (⚠️ dead). A hand reaches out and **cancels runs 3–15**.
- A clock/timer showing ~2 hours elapsed.
- A thought bubble: "Killed the bad runs. Moving on."

**What's happening:**
- The user starts the sweep at 9am Monday.
- By 10am, live curves appear on W&B.
- By 11am, the user sees run 2 is dead and cancels the rest.
- Total time invested: 2 hours. Runs 3–15 were never started.

**Emotional tone:**
- Fast, decisive, controlled.

## Comparison label

- Left side: "Without live logging: 90 GPU hours wasted, Thursday decision."
- Right side: "With live logging: 2 hours invested, Monday decision, stop early."

## Color and style

- Left side: grays, clocks spinning, red X marks (waste).
- Right side: greens (good curves), quick checkmark (decision made), saved time.
- Curves: show actual line graphs in the right side (compose-rate climbing, direction-cosine high).
- The "cancel" action: bold red button or gesture.

## Key message

This plan's job is to move from "find out too late" to "know early enough to act." The difference is the three live curves. Without them, you're running blind. With them, you can kill bad runs before they waste GPU.

## Audience

A researcher about to run a large unattended sweep. They need to understand: "If I don't set up live logging first, I'm gambling with GPU time."

## Save instructions

**Output file:** `diagrams/figures/why-this-plan-exists.png`

After generating this diagram, save it to the path above. The plan file will reference it in the "Why this plan exists" section.
