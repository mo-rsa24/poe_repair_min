## Query (verbatim)

> could you /demonstrate "Hard (lowest compose-rate as unseen pairs): frog×toad (0.9375), eagle×hawk (0.9375), seal×walrus (0.9375)
> Easy (perfect transfer): leopard×jaguar (1.0), and, from the 11 with no direct read, ones like wolf×husky or lion×tiger as animal-family-diverse easy cases" and create a (e.g qualitative /pair-figure) of each of them in a single figure that shows mono|PoE|LoRA as 3 columns of a single row then we can answer the question for the next nudge in /plain-speak --drip

## Big picture

This sits inside the `animals-compose-transfer` scope: fixing PoE (Product of Experts) diffusion composition, where asking for "a frog and a toad" produces one blended creature instead of two distinct animals. A rank-8 LoRA is trained on 11 animal pairs (Phase 1, `plans/animals-compose-transfer/plans/03a-phase1-pooled.md`) and tested on pairs it never saw, to check whether the fix generalizes rather than memorizing specific pairs.

The scope was deciding how big the next experiment (`plans/03-run-A-leave-one-pair-out.md`, the 15-pair leave-one-pair-out sweep) needs to be: all 11 training pairs, a small 2-4 pair sample, or a "middle path" of 4-6 pairs chosen to span easy and hard cases. This demonstration is the qualitative evidence for that middle path: does the model actually look different on hard vs easy pairs, or does the compose-rate number hide an image that looks the same either way?

## What was asked, and what exists

The query named 6 specific pairs and a specific layout (Mono | PoE | LoRA, one row per pair). All 6 pairs already had real rendered triptych images on disk from Phase 1 training, at exactly step 60000 (the best-scored checkpoint), 8 seeds each. No new GPU render was needed; this script only composes existing images, calling no model code.

## Method

`demo.py` reads `compose_triptych` output PNGs already produced during Phase 1 training (`_inline_sampling.py::compose_triptych`, wired into `train_pooled.py`) and stacks 6 of them into one figure, using the lowest-numbered available seed per pair (seed01 for in_in/training pairs, seed09 for out_out/held-out pairs — the first seed in the trainer's own render order, not cherry-picked). Labels each row with the pair name and its Phase 1 held-out compose-rate from `compose_rate.json`, or "training pair, not held out" for wolf×husky and lion×tiger (which were only ever trained on, never held out, so no compose-rate exists for them).

## Result

The figure (`hard_vs_easy_transfer.png`) shows, for every one of the 6 rows, the PoE (middle) column producing one blended creature (a single frog/toad face, a single eagle/hawk head, a single seal/walrus face), while the LoRA (right) column recovers two visually distinct animal figures side by side, matching the Mono (left) target's composition. Instance separation holds across all 6 rows, including the three hardest pairs by compose-rate (frog×toad 0.9375, eagle×hawk 0.9375, seal×walrus 0.9375), the easy pair (leopard×jaguar 1.0000), and the two untested training pairs (wolf×husky, lion×tiger).

But one row does NOT fully confirm the number: **seal×walrus**. Two distinct animal instances are present (so the instance-count scorer correctly calls it a compose, 0.9375), but the second figure has no tusks, the walrus's single most identifying feature, and reads visually as two seals rather than a seal and a walrus. Frog×toad and eagle×hawk show minor artifacts (a double-body overlap on the top frog; stiffer poses than the Mono target) but both animals are still clearly identifiable as the right species. Seal×walrus is qualitatively worse than its 0.9375 score suggests, because that score only measures "are there two things here," not "is each thing the right animal."

Aggregate context at step 60000 (from `compose_rate.json`, not re-derived here): in-distribution compose-rate 0.960 (n=176), held-out compose-rate 0.961 (n=128) — the two nearly match, meaning transfer to unseen pairs looks as strong as performance on pairs the model trained on. That aggregate is accurate for "separates into two instances," but the seal×walrus row shows it is not automatically accurate for "each instance is the correct animal."

## Limits

- **One seed per pair, not an aggregate.** Each row shows a single seed (the first cached one), not all 8. The compose-rate label is the true per-pair aggregate over all seeds; the image itself is one draw from that distribution. No per-seed compose-rate exists to confirm the shown seed is "typical" — this is stated, not verified.
- **Only 6 of the 15 rotation pairs shown.** This does not cover the full leave-one-pair-out set; it is illustrative evidence for the hard-vs-easy question, not a substitute for plan 03's own sweep.
- **wolf×husky / lion×tiger have no held-out score.** They were included as "easy, family-diverse" comparisons on the assumption that familiar large-mammal pairs are easy, but this is untested — Phase 1 never held them out. The figure shows they render as two distinct animals when trained on, which is a different (weaker) claim than "they'd transfer well as held-out pairs."
- **Step 60000, not the final checkpoint.** Training reached step 100000; steps 70000-100000 are still unscored (open task on plan 03a). The step 60000 read is the best-available data, not the final one.

## What this means in plain terms

The number (compose-rate 0.94-1.0) said the fix works even on the hardest pairs. Looking at the actual images mostly confirms that: five of the six pairs show two clearly separate, correctly-identified animals, not one merged creature. But the sixth, seal×walrus, shows the number can be right about "two things are here" while being silent about "are they the right two things." The walrus lost its tusks and came out looking like a second seal. The score still called this a near-perfect compose.

## Real-life scenario

Without this check, a compose-rate near 1.0 would have been trusted at face value for every pair, including seal×walrus. Looking at the images directly caught something the score could not: the automated scorer counts distinct animal-shaped regions, and two seal-shaped regions still count as "compose," even though one of them was supposed to be a walrus. If plan 03's pair selection used compose-rate alone, seal×walrus would look exactly as safe a choice as leopard×jaguar. This is why the pattern in `dataviz`/`evidence-ladder` pairs every number with an actual example: a good number and a wrong image can coexist, and only looking at both catches it.

## Corroboration prompts

- **Claude:** "In a PoE (Product of Experts) diffusion composition setup, is a vision-based instance-count scorer (e.g. GroundingDINO with a distinct-instance threshold) a reliable proxy for 'the image shows two visually distinct subjects,' or are there known failure modes where instance-count and visual coherence diverge?"
- **Google:** `"instance count" OR "object detection" metric diffusion image composition failure mode visual coherence`

## Reproduce

```
Reproduce
  python docs/evidence/animals-compose-hard-vs-easy/demo.py       # the terminal report above
  python docs/evidence/animals-compose-hard-vs-easy/test_demo.py  # the checks can fail (6/6 passed)
  figure: docs/evidence/animals-compose-hard-vs-easy/hard_vs_easy_transfer.png
  record: docs/evidence/animals-compose-hard-vs-easy/QUERY.md
```
