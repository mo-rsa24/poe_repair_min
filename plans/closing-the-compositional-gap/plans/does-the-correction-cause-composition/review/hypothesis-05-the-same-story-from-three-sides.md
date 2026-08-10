# Review: does the same story show up from three other angles?

**Nothing has run yet.** This file holds the questions, written before the runs. It judges
[../plans/hypothesis-05-the-same-story-from-three-sides.md](../plans/hypothesis-05-the-same-story-from-three-sides.md).
One of its answers fills register slot **F5**; the rest support the central claim rather than
carrying a figure of their own.

Three of the five questions can be answered today from cached data with no GPU. The two that use
the strength-sweep's pictures wait on that sweep's re-score.

## Words this file uses
- **The blend region**: the part of the image-similarity map where one fused animal lands, as
  opposed to where two separate animals land.
- **Additivity gap**: how much the joined prompt "a cat and a dog" differs, in text space, from
  simply adding "a cat" and "a dog". If that gap predicts which pairs need a big correction, the
  difficulty is visible in the text before any picture is made.
- **Binding direction**: whatever is left of the joined prompt after the two separate prompts are
  subtracted. The question is whether every pair leaves the same thing behind.
- **Caption readback**: showing a generated picture to a captioner and seeing which description
  it picks. A blended animal should read back as a blend, a real composition as two animals.

## Run kind
**Tests the claim.** These corroborate the central result from angles that share none of its
machinery, so agreement means something. None of them may overturn it on its own; a disagreement
is a diagnosis, not a verdict.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| three cache-only probes (additivity gap, binding direction, quality check) | Tests the claim | | | can run now, not started |
| two probes over the strength-sweep pictures (map slide, caption readback) | Tests the claim | | | waits on the re-score |

## Written before the run, answered after

- [ ] ⚠️ Does the additivity gap predict how big a correction each pair needs?
      Report the correlation either way. A null is a real finding: it would mean the binding
      information lives in how the model processes the prompt jointly, not in the prompt's
      embedding.
- [ ] ⚠️ Do all pairs leave the same thing behind when the two separate prompts are subtracted?
      One shared direction across pairs would be evidence the correction is a general mechanism
      rather than per-pair bookkeeping.
- [ ] ⚠️ Is a blended animal wrong content rather than a poor-quality picture?
      Measure picture quality on the cached broken and working outputs and expect no gap. This is
      the check that removes "the correction just improves image quality" as an objection, and it
      is the cheapest of the five.
- [ ] ⚠️ Do the pictures slide out of the blend region as the strength rises, while a same-sized
      random push does not? Feeds slot **F5**.
- [ ] ⚠️ Does the caption readback cross over from a blend description to a two-animal
      description as the strength rises?
