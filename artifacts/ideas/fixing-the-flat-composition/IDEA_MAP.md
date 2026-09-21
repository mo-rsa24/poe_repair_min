# Fixing the flat composition

**The idea.** The adapter puts two animals in the picture. The picture comes out flat. The two jobs
happen in the same first ten steps, so they cannot be separated by timing, and the question is
whether they can be separated at all, and by which lever: the adapter's reach, its loss, what it is
aimed at, or something after it.

**Mode.** Default, a claim to check.

## Claims

| # | Claim | Mark |
|---|---|---|
| 1 | Half the evidence is unreadable, and it is fixable | open ← current |
| 2 | The flatness is in the training pairs too, so it is the objective and not only generalisation | holds |
| 3 | Composition and fidelity are separable at all **(load-bearing)** | open |
| 4 | The adapter's reach is a real limit on what it can express | open |
| 5 | Repair after training is exhausted, so the fix has to be in training | open |

## What the ground check found

Contrast of the picture against its own target, measured on `pool43-all50` at step 40,000, as the
gap between the 1st and 99th percentile of brightness:

| Cell | Target | Adapter | Change |
|---|---|---|---|
| lion x meerkat, seed 1 (trained on) | 178 | 157 | -12% |
| typewriter x cactus, seed 1 (trained on) | 226 | 195 | -14% |
| cat x dog, seed 9 (held out) | 208 | 149 | -28% |
| elephant x penguin, seed 10 (held out) | 221 | 172 | -22% |

The flatness is in both. It is about half as bad on the pairs the adapter trained on. So the
objective does not fix it even where it had every chance, and it gets worse on unseen pairs.

## Dead ends already recorded

Five inference-time repairs have come back null: corrector steps on the adapter's own score,
handing the tail to the frozen model, CFG++ re-noising, the APG projection, and rescaling the
corrected prediction to the plain one's size.

## Held claims

None.
