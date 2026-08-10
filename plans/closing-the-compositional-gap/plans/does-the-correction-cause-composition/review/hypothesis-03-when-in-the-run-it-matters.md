# 🔬 Review: the timing pair

Verdicts for [../plans/hypothesis-03-when-in-the-run-it-matters.md](../plans/hypothesis-03-when-in-the-run-it-matters.md). Questions written before
the grids run. Answers are owed by the runs the design plan launches.

## Run kind
**Tests the claim** (Goal 2: when in the denoising run the correction matters).

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| (none yet) | | | `/datasets/.../interaction_term/window/{w1,w2}/` | not started |

## The pre-registered bar

- [ ] ⚠️ Does gating the correction to a sliding window produce a compose-rate curve with an
      interior peak? A peak means there is a time the correction is needed; flat means timing
      does not matter, and flat is a finding, not a failure. Feeds slot F4.

## Written before the run, answered after

- [ ] ⚠️ Does the all-off window reproduce plain PoE exactly? The leak check: if gating
      everything off does not equal plain PoE, the harness contaminates the base path and no
      curve from it may be read.
- [ ] ⚠️ Do the W1 and W2 peaks coincide? W2 gates only the injected correction (conditioning
      always on); W1 gates the conditioning itself. Same peak means one mechanism; different
      peaks mean conditioning and correction are needed at different times. Either answer goes
      in the caption.
- [ ] ⚠️ Does the window peak sit at the fork elbow (step 16, from the cache analyses)? The two
      timing reads corroborate or they disagree, and a disagreement is diagnosed before either
      is printed.
