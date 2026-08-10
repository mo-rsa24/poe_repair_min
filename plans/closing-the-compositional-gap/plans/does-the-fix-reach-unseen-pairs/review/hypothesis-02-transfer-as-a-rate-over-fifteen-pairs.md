# Review: does the fix reach pairs it never trained on?

**Nothing has run yet.** This file holds the questions, written before the runs. It judges
[../plans/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md](../plans/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md),
the main question of this whole claim, and its answers fill register slot **F8**.

It cannot start until the one-epoch smoke in
[instrument-02](instrument-02-three-live-curves-while-training.md) is green, because fifteen
unattended runs with a broken scorer produce fifteen convincing wrong answers.

## Words this file uses
- **Leave one pair out**: train fifteen adapters, each missing a different pair, and test each one
  on exactly the pair it never saw. Fifteen tests instead of one, so the answer is a rate rather
  than an anecdote.
- **The degradation curve**: compose rate plotted against how much of the pool was held back. Its
  shape is the finding: a gentle slope and a cliff are different papers.
- **At the floor**: a held-out pair that composes no better than the broken method. Two causes,
  and telling them apart is why the extra measures exist: the correction never arrived, or it
  arrived pointing the wrong way.

## Run kind
**Tests the claim.** A failure of the bar below closes the plan and opens one follow-on.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| fifteen runs, one per held-out pair | Tests the claim | | leaderboard plus degradation curve | not started |

## The pre-registered bar

- [ ] ⚠️ Do most held-out pairs compose on an adapter that never saw them?
      The claim is transfer, and fifteen held-out points turn it into a rate with a spread rather
      than a single yes.

## Written before the run, answered after

- [ ] ⚠️ What shape does the degradation curve take?
      Report it either way. A gentle decline says the fix generalises smoothly; a cliff says
      there is a minimum pool size, which is a different and still publishable claim.
- [ ] ⚠️ For every pair at the floor: did the correction fail to arrive, or arrive pointing
      wrong? Answered from the two direction measures, so a dead run is never misread as a
      transfer failure.
