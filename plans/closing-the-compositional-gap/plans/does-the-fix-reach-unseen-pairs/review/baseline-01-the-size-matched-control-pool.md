# Review: would any pool of the same size have worked?

**Nothing has run yet.** This file holds the question, written before the run. It judges
[../plans/baseline-01-the-size-matched-control-pool.md](../plans/baseline-01-the-size-matched-control-pool.md),
and its answer is what lets the transfer claim survive the obvious objection.

The objection: if training on fifteen animal pairs helps unseen animal pairs, maybe training on
*any* fifteen pairs would have helped just as much, and the win is really about how much data
there was.

## Words this file uses
- **Size-matched**: the control pool holds exactly as many pairs as the animals pool, so the
  amount of training signal is identical and only the content differs.
- **The mixed pool**: the same count of pairs, built from scenes, styles and objects instead of
  animals.
- **The identical held-out set**: both pools are tested on the same unseen animal pairs. Testing
  them on different sets would make the comparison meaningless.

## Run kind
**Produces a competitor.** It freezes the moment it lands: no tuning it up after our own number
is known, and no tuning it down either.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| one mixed-pool run, same pair count, same held-out set | Produces a competitor | | the contrast per held-out pair | not started |

## The pre-registered bar

- [ ] ⚠️ Does the animals pool beat the size-matched mixed pool on the identical held-out animal
      pairs? A win attributes the transfer to what is in the pool rather than to how much of it
      there is. A tie or a loss bounds the claim, and that boundary goes in the paper as a
      sentence rather than being left out.
