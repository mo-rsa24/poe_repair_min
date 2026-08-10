# Review: can a training run be read while it is still running?

**Two thirds answered; the last third is the gate the fifteen-run sweep waits on.** This file
judges [../plans/instrument-02-three-live-curves-while-training.md](../plans/instrument-02-three-live-curves-while-training.md).

Why it matters more than its size suggests: the sweep trains fifteen adapters unattended. If the
scorer inside the training loop is wrong, all fifteen produce plausible numbers that mean nothing,
and nobody finds out for days. This instrument is what makes that sweep safe to leave alone.

## Words this file uses
- **The three curves**, logged live per evaluation so a run can be read before it finishes:
  - **compose rate**: how often two separate animals appear.
  - **direction agreement**: whether this run's correction points the same way as the pool's
    average correction.
  - **distance reached**: how far toward the target the correction actually moved the prediction.
- **Why three and not one**: a pair sitting at the floor has two very different causes. Either the
  correction never arrived (distance reached stays flat) or it arrived pointing the wrong way
  (direction agreement is low). One curve cannot tell those apart; the paper needs to.

## Run kind
**Not a run: an instrument.** Judged by whether its checks could fail, not by what they found.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| the one-epoch smoke | Instrument | not launched | three separate W&B series | not started |

## The questions

- [x] ✅ Is the scorer wired into the evaluation loop?
      Yes, and proven end to end: the pooled run wrote a per-held-out-pair compose rate
      (`compose_rate.json`), which it could not have done unless the whole path worked.
- [x] ✅ Are the two direction measures wired?
      Yes, in code and importing cleanly. `_inline_sampling.py::direction_metrics` logs both per
      cell plus their means, reusing the existing maths rather than redefining it.
- [ ] ⚠️ Do all three land as three separate live curves on a one-epoch smoke run?
      **This is the gate.** The fifteen-run sweep may not start until this is green, for the
      reason above: a wrong in-loop scorer turns an unattended fan-out into fifteen runs of
      convincing nonsense.
