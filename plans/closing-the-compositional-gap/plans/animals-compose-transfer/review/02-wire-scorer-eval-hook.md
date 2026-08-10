# 🔬 Review: the three live curves

Verdicts for [../plans/02-wire-scorer-eval-hook.md](../plans/02-wire-scorer-eval-hook.md).

## Run kind
**Tests the claim** (instrument wiring; its smoke is the unattended-safety gate for the sweep).

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| (the 1-epoch smoke) | Tests the claim | not launched | three separate W&B series | not started |

## The questions

- [x] ✅ Is the scorer wired into the eval loop? Yes: per-held-out-pair compose-rate logged live
      (`compose_rate.json` from the 03a run proves the path works end to end).
- [x] ✅ Are the two direction metrics wired? Yes, in code and import-clean:
      `_inline_sampling.py::direction_metrics` logs `eval/direction_cosine/...` and
      `eval/frac_distance_reached/...` per cell plus means, reusing `task_d_bridge` maths.
- [ ] ⚠️ Do all three metrics land as separate live W&B curves on a 1-epoch smoke? THE gate:
      the 15-run sweep may not start until this is green, because a wrong in-loop scorer makes a
      fan-out produce garbage that looks like a real null.
