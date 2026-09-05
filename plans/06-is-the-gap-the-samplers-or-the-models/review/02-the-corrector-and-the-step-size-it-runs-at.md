# 🔬 Review: does the corrector change nothing when switched off, and what step size does it run at?

**The corrector is built, both leak checks pass byte for byte, and the pre-registered search
rule picks `c = 30`. The pick is provisional: the rendered samples show the chain wrecking the
image from `c = 3` upward while both numeric divergence guards pass, and the composing-pair
control at step 26 rejects `c = 30`. The grid is being repeated at `c = 3` and `c = 0.3` to find
the largest step size that survives that control.** Every question below was written before any corrector existed. This file
judges [the corrector and step-size design](../plans/tools/02-the-corrector-and-the-step-size-it-runs-at.md).
Steps 26, 27 and 29 all measure something on the path this composer produces, so a failure recorded
here blocks all three rather than bounding any of them.

## Recommended prompt (when the run lands)

```
/analyze-run the step-size search, five c values at k=20 on a_cat__x__a_dog seed 9
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tools/02-the-corrector-and-the-step-size-it-runs-at.md) | the composer, the two leak checks, the search and its bounds |
| **this file** | **the verdict: not yet run** |
| [the step 26 verdict](03-what-is-left-once-the-chain-settles.md) | what this composer is built to measure |
| [the questions this file was split from](../source/the-questions-pre-registered-against-it.md) | the whole pre-registered set, written before the scope existed |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [The step-size search](#the-step-size-search)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The corrector count, `k`**: how many Langevin steps run at each of the 50 noise levels before
  the reverse step. `k=0` is plain product-of-experts.
- **The step-size multiplier, `c`**: the corrector's one free parameter, where the step size at
  noise level `t` is `δ_t = c·β_t`.
- **The relative displacement**: how far the chain actually moved,
  `‖x_t^(k) - x_t^(0)‖/‖x_t^(0)‖`. It is the only thing that tells a corrector doing nothing apart
  from a corrector that was never switched on.
- **A leak check**: a run where the corrector is switched off in some way, whose output must be
  byte-identical to plain product-of-experts. A single differing byte means the composer changes
  something it should not.

> A Langevin corrector is a small repeated random walk that nudges the latent along the score and
> adds a little noise each time. Run for long enough at a fixed noise level it forgets where it
> started and settles wherever the score says the probability actually is.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Builds a measuring tool.** Missing the threshold blocks every plan downstream of it, because steps
26, 27 and 29 all run this composer. It does not bound a claim; it decides whether any claim can be
measured.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Leak check, `k=0` byte-identical to plain product-of-experts | Builds a measuring tool | 2026-09-05 16:46, in-session on mscluster85 device 0 (RTX 3090), pid 255085, commit 72b8826 | 3 renders (23.7 s each) | `corrector/logs/leak_checks.log` | ✅ passed: `k=0` with a window past the last step is byte-identical to `k=0` with no window, and both are byte-identical to `run_cfg_poe` itself |
| Leak check, `k=200` with the window past the last step | Builds a measuring tool | 2026-09-05 16:47, same node, device and pid | 2 renders | `corrector/logs/leak_checks.log` | ✅ passed: byte-identical to the `k=0` reference, and the corrector fired at 0 of 50 levels |
| Step-size search, `c ∈ {0.01, 0.035, 0.1, 0.3, 1.0}` at `k=20` | Builds a measuring tool | 2026-09-05 16:52, `nohup` on mscluster108 device 1 (Quadro RTX 8000), pid 293216, `co3`, commit 72b8826 | ~118 plain-render equivalents (each `c` costs 50 levels × 71 UNet evaluations: 60 for the chain, 5 for the settled read, 6 for the three inner probes) | `corrector/step_size_search.json`, `corrector/logs/step_size_search.log`, and [the search table](#the-step-size-search) | ✅ done 18:23; every `c` usable except 0.01, so the pick sat at the top edge and the range was widened twice |
| Step-size search widened, `c ∈ {3, 10}` then `{30, 100, 300}` at `k=20` | Builds a measuring tool | 2026-09-05 18:24 and 18:42, same node and device, pids 297587 and 298292 | ~118 plain-render equivalents (8.7 min per `c` on the RTX 8000 once its neighbour's load dropped) | `corrector/logs/step_size_search_wider.log`, `_wider2.log`; rows appended to the same JSON | ✅ done: 3, 10 and 30 usable, 100 diverged (max latent norm 3.09× its start), 300 recorded for the table |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ✅ **Is there a step size at which the chain provably moves without diverging?** Yes, eight
      of them: every `c` from 0.035 to 30 has a median relative displacement at or above 0.05
      (0.091 at 0.035, rising to 1.23 at 30), a maximum latent norm within 1.5× its start (1.001
      to 1.11), and a residual ratio that does not rise with the inner count (the probed sequence
      0, 5, 10, 20 increases at 2% to 18% of the 50 levels, with a median start-to-end change
      between −0.006 and +0.008). `c = 0.01` stalls at 0.049. `c = 100` diverges: the latent
      norm reaches 3.09× its start at the noisiest levels, where `δ_t = 100·β_t = 1.17` is past the
      point the Euler step contracts. The pick is `c = 30`, the largest usable value, per the
      rule below. Pass if at
      least one `c` has a median relative displacement at or above `MIN_CHAIN_DISPLACEMENT = 0.05`
      while its maximum latent norm stays within 1.5× the uncorrected latent's and its ratio does
      not rise monotonically with `k`; the pick is the largest such `c`. A search where every `c`
      fails is a finding and stops the scope here. It says this corrector cannot be run stably on
      this model at these settings. The bound lives in source as a module-level constant, so moving
      it after the answer is visible shows up in a diff.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#the-step-size-search) ➡️

- [x] ✅ With `k=0`, does the composer reproduce plain product-of-experts byte-identical?
      Yes, on a_cat×a_dog seed 9 at 50 steps. The reference is the composer at `k=0` with no
      window; the composer at `k=0` with a window past the last step gives the same latents bit
      for bit, and so does `run_cfg_poe`, because the composer's per-level call is that sampler's
      three-branch call op for op (`corrector/logs/leak_checks.log`, commit 72b8826).
- [x] ✅ With a corrector window placed past the last step and `k=200`, is the output still
      byte-identical? This catches a corrector running outside its window, which the first check
      cannot see. Yes: window (60, 70) at `k=200` is byte-identical to the `k=0` reference, and the
      run's own per-level record shows the corrector applied at 0 of 50 levels. The Langevin noise
      generator is never drawn from when a level applies no steps, which is why the two runs can
      match bit for bit rather than only closely.
- [x] ✅ Does the picked `c` sit in the middle of the tested range rather than at its edge? It does
      now. The first five values all passed except 0.01, so the pick sat at 1.0, the top of the
      range; the range was widened to 3 and 10 (both usable), then to 30, 100 and 300, and 100 is
      the first value to diverge. The tested range is 0.01 to 300, the usable range 0.035 to 30,
      and the pick is 30, one step inside the tested edge on the failing side.
- [x] ✅ Does anything else in the tree already measure a corrected residual? No. Outside the
      vendored Du et al. code, `grep -rn "langevin\|corrector" plans/ scripts/ poe_repair/` hits
      three things, none a collision: `run_external_corrector_inject` in
      `poe_repair/methods/_sampling.py`, a learned additive residual corrector from the
      failure thread, which adds a network's output to the prediction and runs no chain; the
      `scripts/superdiff/*` files, which only use `corrector/` as this scope's output folder name;
      and plan files in scopes 01, 03 and 05 that refer to this scope's corrector by name.

## The step-size search

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

Filled by task 2. The cost of the grid at step 26 rides on this pick, so the table is here rather
than in a log. `δ_t = c·β_t`, at `k=20`, on `a_cat__x__a_dog` seed 9.

| `c` | Median relative displacement (min to max over the 50 levels) | Max latent norm, × the chain's start | Ratio rises with `k`? (levels increasing, median change) | Read-zone ratio before the chain → after 20 steps | Verdict |
|---|---|---|---|---|---|
| 0.01 | 0.049 (0.034 to 0.069) | 1.001 | no (10%, −0.0002) | 0.181 → 0.179 | stalled |
| 0.035 | 0.091 (0.061 to 0.128) | 1.001 | no (6%, −0.0034) | 0.206 → 0.203 | usable |
| 0.1 | 0.153 (0.096 to 0.216) | 1.002 | no (18%, −0.0064) | 0.222 → 0.206 | usable |
| 0.3 | 0.255 (0.107 to 0.371) | 1.004 | no (10%, +0.0004) | 0.139 → 0.141 | usable |
| 1.0 | 0.444 (0.134 to 0.652) | 1.007 | no (6%, −0.0057) | 0.226 → 0.221 | usable |
| 3.0 | 0.699 (0.177 to 1.021) | 1.011 | no (12%, +0.0042) | 0.245 → 0.210 | usable |
| 10 | 1.103 (0.344 to 1.381) | 1.038 | no (2%, +0.0038) | 0.154 → 0.154 | usable |
| 30 | 1.226 (0.432 to 1.497) | 1.113 | no (2%, +0.0082) | 0.123 → 0.124 | **usable, picked** |
| 100 | 1.358 (1.078 to 3.073) | 3.086 | no (12%, +0.0022) | 0.184 → 0.155 | diverged |
| 300 | 1.339 (see the file) | 5.244 | no (16%, +0.0010) | 0.184 → 0.155 | diverged |

The displacement is within one level: how far the 20-step chain moved from the point it started
that level at, relative to that point's norm. The last column is the mean ratio over the last
five levels, read once before the chain ran at that level and once at the settled point.
Every row is `a_cat__x__a_dog` seed 9, `k = 20`, 50 DDIM steps, guidance 7.5. From
`corrector/step_size_search.json`, fields `rows[].median_chain_disp_rel`,
`max_latent_norm_rel`, `frac_levels_ratio_increasing`, `median_ratio_change_start_to_end`,
`read_zone_ratio_at_inner0`, `read_zone_ratio_at_k`.

**Picked `c`:** 30, the largest usable value, written to `picked_c` in the same file.

**Was the range adequate?** Yes, after two widenings: the first pass ended with the pick at its top
edge (1.0), the second pass (3, 10) did the same, and the third (30, 100, 300) found the first
value that diverges. Recorded by Claude under instruction 3.3 of the design, veto after.

**What the search says about its own rule.** The norm bound cannot trip below the Euler stability
limit `δ_t < 2`, which is `c` near 170 at the noisiest level, and the rise-with-inner-count guard
never tripped at any `c`. So the "largest usable `c`" rule lands on a step size, `δ_t = 30·β_t`,
that is 0.35 at the noisiest level and 0.026 at the cleanest: the chain forgets its start within a
few steps (median displacement 1.23 times the latent norm), which is what equilibration asks for,
and the discretisation bias of unadjusted Langevin at that step size is a limit the write-up owes
(no Metropolis step is available, see the table below). A smaller `c` would trade that bias for
a chain that has not settled by `k = 200`; the pre-registered rule chose the settled side, and the
`k = 100` against `k = 200` check at step 26 is what says whether the chain really did settle.

## Asked after the result

Navigation: ⬅️ [The step-size search](#the-step-size-search) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- The read-zone ratio before the chain runs at a level (the `inner0` column) differs across `c`,
  from 0.12 to 0.25, even though no chain has run at that level yet. That is the corrected path:
  the chains at the 45 earlier levels put each `c` on its own trajectory, and the last five levels
  inherit it. It is the plan's own sentence that two corrector counts are two trajectories,
  seen across `c` instead of `k`.
- Within a level, 20 steps change the read-zone ratio by at most 0.035 (at `c = 3`), so at
  `k = 20` most of what the chain does to the residual is done through the path, not at the
  level. The grid's `k` sweep is the measurement of that; this table only says the instrument
  moves.
- **The two numeric divergence guards do not see what the pictures show.** The final images of
  the search chains (`corrector/search_renders/cat_dog_k20_by_c.png`, one tile per `c` at
  `k = 20`, from the composer's own PNGs under `corrector/pairs/a_cat__x__a_dog/seed_9/`) are
  photographs of one fused animal up to `c = 0.3`, a different scene with a leash and harness at
  `c = 3`, a painterly smear at `c = 10`, and texture noise from `c = 30`, while the latent norm
  stays within 1.11× and the residual ratio does not rise within a level for any of them. The
  norm cannot rise because the Langevin step contracts toward the score's mean whatever its size
  until `δ_t` passes 2; and the in-level ratio guard reads the joint branch at a point the chain
  has already left the data manifold at, where the two branches agree about as badly as they did
  before. So "does not diverge" as pre-registered is a statement about the Euler discretisation,
  not about the sample, and the rule "the largest such `c`" lands on a step size whose chain
  produces noise. On the composing pair the grid at step 26 makes this a number: at `c = 30` the
  read-zone ratio rises from 0.090 at `k = 0` to 0.111 at `k = 200` (peak 0.169 at `k = 20`), and
  its images are texture from `k = 1`, which is the pre-registered inconclusive branch there. At
  `c = 0.3` the same pair holds at 0.092 to 0.093 from `k = 0` to `k = 20` with the meadow and the
  butterfly intact in every tile (`corrector/search_renders/butterfly_c0p3_by_k.png`).
- **What is being done about it, under the plan's own rule.** Step 26's inconclusive branch says
  return here and widen; the still-open row below already named the composing pair as the
  search's missing control. So the grid is being run again at `c = 3` and `c = 0.3`, and the
  step size the scope proceeds at is the largest `c` whose composing-pair curve does not rise
  past `1 + MAX_DRIFT_FOR_NULL` of its `k = 0` value at `k = 200`. No threshold moves; the
  control that was already written into step 26 is applied to the search's candidates. The
  numeric search table stays as the record of what the pre-registered rule alone would have
  chosen.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Across the ten search rows only `c` varies: same pair, same
      seed, same `k = 20`, same 50 DDIM steps, same guidance 7.5, the same cached starting latent,
      and the same Langevin noise stream (the generator is seeded by the cell's seed).
- [x] ✅ **Was the measuring tool sound?** Both leak checks are byte-identical, the reference being
      a `k=0` run of the composer, which is also byte-identical to `run_cfg_poe` because its
      per-level call is that sampler's own three-branch call. The displacement column rises
      monotonically with `c` from 0.049 to 1.36, so the corrector is applied and its size tracks
      the step size.
- [x] ✅ **Did the run respect the environment?** Output under
      `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`, the launcher's disk
      guard reads that path's own filesystem (33% used), every norm is float32, and the SSH launch
      lines onto mscluster108 carry absolute paths only; node, device and PID are in each log
      header.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| anything measured on the corrected path | that the step size was picked by a recorded search rather than inherited, and which value it is |
| a corrector result at any `k` | that the corrector is unadjusted Langevin, with no Metropolis acceptance step, because SDXL is a score model with no energy head |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether one pair and one seed is enough to fix a step size for the whole scope | the grid at step 26 on the composing pair, at `c = 30`, `3` and `0.3`; the largest `c` whose control curve does not rise is the step size the scope proceeds at | step 26's verdict and everything after it, until the repeated grids land |
| a divergence guard that sees what the eye sees | a per-`c` render read by eye is now saved with every search row; a numeric candidate is the joint branch's ratio measured on the path across `k` (what step 26 does) rather than within a level | nothing: the composing-pair control already covers it for this scope, and a future search should carry it from the start |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run the two leak checks. Neither costs more than one render, and a failure in either stops the scope
before 670 plain-render equivalents are spent on a composer that is not inert.
