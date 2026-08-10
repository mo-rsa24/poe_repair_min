# 🔬 Review: the disk-only analyses

Verdicts for [../plans/05-cache-analyses.md](../plans/05-cache-analyses.md). The design lives
there; the findings live here. Questions written before each analysis ran; answers carry the
numbers that decided them.

## Run kind
**Tests the claim** (Goals 2, 3, 4 of this scope). No pre-registered single bar: each analysis
carries its own expected shape, named per question.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| snr_collapse, fork_curve, climb, spectrum: in-session scripts over the cache | Tests the claim | commits of 2026-08-05 | `outputs/interaction_term/cache_analyses/` | done |
| Mono-path generation for the fork read, 38/38 cells, 0 failed, mscluster109 GPU 1 | Tests the claim | 2026-08-05, `scripts/mechanism_study/generate_fork_paths.sh` | fork paths beside the cache | done |

## The questions

- [x] 🟡 Does the correction's size follow noise level on one shared curve across pairs?
      Partly. Spread 19.7% over 17 pairs (34 curves) under the pre-committed `relative_norm`,
      which is looser than hoped. And the two measures disagree about the peak: the committed
      measure is still rising at the right edge (no interior peak), while raw ‖r_t‖ peaks at
      log-SNR -0.90, because ‖ε_PoE‖ itself falls about 15% along the trajectory. Feeds slot F3
      with the caption claiming the collapse only as far as 19.7% supports it. Do not read either
      curve as the timing answer; timing is plan 04's question.
- [x] ✅ Where do the PoE and Mono paths fork?
      Elbow at step 16 (median over 19 cells, 50-step schedule), tight: 15 of 19 cells land
      between steps 13 and 20. d(0)=0.00 on every cell, the check that both paths start from the
      same pinned init. Guard added while reading: trajectories under 40 steps are skipped, since
      a 20-step smoke run had been pooled in and pulled the median to 15. The comparison against
      plan 04's window is OPEN until that sweep runs. Feeds the F4 vertical band.
- [x] ✅ Does the correction align with the sampling motion?
      Yes, with the sign understood: normalised climb median +0.397, 0/38 cells negative. A DDIM
      step moves along MINUS the prediction, and r_t sits on the opposite side of zero from the
      prediction (cos(r_t, ε_PoE) negative in 38/38 cells, median -0.14). So the correction
      SUBTRACTS from what PoE asks for, which is exactly what "PoE overshoots into a blend"
      predicts. Controls: random-vs-motion +0.000, wrong-step r_t +0.11. Alignment decays 0.92
      to 0.21 across the run, positive throughout.
- [x] ✅ Is the correction low-rank, against a matched random floor?
      Yes by ratio: k=1 carries 5.4% vs 0.2% floor (25x), k=8 24.7% vs 1.7% (15x), k=64 62.6% vs
      13.2% (5x). Read the ratio, never the raw percentage: with N vectors in D dims the energy
      at k is partly forced by N alone. Feeds slot F6, floor shaded.
- [x] ✅ Does a subspace fitted on training pairs carry to held-out pairs?
      No at the vector level, and that licenses nothing about transfer: 6.0% at k=64 on the 6
      unseen pairs against 62.6% on training pairs, while the SAME adapter composes 96.9% on
      those pairs where plain PoE composes 0%. The r_t vectors are mutually near-orthogonal
      (cosine about 0.00 even train-to-train), so no fitted subspace can contain unseen pairs
      whether or not the correction transfers. Full argument:
      `docs/evidence/subspace-vs-transfer/QUERY.md`. Any paper sentence built on "shared
      subspace" wording must be rewritten to this bounded form.

## Still open

- [ ] ⚠️ The spectrum's statistical entity: per-timestep rows or time-averaged rows. A design
      decision made by `/pair-figure`, recorded here only because F6's caption depends on it.
