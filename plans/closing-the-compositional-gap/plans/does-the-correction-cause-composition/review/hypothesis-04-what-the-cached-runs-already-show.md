# Review: what do the already-cached runs tell us?

**All answered, from cached data with no GPU and no queue.** This file judges
[../plans/hypothesis-04-what-the-cached-runs-already-show.md](../plans/hypothesis-04-what-the-cached-runs-already-show.md)
and feeds three register slots: **F3** (does the correction's size track the noise level), **F4**'s
timing band, and **F6** (is it low-rank enough to learn).

Two answers changed what the paper may claim, so read them before writing anything that leans on
them. The size-follows-noise result is looser than hoped. The low-rank result licenses much less
than its name suggests.

## Words this file uses
- **Noise level**: how far through the denoising run a step is, expressed so that different pairs
  can be compared on one axis.
- **The two paths**: the broken one and the working one, walked from the same starting noise. The
  step where they pull apart is the step where the outcome gets decided.
- **Low-rank**: how few directions carry most of the corrections' energy. Fewer means a small
  adapter can learn it. Always read against a same-shape random floor, because the raw percentage
  is partly forced by how many vectors were stacked.

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
- [x] 🔴 Is the correction low-rank, against a matched random floor?
      Not against a floor that controls for the right thing, and the floor used here did not.
      Against the same-shape Gaussian the pooled stack looks strong (k=8 carries 22.6% against
      2.1%, 11x). That floor gives every row the same expected norm, while real ‖r_t‖ runs 8.7
      to 107.6 across the 440 rows, a factor of 4.5 against the median, because it tracks the
      noise level. Rows of unequal size concentrate energy on their own. Against random
      directions carrying the real norms, the stack is 1.5x at k=1, 1.4x at k=8 and 1.1x at
      k=64: no distinguishable structure. Scaling every row to unit norm does leave real
      direction structure (23x at k=1, 7.8x at k=8), but it lives inside single runs, not across
      them: 50 steps of one cell beat the floor 4.8x at k=8, while one step from each of 50
      different cells beats it 1.2x. Full argument and tables:
      `docs/evidence/F6-what-the-spectrum-measures/QUERY.md`. Two claims follow. F6 may not argue
      that a shared low-dimensional structure is what makes the correction learnable, and the
      learnability claim rests on the adapter's measured behaviour instead. Within-run smoothness
      is D1's result and cross-pair orthogonality is D3's, so what the spectrum adds beyond those
      two is a decision the slot still needs.
- [x] ✅ Does a subspace fitted on training pairs carry to held-out pairs?
      No at the vector level, and that licenses nothing about transfer: 13.3% at k=64 on the 6
      unseen pairs against 63.0% on training pairs, while the SAME adapter composes 96.9% on
      those pairs where plain PoE composes 0%. The r_t vectors are mutually near-orthogonal
      (cosine about 0.00 even train-to-train), so no fitted subspace can contain unseen pairs
      whether or not the correction transfers. Full argument:
      `docs/evidence/F6-subspace-vs-transfer/QUERY.md`, whose per-pair table is measured at 3
      seeds and every 3rd step, where the same projection reads 6.0%. The sampling moves the
      percentage and not the gap, which is what carries. Any paper sentence built on "shared
      subspace" wording must be rewritten to this bounded form.

## Design decisions F6's caption rests on

- [x] ✅ The spectrum's statistical entity: one row is the correction at ONE denoising step of one
      cell, a cell being one pair at one seed, flattened over the 4x128x128 latent into 65536
      numbers. A cell averaged over its own timesteps is rejected as the row, for two reasons.
      It matches what the adapter has to do: the LoRA is called once per step and computes a
      fresh r_t from the current latent and the prompt, so the vectors it must represent are
      indexed by step, and a per-cell average is a quantity it never produces. And the averaged
      version has no claim left in it. The steps of one cell share no direction to average over:
      cosine between two steps of the same cell is +0.81 at a gap of 1 step and +0.012 at gaps of
      20 to 49 (median over the 11 training cells, 50 steps each), so a cell's mean is set by
      whichever stretch of the run carries the largest ‖r_t‖ rather than by anything the steps
      agree on. The energy comparison says the same thing: over 11 pairs at 8 seeds keeping every
      10th step, the per-step stack is 440 rows with 22.6% of its energy at k=8 against a 2.1%
      same-shape Gaussian floor, a ratio of 10.7, while averaging each cell leaves 88 rows with
      39.1% against a 9.8% floor, a ratio of 4.0. Both sides are measured against the same
      equal-norm floor, which overstates each of them by the same route (see the low-rank answer
      above), so the comparison holds even though neither ratio should be quoted on its own. So
      F6 keeps the
      per-timestep rows `scripts/spectrum.py` already stacks, and says in the caption that a row
      is one step of one cell.

- [x] ✅ How much of the cache the quoted spectrum is measured on: all 11 training pairs at 8
      seeds, keeping every 10th step, 440 rows. The ratio against the floor depends on this
      choice, so the invocation travels with the numbers everywhere they are quoted. Adjacent
      steps are near-copies (cosine +0.81 one step apart) and near-copy rows concentrate energy
      in a way a Gaussian floor of independent rows does not correct for, so a denser sampling
      flatters the claim: at k=8 the ratio is 21.8 keeping every step at 1 seed, 14.6 keeping
      every 3rd at 3 seeds, and 10.7 at the chosen setting, whose kept steps sit at cosine +0.11.
      The claim holds at all three, which is why this was a question of what is quotable rather
      than of whether F6 stands.
