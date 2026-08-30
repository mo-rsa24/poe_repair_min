# 📊 Review: what do the already-cached runs tell us?

**All answered, from cached data with no GPU and no queue.** This file judges
[../plans/hypothesis-04-what-the-cached-runs-already-show.md](../plans/hypothesis-04-what-the-cached-runs-already-show.md)
and feeds three places in the figure register. **F3** asks whether the correction's size tracks the
noise level, **F4** carries the timing band, and **F6** asks whether the correction is low-rank
enough to learn.

Two answers changed what the paper may claim, so read them before writing anything that leans on
them. The size-follows-noise result is looser than hoped. The low-rank result licenses much less
than its name suggests.

## Recommended prompt (to write the figures)

```
/design-figure F6 the spectrum, energy at k against a norm-matched random baseline
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis-04-what-the-cached-runs-already-show.md) | the four analyses over the cache, and the shape each expects |
| **this file** | **the verdict: three hold, the size-follows-noise result is loose, the low-rank result licenses much less than its name** |
| [the timing answer](hypothesis-03-when-in-the-run-it-matters.md) | the correction moved across nine window positions, which disagrees with the fork step and is not a contradiction |
| [the register](../../../paper/iclr/figures.md) | rows F3, F4 and F6 |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️
- **Noise level**: how far through the denoising run a step is, expressed so that different pairs
  can be compared on one axis.
- **The two paths**: the broken one and the working one, walked from the same starting noise. The
  step where they pull apart is the step where the outcome gets decided.
- **Low-rank**: how few directions carry most of the corrections' energy. Fewer means a small
  adapter can learn it. Always read against a same-shape random baseline, which is what the
  measure reads when there is no real structure, because the raw percentage is partly forced by
  how many vectors were stacked.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️
**Tests the claim** (Goals 2, 3, 4 of this scope). There is no single pre-registered threshold
here. Each analysis carries its own expected shape, named per question.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| snr_collapse, fork_curve, climb, spectrum: in-session scripts over the cache | Tests the claim | commits of 2026-08-05 | no GPU, no queue: reads the cache | `outputs/interaction_term/cache_analyses/` | done |
| Mono-path generation for the fork read, 38/38 runs, 0 failed, mscluster109 GPU 1 | Tests the claim | 2026-08-05, `scripts/mechanism_study/generate_fork_paths.sh` | 38 runs | fork paths beside the cache | done |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**There is no single threshold here, and inventing one would misrepresent the plan.** Each analysis
carries its own expected shape, named inside its own question below. Two of the answers changed
what the paper may claim, so read those before writing anything that leans on them: the
size-follows-noise result is looser than hoped, and the low-rank result licenses much less than
its name suggests.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] 🟡 Does the correction's size follow noise level on one shared curve across pairs?
      Partly. Spread 19.7% over 17 pairs (34 curves) under the pre-committed `relative_norm`,
      which is looser than hoped. And the two measures disagree about the peak: the committed
      measure is still rising at the right edge (no interior peak), while raw ‖r_t‖ peaks at
      log-SNR -0.90, because ‖ε_PoE‖ itself falls about 15% along the trajectory. Feeds F3
      with the caption claiming the collapse only as far as 19.7% supports it. Do not read either
      curve as the timing answer; timing is plan 04's question.
- [x] ✅ Where do the PoE and Mono paths fork?
      Elbow at step 16 (median over 19 runs, 50-step schedule), tight: 15 of 19 runs land
      between steps 13 and 20. d(0)=0.00 on every run, the check that both paths start from the
      same pinned init. Guard added while reading: trajectories under 40 steps are skipped, since
      a 20-step first short run had been pooled in and pulled the median to 15. The comparison
      against plan 04's window is OPEN until those window runs happen. Feeds the F4 vertical band.
- [x] ✅ Does the correction align with the sampling motion?
      Yes, with the sign understood: normalised climb median +0.397, 0/38 runs negative. A DDIM
      step moves along MINUS the prediction, and r_t sits on the opposite side of zero from the
      prediction (cos(r_t, ε_PoE) negative in 38/38 runs, median -0.14). So the correction
      SUBTRACTS from what PoE asks for, which is exactly what "PoE overshoots into a blend"
      predicts. Controls: random-vs-motion +0.000, wrong-step r_t +0.11. Alignment decays 0.92
      to 0.21 across the run, positive throughout.
- [x] 🔴 Is the correction low-rank, against a matched random baseline?
      Not against a baseline that controls for the right thing, and the one used here did not.
      Against the same-shape Gaussian the pooled stack looks strong (k=8 carries 22.6% against
      2.1%, 11x). That baseline gives every row the same expected norm, while real ‖r_t‖ runs 8.7
      to 107.6 across the 440 rows, a factor of 4.5 against the median, because it tracks the
      noise level. Rows of unequal size concentrate energy on their own. Against random
      directions carrying the real norms, the stack is 1.5x at k=1, 1.4x at k=8 and 1.1x at
      k=64, so there is no distinguishable structure. Scaling every row to unit norm does leave
      real direction structure (23x at k=1, 7.8x at k=8), and that structure lives inside single
      runs rather than across them: 50 steps of one run beat the baseline 4.8x at k=8, while one
      step from each of 50 different runs beats it 1.2x. Full argument and tables:
      `artifacts/results/which-way-the-correction-points/what-the-spectrum-measures/QUERY.md`. Two claims follow. F6 may not argue
      that a shared low-dimensional structure is what makes the correction learnable, and the
      learnability claim rests on the adapter's measured behaviour instead. Within-run smoothness
      is D1's result and cross-pair orthogonality is D3's, so what the spectrum adds beyond those
      two is a decision F6 still needs.
- [x] ✅ Does a subspace fitted on training pairs carry to held-out pairs?

      > Held-out means a pair the adapter never trained on. It is the only kind that tests
      > whether the fix reaches something new.

      No at the vector level, and that licenses nothing about transfer: 13.3% at k=64 on the 6
      unseen pairs against 63.0% on training pairs, while the SAME adapter composes 96.9% on
      those pairs where plain PoE composes 0%. The r_t vectors are mutually near-orthogonal
      (cosine about 0.00 even train-to-train), so no fitted subspace can contain unseen pairs
      whether or not the correction transfers. Full argument:
      `artifacts/results/which-way-the-correction-points/does-the-subspace-test-predict-transfer/QUERY.md`, whose per-pair table is measured at 3
      seeds and every 3rd step, where the same projection reads 6.0%. The sampling moves the
      percentage and not the gap, which is what carries. Any paper sentence built on "shared
      subspace" wording must be rewritten to this bounded form.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the results themselves raised. **Nothing here may ever become the question above**, because it was
written with the answers already visible.

- [ ] ⚠️ What does the [spectrum](/home-mscluster/mmolefe/goal-setting/learning/spectral-structure-of-the-correction/plans/09-energy-at-k-and-the-floors.md) add beyond within-run smoothness (D1's result) and cross-pair
      orthogonality (D3's result)? Raised by the low-rank answer collapsing against the right
      baseline. Until it is answered, F6 has no claim of its own.
- [x] ✅ Should trajectories under 40 steps be pooled into the fork read? No. A 20-step first
      short run had been pooled in and pulled the median from 16 to 15. A guard was added while
      reading.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** The fork read starts both paths from the same pinned init,
      checked rather than assumed: d(0)=0.00 on every run. The low-rank answer is where fairness
      failed and was repaired. The same-shape Gaussian baseline gives every row the same expected
      norm while real ‖r_t‖ runs 8.7 to 107.6 across the 440 rows, so that baseline was measuring
      row size rather than direction structure. The verdict rests on the norm-matched baseline
      instead.
- [x] ✅ **Was the measuring tool sound?** Two faults were found by reading rather than by luck, and
      both are fixed: the 20-step run pooled into the fork median, and the equal-norm baseline
      under the spectrum. The
      climb result carries its own controls (random-against-motion +0.000, wrong-step r_t +0.11)
      rather than resting on the headline number alone.
- [x] ✅ **Did the run respect the environment?** Outputs under
      `outputs/interaction_term/cache_analyses/`. The four analyses need no GPU and no queue, which
      is why they were the cheapest work available. The one generation step ran 38 of 38 runs with
      0 failed on mscluster109 GPU 1, in-session rather than queued.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| F3, the correction's size follows the noise level | the collapse only as far as 19.7% spread over 17 pairs supports it, and that the two measures disagree about the peak. Neither curve may be read as the timing answer |
| F4's vertical band at step 16 | that moving the correction across window positions puts the moment that matters at step 5, seven times better than step 15. The two estimates disagree, so neither may be printed as confirming the other |
| F6, the correction is low-rank | it may not argue that a shared low-dimensional structure is what makes the correction learnable. Against a norm-matched random baseline the stack is 1.5× at k=1 and 1.1× at k=64, so there is no distinguishable structure. The learnability claim rests on the adapter's measured behaviour instead |
| any sentence using "shared subspace" wording | rewriting to the bounded form. A subspace fitted on training pairs reads 13.3% at k=64 on unseen pairs against 63.0% on training pairs, while the same adapter composes 96.9% on those pairs. The r_t vectors are mutually near-orthogonal, so no fitted subspace can contain unseen pairs whether or not the correction transfers |
| the spectrum's numbers | the invocation they were measured under, which travels with them everywhere. See the two decisions below |

**A row is one denoising step of one run, never a run averaged over its steps.**


- [x] ✅ The spectrum's statistical entity: one row is the correction at ONE denoising step of one
      run, a run being one pair at one seed, flattened over the 4x128x128 latent into 65536
      numbers. A run averaged over its own timesteps is rejected as the row, for two reasons.
      It matches what the adapter has to do, since the LoRA is called once per step and computes a
      fresh r_t from the current latent and the prompt. The vectors it must represent are
      therefore indexed by step, and a per-run average is a quantity it never produces. The
      averaged version also has no claim left in it, because the steps of one run share no
      direction to average over. Cosine between two steps of the same run is +0.81 at a gap of 1
      step and +0.012 at gaps of 20 to 49 (median over the 11 training runs, 50 steps each), so a
      run's mean is set by whichever stretch of the denoising carries the largest ‖r_t‖ rather
      than by anything the steps agree on. The energy comparison says the same thing. Over 11
      pairs at 8 seeds keeping every 10th step, the per-step stack is 440 rows with 22.6% of its
      energy at k=8 against a 2.1% same-shape Gaussian baseline, a ratio of 10.7, while averaging
      each run leaves 88 rows with 39.1% against a 9.8% baseline, a ratio of 4.0. Both sides are
      measured against the same equal-norm baseline, which overstates each of them by the same
      route (see the low-rank answer above), so the comparison holds even though neither ratio
      should be quoted on its own. So F6 keeps the
      per-timestep rows `scripts/spectrum.py` already stacks, and says in the caption that a row
      is one step of one run.

- [x] ✅ How much of the cache the quoted spectrum is measured on: all 11 training pairs at 8
      seeds, keeping every 10th step, 440 rows. The ratio against the baseline depends on this
      choice, so the invocation travels with the numbers everywhere they are quoted. Adjacent
      steps are near-copies (cosine +0.81 one step apart) and near-copy rows concentrate energy
      in a way a Gaussian baseline of independent rows does not correct for, so a denser sampling
      flatters the claim: at k=8 the ratio is 21.8 keeping every step at 1 seed, 14.6 keeping
      every 3rd at 3 seeds, and 10.7 at the chosen setting, whose kept steps sit at cosine +0.11.
      The claim holds at all three, which is why this was a question of what is quotable rather
      than of whether F6 stands.

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| what F6 claims at all, now that the low-rank argument has collapsed against a norm-matched random baseline | deciding what the spectrum adds beyond within-run smoothness (D1) and cross-pair orthogonality (D3) | F6 itself, which currently has a figure and no claim of its own |
| whether the fork step and the nine window positions can be reconciled | nothing further, since they measure different things: a cause acting early and a difference becoming visible later. What is open is F4's layout, which draws them as agreeing | F4's caption and the register row behind it |
| how loose the size-follows-noise collapse is allowed to be before F3 stops claiming it | a judgement call on the 19.7% spread over 17 pairs | F3's caption |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Decide what F6 claims, given the spectrum no longer supports the learnability argument. Until
that is written down, building the figure would be drawing a panel with no sentence under it.
