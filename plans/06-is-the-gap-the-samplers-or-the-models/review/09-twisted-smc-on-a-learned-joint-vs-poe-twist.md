# 🎲 Review: twisted SMC on a learned joint-versus-PoE twist

This file judges [the design](../plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md):
a particle sampler over the plain PoE score that only selects, using a learned estimate of the
joint-to-PoE density ratio as its weight.

## Recommended prompt (when the run lands)

```
/analyze-run the twisted-SMC run in prime_lab/poe-repair-animals-compose, group twisted-smc: val/acc, eval/ess_mean, the compose curves and the strips
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md) | the bank, the head, the sampler, the strip cadence, the bars |
| **this file** | **the verdict** |
| [step 26's verdict](03-what-is-left-once-the-chain-settles.md) | the Langevin read of the same question |

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

- **The twist**: `log psi_t(x_t)`, the head's logit, an estimate of
  `log p_J,t(x_t) − log p_PoE,t(x_t)` for a noisy latent at timestep `t` given the joint prompt.
- **The PoE control**: K particles on the product-of-experts score with DDIM at `eta` 1.0,
  the same noise draws as the SMC panel, no weights and no resampling.
- **Compose fraction**: the share of images the validated detector counts two or more animal
  instances in. The run itself scores the shown particle (the largest-weight one per render cell,
  three per checkpoint); `scripts/twisted_smc/score_all_particles.py` scores every saved
  particle afterwards (four per cell, twelve per checkpoint).
- **ESS**: effective sample size, `1 / sum(w²)` over normalised particle weights; below half of K
  the population is resampled.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** A null is a finding about whether the product's proposals contain composing
states; an unlearned twist is inconclusive, not a null.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Smoke, `scripts/twisted_smc/train_twist.sbatch smoke`, Slurm job 49849 on mscluster110 (`co3_bw`) | Builds a measuring tool | 2026-09-05 06:38 | 3 min wall: 40 steps, K 2, 512², 4 render steps, 2 render cells | run dir `outputs/interaction_term/twisted_smc/smoke_twist_w64_b4_lr1e-04_s40_20260905-063844/`: `references/` (Mono and PoE, both particles), `samples/step_{000000,000020,000040}/` each with the strip, both SMC particles and the resample record, `checkpoints/twist_step_000040.pt`, `verdict.json` ("inconclusive: twist did not learn", as a 40-step run must). W&B `prime_lab/poe-repair-animals-compose/runs/5isfz35a`: 6 strip images, `strips-step000040` artifact (33 files). The images are noise at 4 steps and 512², which is the plumbing test, not a read | ✅ the whole path runs |
| First full launch, Slurm job 49850 on mscluster110, W&B `cyhd80oz` | Tests the claim | 2026-09-05 06:42 | 5 min: references rendered in 71 s (six K-4 runs at 20 steps, 1024²) | nothing kept | ⏹ cancelled at 06:49: the default render cells were `a_bear__x__a_salmon`, `a_butterfly__x__a_flower_meadow` and `a_barn__x__pencil_drawing_style`, and the instance detector counts animals, so two of the three could never read as compose. Also found: the heldout split repeats six training pairs, so the validation bank now excludes any pair present in train |
| Re-score of every saved particle, `scripts/twisted_smc/score_all_particles.py <run dir>`, mscluster85 device 0 (`co3`), PID 251266 | Builds a measuring tool | 2026-09-05 16:30 | 6 min: 156 images through the detector | `all_particles_scores.json` in the run dir, copied to `artifacts/results/is-the-gap-the-samplers-or-the-models/twisted-smc-all-particles-scores.json`. Mono 12 of 12, PoE control 0 of 12, SMC 0 of 12 at 100k; SMC 3 of 12 at 40k and 80k, 2 of 12 at 60k and 70k, 0 elsewhere. All 10 hits opened: each is one fused animal with a second box on a collar tag or coat patch | ✅ done |
| Full, `train_twist.sbatch full`, Slurm job 49853 on mscluster110 (`co3_bw`) | Tests the claim | 2026-09-05 06:51 | 100k steps at batch 16+16; renders at K 4, 20 steps, 1024² every 10k steps on `a_wolf__x__a_husky` and `a_lion__x__a_tiger` (training, seed 1) and `a_cat__x__a_dog` (held out, seed 1) | run dir `outputs/interaction_term/twisted_smc/twist_w64_b16_lr1e-04_s100000_20260905-072331/`, W&B `prime_lab/poe-repair-animals-compose/runs/3cwrxlw0`. 100k steps in 45 min (about 37 steps per second, renders 31 s each). Eleven strips per cell (steps 0 and every 10k), `checkpoints/twist_step_100000.pt`, `verdict.json`: compose fraction over the three shown particles (the largest-weight particle of each render cell) Mono 1.0, PoE control 0.0, SMC 0.0 at step 100k; the single non-zero SMC read was 0.33 at step 40k (cat×dog, a second face inside a cat's fur) and did not recur. The other K−1 particles per cell are saved but not scored by the run. Resampling fired 5 to 8 times per 20-step run at every checkpoint (`eval/n_resample_mean`), mean ESS 2.4 to 2.8 of K 4. Training accuracy 1.0 and per-bucket training loss 0.00 in the mid and near-clean buckets and 0.03 to 0.04 at high noise from step 30k on (chance is 0.69); validation accuracy 0.53 to 0.66 across the run, 0.5625 at the final validation, validation loss rising from 2.2 to 5.5 | ✅ finished; `verdict.json` says "inconclusive: twist did not learn" (final validation accuracy under the 0.60 bar) |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does resampling the plain PoE sampler on the learned twist raise the compose fraction
      over the same particles left unweighted?** Read from `verdict.json` at the final render:
      pass if the SMC fraction exceeds the control's by 0.25 or more; null if the two are within
      0.10 while validation accuracy is at or above 0.60; inconclusive if validation accuracy
      never reaches 0.60. The bars are `PASS_MARGIN`, `NULL_MARGIN` and `MIN_VAL_ACC` in
      `poe_repair/experiments/twisted_smc/train.py`.
      **Inconclusive by the code's own bar, with the direction pointing at null.** Final
      validation accuracy 0.5625, so the head did not learn a ratio that carries to unseen pairs.
      What it did learn separated the classes by memorising the 120 training latents: training
      loss 0.00 in the mid and near-clean buckets and 0.03 to 0.04 at the highest noise, where
      the two distributions coincide and no content-based classifier can score. On the reweighting that head produced, the
      compose fraction of the SMC panel was 0.0 at ten of eleven checkpoints against 0.0 for the
      control and 1.0 for Mono, with resampling firing 5 to 8 times per run. Over every saved
      particle (12 per checkpoint) the control is 0 of 12, Mono 12 of 12, and SMC 0 of 12 at the
      final checkpoint; the detector fires on 10 of the 132 SMC particles at 40k to 80k, and each
      of those, opened and read, is one fused animal with the second box on a collar tag or a
      coat patch. So a selection-only sampler driven by this twist did not find composing states
      among the product's proposals at K 4; whether a content-based twist would is what the next
      version has to test.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ⚠️ Did the smoke run produce references, three strips, a checkpoint and the W&B artifact?
      Yes, all four (Runs table, job 49849).
- [x] ⚠️ Does validation accuracy climb above 0.60 and flatten, and by which step? W&B run id?
      No. Run `3cwrxlw0`: 0.60 at step 1k, a high of 0.66 at 31k, 0.53 at 61k, 0.5625 at 100k.
      It never flattened above the bar.
      📊 Drawn in [Figure 3 of the figure explainer](../../../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer.md#figure-3-what-the-twist-head-learned-accuracy-and-loss-by-noise-level).
- [x] ⚠️ Is the per-timestep-bucket loss ordered as the theory says: lowest near clean, highest at
      high noise, where the two distributions coincide? If the high-noise bucket is as low as the
      near-clean one, the head is separating on an artefact. **The artefact case.** Mid and
      near-clean buckets at 0.00 from step 10k; the high-noise bucket at 0.17 at 10k and 0.03 to
      0.04 from 30k to 100k, against 0.69 at step 1. A classifier that separates re-noised Mono from re-noised PoE at
      timestep 900 is reading the identity of one of 120 latents, not composition.
- [x] ⚠️ Does resampling fire at all (`eval/n_resamples` above 0, `eval/ess_mean` below K)? If
      never, the weights are flat and the sampler is the control. Yes: 5 to 8 resamples per
      20-step run at every checkpoint, mean ESS 2.4 to 2.8 of 4.
- [x] ⚠️ Before any resample fires, is particle `k` of the SMC render the same file as particle
      `k` of the PoE reference (`*__smc__p{k}.png` against `*__poe__p{k}.png`)? They share the
      noise path, so a difference there is a bug, not a result. The strip panels themselves may
      differ, since the control shows particle 0 and the SMC panel shows the largest-weight
      particle. Checked on the smoke run at step 0 (no resample fired): both particles
      byte-identical by md5.
- [x] ⚠️ Does the compose fraction of the SMC panel move across checkpoints while the control's
      stays fixed? Control fixed at 0.0 and Mono at 1.0 throughout. SMC 0.0 at every checkpoint
      except 0.33 at step 40k, which did not recur.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- [ ] ⚠️ Would a twist that cannot memorise reach the bar? Two routes: more positives (the
      held-out Mono finals of pairs not used for validation, or scorer-labelled composing images
      across many seeds), and a head that sees the latent only through a frozen encoder or a
      random-projection bottleneck, so pair identity is not a cheap feature.
- [ ] 🟡 Does the SMC panel's fresh chimera at every checkpoint (a different single animal from
      the control's, on all three cells) say anything on its own? It says the weights are
      informative enough to move the selection and that every selected state is still a chimera,
      which is consistent with the product proposing no composing state at K 4.
- [ ] 🟡 Would K 16 or 64 change the read? Unknown; K 4 was chosen for the strip cadence, not
      for coverage.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ⚠️ **Was the comparison fair?** Same K, `eta`, noise draws, steps, guidance and
      resolution between the control and the SMC panel; only the weighting differs. Mono uses the
      same sampler, so the target panel is not the repo's deterministic render. Yes, and the
      noise-path check passed by md5 on the smoke run.
- [x] ⚠️ **Was the measuring tool sound?** The run scored only the shown particle; the re-score
      row in the Runs table read all 156 saved images. The eye agrees with the detector on every
      Mono and control image and on 122 of 132 SMC particles; on the other 10 the detector draws
      its second box on a collar tag or a coat patch of one fused animal. The detector is the
      pinned rule, so those 10 stay counted as composes in the numbers and are named as scorer
      errors beside them.
- [x] ⚠️ **Did the run respect the environment?** Everything under `/datasets`, `co3_bw` on the
      Blackwell node, the disk guard on the output root, curves on W&B and ids here. Yes.
- [x] ⚠️ **Did the twist see the held-out render pair?** The held-out render cell is in the
      validation bank, which the loss never trains on; the two training render cells are in the
      training bank. Read the held-out strip separately from the two training strips. cat×dog is
      not among the 18 training pairs and its heldout cells are excluded from nothing but
      training; the validation bank excludes the six pairs the heldout split shares with train.
      The held-out strip reads the same as the two training strips.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a selection-only sampler composes, or does not | the compose fractions per panel with the particle count, the render cells, and `eta` |
| the corrector and the twist are one object | the identity `r_t ∝ ∇ log psi_t`, and which of the two composed on the same cells |
| the twist was learned contrastively | the positive count (120 Mono finals), the validation accuracy, and the amortised-twist paper it follows |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [ ] Whether a non-memorising twist (more positives, a bottlenecked head) clears 0.60 on
      unseen pairs and changes the compose read. Until it does, this plan's answer is the
      inconclusive above, and the scope's sampler-side question stays open from this side.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Decide whether to spend a second run on the twist's data problem. If yes: positives from every
Mono final in both splits except the validation pairs, plus scorer-labelled composing renders
across seeds; a head with a frozen or random-projection front end; the same bars.
