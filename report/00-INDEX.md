# 🔬 Report: poe_repair_min

What this project found, one file per question, every claim paired with a figure and the
statistic that backs it. What the work *means* lives in [context/](../context/00-INDEX.md),
how to redo it in [runbook/](../runbook/00-INDEX.md); this folder is only the findings.

## What was found

| The question | Verdict | The number | The finding |
|---|---|---|---|
| Where does each condition land in the scorer's image space, and does the correction move PoE toward the joint prompt? | ❌ null on both pre-registered bars; ✅ support on the post-hoc reads | cat × dog both-ness (projection toward the joint-prompt centroid, DINOv2 cosine units, mean of 8 seeds): PoE 0.21, corrected 0.42, joint 0.52; on the 8 held-out pairs the both-ness bands overlap on 5 of 7 counted pairs (null), while the instance count on the same images reads PoE 0.00 to 0.25 and corrected 0.75 to 1.00 on every pair (post-hoc) | [where-does-each-condition-land.md](when-does-the-outcome-lock-in/where-does-each-condition-land.md) |
| What is the correction made of, and could PoE have supplied it by re-weighting its own predictions? | ❓ inconclusive on the pre-registered bar | orthogonal share of the correction's squared norm outside the span of the three predictions PoE has, mean over seeds 9 to 16 and steps 0 to 10: 0.374 against bars at 0.5 and 0.25; the reachable part weights each expert at 1 to 3 where PoE weights 7.5; the adapter's orthogonal share 0.19 against the target's 0.37 | [what-is-the-correction-made-of.md](when-does-the-outcome-lock-in/what-is-the-correction-made-of.md) |
| Is the held-out gap of the pooled rank 8 adapter a fit, a drift, or a pair problem? | ✅ support: pair problem; ⚪ null on drift | fit cosine on cached states: training pairs 0.97, seen words in an unseen pairing 0.84, no seen word 0.80; on the adapter's own corrected trajectory 0.82 against 0.86 cached, cat x dog | [is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md](does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md) |
| Does the interaction strength read from the cache (Mitra et al. Theorem 1 turned around) predict which pairs blend? | ⚪ null | lower bound on G·M over steps 10 to 40 spans 100 to 3200 across pairs that all fail 8 of 8; near-synonym pairs sit ten times below distinct-animal pairs | [does-interaction-strength-predict-which-pairs-blend.md](does-the-fix-reach-unseen-pairs/does-interaction-strength-predict-which-pairs-blend.md) |
| Which checkpoint of the pooled adapter composes cat x dog best, and does more correction or an early-only window help? | ✅ 7 of 8 at rank 8 @ 30k, rank 16 @ 50k, rank 32 @ 30k to 90k · ⚪ null on λ above 1 and on the early window | seeds composing of 8 at λ 1, all 50 steps: 7 / 7 / 7 against 0 for plain PoE; λ 1.2 to 2.0 never adds a seed on a healthy checkpoint; the early window (steps 0 to 9) reaches at most 5 of 8 | [which-checkpoint-composes-best-and-does-more-correction-help.md](does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md) |
| Does training the pooled adapter longer keep improving the held-out renders? | ❌ no: fidelity decays while the weights grow without bound | 8-seed DINOv2 drift at λ 1 (negative = nearer the joint-prompt image): rank 32 −0.091 at 30k to −0.013 at 90k; rank 8 −0.132 at 30k to +0.041 at 280k; LoRA weight norm rank 8 35.9 at 10k to 93.2 at 450k with weight decay 0 | [does-training-longer-keep-improving-the-held-out-fix.md](does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md) |
| Does choosing among the product's own proposals, with no correction added, produce two animals? | ❓ inconclusive | compose fraction over all 12 particles at step 100k: Mono 1.0, PoE control 0.0, twisted SMC 0.0 (10 detector hits of 132 SMC particles at 40k to 80k, each one fused animal by eye); the twist's validation accuracy 0.56, under the 0.60 bar | [does-selecting-among-poe-proposals-compose.md](is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md) |
| Does steering the plain product's particles on the compose scorer (Feynman-Kac steering, Singhal et al.) find a composing proposal? | ⚪ null | cat × dog, seeds 9 to 16: compose rate 0.0 for steering at K 4 and K 16 against 0.0 for the unweighted control and 1.0 for Mono; 0 of 128 unweighted particles composed; the step-10 reward agrees with the final verdict on 0.99 of particles, so the reward was not blind | [does-steering-on-the-scorer-find-a-composing-proposal.md](is-the-gap-the-samplers-or-the-models/does-steering-on-the-scorer-find-a-composing-proposal.md) |
| Does SuperDiff compose at its own defaults, and what is it missing when it does not? | ⚪ null at 200 steps; ✅ the missing piece is the same residual | cat × dog, seed 9: detector count 1 (blend) at 200 steps, 2 (compose) at 50; adding back its own residual `r_t^SD` separates 4 of 4 seeds by λ 0.75 (cat × dog) and by 0.25 (butterfly × meadow) | [does-superdiff-compose-at-its-own-defaults.md](is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md) |
| Does the PoE-trained correction carry into SuperDiff? | ⚪ null, it hurts | first separating λ "never" on 12 of 12 seeds against 0.5 to 0.75 for SuperDiff alone; median cosine between the adapter's correction and SuperDiff's missing residual 0.10 / 0.34 / 0.18 / 0.06 by step bucket, norms the same order | [does-the-poe-trained-correction-carry-into-superdiff.md](is-the-gap-the-samplers-or-the-models/does-the-poe-trained-correction-carry-into-superdiff.md) |
| Does an adapter trained on SuperDiff's own residual compose inside SuperDiff? | ❓ inconclusive: separates early, smears with training, runs stopped at 61k/62k/36k of 100k | seeds of 4 with two bodies at λ 1, cat × dog: rank 8 peaks at 4 (30k) and holds 3 at 60k; rank 16 4 at 10k, 0 from 50k; rank 32 4 at 10k, 0 from 20k; loss plateaus from 30k to 40k without a matching rise | [does-an-adapter-trained-on-superdiffs-own-residual-compose.md](is-the-gap-the-samplers-or-the-models/does-an-adapter-trained-on-superdiffs-own-residual-compose.md) |
| Does searching over the initial noise, with the compose scorer as verifier, produce two animals? | ❓ inconclusive on the pre-registered bar; a null in everything but its letter | cat × dog seeds 9 to 16: random search over the 8 seeds finds 0 composing; one round of zero-order search (8 candidates per seed at σ 0.1 and 0.3, plain PoE, DDIM eta 0) keeps a counted compose on 1 of 8 seeds at σ 0.1 (0.125 against a 0.10 null margin and 0.25 support margin) and 0 of 8 at σ 0.3; the one counted image is one body with two heads; the scorer read on x0-hat at step 10 keeps 0 of 8 at both σ and loses the butterfly on 3 of 8 control seeds | [does-searching-over-the-initial-noise-compose.md](is-the-gap-the-samplers-or-the-models/does-searching-over-the-initial-noise-compose.md) |
| Does searching over the initial noise, with the correction attached, give a crisper two-animal render? | ✅ support on the pre-registered bar; half the seeds by eye | cat × dog seeds 9 to 16, rank-32 step-30050 adapter at λ 1.2: the adapter's own render composes 6 of 8, the best of 8 nearby noises 8 of 8 at σ 0.1 and 0.3 (the count-then-confidence rule reaches the same 8 of 8); median per-seed sharpness ratio kept over adapter 1.47 and 2.88 against a 1.10 bar; by eye a cleaner cat and dog on 4 of 8 seeds, equal on 2, worse on 2 where the edge measure picks drawn texture; the adapter itself loses the butterfly on 4 of 8 control seeds by count | [does-searching-over-the-noise-sharpen-the-corrected-render.md](is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md) |
| Does a Langevin corrector remove part of the correction and leave part of it? | ❓ inconclusive at every step size tried | read-zone mean of `‖eps_J − eps_PoE‖ / ‖eps_PoE‖` over the last five steps, cat × dog seed 9 at step-size multiplier 3, by corrector count 0 to 200: 0.153, 0.138, 0.261, 0.203, 0.103, 0.125; the two largest counts differ by 21% against the 5% bar, and the same uncorrected cell reads 0.153, 0.169 and 0.180 on three GPUs | [does-a-langevin-corrector-remove-part-of-the-correction.md](is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md) |
| Does a Langevin corrector, adding nothing to the score, produce two animals? | ⚪ null | cat × dog: 0 of 4 seeds composed in every one of the nine ten-step corrector windows and with the corrector on all 50 steps; on the eight held-out seeds joint prompt 8 of 8, plain product-of-experts 0 of 8, corrector 0 of 8; the control pair holds 8, 8, 8 | [does-a-corrector-alone-produce-two-animals.md](is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md) |
| Can a corrector, or handing the tail to the frozen model, sharpen the adapter's renders? | ⚪ null on both bars | corrector on the adapter's own tail: mean Laplacian variance 57.1 to 61.7 over 8 seeds, +8.0% against a 10% bar, composed 7, 8, 7 of 8; clean tail: DINOv2 distance to the joint render 0.472 (adapter alone) to 0.434 at best, a gain of 0.038 against a 0.05 bar, composition held within one seed | [can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md](is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) |
| Does a Langevin corrector on the product-of-experts score remove part of the correction and leave part of it? | ❓ inconclusive at all three step sizes | cat × dog seed 9, the correction's size relative to the product-of-experts prediction over the last five denoising steps, by corrector count 0 to 200 at step-size multiplier 3: 0.153, 0.138, 0.261, 0.203, 0.103, 0.125; the 100-step and 200-step values differ by 21% against the 5% bar, and the same uncorrected cell reads 0.153, 0.169 and 0.180 on three GPUs, so one seed's scatter exceeds every bar | [does-a-langevin-corrector-remove-part-of-the-correction.md](is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md) |
| Does a Langevin corrector, adding nothing to the score, produce two animals? | ⚪ null | cat × dog: 0 of 4 seeds composed in every one of the nine ten-step corrector windows and in the all-50 column (40 renders, every one counting a single animal), and 0 of 8 held-out seeds against plain product-of-experts' 0 of 8 and the joint prompt's 8 of 8; the control pair holds 8 of 8 | [does-a-corrector-alone-produce-two-animals.md](is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md) |
| Can a corrector, or handing the tail to the frozen model, sharpen the rank-32 adapter's renders? | ⚪ null on both bars | corrector on the adapter's own score over steps 35 to 49, 8-seed mean Laplacian variance: 57.1, 61.1, 61.7 at 0, 5, 20 steps, a rise of 8% against the 10% bar, composed 7, 8, 7 of 8; the adapter switched off after step 29 with a frozen tail moves the 8-seed mean DINOv2 distance to the joint render from 0.472 to 0.434, a gain of 0.038 against the 0.05 bar, and adding corrector steps there moves it back to 0.490 | [can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md](is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) |

## Figures still missing

The evidence pairs waiting on a render, so they are visible rather than quietly absent.

None. Every finding's evidence pairs have their render.

## Reading the pictures

Each finding's figures are read one at a time, in plain words, in a figure explainer beside the
images: [where each condition lands, what these pictures mean](../artifacts/results/where-does-each-condition-land/figure-explainer.md)
[is the gap the sampler's or the model's, what these pictures mean](../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer.md)
and [what the correction is made of, what these pictures mean](../artifacts/results/what-the-correction-is-made-of/figure-explainer.md).
The path from two X posts to the twisted-SMC run, for a reader who was not in that conversation, is
[the note beside those pictures](../artifacts/notes/from-an-x-post-to-a-twisted-smc-baseline/note.md).
How a finding, its card and its explainer are laid out, with that one as the worked instance, is
in [the finding-and-explainer template](finding-and-explainer-template.md).

## The question groups

One folder per high-level research question, mirroring the `artifacts/results/` grouping the
findings embed from; a finding with no sibling yet sits flat.

| Group | The question it answers | Findings |
|---|---|---|
| [does-the-fix-reach-unseen-pairs/](does-the-fix-reach-unseen-pairs/) | Why does the pooled adapter's correction carry to some unseen pairs and not others | 2 |
| [does-training-longer-help-the-pooled-lora/](does-training-longer-help-the-pooled-lora/) | Which checkpoint of the pooled adapter to show, and what training past it does to the held-out renders | 2 |
| [when-does-the-outcome-lock-in/](when-does-the-outcome-lock-in/) | When does a run commit to one animal or two, where does each condition end up, and what is the correction made of | 2 |
| [is-the-gap-the-samplers-or-the-models/](is-the-gap-the-samplers-or-the-models/) | Is the gap the samplers or the models | 10 |

## Still open

- [ ] Six older documents in this folder predate the finding format. Against the format's
      boundary with the results folder and this repo's own `CLAUDE.md` (which keeps
      pre-registrations, instrument provenance and results summaries here), they sort as:
      `experiments-log.md` and `normalization_preregistration.md` are pre-registrations and stay;
      `instrument_smoke.md` is instrument provenance and stays; `RESULTS_SUMMARY.md` is a results
      summary and stays, though every verdict it carries is a candidate finding.
      `decision-timeline.md` is a narrative of decisions in order, which the format files under
      the plan tree's history (`CHANGELOG.md` or `artifacts/notes/`), and
      `paper-evidence-index.md` is the manuscript's figure register, which belongs beside
      `paper/iclr/figures.md`. Both carry inbound links from plans, context and the archive, so
      the move is a `tidy-repo` pass with a `RENAMES.md` row each, not a quiet relocation.
- [ ] The dose-response result (root step 4), the transfer result (root step 10) and the
      three-sides result (root step 7) have review verdicts and figures under
      `artifacts/results/` and no finding file.
- [ ] The literature verdict on the three remaining failures (blend, one-animal basin, late haze)
      lives in [the pressure-test route](../artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md)
      and is a reading, not a measured finding, until a run tests one of its predictions.
- [ ] The energy-penalty null (root step 64, scope 01 plan 22: a Girsanov-weighted running cost on the
      adapter's correction cut its on-policy energy 27% with the fit kept and moved the held-out renders
      toward plain PoE; the adapter already spends half the true correction's energy) has its verdict in
      the review file, its figures under `artifacts/results/does-charging-the-adapter-for-its-energy-sharpen-the-fix/`
      and W&B run `2cfdtdnl`, and no finding file yet.
- [ ] No runbook recipe regenerates the twisted-SMC number.
- [ ] The cross-device fp16 pixel difference (about 2 of 255) quoted in
      [which checkpoint composes best](does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md)
      is from the session record, not a file on disk.
- [ ] The decay finding infers its cause (norm growth under weight decay 0); no run has varied
      weight decay or the learning rate, and the trainer has no flag for weight decay yet.
- [ ] The SuperDiff-residual adapters' pre-registered λ sweep on finished checkpoints never ran (trainings stopped at 61k/62k/36k of 100k); the stopped checkpoints and the cache stay under `corrector/superdiff/`.
- [ ] Two numbers in the SuperDiff findings are lifted from review files rather than re-read from scorer output: the eight-cell detector counts and the cache check; each finding's `Still open` names the file to open.
- [ ] The landing figures' three open rungs: RAE axis pictures, per-step tracks with a commit
      step, and the unseen-pair render with trajectories saved (root step 52, tasks 2 to 4).
