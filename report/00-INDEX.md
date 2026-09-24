# 🔬 Report: poe_repair_min

What this project found, one file per question, every claim paired with a figure and the
statistic that backs it. What the work *means* lives in [context/](../context/00-INDEX.md),
how to redo it in [runbook/](../runbook/00-INDEX.md); this folder is only the findings.

## Read it in this order

The table below is a lookup, sorted by nothing. This section is the thread: eight rungs, each one
the question you can only ask because the rung before it was answered. Read it top to bottom once
and the rest of the folder becomes navigable.

A rung marked ⬜ has its verdict and its figures on disk but no finding file written yet, so it
names the review file and the evidence folder instead. Those are the holes, and they are at the
front of the arc rather than the back.

**1. What product-of-experts is for, and when it works**

Multiplying two experts' predictions is supposed to put both things in one picture, and sometimes
it does. The control pair used throughout the project, a butterfly over a meadow, composes on 8 of
8 held-out seeds, and the joint prompt (one sentence naming both animals, no product at all) also
composes 8 of 8. Those two columns are what every failure below is measured against.

⬜ No finding file. The successful columns live inside
[does a corrector alone produce two animals](is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md)
as the control, never on their own. What the method is and what a compose rate counts is in
[what product-of-experts composition is](../context/world/poe-composition.md) and
[the compose rate](../context/world/compose-rate.md).

**2. Where it breaks, and how we know it broke**

Ask for a cat and a dog and the product renders one fused animal: a cat's face on a dog's body.
This is the failure the whole project is about, and it is not a one-off. It recurs across 17 animal
pairs at 8 seeds each.

⬜ No finding file. The gallery of 136 uncorrected renders, one per pair per seed with its sampler
settings beside it, is [the blend gallery](../artifacts/results/poe-blends-instead-of-composing/README.md).
The instrument that decides a render failed, a GroundingDINO instance count validated against
labelled images, is [the compose-score validation set](../artifacts/results/can-we-trust-the-compose-score/README.md),
with its verdict in [what the current benchmarks score](../plans/02-can-we-trust-the-compose-rate/review/03-what-the-current-benchmarks-score.md).
Read the instrument before any rate below, because every number in this folder is counted by it.

**3. The correction exists, and it is what causes composition**

Subtract the plain product's prediction from the joint prompt's, step by step, and what is left is
the correction. Add it back and the pair composes. Four checks say the effect is the correction and
not the act of perturbing anything: more of it composes more, matched substitutes of the same size
pointing elsewhere do nothing, it only works in an early window, and the same story holds read from
three sides.

⬜ No finding file, three verdicts. The amount question is in
[more correction, more composition](../plans/03-does-the-correction-cause-composition/review/03-more-correction-more-composition.md)
with figures under [how much correction is needed](../artifacts/results/how-much-correction-is-needed/README.md);
the timing question is in [when in the run it matters](../plans/03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md)
with figures under [when the correction must arrive](../artifacts/results/when-the-correction-must-arrive/README.md);
the direction and substitute controls are in
[the same story from three sides](../plans/03-does-the-correction-cause-composition/review/06-the-same-story-from-three-sides.md)
with figures under [does the interaction term cause composition](../artifacts/results/does-the-interaction-term-cause-composition/README.md).

**4. What the correction is made of, and where it moves the picture to**

Two findings, and the first one written should be read second. Together they answer whether the
product could have supplied this correction by re-weighting predictions it already has (mostly yes
in principle, not at the weights it uses), and whether the correction moves the render toward the
joint prompt in the scorer's own image space (yes on cat × dog, overlapping bands on held-out
pairs).

- [What is the correction made of, and could PoE have supplied it?](when-does-the-outcome-lock-in/what-is-the-correction-made-of.md) ❓ inconclusive on the pre-registered bar
- [Where does each condition land in the scorer's image space?](when-does-the-outcome-lock-in/where-does-each-condition-land.md) ❌ null on both pre-registered bars, ✅ support on the post-hoc reads

**5. An adapter learns the correction, and which one to show**

The correction is computed from a joint prompt the deployed method never gets to see, so it is not
a fix on its own. A low-rank adapter trained to predict it is. These two say which checkpoint
composes best and what happens if you keep training past it.

- [Which checkpoint composes best, and does more correction help?](does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md) ✅ 7 of 8 seeds, ⚪ null on more correction and on the early-only window
- [Does training longer keep improving the held-out fix?](does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md) ❌ no: fidelity decays while the weights grow without bound

**6. Whether the fix reaches pairs it never trained on**

The adapter is trained on a pool of pairs and asked about pairs outside it. It carries to some and
not others, and these two ask why: is the gap a fit problem, a drift problem or a property of the
pair, and can the cache predict which pairs will blend before you render anything.

- [Is the held-out gap a fit, a drift, or a pair problem?](does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md) ✅ pair problem, ⚪ null on drift
- [Does interaction strength predict which pairs blend?](does-the-fix-reach-unseen-pairs/does-interaction-strength-predict-which-pairs-blend.md) ⚪ null

⬜ The transfer rate over fifteen pairs has its verdict in
[does one pooled fix transfer at all](../plans/04-does-the-fix-reach-unseen-pairs/review/03-does-one-pooled-fix-transfer-at-all.md)
and its figures under [does the fix reach unseen pairs](../artifacts/results/does-the-fix-reach-unseen-pairs/README.md),
with no finding file.

**7. Everything tried instead, and what each one ruled out**

Twelve findings, and the reason there are twelve is that a reviewer will ask whether a cheaper fix
works. Most are ways of getting two animals without adding a learned correction; the last two in
the search family keep the correction and spend inference compute on top of it, and the last of all
asks not whether a cheaper fix composes but whether any of them repairs the picture quality the
correction costs. They are ordered by family, and ten of the twelve are nulls or inconclusive. The
two that support both buy their result with 16 or more draws per seed, and neither reaches past what
the correction already delivers at full dose, which is the point: the correction is not obviously
replaceable.

*Correctors, which re-sample the latent without changing the score:*

- [Does a corrector alone produce two animals?](is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md) ⚪ null
- [Does a Langevin corrector remove part of the correction?](is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md) ❓ inconclusive at every step size
- [Can a corrector or a clean tail sharpen the adapter's renders?](is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) ⚪ null on both bars
- [Can any render-time fix recover the picture quality the adapter costs?](is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md) ❓ inconclusive: eight fixes, the edge measure passes two of them, the pictures do not

*A different composition method, SuperDiff, and whether its residual is the same object:*

- [Does SuperDiff compose at its own defaults?](is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md) ⚪ null at 200 steps, ✅ the missing piece is the same residual
- [Does the PoE-trained correction carry into SuperDiff?](is-the-gap-the-samplers-or-the-models/does-the-poe-trained-correction-carry-into-superdiff.md) ⚪ null, it hurts
- [Does an adapter trained on SuperDiff's own residual compose?](is-the-gap-the-samplers-or-the-models/does-an-adapter-trained-on-superdiffs-own-residual-compose.md) ❓ inconclusive, runs stopped early

*Search and selection, which spend compute at inference instead of training anything:*

- [Does selecting among the product's own proposals compose?](is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md) ❓ inconclusive
- [Does steering on the scorer find a composing proposal?](is-the-gap-the-samplers-or-the-models/does-steering-on-the-scorer-find-a-composing-proposal.md) ⚪ null
- [Does steering the corrected sampler produce a clean cat and dog?](is-the-gap-the-samplers-or-the-models/does-steering-the-corrected-sampler-produce-clean-pairs.md) ✅ support at half dose; ⚪ null on what the resampling itself adds
- [Does searching over the initial noise compose?](is-the-gap-the-samplers-or-the-models/does-searching-over-the-initial-noise-compose.md) ❓ inconclusive, a null in everything but its letter
- [Does searching over the noise sharpen the corrected render?](is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md) ✅ support, 8 of 8 seeds

**8. What is still open**

The one result with a verdict, figures and a W&B run and no home in the arc yet is the energy
penalty: charging the adapter for the size of its correction cut its on-policy energy 27% and moved
the held-out renders back toward plain product-of-experts. Its verdict is in the scope 01 review
file, its figures under
[does charging the adapter for its energy sharpen the fix](../artifacts/results/does-charging-the-adapter-for-its-energy-sharpen-the-fix/),
and the run is `2cfdtdnl`. The rest of the holes are in [Still open](#still-open) at the foot of
this file.

**The illustrated version** is two pictures waiting to be rendered, in
[the report's illustrated map](diagram-prompts.md): the eight rungs above as one line, and what a
single corrector step does inside one noise level.

## What was found

| The question | Verdict | The number | The finding |
|---|---|---|---|
| Can real photographs of both animals train the composition instead of the model's own joint-prompt prediction? | ❌ dead: the corpus is reachable and unusable | of the images an automatic filter accepted, 7 of 16 cat × dog are genuine photographs of both animals and 0 of 11 turtle × tortoise are, the rest watermarked stock and merchandise; fed single-species images the same filter reports both species at 0.323 for turtle × tortoise against a 0.10 bar | [05-training-on-real-photographs.md](designing-the-correction-loss/05-training-on-real-photographs.md) |
| Where does each condition land in the scorer's image space, and does the correction move PoE toward the joint prompt? | ❌ null on both pre-registered bars; ✅ support on the post-hoc reads | cat × dog both-ness (projection toward the joint-prompt centroid, DINOv2 cosine units, mean of 8 seeds): PoE 0.21, corrected 0.42, joint 0.52; on the 8 held-out pairs the both-ness bands overlap on 5 of 7 counted pairs (null), while the instance count on the same images reads PoE 0.00 to 0.25 and corrected 0.75 to 1.00 on every pair (post-hoc) | [where-does-each-condition-land.md](when-does-the-outcome-lock-in/where-does-each-condition-land.md) |
| What is the correction made of, and could PoE have supplied it by re-weighting its own predictions? | ❓ inconclusive on the pre-registered bar | orthogonal share of the correction's squared norm outside the span of the three predictions PoE has, mean over seeds 9 to 16 and steps 0 to 10: 0.374 against bars at 0.5 and 0.25; the reachable part weights each expert at 1 to 3 where PoE weights 7.5; the adapter's orthogonal share 0.19 against the target's 0.37 | [what-is-the-correction-made-of.md](when-does-the-outcome-lock-in/what-is-the-correction-made-of.md) |
| Is the held-out gap of the pooled rank 8 adapter a fit, a drift, or a pair problem? | ✅ support: pair problem; ⚪ null on drift | fit cosine on cached states: training pairs 0.97, seen words in an unseen pairing 0.84, no seen word 0.80; on the adapter's own corrected trajectory 0.82 against 0.86 cached, cat x dog | [is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md](does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md) |
| Does the interaction strength read from the cache (Mitra et al. Theorem 1 turned around) predict which pairs blend? | ⚪ null | lower bound on G·M over steps 10 to 40 spans 100 to 3200 across pairs that all fail 8 of 8; near-synonym pairs sit ten times below distinct-animal pairs | [does-interaction-strength-predict-which-pairs-blend.md](does-the-fix-reach-unseen-pairs/does-interaction-strength-predict-which-pairs-blend.md) |
| Which checkpoint of the pooled adapter composes cat x dog best, and does more correction or an early-only window help? | ✅ 7 of 8 at rank 8 @ 30k, rank 16 @ 50k, rank 32 @ 30k to 90k · ⚪ null on λ above 1 and on the early window | seeds composing of 8 at λ 1, all 50 steps: 7 / 7 / 7 against 0 for plain PoE; λ 1.2 to 2.0 never adds a seed on a healthy checkpoint, and on the shipped rank-32 step-30050 checkpoint it changes the scene instead (7 / 6 / 6 / 7 / 6 at λ 1.0 / 1.2 / 1.3 / 1.5 / 2.0, with one seed's animals shrinking out of the detector's reach); the early window (steps 0 to 9) reaches at most 5 of 8 | [which-checkpoint-composes-best-and-does-more-correction-help.md](does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md) |
| Does training the pooled adapter longer keep improving the held-out renders? | ❌ no: fidelity decays while the weights grow without bound | 8-seed DINOv2 drift at λ 1 (negative = nearer the joint-prompt image): rank 32 −0.091 at 30k to −0.013 at 90k; rank 8 −0.132 at 30k to +0.041 at 280k; LoRA weight norm rank 8 35.9 at 10k to 93.2 at 450k with weight decay 0 | [does-training-longer-keep-improving-the-held-out-fix.md](does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md) |
| Does choosing among the product's own proposals, with no correction added, produce two animals? | ❓ inconclusive | compose fraction over all 12 particles at step 100k: Mono 1.0, PoE control 0.0, twisted SMC 0.0 (10 detector hits of 132 SMC particles at 40k to 80k, each one fused animal by eye); the twist's validation accuracy 0.56, under the 0.60 bar | [does-selecting-among-poe-proposals-compose.md](is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md) |
| Does steering the plain product's particles on the compose scorer (Feynman-Kac steering, Singhal et al.) find a composing proposal? | ⚪ null | cat × dog, seeds 9 to 16: compose rate 0.0 for steering at K 4 and K 16 against 0.0 for the unweighted control and 1.0 for Mono; 0 of 128 unweighted particles composed; the step-10 reward agrees with the final verdict on 0.99 of particles, so the reward was not blind | [does-steering-on-the-scorer-find-a-composing-proposal.md](is-the-gap-the-samplers-or-the-models/does-steering-on-the-scorer-find-a-composing-proposal.md) |
| Does selecting among the corrected sampler's own draws produce a clean cat and dog? | ✅ support on the pre-registered bar; ⚪ null on what the resampling itself adds | cat × dog, seeds 9 to 16, rank-32 correction at λ 0.5 inside the proposal: 8 of 8 seeds read as two animals for steering at K 16 against 2 of 8 for the unweighted first particle and 3 of 8 for the correction alone, a gap of 0.75 against a 0.25 bar; the best of the same 16 unweighted particles is also 8 of 8, so the draws compose and the weights only choose which one survives (mean ImageReward +0.75 steered, +0.45 best of 16, +0.24 the joint prompt, −0.77 the correction alone); at λ 1.2 the control is already 8 of 8 and the gap is 0.00 | [does-steering-the-corrected-sampler-produce-clean-pairs.md](is-the-gap-the-samplers-or-the-models/does-steering-the-corrected-sampler-produce-clean-pairs.md) |
| Does SuperDiff compose at its own defaults, and what is it missing when it does not? | ⚪ null at 200 steps; ✅ the missing piece is the same residual | cat × dog, seed 9: detector count 1 (blend) at 200 steps, 2 (compose) at 50; adding back its own residual `r_t^SD` separates 4 of 4 seeds by λ 0.75 (cat × dog) and by 0.25 (butterfly × meadow) | [does-superdiff-compose-at-its-own-defaults.md](is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md) |
| Does the PoE-trained correction carry into SuperDiff? | ⚪ null, it hurts | first separating λ "never" on 12 of 12 seeds against 0.5 to 0.75 for SuperDiff alone; median cosine between the adapter's correction and SuperDiff's missing residual 0.10 / 0.34 / 0.18 / 0.06 by step bucket, norms the same order | [does-the-poe-trained-correction-carry-into-superdiff.md](is-the-gap-the-samplers-or-the-models/does-the-poe-trained-correction-carry-into-superdiff.md) |
| Does an adapter trained on SuperDiff's own residual compose inside SuperDiff? | ❓ inconclusive: separates early, smears with training, runs stopped at 61k/62k/36k of 100k | seeds of 4 with two bodies at λ 1, cat × dog: rank 8 peaks at 4 (30k) and holds 3 at 60k; rank 16 4 at 10k, 0 from 50k; rank 32 4 at 10k, 0 from 20k; loss plateaus from 30k to 40k without a matching rise | [does-an-adapter-trained-on-superdiffs-own-residual-compose.md](is-the-gap-the-samplers-or-the-models/does-an-adapter-trained-on-superdiffs-own-residual-compose.md) |
| Does searching over the initial noise, with the compose scorer as verifier, produce two animals? | ❓ inconclusive on the pre-registered bar; a null in everything but its letter | cat × dog seeds 9 to 16: random search over the 8 seeds finds 0 composing; one round of zero-order search (8 candidates per seed at σ 0.1 and 0.3, plain PoE, DDIM eta 0) keeps a counted compose on 1 of 8 seeds at σ 0.1 (0.125 against a 0.10 null margin and 0.25 support margin) and 0 of 8 at σ 0.3; the one counted image is one body with two heads; the scorer read on x0-hat at step 10 keeps 0 of 8 at both σ and loses the butterfly on 3 of 8 control seeds | [does-searching-over-the-initial-noise-compose.md](is-the-gap-the-samplers-or-the-models/does-searching-over-the-initial-noise-compose.md) |
| Does searching over the initial noise, with the correction attached, give a crisper two-animal render? | ✅ support on the pre-registered bar; half the seeds by eye | cat × dog seeds 9 to 16, rank-32 step-30050 adapter at λ 1.2: the adapter's own render composes 6 of 8, the best of 8 nearby noises 8 of 8 at σ 0.1 and 0.3 (the count-then-confidence rule reaches the same 8 of 8); median per-seed sharpness ratio kept over adapter 1.47 and 2.88 against a 1.10 bar; by eye a cleaner cat and dog on 4 of 8 seeds, equal on 2, worse on 2 where the edge measure picks drawn texture; the adapter itself loses the butterfly on 4 of 8 control seeds by count | [does-searching-over-the-noise-sharpen-the-corrected-render.md](is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md) |
| Does adapting self-attention as well as cross-attention change the result? | ❓ inconclusive: the available comparison is mixed | DINOv2 drift at step 50,000, six-cell mean: cross and self-attention -0.269 against -0.329 for the best cross-attention run; best on cat and dog seed 9 (-0.527 against -0.488), worst on elephant and penguin seed 9 (+0.174 against -0.234); the cross-and-self run also decays its rate and resumed at step 5,000, so two things moved beside the layers | [does-adapting-self-attention-change-the-result.md](what-the-adapter-ablations-show/does-adapting-self-attention-change-the-result.md) |
| How much rank does the correction need? | ❓ inconclusive: no trend in rank, every comparison mixed | DINOv2 drift, cat and dog seed 9 at ranks 8, 16, 32: -0.252, -0.516, -0.408; seed 10: +0.222, +0.428, +0.319; the three runs differ in loss and length as well as rank; trainable parameters 0.19%, 0.38% and 0.77% of the model, and 1.38% with self-attention added | [how-much-rank-the-correction-needs.md](what-the-adapter-ablations-show/how-much-rank-the-correction-needs.md) |
| Does the joint prompt always draw both concepts? | ❌ no | 11 failing cells of 32 rendered joint prompts on held-out pairs: chess board and hourglass has no hourglass on all 5 seeds, cat and fox has no cat on 4 of 6, dog and dog puts a person in the frame at seed 1, elephant and penguin drops the penguin at seed 9; on the chess cell at seed 10 the plain product holds both objects where the target holds one | [does-the-joint-prompt-always-draw-both-concepts.md](is-the-adapter-just-copying-the-joint-prompt/does-the-joint-prompt-always-draw-both-concepts.md) |
| Does the adapter return two objects when both prompts name the same concept? | ✅ support | one concept asked for twice, 5 cells: the plain product draws one animal in 5 of 5, the adapter draws two in 4 of 5 (the exception is dog and dog at seed 1, where the second animal is horse-like), the joint prompt draws two in 3 of 5; adapter v58-03-self at step 50,000, correction on all 50 steps | [does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md](is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md) |
| Does a Langevin corrector remove part of the correction and leave part of it? | ❓ inconclusive at every step size tried | read-zone mean of `‖eps_J − eps_PoE‖ / ‖eps_PoE‖` over the last five steps, cat × dog seed 9 at step-size multiplier 3, by corrector count 0 to 200: 0.153, 0.138, 0.261, 0.203, 0.103, 0.125; the two largest counts differ by 21% against the 5% bar, and the same uncorrected cell reads 0.153, 0.169 and 0.180 on three GPUs | [does-a-langevin-corrector-remove-part-of-the-correction.md](is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md) |
| Does a Langevin corrector, adding nothing to the score, produce two animals? | ⚪ null | cat × dog: 0 of 4 seeds composed in every one of the nine ten-step corrector windows and with the corrector on all 50 steps; on the eight held-out seeds joint prompt 8 of 8, plain product-of-experts 0 of 8, corrector 0 of 8; the control pair holds 8, 8, 8 | [does-a-corrector-alone-produce-two-animals.md](is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md) |
| Can a corrector, or handing the tail to the frozen model, sharpen the adapter's renders? | ⚪ null on both bars | corrector on the adapter's own tail: mean Laplacian variance 57.1 to 61.7 over 8 seeds, +8.0% against a 10% bar, composed 7, 8, 7 of 8; clean tail: DINOv2 distance to the joint render 0.472 (adapter alone) to 0.434 at best, a gain of 0.038 against a 0.05 bar, composition held within one seed | [can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md](is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) |
| Can any render-time fix recover the picture quality the adapter costs? | ❓ inconclusive: the edge measure passes two of them, the pictures do not | median Laplacian variance over 8 held-out cells (greyscale 0 to 1, ×1e-4), as a ratio to the adapter's own render: repaint at 0.25 1.14 and at 0.40 1.24, adapter for 20 steps 1.09, guidance 10 1.45 and 12 1.80, experts weighted 3.0 and 3.0 0.78, the joint prompt itself 1.18, against a 1.10 bar borrowed from the noise-search finding after these images existed; by eye the re-weighting smears both subjects on 8 of 8 cells and guidance changes what is in the picture on 3 of 8 | [does-any-render-time-fix-recover-picture-quality.md](is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md) |

## Figures still missing

The evidence pairs waiting on a render, so they are visible rather than quietly absent.

None. Every finding's evidence pairs have their render.

## Reading the pictures

Each finding's figures are read one at a time, in plain words, in a figure explainer beside the
images: [where each condition lands, what these pictures mean](../artifacts/results/where-does-each-condition-land/figure-explainer.md)
[is the gap the sampler's or the model's, what these pictures mean](../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer.md)
[what the correction is made of, what these pictures mean](../artifacts/results/what-the-correction-is-made-of/figure-explainer.md)
[the corrector figures, what these pictures mean](../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer-the-langevin-corrector.md)
and [which fidelity fix wins, what these pictures mean](../artifacts/results/which-fidelity-fix-wins/figure-explainer.md).
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
| [is-the-gap-the-samplers-or-the-models/](is-the-gap-the-samplers-or-the-models/) | Is the gap the samplers or the models | 12 |
| [what-the-adapter-ablations-show/](what-the-adapter-ablations-show/) | Which layers the adapter needs and how much rank the correction takes | 2 |
| [is-the-adapter-just-copying-the-joint-prompt/](is-the-adapter-just-copying-the-joint-prompt/) | Whether the target itself is right, and what the adapter adds beyond it | 2 |

## The documents that are not findings

Six documents here predate the finding format and answer no question of their own. Two of them are
load-bearing anyway: the bars several findings are judged against were written in them, before the
runs. Where each one should eventually be filed is argued in [Still open](#still-open); this
section is only so a reader can reach them.

**[The stage-by-stage results summary](RESULTS_SUMMARY.md)**

One screen per stage of the work as it stood on 2026-07-22: status, what the stage produced, its
W&B runs, and commands that walk the outputs. Every verdict in it is a candidate finding that has
not been written as one. Its opening warning still bites: artifacts are split across the repo and
`/datasets`, neither root complete, so a bare `artifacts/...` path resolves for some runs and
silently fails for others.

**[The commitment-timing pre-registration](experiments-log.md)**

Written before any of those runs. It fixes the axes (pair, seed, step, condition, space, adapter
strength) and the falsification rule for each of the five experiments, and it says the rules are
not revised after seeing results. Read it before quoting a bar from rungs 3 and 4 of the arc.

**[How correction size is measured](normalization_preregistration.md)**

Committed 2026-08-05, before any cross-type plot existed. It pins the measure to `‖r_t‖` over
`‖ε̃_PoE‖`, median over steps for a per-run number then median over seeds for a per-pair number,
every tensor upcast to fp32 first. Medians rather than means because the early steps carry heavy
fp16 cancellation noise. Every size number in this folder is measured this way, so this one is
still live rather than historical.

**[What the measuring scripts do on real data](instrument_smoke.md)**

The recorded output of each measuring script on `a_cat__x__a_dog` seed 9, with the command above
each result, dated 2026-08-05 on node mscluster85. This is instrument provenance: it is what says
the scripts were checked against real cells rather than assumed to work.

**[The decision timeline](decision-timeline.md)**

The account, in order, of what each objective asked, what ran, what came back, and the decision it
forced, append-only with superseding banners. The arc above deliberately does not follow this
order, because reading order and the order the work happened are different things.

**[What the paper does with each piece of evidence](paper-evidence-index.md)**

One row per evidence folder, saying which figure of the manuscript it feeds and what state it is
in. It carries two live warnings the findings do not: the shared-structure argument behind figure 6
does not stand once the control accounts for the spread in `‖r_t‖`, and several captions have to be
rewritten before the paper may use their result.

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
