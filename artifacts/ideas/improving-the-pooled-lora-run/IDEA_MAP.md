# 💡 Improving the pooled LoRA run

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1 | wrong as stated, split; 1a held |  **run 1: the on-cache gap is 0.17 and lives in steps 25 to 49; fit does not predict which held-out seeds fail. Run 2: on the adapter's own corrected path the fit is 0.82 against 0.86 on the cache, so leaving the cache costs 0.04 and the adapter has learned a rule that holds off-cache. The render-time 0.54 and 0.40 are cos(Δ̂, pool-mean Δ̄), a shared-component share, not a fit. Split: 1a more pairs raise the on-cache late-step fit (test: the 20-new-pair cache); 1b closed: exposure bias is 0.04, on-policy training is demoted to a polish** |
| **2 (current). pair diversity, not more seeds, is the axis that moves the held-out gap** | **needs a check, reworded 2026-09-08: "adding the three cached pairs that contain neither cat nor dog helps", tested by parent vs C4. The interaction bound is measured and reported beside the results and makes no claim until examples are laid against it** | |
| 3. the late fidelity decay is an optimizer artefact (weight decay 0, constant LR), and weight decay plus early stopping on held-out drift removes it | open, sharpened by route 1: norm growth plus off-trajectory evaluation, rank 32 first because alpha = rank | |
| 4. token-disjointness can be relaxed inside the train pool without contaminating the held-out test** | **holds inside train, and unnecessary against the held-out test: three cached high-bound pairs (lion x horse, bear x salmon, wolf x horse; 29 cells at 1583 to 4113) touch neither cat nor dog** | |
| 5. filtering training cells by the quality of their mono target improves the learned correction | open | |
| 6. every change is made one axis at a time | open | |
| 7. cells whose joint-prompt render does not itself show both concepts are teaching the wrong target | **split. The observation is verified: on every visually separable pool pair, many targets show one concept twice. The training-target reading is unproven and demoted from load-bearing; two cheap reads (run 6) are null and only a training ablation settles it. It survives free as an evaluation change** | **run 5: on every pair whose two animals are visually separable, many targets show one concept twice. cheetah x cougar shows two spotted cats on every seed sampled, horse x zebra two striped animals on three of four, cat x dog two dogs or two cats on three of eight. The automated read's 80% train figure is an upper bound: it misses good targets on similar pairs (2 of 8 wrong on lion x tiger) and cannot decide rabbit x hare at all** |
| 9. a joint sampler made to compose is a better teacher than the joint prompt | open, appended 2026-09-08; the follow-on if the parent-versus-C1 gap says wrong targets matter | |
| 10. weighting the loss toward the part of the correction outside the two experts' span makes the adapter learn the new direction it currently gets half of | needs a check: child C3 against the parent | |
| 11. guidance restricted to the middle noise levels removes the over-committed look without touching composition | needs a check: rendered into the stacked sampler, read on the baseline first | |
| **8 (current). softness is trained in, so the adapter can be taught to spend less of itself in the low-noise steps** | **8.2 located the window at [25, 50) with the gain at the bar on clean-reference seeds; 8.4 holds; 8.1 (an instrument) still open; 8.3 (why a well-aimed late correction softens) still the unknown** | |

Claim 1 in full: the held-out gap in fit (cosine of the LoRA's correction against the cached target on the cached inputs: 0.969 train, 0.800 held-out at rank 8 @ 30k) is data-limited, so more training pairs move it.

Load-bearing, from 2026-09-08: claim 2. Claim 7's observation is verified and its training-target
reading is unproven, so the load moved to the one recorded fact about the pool that still stands:
nine of its eleven pairs sit at an interaction bound of 100 to 430 while the reference test sits
at 2656 (run 3). Target quality turned out to be uncorrelated with the bound, so it is a separate
axis and not part of this claim.

Superseded, from 2026-09-07: claim 7. The deliverable is a checkpoint whose renders on unseen
pairs are good enough to publish, and the two failure modes behind that are one animal instead of
two (claims 7 and 5) and two animals rendered badly (claim 3's low-noise steps). Claim 1a, the
on-cache held-out cosine, no longer sits under either: run 1 broke the link between fit and
composition, and run 4 showed the gap is per pair rather than per concept, so a bigger pool raises
a number that does not predict the failure. Claim 1 as originally load-bearing: If more pairs do not move the held-out cosine, the 100-pair cache is wasted and the plan reduces to claim 3 alone.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [The run set](#the-run-set)
- [The two failure modes, and which change serves which](#the-two-failure-modes-and-which-change-serves-which)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Routes](#routes)
- [Sources](#sources)
- [Next step](#next-step)
- [Compiled](#compiled)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

A pooled cross-attention LoRA trained on 11 animal pairs composes 7 of 8 held-out cat x dog seeds at its best checkpoints, then loses fidelity as training continues. The idea is that more and cleaner training pairs close the held-out gap, and a tamer optimizer stops the decay.

**Where the walk is**

Compiled 2026-09-08 to destination A: the idea stands, with claim 2 load-bearing and its check
named as the parent-versus-C4 run. The compiled statement, ledger, build path and the
`/init-master-plan` prompt are the last section of this file; the run set itself is
[the run design](run-design-parent-and-children.md).

**What compile produced**

Destination A. Seven runs of the walk settled which changes serve which failure, located the
softness window, rejected three instruments and the walk's own first mechanism, and left one
scope to build: five training runs in a chain on one Blackwell, two samplers, one strip per seed.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

One LoRA on the cross-attention projections (attn2 to_q, to_k, to_v; rank 8, 16 or 32) is trained to predict the interaction term, the correction the product of experts drops, on 11 train pairs x 8 seeds. On held-out cat x dog it reaches 7 of 8 composed seeds at 30k to 60k steps, but its learned-vs-target cosine on held-out cells sits at 0.40 against 0.54 on train cells from 10k onwards, and past roughly 70k (rank 32) or 160k (rank 8) the held-out renders drift back toward plain PoE and go hazy. The idea has two halves. The generalization gap is a data-size problem, so more pairs (11 toward 100), cleaner targets, and a train pool that reuses animal words freely should move the 0.40. The decay is an optimization problem, so weight decay and a stopping rule on held-out drift should remove it. Each half is tested on its own axis, starting with the pairs already cached.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#the-two-failure-modes-and-which-change-serves-which) ➡️

### 1. 🔍 The held-out gap is data-limited

Mark: wrong as stated, round 1. The cached animal pairs outside the 11 all reuse a word that is
either in train already or in a held-out pair. Keeping every held-out word out of train leaves
two usable extra pairs with 8 or more seeds (wolf x horse, lion x horse), so an 11-to-13 step
cannot test data-limitation. Repaired claim: the gap is data-limited if the held-out cosine at
30k falls when the train pool shrinks from 11 pairs to 5. That run costs no cache and about 3 h
on the Blackwell. If it falls, a 20-pair token-disjoint cache (160 cells, about 6 GB, 5 to 8
RTX 8000 GPU-hours) is the next arm. If it stays at 0.40, more pairs will not move it either.

Round 2, from [route 1](routes/01-pressure-test-poe-failures.md). The 0.54 and 0.40 are read
along the corrected trajectory, which leaves the cached states, and the adapter has only ever
seen cached states. The field's name for this is exposure bias, the compounding error of
imitation learning (DAgger). The target is free at any state, one joint-prompt forward, so the
cure is training on states the adapter itself produces, and the cache is a convenience rather
than a requirement. Split: 1a, more pairs lift the on-cache held-out fit, read with run 1's
script against 0.800 with 0.081 as the noise floor; 1b, on-policy training lifts the render-time
cosine. 1b is load-bearing: if it holds, pair count is a second-order axis. Route 1 also places
the one-animal render outside this idea: seed 14 fits at 0.867 and still fails, so it is a basin
of the deterministic trajectory (scopes 05 and 06), not a fit problem.

- [x] 1.3 the on-policy fit read (run 2: 0.82 on the corrected path against 0.86 cached; live target orthogonal to cached target late, cosine 0.015): roll out cat x dog with rank 8 @ 30k, true r_t at each corrected state from one joint forward, cos(r_hat, r_t) against step beside run 1's cached-state curve
- [ ] 1.4 demoted by run 2 to at most 0.04 of cosine: on-policy training, half of each batch from the adapter's own corrected-trajectory states, refreshed every few hundred steps; read the on-policy cosine and the 8-seed cat x dog grid
- [x] 1.2 the in-distribution fit, cos(Δ̂, Δ_t) on the cached inputs themselves, per pair, so seen-vs-unseen is not mixed with wolf x husky-vs-cat x dog (run 1)
- [ ] 1.1 the seed-to-seed noise of the held-out cosine at 30k on the 11-pair config, needed to read a 5-pair drop as real

### 2. 🔍 Pair diversity, not seeds, is the axis

Mark: open, and load-bearing from 2026-09-08.

Round 1, 2026-09-08. Target quality and the interaction bound are two uncorrelated axes:
cheetah x cougar and lion x tiger sit at bounds of 1766 and 2222, beside cat x dog at 2656, and
have the worst targets in the pool; dolphin x porpoise sits at 100 and is flagged on half. What
survives from run 3 is unrepresentativeness alone: nine of eleven train pairs at 100 to 430, two
at 1766 to 2222, the reference test at 2656, and six of the other seven held-out pairs at 218 to
392 where the pool already lives.

Said plainly: the held-out gap moves when the pool covers the range of bound the test pair sits
at, not when seeds are added to pairs already there. The nearest field name is covariate shift
on one scalar per pair.

What it commits the idea to, in order of how little is known: that the bound is the axis
transfer happens along (unmeasured; run 1's per-pair fit does not track it, but fit does not
predict composition either); that enough high-bound animal pairs exist (run 3 names five already
cached at 1300 to 3200, three reusing cat or dog, so claim 4 is what makes them usable and
elephant x penguin at 2558 becomes the read); that the deliverable's unseen pairs are high-bound
(cat x dog and elephant x penguin are, the other six are not, and on those six the checkpoint
already scores 0.75 to 1.00).

- [x] 2.0 the bound of every cached non-pool pair, recomputed from the residuals with run 3's
  formula (cheetah x cougar reproduces at 1716 against run 3's 1766, so the method agrees).
  **No low-bound control arm exists in cache**: every cached extra is high-bound (horse pairs 1424
  to 4113, bear x salmon 1812, dog x duck 1583, the cat/dog/lion/tiger pairs 2438 to 4037), and
  the only low one is butterfly x meadow at 122, a compose-by-default control. No seed-depth
  control either: every train pair has exactly 8 seeds cached bar wolf x husky's 9
- [ ] 2.1 rebuilt after 2.0, one arm: the 11 train pairs plus lion x horse, bear x salmon and
  wolf x horse (29 new cells, bounds 1583 to 4113, none touching cat or dog), rank 32 to 30k,
  weight decay 0, everything else fixed. Read compose count on elephant x penguin at 2558, the
  only other high-bound held-out pair, with the six low-bound held-outs as the do-no-harm check.
  The pool-size confound stays without a control arm, so a gain is suggestive rather than clean;
  breaking the tie needs a 29-cell duplication arm that does not exist in cache

### 3. 🔍 The late decay is an optimizer artefact

Mark: open.

### 4. 🔍 Token-disjointness can be relaxed inside train

Mark: holds inside the train pool, and the deliverable does not need it against the held-out
test. Round 1, 2026-09-08.

The measured half is weaker than it looks and is not leaned on. Run 4 puts both-words-seen at
0.84, one-word-seen at 0.77 and neither-seen at 0.80, a spread of 0.07 inside the 0.081
cell-to-cell noise of the held-out fit, so seeing a word buys nothing measurable. But that is a
fit number, and this walk has twice established that fit does not predict which seeds compose. A
reviewer objecting that cat x dog is not held out if cat was trained on is making a validity
argument a fit number cannot answer.

Task 2.0's lookup makes the argument unnecessary. Three cached high-bound pairs touch neither cat
nor dog: lion x horse (bound 4113, 12 seeds), bear x salmon (1812, 5 seeds), wolf x horse (1583,
12 seeds). That is 29 cells at 1583 to 4113 against a train pool at 100 to 430. Two of the three
reuse words already inside train, which is the relaxation this claim is about, and none touches a
held-out word, so cat x dog stays a clean concept-disjoint test.

Two cached pairs stay out and it is worth naming why: cat x horse would put cat into training and
cost the reference test; cow x horse would put cow into training and cost cow x buffalo.

### 5. 🔍 Filtering cells by mono quality helps

Mark: open.

### 6. 🔍 One axis per change

Mark: open.

### 7. 🔍 Cells whose joint-prompt target does not show both concepts

Mark: split, 2026-09-08. The observation is verified and the training-target reading is unproven.

**Verified.** On every pool pair whose two animals are visually separable, many joint-prompt
targets show one concept twice (run 5, by eye).

**Unproven and demoted from load-bearing.** That these cells teach the adapter a wrong correction
does not follow from the pictures, and run 6's two cheap reads are both null. Four objections
stand, listed in `run-06-does-a-bad-endpoint-mean-a-bad-target.md`. Only a training ablation
settles it, which is no longer a free change.

**Free and worth doing regardless.** A held-out seed whose own joint-prompt target shows one
concept twice cannot count as a compose failure. That is an evaluation change at no cost.

Appended after claim 6 on 2026-09-07; nothing renumbered.

Every training cell's target is the joint-prompt prediction on that seed's trajectory, and the
picture of that target is the cell's `mono.png`. Where the picture shows one animal, the
correction the adapter is fitted toward at all 50 steps of that cell points at one animal. The
field's name is label noise, in the imitation-learning case where the demonstrator is wrong
rather than noisy.

What it commits the idea to: that the loss decomposes per cell, so dropping cells is a clean
operation (it does), and that the count of bad cells is large enough to matter. Eleven train
pairs at eight seeds is 88 cells; three bad ones move nothing, twenty change the run, and nobody
has counted.

One correction to the premise the claim arrived with. Seed 14 is a held-out cell, not a train
cell, so cleaning the training pool cannot fix seed 14. The seed-14 evidence supports the
training-target reading by analogy and the evaluation reading directly: a held-out seed whose
joint-prompt target is two cats does not belong in a compose count.

Round 3, 2026-09-07, from run 5. The claim holds on the one pair with ground truth: three of the
eight held-out cat x dog targets show one concept twice, and they are seeds 10, 13 and 14, the
three the failure was first noticed on. What did not exist was an instrument. Two of the three
candidate reads find none of the three bad targets, and the third finds all three. The sweep over
all 790 cached targets is what turns this into a count.

Round 2, 2026-09-07. The check as first written was wrong. `count_instances()` queries the generic
term `INSTANCE_QUERY = "animal"` (`detection_scorer.py:32`), so a target showing two cats returns
2 and reads as a good target. That is exactly the failure claim 7 is about: seed 14 (one animal)
would be caught, seed 15 (two cats) would pass. The generic query is deliberate in the compose
scorer and must stay there, because the same file records that per-query box-IoU under-counts real
composes on the hard pairs, a measurement bias. So target quality needs its own instrument rather
than a change to the compose scorer.

The named-class read is already available: `vmetrics.detect_boxes(path, [query_a, query_b])`
returns per-label boxes, and `score_output_instances` already computes `conf_a` and `conf_b` and
discards them for the label. A target is good when the picture carries at least one box for each
named animal, not two boxes of anything.

DINOv2 cannot do this job. It has no text encoder, so there is no prompt to give it, and the
scorer's own docstring records that the whole-image embedding read (DINOv2 and CLIP) was nulled
because for two similar animals "one chimera" and "two separate animals" sit in the same feature
region.

- [ ] 7.3 eye pass over the pool's 152 targets, nineteen contact sheets of eight, to turn the
  upper bound into a count on the pairs the forced choice cannot separate ← the next run
- [x] 7.1 score every cached `mono.png` (790 across 18 train and 58 held-out pairs) two ways: the
  generic `count_instances()` and a per-name read requiring one box for each of the pair's two
  animals. Report how many train targets show one animal, how many show two of the same animal,
  and whether the held-out cells failing either read are the cells the adapter drops
- [x] 7.2 validated three reads against the eight cat x dog targets labelled by eye (run 5). The
  per-name read is rejected and inverted; the forced-choice read scores 7 of 8 with its one error
  a false negative, so it can only drop usable cells and never keep broken ones

### 8. 🔍 Softness is trained in, not only sampled in  ← current

Mark: open, expanded 2026-09-08. Appended 2026-09-07 after the goal split; nothing renumbered.
The trainer's loss is `‖Δ̂ − Δ_t‖²` averaged over the batch with no timestep weighting
(`poe_repair/experiments/one_pair_one_seed/trainer.py:9`), so every step counts the same.

- [ ] 8.1 an instrument for crispness. Still open, and the gate for the rest. Ruled out: the
  instance count (blind to it), Laplacian variance (heavy-tailed; drawn-texture seeds dominate),
  and, from run 7, the paired high-frequency share against the seed's joint render (per-seed
  ratios swing 0.15 to 4.2; it counts fur texture, and ranks hand-off@20 best where DINOv2 ranks
  it worst). Weakened: DINOv2 nearness-to-joint, because its reference is a wrong picture on
  seeds 10, 13 and 14, the cells claim 7 flagged, and the 0.038 was averaged over them. Next
  candidate: re-read the recorded clean-tail DINOv2 numbers on the five seeds whose reference is
  right, then decide whether nearness on a correct reference is instrument enough
- [x] 8.2 softness sits in a locatable window: [25, 50). Run 7 swept the switch step over 20,
  25, 30, 35, 40, 45, 50. The nearness gain is single-peaked at 25 (+0.041 on eight seeds, +0.050
  on the five whose reference is a real cat and dog, which is the bar exactly) and drains to zero
  by 50. No composition lost on the five clean seeds at any cutoff; the two seeds that need the
  adapter past 30 both have a wrong reference. A training change now has a window to stay out of
- [ ] 8.3 the late correction hurts by direction, not size. The unknown that could kill the
  claim. Run 2 puts the late correction at cosine 0.815 to the true one at 0.79 of its magnitude,
  yet shrinking it drifts toward plain PoE (energy-penalty null) and removing it moves renders
  nearer the joint prompt (hand-off). Nothing recorded explains a correction that helps when
  removed and hurts when shrunk. Check: decompose the late correction into in-span and
  orthogonal parts with the span-share scripts and ask which part the hand-off removes
- [x] 8.4 a training-side change is buildable, three ways, cheapest first: train on steps 0 to 29
  only and use the windowed sampler at inference (the mask is new; `commit_window` (5, 25) is a
  reporting bucket, not a mask); weight the loss by signal-to-noise so low-noise steps count less
  (Min-SNR weighting, Hang et al. 2023, confident on the method, verify it has been pointed at a
  correction target); or fit a target that is r_t early and zero late

What would surprise me: 8.1 ranking the hand-off below adapter-alone on sharpness while DINOv2
says it is nearer. Then "nearer the joint picture" and "crisper" are different things and the
deliverable wants the second.

Handing the last twenty steps to the frozen model moves renders 0.038 nearer the joint prompt in
the compose scorer's DINOv2 embedding, against a 0.05 bar, with composition held
([the clean-tail finding](../../../report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md)).
The lever is real and under-powered as a pure inference switch. The claim is that the same lever
exists on the training side: weight the loss down over the low-noise steps, or train the adapter
against a target that is already handed off, so the adapter never learns to spend itself there.

Every other change on the list is data-side or optimizer-side. This is the only one aimed at where
softness was actually measured to live.

### 9. 🔍 A joint sampler made to compose is a better teacher

Mark: open, appended 2026-09-08 from the ideas round; nothing renumbered.

The adapter is trained to reach the joint prompt's prediction, and on the pairs that matter the
joint prompt itself draws one concept twice. Replacing the teacher with an attention-guided joint
sampler (Attend-and-Excite, Chefer et al. 2023, confident on the method) that is made to attend to
both subjects gives a target that composes at the same cached states. Costs a new cache and a
validation that the guided sampler composes where the plain one does not. Held for the follow-on.

### 10. 🔍 Weight the loss toward the new direction

Mark: needs a check, appended 2026-09-08. The recorded finding is that the adapter reproduces the
part of the correction that lies along the two experts' own predictions almost exactly and only
about half of the part orthogonal to them. The plain loss weights both equally, so the easy part
dominates. Weighting the orthogonal error up by a factor (3 in the parent run) aims capacity at
the measured deficit. Settled by child C3 against the parent.

### 11. 🔍 Guidance only in the middle noise levels

Mark: needs a check, appended 2026-09-08. Kynkäänniemi et al. (NeurIPS 2024) show guidance harms
the earliest steps, is unnecessary in the last ones, and helps in between, on SDXL among others.
This project runs guidance 7.5 at every step. Dropping it outside steps 5 to 35 is the same shape
as the hand-off applied to guidance and targets the over-saturated, over-committed look. It is
part of the stacked sampler and reads on the baseline before any training run.

## The run set

Navigation: 📋 [TOC](#table-of-contents)

One parent with everything on, four children each switching one thing off, one no-training
baseline, every run rendered under the shipped sampler and the stacked one, one strip per seed to
W&B. The full design, the words it uses, the read written before launch, the three trainer
changes it needs and its cost are in
[the run design](run-design-parent-and-children.md).

## The two failure modes, and which change serves which

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

The deliverable is a checkpoint whose renders on unseen pairs are publishable. That splits into two
failures with different causes and different instruments.

**A. One animal instead of two.** Instrument exists and is validated: the GroundingDINO instance
count. The rank-32 checkpoint at step 30050 already scores 0.75 to 1.00 on the eight held-out pairs
against 0.00 for plain PoE on six of seven.

**B. Two animals, rendered too softly to publish.** No instrument. The nearest available read is the
DINOv2 cosine distance to that seed's joint-prompt render with a 0.05 bar, which is a proxy for
nearness rather than sharpness. Laplacian variance cannot serve, because on the plain-PoE
references it is heavy-tailed (mean 152, standard deviation 183 over eight seeds). Naming B's
instrument is owed by this walk.

| The change | Serves | Why |
|---|---|---|
| Clean cells whose `mono.png` shows one animal (claim 7) | A | The only change with a recorded cause. Also an evaluation change: a held-out seed whose own joint target shows one animal cannot count as a compose failure. |
| More pairs, spread over the interaction bound (claims 1a, 2) | A, only if the read changes | Run 4 puts the gap on the pair axis, so unseen-pair work is pair-limited. The held-out cosine it was to be judged by predicts neither mode; judged on compose count over new unseen pairs it serves A. |
| Relax token-disjointness (claim 4) | neither, alone | An enabler at no cost. It rides inside the pool change and is never an arm of its own. |
| Filter cells by target quality (claim 5) | B | Boundary against claim 7: claim 7 drops targets showing the wrong thing, claim 5 drops targets showing the right thing badly. |
| Weight decay plus a stopping rule (claim 3) | B, conditionally | 30050 sits before the decay, so at the current pool this buys nothing. It becomes necessary once the pool grows or the run goes long. |
| One axis per change (claim 6) | neither | A rule about how arms are run, not an arm. |
| Live rather than cached residuals | A, as the enabler of the pool change | As a fit improvement it is worth 0.04 of cosine and serves neither mode. As a pipeline change it removes the cache entirely: a new pair costs train-time compute instead of ~5.8 GB and cache-building GPU hours, which is what currently blocks claims 1a and 2. Costs a joint forward per step. |

**No read predicts which seeds compose.** Fit on cache does not (run 1: seed 14 at 0.867, above the
held-out mean, fails). Fit on the adapter's own path does not (run 2: seed 14 at 0.77, above seed 9
at 0.732, which composes). The interaction bound does not (run 3: every red pair fails 8 of 8 at
bounds from 100 to 3200). One candidate exists and it is claim 7's check: run 2's render notes put
one animal on seed 14 and two cats on seed 15, the two cat x dog seeds the walk keeps flagging.
That is n = 2, so task 7.1 either makes it a read or kills it.

## What the words are

Navigation: ⬅️ [The two failure modes](#the-two-failure-modes-and-which-change-serves-which) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| the correction the LoRA learns | the interaction term | what a product of experts drops when it treats the two concepts as independent | confident, project's own term |
| held-out cosine stuck at 0.40 | alignment with the pool-mean correction | `_inline_sampling.direction_metrics` takes cos(Δ̂_t, Δ̄_t) against the mean correction over every train pair and seed; it is the share of the correction along the pool average, a shared-component measure that cannot reach 1 for a state-specific target | verified in code, run 2 |
| the fit on the adapter's own path | exposure bias | states reached with the correction on differ from the cached ones (drift 0.7 to 0.86 by step 49) and the adapter is evaluated there; run 2 measures the cost at 0.04 of cosine | measured, run 2 |
| on-cache held-out cosine 0.80 against 0.97 train | generalization gap | the train-to-held-out difference in fit that more or more varied data is expected to shrink | confident |
| the blend | semantic leakage | features of two similar subjects merged into one; under Bradley et al. it is the product target itself | confident |
| the one-animal render | catastrophic neglect | one subject dropped; a basin of the deterministic trajectory chosen by the seed in the first steps | confident |
| fidelity decay late in training | overfitting via norm growth | with no weight decay and a constant learning rate the adapter weights grow without bound and fit train cells at the cost of unseen ones | likely, verify with the weight-decay arm |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|
| 1a | whether more pairs lift the on-cache late-step held-out cosine off 0.800 (noise floor 0.081) | the 5-pair downward run, or a 20-pair cache. Held rather than walked because the cosine does not predict which seeds compose (run 1: seed 14 fits 0.867 and fails), so lifting it is not known to serve the deliverable |
| 1b | nothing; closed | run 2 put the whole prize at 0.04 of cosine, on a quantity that does not predict composition. Dropped rather than carried |

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|
| 1 | train on the ~9 extra animal pairs already cached, zero cache cost | 7 of the 9 reuse dog, cat or cow, which are held-out words; the two that survive (wolf x horse, lion x horse) make 13 pairs, too small a step to read |
| 1 | free cow by dropping cow x buffalo from held-out | buys one pair (cow x horse), 14 total, still too small |
| 1b | live residuals judged as a fit improvement | run 2 priced that prize at 0.04 of cosine, on a quantity that predicts neither failure mode. Dead as a fit argument, live as a pipeline argument: see the row in the failure-mode table |

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 7 | score every cached `mono.png` (790 images) with both the generic instance count and a per-name read requiring one box per named animal; one GPU, tens of minutes, no cache cost | more than a handful of one-animal train targets: drop or re-render them and retrain, one axis; a handful only: the claim becomes an evaluation change (exclude unwinnable held-out cells from the compose count and say so); held-out one-animal targets coinciding with the seeds the adapter drops: the first read that predicts which seeds compose |
| 1b | roll out cat x dog with rank 8 @ 30k, compute the true r_t at each corrected state with one joint forward, plot cos(r_hat, r_t) against step beside run 1's cached-state curve; 8 renders plus 400 joint forwards, under an hour | starts near 0.9 and decays toward 0.4 along the run: the 0.40 is exposure bias, go to the next row; flat from step 0: the render-time proxy measures something else, re-derive what |
| 1b | rerun the 11-pair training with half of each batch drawn from the adapter's own corrected-trajectory states, refreshed every few hundred steps, targets from a joint forward at those states; read the on-policy cosine and the 8-seed cat x dog grid | on-policy cosine above 0.6 with no seed lost: exposure bias confirmed, the cache framing goes; unchanged: the divergence is intrinsic to the corrected path |
| 1a | rank 8, 30k steps, 5 of the 11 train pairs (one per family), same seeds, weight decay 0, read held-out learned-vs-target cosine at 30k against 0.40 | falls by more than the run-to-run noise: data-limited, build the 20-pair cache; stays at 0.40: not data-limited at this scale, drop the cache and spend on claim 3 |
| 1 | cache 20 new token-disjoint animal pairs x 8 seeds (160 cells, one `scripts/build_training_cache.py` call each, ~90 s on an RTX 8000 or ~30 s on the Blackwell, inferred from the 51.5 s tracking render; ~5.8 GB), keep the pairs whose `poe.png` blends on 6 or more of 8 seeds, train rank 8 to 30k on 11 + those, read the same held-out cosine and the 8-seed cat x dog grid | held-out cosine above 0.40 by more than the noise floor and seed 14 composing: data-limited, go to 100; unchanged: the gap is not on the data axis |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 2 | claim 1b | `scripts/showcase/on_policy_fit_cosine.py`, rank 8 @ 30k, cat x dog seeds 9-16, λ=1 all steps; slurm 49421 on mscluster72, output `/datasets/mmolefe/poe_repair_min/outputs/showcase/on_policy_fit_r8_030000/` | done 2026-09-03 17:15, table and reading in `run-02-on-policy-fit-r8-030000.md` beside this map | cos(Δ̂, live Δ) 0.817 mean (early 0.855, commit 0.810, late 0.815) against 0.860 on cached states; state drift 0.69 to 0.86 by step 49; live target vs cached target late cosine 0.015; seed 14 fits 0.77 and still renders one animal. The 0.54 / 0.40 are pool-mean alignment, not fit. |
| 3 | route 1, use 1 | `scripts/showcase/interaction_bound_from_cache.py`: Mitra et al. Theorem 1 turned around, the lower bound on G·M = ‖r_t/σ_t‖²/4 at every cached state of 25 pairs (all train-split pairs with 3+ seeds plus the 8 animal held-outs and butterfly x meadow), median over seeds; CPU on this node, output `/datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/` (`interaction_bound_mid.png`, `interaction_bound.json`) | done 2026-09-03 | the bound does not predict the plain-PoE fail rate (every red pair fails 8 of 8 at bounds from 100 to 3000). From step 10 the pairs split into two bands ten times apart: near-synonym pairs (wolf x husky, rabbit x hare, seal x walrus, eagle x hawk, frog x toad, crow x raven, dolphin x porpoise) and the composing control butterfly x meadow sit low at 100 to 400; cat x dog, lion x tiger, cheetah x cougar, cat x lion, tiger x dog, lion x dog, elephant x penguin, bear x salmon, dog x duck sit high at 1300 to 3200. The bound measures how far apart the two experts' predictions are, so the train pool is mostly pairs where the target is small and the reference test is a pair where it is large. |
| 4 | claim 1a | `scripts/showcase/fit_cosine_on_cache.py` on rank 8 @ 30k over 8 cached pairs the token-disjoint rule excluded: tier 1, both words seen in training but the pair unseen (wolf x horse, lion x horse); tier 1.5, one word seen (lion x dog, tiger x dog, cat x lion, dog x horse, cat x horse, cow x horse); 12 seeds each; slurm job 49851 on mscluster52, output `/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_tiers_r8_030000/` (`fit_by_tier.png` puts the four tiers on one axis) | done 2026-09-03 | tier 1 fits 0.84 (0.81, 0.86), tier 1.5 fits 0.77 (0.70 to 0.83), tier 2 from run 1 fits 0.80 (0.68 to 0.88), train fits 0.97. Seen words in an unseen pairing generalize no better than unseen words. The adapter learns a per-training-pair correction, not a per-concept one, so the held-out gap is on the pair axis and the 20-pair cache can be built from new pairings of words already in the pool, at no token-disjointness cost. |
| 8 | claim 8, tasks 8.1 and 8.2 (done) | `scripts/showcase/crispness_paired_hf.py` (CPU, existing renders) and `scripts/showcase/handoff_switch_sweep.py` (this node's GPU, new renders at cutoffs 25, 35, 40, 45), λ 1.2, rank 32 @ 30050, cat x dog seeds 9 to 16; output `/datasets/mmolefe/poe_repair_min/outputs/showcase/crispness/` | both done 2026-09-08; reading in `run-07-crispness-instrument-and-switch-sweep.md` beside this map | 8.1 rejected: the paired high-frequency read counts texture, not crispness. Found by looking: the DINOv2 nearness reference is a wrong picture on seeds 10, 13, 14. 8.2 located: nearness gain single-peaked at switch step 25, +0.050 on the five clean-reference seeds (the bar), no composition lost there; softness lives in steps 25 to 50 |
| 7 | claim 7, task 7.3 | eye pass over the pool's 152 targets, nineteen contact sheets of eight | not started, and no longer urgent: the count only matters if the training-target reading is true | |
| 6 | the `break` on claim 7 | `scripts/showcase/residual_direction_by_target_quality.py`, cached tensors only, no model forward; two reads on lion x tiger against run 5's eye labels | done 2026-09-08, tables and reading in `run-06-does-a-bad-endpoint-mean-a-bad-target.md` beside this map | null on both. Cross-cell cosine is confounded: the same-pair good-to-good ceiling (0.004 commit, 0.001 late) sits on the different-pair floor (0.007, 0.002), because corrections at different states are orthogonal whatever the target shows. The within-cell joint-lean read is flat: the one bad cell leans 0.457 in the commit window against 0.539 for a good one. Neither refutes nor supports the claim; only a training ablation settles it. |
| 5 | claim 7, tasks 7.2 then 7.1 | `scripts/showcase/target_quality_scan.py`, in-session on this node's free GPU, output `/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/`; three candidate target-quality reads validated against eight cat x dog `mono.png` targets labelled by eye, then the surviving read swept over all 790 cached targets | done 2026-09-07, tables and reading in `run-05-target-quality-instrument.md` beside this map | 3 of 8 held-out cat x dog targets show one concept twice (seeds 10, 13, 14). The generic instance count returns 2 on all eight and finds none of them. Per-name detection is inverted: the two-cat target returns "a dog" at 0.787, above the 0.615 the real dog gets. Forced choice between the pair's two names on each detected box finds 3 of 3 on cat x dog and scores 7 of 8 there, but its false-negative rate rises on similar pairs (2 of 8 wrong on lion x tiger) and it cannot decide rabbit x hare, so its 80% train and 59% held-out figures are upper bounds. The eye confirms the substance: on every visually separable pair, wrong targets are common. |
| 1 | claim 1 | `scripts/showcase/fit_cosine_on_cache.py` on rank 8 @ 30k: cos(Δ̂, cached Δ_t) on the exact cached (x_t, t), 10 steps per cell, all 88 train and 64 held-out cells; mscluster108 device 1, output `/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_r8_030000/` | done 2026-09-03 16:42, table copied to `run-01-fit-cosine-r8-030000.md` beside this map | in-distribution cosine 0.969 train / 0.800 held-out (sd across cells 0.011 / 0.081). Gap sits in the late steps: early 0.985 vs 0.925, commit 0.975 vs 0.832, late 0.961 vs 0.748. cat x dog 0.860, not the hardest pair (seal x walrus 0.677); the failing seed 14 scores 0.867, above the held-out mean, so fit does not predict the compose failure. The render-time 0.54 / 0.40 were trajectory-divergence proxies, not fit. |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | The ask | Return line | State |
|---|---|---|---|---|
| 2 | the whole walk | the session record: every step of 2026-09-05 in order, what it asked, did, and left behind, with the X post in the sources register placed | [what this walk tried](routes/02-what-this-walk-tried-2026-09-05.md) | written 2026-09-05 |
| 1 | claims 1 and 2 | `/pressure-test`: are the held-out failures (blend, one-animal basin, late haze) known PoE failures with known sampler-side or data-side fixes, and does the literature expect more pairs to fix them | [the verdict](routes/01-pressure-test-poe-failures.md): promising but needs rework; the 0.40 is exposure bias, the blend is the product target, the one-animal render is a trajectory basin, the haze is norm growth plus off-trajectory evaluation | run and folded 2026-09-03 |

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| `artifacts/results/does-the-fix-reach-unseen-pairs/pair_pool.yaml` | the 11 train and 8 held-out pairs, and the token-disjoint rule as a comment | verified |
| `poe_repair/experiments/one_pair_one_seed/config.py:43` | `OptimConfig.weight_decay = 0.0`, no learning-rate schedule anywhere in the pooled trainer | verified |
| `/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/` | cell counts per pair, both splits | verified 2026-09-03 |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

The walk has a run set and a launch plan, both in
[the run design](run-design-parent-and-children.md): one Blackwell (`mscluster112`, free on
2026-09-08; 110 busy, 111 faulted), `co3_bw`, five runs in a chain overnight, six ordered steps
before the launch line, and what each run logs to W&B and from where. `compile` turns this into
the `/init-master-plan` prompt for one scope.

Still owed: an instrument for crispness (8.1). Until one exists, "clearly better" is judged on
the strips by eye against a fixed rubric, and the plan says so.

## Compiled

Navigation: ⬅️ [Next step](#next-step) | 📋 [TOC](#table-of-contents)

Destination A, 2026-09-08. Claim 2 is load-bearing and its check is named and runnable (the
parent against child C4 in the run design), so the idea stands. What follows is the statement a
colleague who missed every round would accept, the ledger, the build path, what is still open,
and the route out. The `/init-master-plan` prompt was emitted in chat the same day and is
reproduced in the route-out paragraph below in summary; the full prompt is what the scope was
initialised from.

**The idea, said plainly.** On most held-out cat x dog seeds the rank-32 adapter at step 30050
produces some composition but not a clean cat and a clean dog, and where both appear the picture
carries details the training distribution would never produce (the problem catalogue in the run
design has one row per seed). The count scorer calls most of these composed, so it is not the
read. Two changes on the data side are cheap: three cached pairs containing neither cat nor dog
are added, and cells whose joint target shows one concept twice are dropped; the correction sizes
of the pairs are measured and reported, not used as a reason. The second fault sits in the
adapter's own last 25 denoising steps, located by sweeping the hand-off point, and is removed at
inference by handing those steps to the frozen model, and at training by never fitting them. A
better teacher than the joint prompt, which itself draws one concept twice on the pairs that
matter, is the strongest follow-on and waits on a new cache. None of the adapter's fit numbers
predict which seeds compose, so the read is the compose count plus a per-region check that both
names are present, on strips, with softness judged by eye until an instrument exists.

**Ledger.** The ledger is this walk's bookkeeping and stays here; the master plan carries only the
problem catalogue, the run set, the read and the environment facts. 1 wrong as stated, split (fit does not predict composition; 1a held, 1b dead as fit
and live as the pipeline enabler). 2 needs a check, load-bearing (parent vs C4). 3 open, carried
as safety in the parent (weight decay 1e-2, EMA), not ablated. 4 holds inside train, unnecessary
against the held-out test. 5 open, boundary against 7 redrawn: 7 drops wrong targets, 5 drops
badly rendered right ones; not in this batch. 6 holds, run as the leave-one-out design. 7 split:
observation verified (run 5), training reading unproven (run 6 null twice), evaluation consequence
free and taken (seeds 10, 13, 14 excluded from nearness). 8: 8.2 located [25, 50) at the bar on
clean seeds, 8.4 holds, 8.1 open (three instruments rejected), 8.3 the unknown. 9 open, held for
the follow-on. 10 needs a check (parent vs C3). 11 needs a check (baseline under the stacked
sampler).

**Build path.** In the run design: three trainer flags and the guidance interval; the exclusion
list with its count; the 14-pair pool with every cell checked; the baseline under the stacked
sampler; a 200-step smoke on `mscluster112` under `co3_bw`; the chain P, C2, C1, C3, C4; the
probe pass that logs the stacked strips, both-names and clean-seed nearness into each W&B run;
the comparison sheet against 30050.

**Still open.** 8.1, an instrument for crispness (a vision-language judge validated against eye
labels on the eight cat x dog seeds is the next candidate). 8.3, why a late correction aimed at
0.8 of the true direction still softens. 2's pool-size confound, which no cached control arm can
break. 9, the better teacher.

**Route out.** `/init-master-plan` for one scope, "improve the rank-32 adapter and show it beat
30050", carrying the statement above, the ledger, the run design file as the plan's spine, and
the read written before launch.
