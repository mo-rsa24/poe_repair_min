# Layer 12, step 4: frame the hypothesis

⬅️ [Layer 12, step 4](../12-the-correction-at-scale.md#4-you-then-the-experiment-only-if-the-gate-passes)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`, only if step 2 recorded a gap.

```text
/frame-hypothesis the four training arms in artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md
claim 8 (including upweighting the real drainless films), with stress-testing before augmentation,
and the dial, memorisation and test-size checks from
artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md,
with the claim that the toy's overlap-dial curve predicts the share of real films below which
composition recovers the cell and joint conditioning does not

Background
Layer 12 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) asks whether a
correction for composition helps on chest X-rays. Step 2 measured a drain gap of at least 5 points
with its interval excluding 0, so the drain case continues. Step 3 named the shortcut to target in
claim 8 of the idea map. This sharpens the experiment into a falsifiable, ordered plan for the
thesis chapter before anything trains.

Objectives
1. Sharpen the claim in the skill line into a falsifiable, ordered plan over refinement rounds.
2. Order stress-testing before augmentation.
3. Compile the five-part result and save it to
   artifacts/ideas/when-poe-reaches-a-and-b/HYPOTHESIS.md.

What to read
- Step 2's verdict, in instruction 11.1 of
  ~/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/b1-drain-shortcut-in-pneumothorax/plans/01-measure-the-drain-gap.md:
  the gate this runs behind.
- artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 8 as step 3 left it, with Checks
  outstanding, claims 2, 2b, 6 and route 5: the arms and the shortcut.
- artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md:
  the six arms, the dial, the checks and the baselines.
- The overlap-dial idea map, claim 4: the sampler caveat.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Claim 8's check names four arms, each one pneumothorax classifier trained on: the original NIH
  films; the real drain-free films upweighted; plus drain-removed edits of positive films; plus
  composed pneumothorax-without-drain films (the map, Checks outstanding, claim 8).
- All four are tested on real drain-free pneumothorax films (NEATX labels) beside the drain films,
  and on the same films with only the drain removed, scored against a VAE round trip of the original
  (the map, Checks outstanding, claim 8).
- What each outcome means there: upweighting closes the gap, so no generated data is needed; edits
  close it and composition adds nothing; composition closes it further, so the untreated look
  matters; none closes it, so data of no kind fixes the shortcut (the same row).
- Route 6's return proposes six arms: ERM, prevalence recalibration or DFR, drain-removed edits,
  edits that remove the drain and grow the pneumothorax, the jointly conditioned model, and composed
  samples (construction 1 with a "no drain" label, annealed MALA or UHA graded on the toy first)
  (the route 6 return).
- The dial is the share of real drainless films the generators see, 100, 25, 5, 1 and 0% (the route
  6 return).
- The null is composition tracking joint conditioning at every share (the route 6 return).
- The toy's prediction: construction (1) holds 0.989 of the product in the rare corner down to a
  0.001 share on the four-quadrant toy (the map, claim 2, run 1).
- Separately fine-tuned experts reach at most 0.5 when each one's leftover density sits in the two
  adjacent quadrants, which is the answer key's case; leftover on the top-right alone gives 0.9997
  (ladder layer 5's corrected answer key; the map, claim 2, run 1).
- Before any classifier trains, compare pneumothorax size in each arm's films against real drainless
  films (SIIM-ACR masks or a segmenter), and check nearest-neighbour distance to the training films
  for memorisation (the route 6 return).
- Test size is about ±0.14 at 25 films against 25 and ±0.07 at 100 against 100 for AUROC 0.75 (the
  route 6 return).
- Negation with "contains" labels is unbounded, p(drain|x)^(−w) growing as p(drain|x) → 0, so use
  the positive label "no drain", whose factor is at most 1 (the route 6 return).
- The baselines to beat are FastDiME's AUROC 0.98 on films that keep the drain association and 0.65
  without it, and Kina and Petersen's recalibration from 0.226 to 0.726 (the route 6 return).
- Scoop risk: Weng et al. name training on counterfactuals as their next step, and Petersen is on
  both the FastDiME and the recalibration paper (the route 6 return).
- Stress-testing comes before augmentation, because the treated-looking lung without a drain is the
  counterfactual a stress test wants, and a training risk only where the air rim vanishes (the map,
  claim 2b and route 5).
- 1,906 drainless films exist, so every share below 100% is deliberate subsampling (claims 6 and 8).
- Samplers recover a third to four fifths of the exact A∩B mass at 2% to 10% overlap (the
  overlap-dial map, claim 4).
- Trained experts cost more under a better sampler: the interaction is +0.054 total variation at
  r = 0.05 and +0.232 at r = 0.001 (the map, run 7).

Rules
- Run this only if step 2 recorded a gap.
- Train and generate nothing.
- Give no novelty verdict, since route 6 holds one.
- Write only HYPOTHESIS.md.
- No commit.
- Each round states what is solid, what needs sharpening and what is not yet falsifiable, with the
  routing verdict, before the next round.

What to hand back
- The five-part compile, saved to artifacts/ideas/when-poe-reaches-a-and-b/HYPOTHESIS.md after it
  compiles (frame-hypothesis compiles inline, so the save is asked for here).
- HYPOTHESIS.md is what hypothesis-to-scope or experiment-planner reads next.
- The proof: HYPOTHESIS.md exists and its falsification rule carries a number in each of the three
  branches; I read it.

What you will see
Along the way: one refinement round per message, each with what is solid, what needs sharpening,
what is not yet falsifiable, and the routing verdict.
At the end, a closing report: what was asked, in one sentence; what was done, with HYPOTHESIS.md's
path; PASS or MISS against a three-way rule (supports, null, inconclusive) with a number in each
branch on the drainless AUROC per arm at each dial share, and the films per cell counted; what is
still open; at most one question for me.

I judge, from HYPOTHESIS.md: done when its rule on drainless AUROC per arm at each dial share has a number in each of supports, null and inconclusive, and the films per cell are counted.
```

added: the seven sections, from the field-labelled version · the Background paragraph naming the gate and step 3 as what this runs behind (from the old Task and Starts from lines)
dropped: nothing
