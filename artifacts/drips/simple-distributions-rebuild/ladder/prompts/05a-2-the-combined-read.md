# Layer 5a, step 2: the combined read

⬅️ [Layer 5a, step 2](../05a-reading-stop-projective-composition.md#2-you-the-combined-read)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after quiz R.1.

```text
/drip-read --combine --lean "projective composition, factorized conditionals and local scores" with
arXiv 2502.04549 and 2509.16447, read against the quadrant toy and the parts world of
artifacts/drips/simple-distributions-rebuild/

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ found in layer 5 that an
expert loose outside its own region needs no overlap. The two Bradley papers give the mathematical
reason: 2502.04549 proves when summed scores are exact (factorized conditionals, Theorem 5.3) and
when a composition is exact in a feature space (Theorem 6.1), and 2509.16447 ties it to a local
score. This read takes the one idea across both papers, so I can place both toys against the
condition. I have passed quiz R.1, and its misses are the first ideas to slow on.

Objectives
1. Read the one idea across both papers in combine mode, one idea of the spine per round, each
   passage as printed with its plain reading and the four comparison lines.
2. At message 1, take my reason: to state when summed scores are exact and place the quadrant toy
   and the parts world against that condition.
3. Keep both papers through the whole read.
4. For "your plan says", offer two candidates at message 1 and let me choose: the vault note
   "Composing Models With A Background", and journey 06's plan
   plans/05-composition-and-the-experiment/plans/04-projective-composition-condition.md in
   ~/goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/.
5. Close the map with the read's closing check against the five statements in What you will see.

What to read
- Quiz R.1's results: the misses that become the first ideas to slow on.
- In the journey folder
  ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/, the
  read-aheads sources/mechanisms-of-projective-composition/read-ahead/ and
  sources/local-mechanisms-of-compositional/read-ahead/: the papers taken apart chunk by chunk, with
  answer keys in each paper's checks/ folder.
- The same journey's drips/read-mechanisms-of-projective-composition/READ_MAP.md: the earlier read's
  map, where it stopped, and its slips.
- The two PDFs on the shelf (see below).

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The PDFs are on the shelf under ~/goal-setting/learning/papers/, in
  deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/ as
  mechanisms-of-projective-composition-of-diffusion-models.pdf and
  local-mechanisms-of-compositional-generalization-in-conditional-diffusion.pdf, with PDF page equal
  to printed page (Papers.md).
- The idea sits at 2502.04549 Definition 4.1 p5; Definitions 5.1 and 5.2 and Theorem 5.3 p6;
  Remark 5.5, the importance-of-background paragraph and Lemma 5.7 p7; Theorem 6.1 and Lemma 6.2 p8;
  and 2509.16447 Definitions 1 and 2 p5; Lemma 1, its relaxation and Corollary 1 p6 (Papers.md).
- The earlier read of 2502.04549 stopped at abstract sentence 6 of 9, so Definition 4.1 onward is
  unread by me and no round may assume it (READ_MAP.md, Position).
- Definition 4.1 says p̂ is a projective composition when Π_i♯p̂ = Π_i♯p_i for every i; it is a
  property, with none or many solutions, allowed outside every p_i's support (read-ahead 04, chunks
  37 and 38).
- Definition 5.1 is C[p_b, p_1, ...](x) = p_b(x)·∏ p_i(x)/p_b(x) / Z, and with p_b the
  unconditional it is the Bayes composition (read-ahead 05, chunks 42 and 43).
- Definition 5.2 is a partition M_b, M_1, ..., M_k with (1) x on M_i independent of the rest under
  p_i, (2) all blocks mutually independent under p_b, and (3) p_i equal to p_b off M_i (read-ahead
  05, chunk 45).
- Theorem 5.3 says that under Definition 5.2, reverse diffusion with the composed noised scores
  samples p̂(x) = p_b(x on M_b)·∏ p_i(x on M_i), and the sampling rests on C[N_t p] = N_t C[p]
  (read-ahead 05, chunks 47 and 49).
- Remark 5.5 says Definition 5.2 implies orthogonal score differences and is strictly stronger;
  deltas confined to separate coordinates over a background with correlation 0.5 give
  N(2.1818, 0.2364) per coordinate against the target N(2, 0.25) (checks/separation_not_enough.out;
  read-ahead 05, chunks 45 and 51).
- The same experts compose with an empty background and fail with the unconditional (Figure 3, A
  against B) (read-ahead 05, chunk 53).
- There are two printed slips: Lemma 5.7 is false as printed (KL log(5/3) = 0.5108 against the
  bound log(4/3) = 0.2877 on two binary pixels), and Definition 5.2 prints as "Theorem 5.2" in three
  places (READ_MAP.md, Disagreements and slips).
- Theorem 6.1 says a C¹ diffeomorphism A that makes the pushed-forward inputs satisfy Definition 5.2
  makes C[p] projective with Π_i(x) = A(x) on M_i, with no sampling claim, and Lemma 6.2,
  C[A♯p] = A♯C[p], holds for C and fails for the simple product (ratio max/min 1.0000 against
  17.0000, checks/reparam.out) (read-ahead 06, chunks 63 and 66).
- In 2509.16447, Lemma 1 says a conditional projective composition (Definition 2) holds exactly when
  the score is a local conditional score (Definition 1); Definition 2 drops Definition 5.2's
  condition 2, the "only if" half needs it back, and the approximation improves at higher noise
  (read-ahead 03 of 2509.16447, chunks 32, 35 and 37).
- The layer 5 quadrant toy has A the left half and B the top half; expert A's 800 points have
  x1 ~ U(−1, 0), a share f with x2 ~ U(0, 1) and the rest x2 ~ U(−1, 0), B mirrors it, and the
  ideal experts are smoothed by σ = 0.05 and composed as the plain product (step file 05, steps 1
  and 2).
- In 2D the background is uniform, so dividing by it changes the plain product only by a constant
  (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, claim 2).
- On the four-quadrant answer key (blobs at (±0.5, ±0.5), std 0.15), experts that model only their
  own coordinate and take the other from the background marginal put 0.9999 (plain) and 0.989
  (background-divided) in the top-right blob with no A∩B data
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md).
- The parts world has 16 pixels, cat +1 and dog −1, variance 0.01 on the animal's half, and the
  other half mean 0 with variance 0.01 (empty) or 1.0 (loose); empty at 0% overlap gives a ghost
  with half-means (0.5, −0.5) and A∩B weight 0.000, 1.000 from 0.1%, and loose gives 1.000 at every
  overlap including 0% (step file 05, step 6; p5.ghost_mismatch_nats 400.0,
  p5.matching_pair_advantage_nats 386.2).
- On the parts-world mosaic at ᾱ = 0.5 the summed noised scores put the head mean at 0.2326 where
  the noised product has 0.6931 (step file 04, step 6).
- Construction (1) targets p(x|A)·p(x|B)/p(x), the operator with the unconditional as background,
  construction (2) is the plain product, they agree only where p(x) is flat, and across an
  invertible warp of the four-quadrant toy construction (1) changes by 4e-17 and the plain product
  by 0.035 to 0.126 total variation (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3
  and run 6).

Rules
- Combine mode builds nothing, so no code, notebook or object.
- Write nothing into either READ_MAP.md or the read-ahead folders.
- Never edit rebuild/ or the step files.
- No commit.
- Each round names one idea of the spine and ends on its four comparison lines; the differences
  table gains a row per difference named; a test prints its numbers labelled ran.

What to hand back
- COMBINE_MAP.md under
  ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/drips/ in its
  combine-<idea-slug>/ folder, holding the Sources table, the idea spine with every idea walked,
  compared or tested, the symbol table and the differences table.
- combined-reading.md in the journey's Obsidian folder, written by the skill at the last idea.
- The retrospective the read closes on, and its /drip-retro command.
- The COMBINE_MAP.md path, which step 4's scene and step 5's cross-links read.
- The proof: the closing check prints its pass or miss line against the five statements, and every
  idea in COMBINE_MAP.md's spine carries a state.

What you will see
Along the way: one idea per round, each ending on its four comparison lines and the drip's control
line.
At the end, a closing report: what was asked, in one sentence; what was done, with the
COMBINE_MAP.md and combined-reading.md paths; PASS or MISS against 5 of 5 statements matched to the
papers (Definition 5.2's three conditions, Theorem 5.3's conclusion, Theorem 6.1's conclusion, and
for each of the quadrant toy and the parts world whether it satisfies or breaks the condition and
why); what is still unclear to me; at most one question.

I judge, from the read's closing check: done when I state all five things with the map closed and each matches the papers' statements.
```

added: the seven sections, from the field-labelled version · the stop's reason as Background (from the stop's "Why the read is critical" line)
dropped: nothing (the earlier version had already replaced "you state the condition and place both toys against it", which had no count, with the five-statement target)
