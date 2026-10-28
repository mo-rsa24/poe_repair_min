to: the forward-process-and-the-score walk (dispatch.md, on integrate)
from: keep, all five taught pieces, one writer in sequence

destination: artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md, Step 4 (uncommitted; `git diff` on that file is the review)

## Piece 01: the forward process > scaling a draw

kept: artifacts/drips/forward-process-and-the-score/kept/01-scaling-a-draw.md (rung from message 4f7410b9, the real-schedule example from e5e8e602, the t = 75 rounding correction from 7eef2c2b)

**Where it went.** Step 4, "Derivation of the one-shot formula", as the passage anchored `#step4-the-scaling-rule` ("The first rule: scaling a draw scales its variance by c²").

**What it merged with.** The derivation's old one-line opening now names the two rules it needs. The Definition's "Why the square roots are there" keeps its wording and gains a pointer to the derivation. The derivation's own "Why the square roots are there" now says "by the scaling rule". The t = 75 correction is written as a fact: the exact c gives (0.2793, −1.1170) and mean square 0.8665, while rounding c to 0.9309 first gives 0.8666.

**Steps and examples added.** The variance definition and the four-line derivation, both with underbraces. The coin at c = 0.7223. Step 50, with the dropped-root case giving 0.2722. Step 25, with the share slip giving 0.0270. β₀ = 0.000631. The four-value draw at t = 75, with squares 1.29976 and 0.43325 computed in python. The 2D toy at t = 25, 50 and 75. The ASCII bars of c against c², with a "How to read this".

**Links on words.** Variance (vault note, #what-variance-measures). Theorem 3.5 (ii) (probability for data science p. 151, read). Derive 2 and the first page's chart Dog fades, noise grows (scene).

## Piece 02: the forward process > independent variances add

kept: artifacts/drips/forward-process-and-the-score/kept/02-independent-variances-add.md (message e5e8e602)

**Where it went.** The same derivation, anchored `#step4-variances-add`.

**What it merged with.** The old opening's "variances add" link to the matrix cookbook p. 41 became a numbered pair on the boxed rule's sentence: [1] MML p. 200, §6.4.5; [2] cookbook p. 41.

**Steps and examples added.** The derivation with every move named. The step a reader stalls on, E[XY] = E[X]E[Y] for independent draws, links to probability for data science p. 280, which proves it. Two coins. Y = X giving 4, with Derive 3's preset Noise copies the picture (a = 0.6, b = 0.8, giving 1.96; preset confirmed in the live page source). 0.4054ε_a + 0.7223ε_b = 0.6861, typical miss 0.8283 against 1.1277.

**Links on words.** The variance of a sum and the cross term (vault Sums Of Independent Gaussians, #why-independence-kills-the-cross-term). Independent (plan 01 §2, the joint tables). Chapter 9, section 9.1 (ours).

## Piece 03: the forward process > two steps, then every step

kept: artifacts/drips/forward-process-and-the-score/kept/03-two-steps-then-every-step.md (message 33967dd8)

**Where it went.** The same derivation, anchored `#step4-two-steps-then-every-step`. It runs from "Two steps" through "Every step, by induction" to "The stack is one draw" (piece 04's first section, filed here because it closes the derivation).

**What it merged with.** The old aligned two-step equation and its "What the equation says" became the piece's fuller z, j, j + 1 derivation in s_t notation, with the name-clash paragraph at the top of the derivation. The Derive 5 still, the β = 0.3 and 0.5 paragraph and the four "Why" paragraphs are kept.

**Steps and examples added.** Steps 0 and 1. Step 50, with the typical misses 0.7001 and 0.1776 against 0.8777. The first case on numbers (ᾱ₀ = 0.9994). The induction move checked at t = 1, (1 − 0.0011169) × 0.9993687 = 0.9982525. The broken run extended, computed: forgetting the shrink at every step gives total variance 1.2085 at step 50 and 7.879 at step 99.

**Links on words.** UDL §18.2.1, eqs. 18.4 to 18.6, p. 366 (read; its z_t, x and α_t translated in the line). From Many Steps To One Jump (vault, #how-two-steps-become-every-step). Derive 6 (scene). "A sum of independent Gaussians is itself a Gaussian" goes to the vault Sums note with a numbered pair: [1] MML p. 207, eqs. 6.78 and 6.79; [2] probability for data science p. 299, Theorem 5.12, both read.

## Piece 04: the forward process > the one-shot draw and its density

kept: artifacts/drips/forward-process-and-the-score/kept/04-the-one-shot-draw-and-its-density.md (message 07b1a108)

**Where it went.** "#### The density: from the draw to q(x_t | x_0)", anchored `#step4-the-one-shot-density`. Another writer added this heading to the file while this keep was running, so it holds the piece in place of "All the words together: the formula". The formula heading is unchanged.

**What it merged with.** That heading already had the name, the centre, the spread, the shape (Chan Theorem 2.2), the density, the t = 50 check and the cancellation into the score. Those were kept, and the t = 50 check gained the noise vector and the step-by-step numbers. My own duplicate Chan link in the derivation was removed.

**Steps and examples added.** The general 2D density read term by term, with the front factor as a product of two 1D fronts (probability for data science p. 307, read). The 1D case, peak 0.5523. t = 25 (q = 0.4505). t = 75, computed: x₇₅ = (0.2793, −0.9344), |d|² = 1.3258, q = 0.0855. The exponent fixed at −|ε|²/2 = −0.765 with the cancellation drawn as underbraces. The broken centre, exponent −0.5987 and q = 0.1676, computed. The still the-one-shot-draw-and-its-density.png, with a "How to read this".

**Links on words.** The exponent (vault The Density Of The One Jump, #the-exponent-at-the-drawn-point).

## Piece 05: the cosine schedule > the boundary values C_k by hand

kept: artifacts/drips/forward-process-and-the-score/kept/05-the-cosine-boundary-values-by-hand.md (message 7a43103b, plus the squaring correction from ffaeba2b)

**Where it went.** Step 4's "The cosine schedule" paragraphs. On-paper 1 to 5 are anchored `#step4-the-quarter-circle-by-hand` and the worked posts `#step4-four-posts-by-hand`. The shape reading went under "#### Then the shape word: cosine".

**What it merged with.** The fence posts went into the opening "The cosine schedule" paragraph. Why δ went into "Why the offset is there", which keeps its 0.000631, 0.0251, 0.000247 and 0.0157. On-paper 6 went into "Why the outer denominator is there", which keeps its two sentences. The term-by-term reading of the formula was already said by the existing "Why k/T", "Why 1 + δ" and "Why π/2" paragraphs, so it was not repeated. The π/2 = arc length point sits in On paper 3.

**Steps and examples added.** On paper 1 to 6 as paragraphs, with underbraced equations for cos² + sin² = 1, the position p_k, and the 46.25° point against (√ᾱ₅₀, √(1 − ᾱ₅₀)). The four posts in one aligned block. β₅₀ worked as a fraction, computed, showing that 0.99984 cancels: 1 − 0.47819/0.49377 = 0.0315. The slope of cos² derived with the chain rule, with values computed: −0.0249 per rad at post 0 against −0.9990 at 46.25°. The paper's reason and the pixel-bin rule, quoted from p. 4. The pixel-bin rule checked against T, computed: √β₀ = 0.0064 at T = 1000 and 0.0251, about 3.2 bins, at T = 100. Under the cosine heading, "What the shape buys" with the paper's design aim, quoted.

**Links on words.** A boundary value (vault The Cosine Noise Schedule, #the-boundary-values-c_k). The quarter circle (vault The Quarter Circle Under The Cosine Schedule, #build-the-quarter-circle-by-hand). The unit circle (trig handbook p. 10). Radians (trig handbook p. 7). Pythagoras (pre-algebra handbook p. 72). Its slope (calculus handbook p. 19, d/dx cos u). Section 3.2, p. 4 (arxiv). Timetable 2 and Timetable 5 (scene).

**Foot list.** The cosine heading's list gains a 🗺️ Ours line for both vault notes.

**Left out on purpose.** The message's "Scene: being added" lines. The circle frames, ruler, every-15° presets and fraction bars they describe do not exist on the live page (version 1792566272-bc54 has Derive 1 to 9 and Timetable 1 to 5 only), so nothing links them.

## Links placed in the other direction

All are link-only, each wrapping words already in the target, and each checked against the disk (file exists, `<a id>` present):

- Variance.md: "why the forward formula takes a square root" → #step4-the-scaling-rule
- Sums Of Independent Gaussians.md: "The diffusion forward process" → #step4-variances-add
- From Many Steps To One Jump.md: "the product this note proves" → #step4-two-steps-then-every-step
- The Density Of The One Jump.md: "Step 4" (in "Step 4 writes q(x_t | x_0)") → #step4-the-one-shot-density
- The Cosine Noise Schedule.md: "traced to where it came from" → #step4-four-posts-by-hand
- The Quarter Circle Under The Cosine Schedule.md: "a length you can measure with a ruler" → #step4-the-quarter-circle-by-hand
- the step file's Definition, "Why the square roots are there" → #step4-the-scaling-rule
- the step file's Step 5: "the forward displacement" → #step4-the-one-shot-density; "a variance with the standard deviation" → #step4-the-scaling-rule

Every relative link in the step file's diff (44 unique targets, including the other writer's) resolves on disk with its anchor. Every PDF page linked was opened as text first.

## The squaring question

**The step file says** (Step 4, "Why π/2 and the square are there", unchanged): "Squaring keeps the boundary nonnegative and makes it represent a variance share; without the square, the schedule would follow a different curve."

**The walk says** (kept/05, the correction message ffaeba2b; dispatch.md, the "correction [vault writer]" line): on a quarter turn cos θ is already in [0, 1], so squaring is not what keeps it non-negative. On the circle, cos θ is the signal multiplier √ᾱ and cos²θ is its variance share (share = multiplier², the scaling rule). Nichol and Dhariwal call the cos² shape an arbitrary choice. The paper confirms that last point on p. 4: "We chose to use cos2 in particular because it is a common mathematical function with the shape we were looking for. This choice was arbitrary".

**Question.** Replace the sentence with the walk's reading, roughly: "On a quarter turn cos θ is already between 0 and 1. The square turns the signal multiplier cos θ into its variance share, by the scaling rule, and Nichol and Dhariwal call the choice of cos² itself arbitrary. Without the square the schedule would follow a different curve." Or keep it as is?

## Citation verdict

Verified against the arxiv PDF of Nichol and Dhariwal (2021), 2102.09672, p. 4, §3.2. The PDF is not on the paper shelf; it was fetched to the scratchpad and read as text.

- The offset prevents β_t from being too small near t = 0 because "having tiny amounts of noise at the beginning of the process made it hard for the network to predict ϵ accurately enough". This matches what the walk recalled.
- On s = 0.008: "we selected s such that √β0 was slightly smaller than the pixel bin size 1/127.5, which gives s = 0.008". The walk's "a tuned choice, with no derivation behind it" is too strong, because a stated selection rule exists. The step file writes the rule.
- The rule holds at T = 1000 (√β₀ = 0.0064 < 0.0078, computed) but not at this page's T = 100 (0.0251). The paper does not say which T the selection used, and it trains with T = 4000 elsewhere, where the value is 0.0031.

## Not done

- No new video moments. Step 4's existing caption-checked tables already cover the two-step proof (CME296 39:07) and the forward process, and adding a new moment would need its captions read.
- No textbook page teaches proof by induction on the shelf, so "induction" carries no link.
- The other writer's density heading is already reflected above. Its "Into the density" line names the walk, which keep did not change.
- The squaring sentence is unchanged, per the question above.
