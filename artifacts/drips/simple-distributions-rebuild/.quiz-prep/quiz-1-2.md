# 📋 Quiz 1.2 preparation: one diffusion expert
Standard: QUIZ_PREP_FORMAT v1

⬅️ [Layer 1: one diffusion expert](../ladder/01-one-diffusion-expert.md) · 🏠 [the guide](../PROMPTS.md) · [the banked solutions](quiz-1-2-solutions.md) ➡️

This file says, for every question quiz 1.2 asks, whether the material that prepares for it exists and teaches in full steps. `quiz-cadence` reads it before publishing and builds the page's read-first list from it. The quiz tests layer 1 itself: the forward process, the sampled-noise identity against the marginal one, and the denoising loss, in the notation of [the step file](../01-one-expert-from-scratch.md).

| Quiz | Goal | Time | Questions | Ready | Blocked | Unprepared by choice | Nodes | Gaps |
|---|---|---|---|---|---|---|---|---|
| 1.2 | derive the ε-score identity and train one 2D expert | 90 min | 8 written + 35 banked = 43 | 43 | 0 | 0 | 32 | 0 |

## Table of contents

- [Words this uses](#words-this-uses)
- [Every question at a glance](#every-question-at-a-glance)
- [Read first: the short path and the full path](#read-first-the-short-path-and-the-full-path)
- [Q1. The one-shot form of the forward process](#q1-the-one-shot-form-of-the-forward-process)
- [Q2. Reading the schedule](#q2-reading-the-schedule)
- [Q3. The toy forward draw](#q3-the-toy-forward-draw)
- [Q4. The sampled-noise identity](#q4-the-sampled-noise-identity)
- [Q5. The squared-error optimum](#q5-the-squared-error-optimum)
- [Q6. The marginal identity](#q6-the-marginal-identity)
- [Q7. Which identity the network learns](#q7-which-identity-the-network-learns)
- [Q8. The denoising loss](#q8-the-denoising-loss)
- [Banked problems](#banked-problems)
- [Gaps](#gaps)
- [Fills](#fills)
- [Dropped problems](#dropped-problems)
- [What the grades rest on](#what-the-grades-rest-on)

## Words this uses

**Q1 to Q8** are the quiz's written questions, one per concept the quiz content names. **B1 to B35** are problems banked from the learner's books, each a question of its own. **R1 to R8** are the concepts the written questions test (Q5's concept is N19 itself, so there is no R5). **N1 to N25** are the concepts those lean on, shared across questions and written once.

**The kinds.** *Learning plan* holds the step file (a walked plan, `/drip-execute-plan`) and the journey plans; *Textbook* holds the journey textbooks and the shelf's PDF pages; *Obsidian note* is the vault; *Video* is a timestamped segment from the video lessons index.

**The grades.** *full* means the passage has all five parts of the full-steps standard: plain-words opening, a worked example with every intermediate, numbered steps, one way it goes wrong with its number, and a check. *partial* names what is missing. *mention* means named or used, not taught. *none* means no passage.

**A node's verdict.** *ready* means at least one kind is full. *gap* would mean none is, which blocks every question that leans on it; no node is a gap now. No node is *held*: no plan in any journey below has a ticked task.

**Path prefixes in the cells.** `J01` is the poe-derivation-foundations journey, `J02` the sampler-correctors journey, `CE` the conditional-expectation journey, `SF` the step file.

## Every question at a glance

| Question | What it tests | Depth | Nodes in its tree | Gaps | Verdict |
|---|---|---|---|---|---|
| Q1 | derive q(x_t \| x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I) from the one-step rule | build | 8 | 0 | ready |
| Q2 | ᾱ_t as a product, its telescope to C_{t+1}, ᾱ_0 = 0.9994, and why the square roots | build | 9 | 0 | ready |
| Q3 | the toy draw x_50 = (0.2167, −0.521) | build | 12 | 0 | ready |
| Q4 | derive ε = −√(1 − ᾱ_t)·∇log q(x_t \| x_0) and its toy value (−0.4153, 1.6613) | build | 13 | 0 | ready |
| Q5 | the minimiser of E\|f(x_t) − ε\|² is E[ε \| x_t] | own | 4 | 0 | ready |
| Q6 | derive E[ε \| x_t] = −√(1 − ᾱ_t)·∇log q_t(x_t) by averaging over the clean point | build | 22 | 0 | ready |
| Q7 | why the trained network learns the marginal identity and not the sampled noise | build | 23 | 0 | ready |
| Q8 | the denoising loss: each part, 0.79 at initialisation, and why it stays above zero | build | 21 | 0 | ready |
| B1 to B35 | banked problems, one row each in [Banked problems](#banked-problems) | | | | 35 ready |

All 43 questions are ready. Six nodes had no full passage at the audit; [chapter 13 sections 13.5 and 13.6](#fills) were written for them, passed their check and merged, and the re-audit grades all six full.

## Read first: the short path and the full path

Minutes are estimates, from each passage's length plus doing its worked example by hand. The list runs bottom up.

**Short path (needed now only), about 245 min.** N6, N7, N14, N16, N17, N18, N19, R1, R2, R4, R6, R7, R8. This is more than the quiz's 90 minutes. The layer's own maths drip (step 2 of [layer 1](../ladder/01-one-diffusion-expert.md)) walks R1, R2, R4 and R6 before the quiz, which takes about 85 minutes off this list once it is done.

**Full path (every node), about 440 min.** The short path, plus N1 to N5 and N10 to N13 (tested by quiz 1.1), N8, N9, N15, N20 to N25.

## Q1. The one-shot form of the forward process

**The question.** Given $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t$, derive $q(x_t\mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$ and the sampling form $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| R1 the one-shot form | N6, N3, N7, N5 | needed now | 20 | [SF step 4](../01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule): partial, the two-step variances are not worked on numbers · [J01 plan 09 §3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/09-the-forward-process.md#3--derive-the-jump-on-paper-before-anything-runs): partial, its failure has no number | [J01 ch 9 §9.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#92-reading-the-image-ladder-and-the-one-jump-form): **full** · [UDL §18.2.1, PDF p. 366](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=366): partial, no numbers, no check · [Little Book §4.1.1, PDF pp. 70 to 71](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=70): partial, no numbers | [The Forward Process](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md), "Try the jump yourself": partial, induction not shown | [CME296 lecture 1, 39:07 to 45:40](https://www.youtube.com/watch?v=tr-CUpw--ck&t=2347s): partial, no numbers or check · [CS236 lecture 16, 16:23 to 19:05](https://www.youtube.com/watch?v=VsllsC2JMGY&t=983s): partial | ready |
| N6 the one-step rule, and that it keeps unit variance | N2, N3 | needed now | 15 | [J02 plan 06 §1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/06-the-forward-process.md#1--the-forward-process-derived): partial, unit variance shown only by numbers | [J01 ch 9 §9.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#92-reading-the-image-ladder-and-the-one-jump-form): **full** (the α form of the step) · [UDL eq. 18.1, PDF p. 363](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=363): mention | The Forward Process, "One small step, fifty of them, and one jump": partial, no step worked by hand | [CME296 lecture 1, 35:19 to 37:57](https://www.youtube.com/watch?v=tr-CUpw--ck&t=2119s): partial | ready |
| N7 the cumulative product ᾱ_t and the induction | N6 | needed now | 15 | [SF step 4](../01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule): **full** (product, telescope, 0.9685 against 0.4783, the swapped-complement failure, printed check) · J01 plan 09 §3: partial | [J01 ch 9 §9.3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#93-abar-from-a-beta-schedule-including-the-log-sum-shortcut): **full** (scaled-linear schedule) | The Forward Process: partial, induction not shown | CME296 lecture 1, 39:07 to 45:40: partial | ready |
| N3 independent Gaussians add their variances | N2, N1 | when it bites (quiz 1.1) | 15 | J01 plan 09 §3: partial | [J01 ch 9 §9.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#91-independent-gaussian-noises-add-their-variances): **full** · [matrix cookbook PDF p. 41](../../../../../../goal-setting/learning/mathematics/textbooks/linear-algebra/matrix-cookbook.pdf#page=41): mention | [The Forward Process](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md), "How the jump works": **full** for the variances (the sum's shape is asserted) | [3Blue1Brown, 3:38 to 8:27](https://www.youtube.com/watch?v=d_qvLDhkg00&t=218s): partial, no numbers | ready |
| N2 mean and variance of aX + b | none held below | when it bites (quiz 1.1) | 10 | J02 plan 06, Engagement Instructions: mention | [Chan PDF p. 200](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=200): not graded, quiz 1.1's node | [Sampling A Diagonal Gaussian](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Sampling%20A%20Diagonal%20Gaussian.md): **full** | none: no listed creator works Var(aX + bY) | ready |
| N1 the isotropic Gaussian density | none held below | when it bites (quiz 0) | 15 | [J01 plan 03 §5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/03-densities-and-the-gaussian.md#5--the-gaussian-by-hand-then-drawn): partial, no numbered method, failure has no number | [Chan PDF pp. 306 to 307](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=306): not graded, quiz 0's node | [Diagonal Covariance](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Diagonal%20Covariance.md): **full** (the diagonal case; σ²I is its special case) | index row "The multivariate normal and its covariance matrix": not graded | ready |
| N5 sampling by μ + σε | N2 | when it bites (quiz 1.1) | 10 | J01 plan 09 §1: mention | Little Book PDF p. 71: mention | [Sampling A Diagonal Gaussian](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Sampling%20A%20Diagonal%20Gaussian.md): **full** | none: a convention with no segment in the index | ready |
| N4 standard deviation against variance | N2 | when it bites (quiz 0) | 5 | CE plan 02 §5: mention | [J01 ch 9 §9.6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#96-the-units-check) with 9.2b's failure line: **full** | Sampling A Diagonal Gaussian: **full** | none | ready |

## Q2. Reading the schedule

**The question.** From $\beta_t$ and $\bar\alpha_t = \prod_{j \le t}(1-\beta_j)$: why $\bar\alpha_t = C_{t+1}$, why $\bar\alpha_0 = 0.9994$ and not 1, why the clip at 0.999 changes $\bar\alpha_{99}$, and why the clean point is scaled by 0.6916 and not 0.4783 at $t = 50$. Leans on N7 and N4 (above).

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| R2 reading the schedule table | N7, N8, N9, N4 | needed now | 15 | [SF step 4](../01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule): **full** (table, telescope, clip to 2.43e-7 and multiplier 2029, printed check against `p1.abar_*`) | J01 ch 9 §9.3: partial for this schedule (it teaches the scaled-linear one) | The Forward Process: partial; its own schedule gives 0.493844 at level 50, not reconciled with 0.4783 | [CME296 lecture 1, 38:00 to 39:06](https://www.youtube.com/watch?v=tr-CUpw--ck&t=2280s): partial · [CS236 lecture 16, 47:58 to 51:46](https://www.youtube.com/watch?v=VsllsC2JMGY&t=2878s): mention | ready |
| N8 the cosine schedule and its clip | none below | link only | 10 | [SF step 4](../01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule): **full** (offset 0 gives 0.000247 against 0.000631; clip) | none: no shelf book works the cosine schedule | [The Order Of Multiplying And Noising](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Order%20Of%20Multiplying%20And%20Noising.md): mention (code only) | none: the index confirms no segment names the cosine schedule | ready |
| N9 code step t = 0 is already noisy | N7 | link only | 5 | [SF step 4, "Indexing convention"](../01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule): **full** (ᾱ_0 = 0.9994 = C_1, the C_t failure, printed check) · J02 plan 07 §2: partial, a different convention | none | The Order Of Multiplying And Noising: mention, and the opposite convention (`abars[0] = 1` is clean) | none: a notation convention no video can carry | ready |

## Q3. The toy forward draw

**The question.** With $x_0 = (0, 0.5)$, $\varepsilon = (0.3, -1.2)$ and $t = 50$, compute $x_{50}$. Leans on R1, R2, N4, N9 (above).

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| R3 one forward draw by hand | R1, R2 | needed now | 5 | [SF step 5, Toy](../01-one-expert-from-scratch.md#step-5-the-noise-is-the-score-rescaled): partial, the result is stated without its products (the [banked solution S1](quiz-1-2-solutions.md#s1-the-toy-forward-draw-b1) writes them) | [J01 ch 9, worked examples 9.2a and 9.2b](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#92-reading-the-image-ladder-and-the-one-jump-form): **full** (the same draw on x_0 = 2.0, ε = 0.5) | The Forward Process, "Try the jump yourself": **full** | none | ready |

## Q4. The sampled-noise identity

**The question.** For one clean point and one sampled noise, derive $\varepsilon = -\sqrt{1-\bar\alpha_t}\,\nabla\log q(x_t\mid x_0)$ and compute the toy score $(-0.4153, 1.6613)$. Leans on R1 (above).

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| R4 the conditional identity | R1, N10, N11, N12, N13 | needed now | 20 | [SF step 5](../01-one-expert-from-scratch.md#step-5-the-noise-is-the-score-rescaled): **full** (the toy, the dropped-divisor and dropped-sign failures, the autograd check against `p1.example_score`) · [J01 plan 11 §3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/11-noise-prediction-is-the-score.md#3--the-identity-derived-twice): partial (conditional half only) | [J01 ch 9 §9.5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#95-eps---sigma_t-grad-log-q): **full** (−0.5521 both ways at four steps, `checks/11_eps_is_score.py`) · [Little Book PDF p. 124](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=124): partial, unshrunk form, no numbers | The Order Of Multiplying And Noising, step 5: mention | [Vizuara lecture 6, 32:45 to 38:19](https://www.youtube.com/watch?v=6ELzfPXUqps&t=1965s): partial, unshrunk form · [38:19 to 40:25](https://www.youtube.com/watch?v=6ELzfPXUqps&t=2299s): partial | ready |
| N10 the log of a Gaussian density | none held below | when it bites (quiz 1.1) | 10 | [J02 plan 03 §1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/03-the-score-of-a-gaussian.md#1--the-score-and-the-score-of-a-gaussian): **full** | J01 ch 9 §9.5 steps 2 and 3: **full** | [The Log Of A Power And Of A Quotient](../../../../../../Obsidian/Mathematics/Algebra/Logarithms/The%20Log%20Of%20A%20Power%20And%20Of%20A%20Quotient.md): partial, no failure | [CME296 lecture 2, 8:17 to 13:22](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=497s): partial | ready |
| N11 the gradient of a squared distance | none held below | when it bites (quiz 0) | 10 | [J02 plan 02 §1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/02-the-gradient-as-a-field.md#1--the-gradient-as-a-field-you-can-draw): **full** | J01 ch 9 §9.5 step 4: partial, the rule is stated · [calculus textbook PDF p. 988](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-textbook.pdf#page=988): not graded, quiz 0's node | The Log Of A Power And Of A Quotient: mention | none graded | ready |
| N12 the score is the gradient of the log density | N10, N11 | when it bites (quiz 1.1) | 10 | J02 plan 03 §1: **full** · J01 plan 07 §3: partial | [Little Book PDF p. 114](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=114): partial, no failure, no check | [The Tempered Density And Its Score](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Tempered%20Density%20And%20Its%20Score.md): partial | [CME296 lecture 2, 6:20 to 14:30](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=380s): partial | ready |
| N13 the score of a Gaussian | N12 | when it bites (quiz 1.1) | 10 | [J01 plan 08 §3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/08-score-of-a-gaussian.md#3--the-score-of-a-gaussian-derived-twice): **full** · J02 plan 03 §1: **full** | Little Book PDF pp. 114 to 115: partial | The Tempered Density And Its Score: partial | [CME296 lecture 2, 21:28 to 24:12](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=1288s): partial | ready |

## Q5. The squared-error optimum

**The question.** Show that, over all functions $f$ of $x_t$, $\mathbb E\lVert f(x_t) - \varepsilon\rVert^2$ is smallest at $f(x_t) = \mathbb E[\varepsilon\mid x_t]$, and name what is left over.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| N19 the squared-error minimiser is the conditional mean (Q5's root) | N18 | needed now | 25 | [CE plan 05 §2](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/05-the-conditional-mean-is-the-best-guess.md#2--prove-it-cold-and-work-both-examples): **full** (die 8/3, 35/12, 11/3; two-point toy floor 0.4496 against the best line 0.5; the cross-term failure; a simulation check) | [CE textbook §5](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/05-the-conditional-mean-is-the-best-guess.md): **full** · [Chan §8.4.3, PDF pp. 539 to 540](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=539): partial, symbols only · [Bishop §1.5.5, PDF pp. 65 to 66](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=65): partial, no numbers · [Little Book PDF p. 126](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=126): mention, stated without proof | none: no vault note found | [Deepia, 1:34 to 5:43](https://www.youtube.com/watch?v=m0OTso2Dc2U&t=94s): partial, open web | ready |
| N18 conditional expectation | N17, N14 | needed now | 15 | [CE plan 03 §2](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/03-conditional-expectation-as-a-function.md#2--work-both-examples-on-paper): **full** · [CE plan 06 §2](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/06-conditioning-with-densities.md#2--the-worked-example-on-paper): **full** | [CE textbook §3, worked examples](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/03-conditional-expectation-as-a-function.md#worked-examples): **full** (the two-point toy, 0.7311 and 0.4621, `verify.py`) · [Chan PDF p. 291](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=291): not graded | none: video index only | [Nazer 4.7, 1:14 to 3:45](https://www.youtube.com/watch?v=86m-cizvK14&t=74s): partial · [CS229 lecture 3, 50:45 to 59:28](https://www.youtube.com/watch?v=Mi8wnYc1m04&t=3045s): partial | ready |
| N17 Bayes' rule for the clean point given the noisy one | N14 | needed now | 20 | CE plan 03 §2: **full** (two-point prior) · CE plan 06 §2: **full** (Gaussian prior) · [GM plan 02](../../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/02-bayes-rule.md): partial · [J02 plan 29 §1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/29-tweedies-formula.md#1--tweedies-formula-derived): partial | [CE textbook §6, worked examples](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/06-conditioning-with-densities.md#worked-examples): **full** (Example 4: posterior 0.7311 from the two spikes) · [Chan PDF p. 105](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=105): not graded | [Kinds Of Composition](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Kinds%20Of%20Composition.md): partial, labels only | [3Blue1Brown, 4:09 to 10:13](https://www.youtube.com/watch?v=HZGCoVF3YvM&t=249s): partial, events not densities · [ritvikmath GMM, 8:47 to 10:13](https://www.youtube.com/watch?v=EWd1xRkyEog&t=527s): partial | ready |
| N14 marginalising over the clean point | none held below | needed now | 15 | [CE plan 02 §3](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/02-joint-and-marginal-summing-out.md#3--integrate-the-noise-out-by-hand): **full** (p(0) = 0.1080; weights dropped gives mass 2.000000) | [UDL eq. 18.10, PDF p. 367](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=367): mention · [Chan PDF p. 266](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=266): not graded | The Order Of Multiplying And Noising, step 6: partial, no plain-words account | [CS229 lecture 3, 23:01 to 26:37](https://www.youtube.com/watch?v=Mi8wnYc1m04&t=1381s): partial, chapter title only | ready |

## Q6. The marginal identity

**The question.** Starting from $q_t(x) = \int q(x\mid u)\,p_{\mathrm{data}}(u)\,du$, derive $\mathbb E[\varepsilon\mid x_t] = -\sqrt{1-\bar\alpha_t}\,\nabla\log q_t(x_t)$. Leans on R4, N19, N18, N17, N14 (above).

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| R6 the marginal identity, in ᾱ notation | R4, N19, N14, N15, N16, N17, N18; N22 and N23 are its other routes | needed now | 30 | [SF step 8](../01-one-expert-from-scratch.md#step-8-the-denoising-loss): partial, no worked numbers and no check ("There are no worked density values for this integral") · [J02 plan 08 §1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/08-noise-prediction-is-the-score.md#1--noise-prediction-is-the-score): partial, the averaging step sits in a tip · [J01 plan 20 §3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/20-score-matching-the-training-objective.md#3--the-objective-derived-three-ways): partial, in σ notation | [J01 ch 13 §13.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#132-denoising-score-matching-and-its-optimum): partial, full in σ notation but not for the shrunk process · [Little Book §6.3, PDF pp. 123 to 127](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=123): partial, no numbers, no failure · [J01 ch 13 §13.5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#135-one-clean-point-against-all-of-them-the-score-the-network-learns): **full** (posterior 0.3339 and 0.6661, four numbered moves, the equal-weight and sampled-noise failures, three routes to (−0.4154, 0.7784)) | none: no vault passage | [Deepia, 5:43 to 10:23](https://www.youtube.com/watch?v=m0OTso2Dc2U&t=343s): partial, open web · [CS236 lecture 13, 29:16 to 44:39](https://www.youtube.com/watch?v=8G-OsDs1RLI&t=1756s): partial, chapter title only | ready |
| N15 differentiating under the integral sign | N14 | when it bites | 10 | [SP plan 01](../../../../../../goal-setting/learning/deep-learning/diffusion-models/06-diffusion-speciation-dynamics/plans/02-speciation/plans/01-tweedie-identity.md): mention · J01 plan 20 §3: mention | [Goodfellow PDF p. 623](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/deep-learning-textbook.pdf#page=623): mention (Leibniz's rule named) · Little Book PDF p. 124: mention · [J01 ch 13 §13.5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#135-one-clean-point-against-all-of-them-the-score-the-network-learns): **full** for the sum (term by term; dropping the prior weights gives (−0.8307, 1.5569); central-difference check), the integral case stated with its condition | none | none | ready |
| N16 the log-derivative identity ∇q = q ∇log q | N12 | needed now | 10 | [J01 plan 06 §7](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/06-the-gradient-field-drawn.md#7--the-gradient-of-a-log-derived-rather-than-quoted): partial, no failure with a number · J02 plan 03 §1, step 8: partial | Little Book PDF p. 118: mention · [J01 ch 13 §13.5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#135-one-clean-point-against-all-of-them-the-score-the-network-learns): **full** (0.2832 × (−0.4154, 0.3358) = (−0.1176, 0.0951); dropping the kernel gives (−0.4154, 0.9987); central-difference check) | none | [CME296 lecture 2, 8:17 to 13:22](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=497s): partial | ready |
| N22 Tweedie's formula | N17, N18, N16 | when it bites | 15 | J02 plan 29 §1: partial · CE plan 06 §2: partial | [CE textbook §6, worked examples](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/06-conditioning-with-densities.md#worked-examples): **full** (Examples 1 and 4: 1.5 − 0.75 = 0.75; 0.5 − 0.0379 = 0.4621; finite-difference check) | none | Deepia, 5:43 to 10:23: partial | ready |
| N23 denoising score matching | N19, N14 | when it bites | 25 | J01 plan 20 §3: **full** (−0.990099, −0.8, −0.2; seed-0 fit −0.9657) | J01 ch 13 §13.2: **full** · [Goodfellow §18.5, PDF p. 635](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/deep-learning-textbook.pdf#page=635): mention | none | CS236 lecture 13, 29:16 to 44:39: partial | ready |

## Q7. Which identity the network learns

**The question.** The toy draw used $\varepsilon = (0.3, -1.2)$. A trained network sees only $x_{50} = (0.2167, -0.521)$: what does it output, and why is that not $(0.3, -1.2)$? Leans on every node of Q4, Q5 and Q6.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| R7 the sampled noise against its posterior average | R4, R6, N19 | needed now | 10 | SF step 5, "Marginal identity": partial, stated, no numbers · J02 plan 08 §1: partial | J01 ch 9 §9.5, note: mention · [J01 ch 13 §13.5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#135-one-clean-point-against-all-of-them-the-score-the-network-learns), worked example 13.5: **full** ((0.3, −1.2) against (0.3, −0.5623)) | none | Deepia, 1:34 to 10:23: partial | ready |

## Q8. The denoising loss

**The question.** Write $\mathcal L = \mathbb E_{x_0,t,\varepsilon}\,\tfrac12\lVert\hat\varepsilon(x_t,t) - \varepsilon\rVert^2$, say what the ½, the expectation and the uniform $t$ each do, read the 0.79 at initialisation, and say why a trained model ends near 0.2 and not 0. Leans on R3 and N19 (above).

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| R8 the denoising loss | R3, N19, N20, N21, N25, N24 | needed now | 20 | [SF step 8](../01-one-expert-from-scratch.md#step-8-the-denoising-loss): **full** (0.79 → 0.2, the `create_graph=False` failure, the printed "45 of 45" and `p1.init_loss_on_bar` check) | [UDL §18.5 and Algorithm 18.1, PDF pp. 376 to 377](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=376): partial, no numbers | [What An Expert Learns From Its Training Set](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/What%20An%20Expert%20Learns%20From%20Its%20Training%20Set.md): partial | [CME296 lecture 1, 1:21:12 to 1:24:04](https://www.youtube.com/watch?v=tr-CUpw--ck&t=4872s): partial · [Vizuara DDPM practical, 30:07 to 39:32](https://www.youtube.com/watch?v=6to_EG3SJgs&t=1807s): partial | ready |
| N20 a loss as an expectation, estimated by a batch mean | none below | when it bites | 10 | SF step 8: partial, the expectation is explained, no batch worked · [J01 plan 20 §1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/20-score-matching-the-training-objective.md#1--the-linear-toy-every-optimum-predicted-then-printed): partial | UDL PDF p. 373: mention · [J01 ch 13 §13.6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#136-the-denoising-loss-by-hand-and-the-floor-it-cannot-go-below): **full** (three draws, batch mean 0.3439; one draw alone reads 0.0214) | What An Expert Learns From Its Training Set: partial, no loss value worked | [CS229 lecture 3, 11:33 to 21:01](https://www.youtube.com/watch?v=Mi8wnYc1m04&t=693s): partial | ready |
| N21 the ½ as a mean over two coordinates | none below | when it bites | 5 | SF step 8: partial, "would double every reported loss" with no number | [J01 ch 13 §13.6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#136-the-denoising-loss-by-hand-and-the-floor-it-cannot-go-below): **full** (0.0214 against 0.0428 without the ½) | What An Expert Learns: mention (sums, then means) | none | ready |
| N25 the loss at its optimum is not zero | N19 | when it bites | 15 | CE plan 05 §2: **full** (floor 0.4496) · [CE plan 04 §3 and §4](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/04-the-tower-law-and-total-variance.md): **full** for E[Var(X \| Y)] | [Bishop eq. 1.90, PDF p. 66](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=66): partial, no numbers · [Chan PDF pp. 541 to 542](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=541): partial · J01 ch 13 §13.2 step 3 (3.2 at the optimum, not zero): partial · [J01 ch 13 §13.6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#136-the-denoising-loss-by-hand-and-the-floor-it-cannot-go-below): **full** in the ε form (ᾱ_t for N(0, I) data; 0.1019 at the two-point toy) | none | none | ready |
| N24 where the ε loss comes from | none below | link only | 30 | [J01 plan 21 §2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/21-the-elbo-to-the-epsilon-loss.md#2--from-the-bound-to-the-weighted-epsilon-loss): **full** | [J01 ch 13 §13.4](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#134-the-per-step-weights-by-hand): not re-graded here · UDL §18.4, PDF pp. 370 to 376: partial | VAE Training Dynamics: mention, the VAE bound | [CME296 lecture 1, 1:18:03 to 1:27:03](https://www.youtube.com/watch?v=tr-CUpw--ck&t=4683s): partial | ready |

## Banked problems

Problems from the learner's books and tutorials, from the easiest that shows the mechanic to the hardest that bites, per concept. Each is a question of its own. Its nodes are the ones above; its verdict follows them. Every row has a matching row in the [worked-examples index](../../../../../../Obsidian/Mathematics/references/exercises/Worked%20Examples.md). "S*n*" means the solution written in [the banked solutions](quiz-1-2-solutions.md).

| Id | Concept, depth | What it asks | Source | Difficulty | Full-steps solution | Nodes | Verdict |
|---|---|---|---|---|---|---|---|
| B1 | forward process, build | the toy draw x_50 from x_0 = (0, 0.5), ε = (0.3, −1.2) | SF step 5, Toy; `p1.example_x_t` | easy | [S1](quiz-1-2-solutions.md#s1-the-toy-forward-draw-b1) | R3 | ready |
| B2 | forward process, build | z = aε₁ + bε₂ has mean 0 and variance a² + b² | UDL Problem 18.2, PDF p. 385 | easy | J01 ch 9 §9.1 (same statement, derived and worked) | N3 | ready |
| B3 | forward process, build | one step keeps variance 1 | UDL Problem 18.1, PDF p. 385 | easy | [S2](quiz-1-2-solutions.md#s2-one-step-keeps-the-variance-at-1-b3-udl-181) | N6, N3 | ready |
| B4 | forward process, build | two-step noise variance both ways, 0.28 | J01 ch 9, worked example 9.1 | easy | in the chapter | N3, N7 | ready |
| B5 | forward process, build | continue to z_3 | UDL Problem 18.3, PDF p. 385 | medium | [S3](quiz-1-2-solutions.md#s3-three-steps-in-one-jump-b5-udl-183) | R1 | ready |
| B6 | forward process, build | four-step toy schedule, iterated and jumped; x_2 = 1.9616 | J01 ch 9, worked example 9.2a | medium | in the chapter | R1, N7 | ready |
| B7 | forward process, build | √(ᾱ_t / ᾱ_{t−1}) = √(1 − β_t) | UDL Problem 18.9, PDF p. 386 | medium | [S4](quiz-1-2-solutions.md#s4-the-ratio-of-consecutive-kept-shares-b7-udl-189) | N7 | ready |
| B8 | forward process, build | ᾱ_t = C_{t+1} by telescoping, and what the clip does to ᾱ_99 | SF step 4 | hard | in the step file | R2, N8, N9 | ready |
| B9 | forward process, build; Python, `cosine_beta_schedule(T, s)` | build the schedule and print ᾱ at 0, 25, 50, 75, 99 | SF step 4, Build | medium | the printed `p1.abar_*` keys | R2 | ready |
| B10 | ε-score identity, build | score of a 1D Gaussian, −(x − μ)/σ² | Little Book PDF p. 114 | easy | in the book | N13 | ready |
| B11 | ε-score identity, build | the toy score (−0.4153, 1.6613) | SF step 5 | easy | in the step file | R4 | ready |
| B12 | ε-score identity, build | the score two ways at step 20, −0.5521, and at three more steps | J01 ch 9, worked example 9.5 | easy | in the chapter | R4 | ready |
| B13 | ε-score identity, build | the score as −Σ⁻¹(x − μ) with Σ = (1 − ᾱ)I | Little Book PDF p. 115 | medium | [S5](quiz-1-2-solutions.md#s5-the-score-through-the-covariance-b13) | R4, N13 | ready |
| B14 | ε-score identity, build | the score grows 24.16-fold while σ times it stays 0.5 | J01 ch 9, worked example 9.6 | medium | in the chapter | R4, N4 | ready |
| B15 | ε-score identity, build | the unshrunk corruption x̃ = x + ε against the forward process | Little Book PDF p. 124 | medium | [S6](quiz-1-2-solutions.md#s6-the-unshrunk-corruption-against-the-forward-process-b15) | R4 | ready |
| B16 | ε-score identity, build; Python, autograd | the score by autograd against −ε/σ | SF step 5, Build | medium | the printed line against `p1.example_score` | R4 | ready |
| B17 | squared-error optimum, own | die guessed from its parity: 8/3, 35/12, 11/3 | CE textbook §5, Example 1; CE plan 05 task 2.2 | easy | in the chapter | N19 | ready |
| B18 | squared-error optimum, own | best linear estimate aY of X from Y = X + N | Chan Exercise 5.16, PDF p. 334 | medium | [S7](quiz-1-2-solutions.md#s7-the-best-straight-line-denoiser-b18-chan-exercise-516) | N19 | ready |
| B19 | squared-error optimum, own | ML, MAP and MMSE for an exponential likelihood and prior | Chan Example 8.22, PDF pp. 543 to 544 | medium | in the book | N19, N17 | ready |
| B20 | squared-error optimum, own | the two-point toy: floor 0.4496, best line 0.5 | CE plan 05 task 2.3; CE textbook §5, Example 3 | medium | in the plan and chapter | N19, N25 | ready |
| B21 | squared-error optimum, own | the conditional mean for a vector target | Bishop Exercise 1.25, PDF p. 83 | medium | [S8](quiz-1-2-solutions.md#s8-the-conditional-mean-for-a-vector-target-b21-bishop-125) | N19 | ready |
| B22 | squared-error optimum, own | Gaussian posterior, its MMSE estimate and its MSE | Chan Exercise 8.5, PDF pp. 553 to 554 | hard | none: to be written (practice only) | N19, N17 | ready (practice only: no solution) |
| B23 | marginal identity, build | the two-point toy by Bayes on densities, then by Tweedie: 0.4621 both ways | CE textbook §6, Example 4 | medium | in the chapter | N17, N22, N16 | ready |
| B24 | marginal identity, build | the smoothed score is the posterior average of conditional scores | Little Book §6.3, PDF pp. 123 to 126 | medium | in the book | R6 | ready |
| B25 | marginal identity, build | the slope a denoiser learns, −1/(1 + σ²) = −0.8 | J01 ch 13, worked example 13.2 | medium | in the chapter | N23 | ready |
| B26 | marginal identity, build | E[ε \| x_t] on N(0, I) data at the toy: (0.1565, −0.3763) | constructed on Chan Theorem 8.3, PDF p. 545 | medium | [S9](quiz-1-2-solutions.md#s9-the-noise-a-network-learns-on-gaussian-data-b26) | R6, R7 | ready |
| B27 | marginal identity, build | E[ε \| x_t] on two clean points: (0.3, −0.5623) | constructed; J01 ch 13, worked example 13.5 | hard | [S10](quiz-1-2-solutions.md#s10-the-noise-a-network-learns-on-two-clean-points-b27) | R6, R7, N15, N16 | ready |
| B28 | marginal identity, build; Python, NumPy | fit a slope by denoising score matching at σ = 0.5; seed 0 gives −0.797997 | J01 plan 20 task 1.2; ch 13 §13.2 step 5 | medium | in the chapter | N23, N20 | ready |
| B29 | denoising loss, build | read the training algorithm term by term | UDL Algorithm 18.1, PDF p. 377 | easy | in the book | R8 | ready |
| B30 | denoising loss, build | one draw's loss with and without the ½: 0.0214 and 0.0428 | constructed on SF step 8 | easy | [S11](quiz-1-2-solutions.md#s11-one-draws-loss-and-what-the-half-does-b30) | N21 | ready |
| B31 | denoising loss, build | a three-draw batch mean, 0.3439 | constructed on SF step 8 | medium | [S12](quiz-1-2-solutions.md#s12-a-batch-mean-stands-in-for-the-expectation-b31) | N20 | ready |
| B32 | denoising loss, build | the reparameterised target's coefficient identity | UDL Problem 18.10, PDF p. 386 | medium | in the book, eq. 18.33, PDF p. 375 | N7 | ready |
| B33 | denoising loss, build | the first step's loss in noise form, factor 0.0251 | UDL Problem 18.11, PDF p. 386 | hard | [S13](quiz-1-2-solutions.md#s13-the-first-steps-loss-in-noise-form-b33-udl-1811) | R8, N7 | ready |
| B34 | denoising loss, build | the floor: ᾱ_t for Gaussian data, 0.1019 at the two-point toy | constructed on Bishop eq. 1.90 and Chan PDF pp. 541 to 542 | hard | [S14](quiz-1-2-solutions.md#s14-the-floor-under-the-loss-b34) | N25, R6 | ready |
| B35 | denoising loss, build; Python, `loss_fn(model, x0)` | write the loss, print 0.79 at initialisation and "45 of 45" | SF step 8, Build | medium | the printed `p1.init_loss_on_bar` | R8 | ready |

**Per concept:** forward process 9 (build), ε-score identity 7 (build), squared-error optimum 6 (own), marginal identity 6 (build), denoising loss 7 (build). Programming exercises: B9, B16, B28, B35.

## Gaps

No node is a gap. Every node has a full passage in at least one kind, and no video is missing for a node that has no other full passage.

| Node | Kinds missing a full passage | Questions it blocks | Who fills it | Fill done |
|---|---|---|---|---|
| none | | | | |

## Fills

| Date | Passage | Path | Owner | State |
|---|---|---|---|---|
| 2026-10-20 | Chapter 13 §13.5 "One clean point against all of them: the score the network learns", which fills R6, R7, N15 and N16 | [J01 ch 13 §13.5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#135-one-clean-point-against-all-of-them-the-score-the-network-learns), with `textbook/verify_ch13_q12.py` and `textbook/figures/ch13-marginal-score.png` | `codex-author`, job 20261020-154432-5781f2 | checked and merged; graded full |
| 2026-10-20 | Chapter 13 §13.6 "The denoising loss by hand, and the floor it cannot go below", which fills N20 and N21 and gives N25 its ε form | [J01 ch 13 §13.6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#136-the-denoising-loss-by-hand-and-the-floor-it-cannot-go-below), with `textbook/figures/ch13-denoising-loss-floor.png` | `codex-author`, same job | checked and merged; graded full |

**How the check ran.** Codex wrote the sections in three rounds; Claude inserted the final additions at the places Codex named without changing a word. `verify_ch13_q12.py` prints every number on the page; all 118 math spans parse in KaTeX 0.16.11 with no error and no stray delimiter; the page was rendered through markdown-it and KaTeX in headless Chromium and read from screenshots, because the VS Code renderer the bridge uses is not installed on mscluster. The marginal-score figure's left panel was re-laid out by Claude so its labels no longer overlap, and its sampled-noise line was recoloured purple to match the page's colour key; no number or label text changed.

## Dropped problems

**UDL Notebooks 18.1 and 18.3** (the diffusion encoder and the reparameterised model, on [GitHub](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap18)). Found, not read, so their exercises cannot be graded.

**Goodfellow §18.5, PDF p. 635.** It defines the smoothed density and sets no exercise.

**UDL Problems 18.4 to 18.8.** They belong to the reverse step and the ELBO, layer 1's step 10 and a link-only node, not to this quiz.

## What the grades rest on

**Read in full by this run:** the step file's steps 4, 5 and 8; J01 chapter 13 §13.5 and §13.6 as merged; J01 textbook chapter 9; chapter 13 §13.2; CE plan 05 §1 and §2; CE textbook §3 Examples 3 and 4 and §6 Example 4 and its opening; UDL PDF pp. 363 to 368, 370 to 378 and 385 to 386; the Little Book PDF pp. 70 to 73 and 114 to 127; Chan PDF pp. 332 to 334 and 536 to 554; Bishop PDF pp. 65 to 67 and Exercises 1.25 and 1.26; Goodfellow PDF p. 635; the video index sections named in the cells.

**Read in full by the two readers this run sent out** (one for plans, one for vault notes, each grading against the same five parts): J01 plans 03, 04, 05, 07, 08, 09, 10, 11, 20, 21; J02 plans 01, 02, 03, 05, 06, 07, 08, 29; CE plans 02, 03, 04, 05; the generative-models Bayes plan; the speciation Tweedie plan; vault notes The Forward Process, Diffusion Models, Multivariate Normal Distribution, Diagonal Covariance, Sampling A Diagonal Gaussian, The Tempered Density And Its Score, Why The Normaliser Has No Slope In X, The Order Of Multiplying And Noising, What An Expert Learns From Its Training Set, The Log Of A Power And Of A Quotient.

**Read in part:** J01 plan 06 (sections 4 to 7), CE plan 06 (to its engagement instructions), the from-scratch VAE plan (sections 2 and 3), three vault composition notes.

**Found, not graded:** the shelf pages marked "not graded" in the cells, all of them quiz 0 or quiz 1.1 nodes; UDL Notebooks 18.1 to 18.4; ten vault notes listed by the vault reader whose text holds none of the missing concepts' terms.

**Video grades rest on the index's own description of each segment**, not on watching: a segment marked "chapter boundary" in the index is good to the chapter, and none works the step file's numbers, so none is graded above partial.
