# Quiz 1.1 preparation: the college maths under layer 1
Standard: QUIZ_PREP_FORMAT v1

⬅️ [Layer 1: one diffusion expert](../ladder/01-one-diffusion-expert.md) · 🏠 [the guide](../PROMPTS.md) · [the step file](../01-one-expert-from-scratch.md) ➡️

| Where | State | Next |
|---|---|---|
| quiz 1.1, audited, banked, gap filled | 22 questions, all ready | `quiz-cadence` builds the page |

## Table of contents

- [What this quiz prepares](#what-this-quiz-prepares)
- [The top table](#the-top-table)
- [Read first](#read-first)
- [Q1: the forward draw is a Gaussian](#q1-the-forward-draw-is-a-gaussian)
- [Q2: the score of a Gaussian](#q2-the-score-of-a-gaussian)
- [Q3: least-squares regression](#q3-least-squares-regression)
- [Q4: the score of the forward draw at the toy](#q4-the-score-of-the-forward-draw-at-the-toy)
- [Q5: what a squared-error fit of the noise returns](#q5-what-a-squared-error-fit-of-the-noise-returns)
- [Banked problems](#banked-problems)
- [Problems searched for and dropped](#problems-searched-for-and-dropped)
- [What the grades rest on](#what-the-grades-rest-on)
- [Gaps](#gaps)
- [Fills](#fills)

## What this quiz prepares

Quiz 1.1 tests the college maths that layer 1 stands on, for the goal "derive the ε-score identity and train one 2D expert", with 40 minutes to prepare. It has three concepts: $\sqrt{\bar\alpha}\,x_0 + \sqrt{1-\bar\alpha}\,\varepsilon$ read as a Gaussian with its mean and variance, the score of a Gaussian $-(x-\mu)/\sigma^2$, and least-squares regression. No quiz page exists yet, so the five questions below are the roots `quiz-cadence` will ask: one mechanic question per concept (Q1 to Q3) and two that join them (Q4 and Q5).

**Notation and numbers.** The notation is the step file's: $q(x_t \mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t) I)$ over $T = 100$ steps of a cosine schedule. Every worked number in an answer comes from the `p1.` keys that `python -m rebuild.walk_numbers` prints (`rebuild/out/walk-numbers.json`), from [the step file](../01-one-expert-from-scratch.md), or from a cited textbook page. The ones the questions use: $\bar\alpha_{50} = 0.4783$ and $\sqrt{1-\bar\alpha_{50}} = 0.7223$ (`p1.abar_50`, `p1.sqrt_1m_abar_50`), $\sqrt{0.4783} = 0.6916$ (step 4), the toy $x_0 = (0, 0.5)$, $\varepsilon = (0.3, -1.2)$, $x_{50} = (0.2167, -0.521)$ (`p1.example_x_t`) and its score $(-0.4153, 1.6613)$ (`p1.example_score`).

## The top table

A question is ready when every node in its tree has a full passage in at least one kind of material, and blocked while any node is a gap.

| Question | Root concept | Nodes | Gaps | Verdict |
|---|---|---|---|---|
| Q1 | the forward draw given $x_0$ is $\mathcal N(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$ | 5 | 0 | ready |
| Q2 | the score of a Gaussian is $-(x-\mu)/\sigma^2$, and $-\Sigma^{-1}(x-\mu)$ in $d$ dimensions | 4 | 0 | ready |
| Q3 | least squares returns the mean: over constants the mean, over functions of the input the conditional mean | 6 | 0 | ready |
| Q4 | the score of $q(x_{50} \mid x_0)$ at the toy is $-\varepsilon/0.7223 = (-0.4153, 1.6613)$ | 10 | 0 | ready |
| Q5 | a squared-error fit of $\varepsilon$ from $x_t$ returns $\mathbb E[\varepsilon \mid x_t]$, not the drawn $\varepsilon$ | 11 | 0 | ready |
| B1.1, B1.2, B1.4, B1.6, B1.7, B1.8 | banked: the forward draw's mean and variance | 3 to 6 each | 0 | 6 ready |
| B2.2 to B2.7 | banked: the score of a Gaussian | 5 to 8 each | 0 | 6 ready |
| B3.1 | banked: the best constant guess | 3 | 0 | ready |
| B3.2, B3.3, B3.4, B3.6 | banked: the least-squares line | 4 each | 0 | 4 ready |

Counts: 22 questions (5 authored, 17 banked); 22 ready, 0 blocked, 0 unprepared by choice. 16 shared nodes plus 17 problem nodes; 0 gaps. N14 was the one gap and is filled by the vault note Least Squares Line. A banked problem's node count is its own node plus the shared nodes it leans on and theirs. The banked rows are listed one by one in [Banked problems](#banked-problems).

## Read first

The *needed now* nodes, bottom up, with minutes to learn (estimate: reading time plus the worked example by hand).

**Short path, needed now only: 53 minutes (estimate).** This is 13 minutes over the 40 the learner has. With 40, read N1, N6, N13 and N11 (40 minutes) and take Q4 and Q5 on the strength of those.

1. N1, the forward draw as a Gaussian, 12 min: [chapter 9, section 9.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#92-reading-the-image-ladder-and-the-one-jump-form)
2. N6, the score of a Gaussian, 8 min: [chapter 6, section 6.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#61-the-gradient-of-a-quadratic-form)
3. N10, the score of the forward draw, 8 min: [step 5 of the step file](../01-one-expert-from-scratch.md#step-5-the-noise-is-the-score-rescaled), from "Conditional identity" to the toy's score
4. N13, the least-squares slope, 5 min: [conditional expectation, section 5, Example 3, step 3](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/05-the-conditional-mean-is-the-best-guess.md#worked-examples)
5. N11, the conditional mean is the best guess, 15 min: [conditional expectation, section 5, the proof and Example 1](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/05-the-conditional-mean-is-the-best-guess.md#the-proof-worth-knowing)
6. N16, the denoising loss read as least squares, 5 min: [step 8 of the step file](../01-one-expert-from-scratch.md#step-8-the-denoising-loss), the definition and the first paragraph of "Derivation of why it learns the score"

**Full path, every node: 154 minutes (estimate).** The short path plus N2 (10), N3 (8), N4 (6), N5 (12), N7 (10), N8 (8), N9 (8), N12 (12), N14 (12) and N15 (15).

## Q1: the forward draw is a Gaussian

The question: given $x_0$, the draw $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ with $\varepsilon \sim \mathcal N(0, I)$ is Gaussian; state its mean and covariance, and evaluate them at $t = 50$ for $x_0 = (0, 0.5)$: mean $(0, 0.3458)$, covariance $0.5217\,I$.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| N1 the forward draw given $x_0$ is $\mathcal N(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$ | N2, N3, N4 | needed now | 12 | [PoE plan 09, instructions 3.2 to 3.6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/09-the-forward-process.md#instructions): partial, no failure with its number (the plan defers it to section 9.1) | [chapter 9, section 9.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#92-reading-the-image-ladder-and-the-one-jump-form): full | [The Forward Process](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md#how-the-jump-works): full | [CME296, lecture 1, 39:07 to 45:40](https://www.youtube.com/watch?v=tr-CUpw--ck&t=2347s): partial, no worked numbers, no failure | full |
| N2 independent Gaussian noises add their variances, and the sum is Gaussian | N5 | when it bites | 10 | PoE plan 09, instructions 3.3 and 3.4: partial, uses the rule and sends the reader to section 9.1 | [chapter 9, section 9.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#91-independent-gaussian-noises-add-their-variances): full | [The Forward Process, how the jump works](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md#how-the-jump-works): partial, the rule is stated and checked, its proof is cited to section 9.1 | [3Blue1Brown, Gaussian + Gaussian = Gaussian, 3:38 to 8:27](https://www.youtube.com/watch?v=d_qvLDhkg00&t=218s): partial, no numbers | full |
| N3 a shifted and scaled Gaussian: $m + s\varepsilon \sim \mathcal N(m, s^2)$ | N5 | when it bites | 8 | none: a gap in this kind, the forward-process plans use it without teaching it | [Chan, follow-up to Example 4.26, PDF pp. 241 to 242](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=241): partial, no failure, no check | [Sampling A Diagonal Gaussian, how sampling works](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Sampling%20A%20Diagonal%20Gaussian.md#how-sampling-works): full | none: the video index records that no listed creator works Var(aX + bY) | full |
| N4 an isotropic covariance $\sigma^2 I$ is one independent bell per coordinate | N3 | link only | 6 | [PoE plan 08, task 1.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/08-score-of-a-gaussian.md#tasks): mention | chapter 9, section 9.1, step 8: mention | [Diagonal Covariance](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Diagonal%20Covariance.md#the-form): full | [ritvikmath, The Covariance Matrix, 2:40 to 5:23](https://www.youtube.com/watch?v=152tSYtiQbw&t=160s): partial | full |
| N5 expectation as a weighted average, and its linearity | none | when it bites | 12 | [conditional expectation plan 01, tasks 1.1 to 1.4](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/01-expectation-and-the-best-constant-guess.md#tasks): full | [conditional expectation, section 1](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/01-expectation-and-the-best-constant-guess.md#the-equations-that-matter): full | none: a gap in this kind | [Stanford CS229, lecture 3, 11:33 to 21:01](https://www.youtube.com/watch?v=Mi8wnYc1m04&t=693s): partial | full |

## Q2: the score of a Gaussian

The question: derive $\nabla_x \log \mathcal N(x; \mu, \sigma^2) = -(x-\mu)/\sigma^2$, then evaluate it at $x = 2$ for $(\mu, \sigma) = (0, 1)$, $(0, 0.5)$ and $(3, 2)$: $-2$, $-8$ and $+0.25$ (chapter 6, section 6.1).

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| N6 the Gaussian score $-(x-\mu)/\sigma^2$, and $-\Sigma^{-1}(x-\mu)$ | N7, N8, N9 | needed now | 8 | [PoE plan 08, instructions 3.3 to 3.6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/08-score-of-a-gaussian.md#instructions): full | [chapter 6, section 6.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#61-the-gradient-of-a-quadratic-form): full; [little book, section 6.1.1, PDF p. 114](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=114): partial, no numbers, no failure, no check | none: a gap in this kind; [The Tempered Density And Its Score](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/The%20Tempered%20Density%20And%20Its%20Score.md) uses the score of $\mathcal N(3, 4)$ without teaching it (mention) | [Vizuara, Principles of Diffusion Models, lecture 6, 32:45 to 38:19](https://www.youtube.com/watch?v=6ELzfPXUqps&t=1965s): partial, no numbers; [CME296, lecture 2, 21:28 to 24:12](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=1288s): mention, stated without derivation | full |
| N7 the score is $\nabla_x \log p(x)$, a vector field toward higher density | none | when it bites | 10 | [PoE plan 07, the score function](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/07-the-score-function.md): found, not read | [PoE chapter 5, section 5.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/05-predicting-a-score-field.md#51-arrows-on-a-two-bump-function): full | none: a gap in this kind | [CME296, lecture 2, 6:20 to 14:30](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=380s): partial | full |
| N8 the log of the Gaussian density, and why its normaliser has no slope in $x$ | N7 | when it bites | 8 | [PoE plan 04, why derivations move to logs](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/04-why-derivations-move-to-logs.md): found, not read | [chapter 6, section 6.3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#63-which-constants-and-shifts-leave-the-score-alone): full | [Why The Normaliser Has No Slope In X](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Why%20The%20Normaliser%20Has%20No%20Slope%20In%20X.md): full | [CME296, lecture 2, 8:17 to 13:22](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=497s): partial | full |
| N9 the gradient of a quadratic form, $\nabla x^\top A x = (A + A^\top)x$ | none | when it bites | 8 | [PoE plan 06, the gradient field drawn](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/06-the-gradient-field-drawn.md): found, not read | [chapter 6, section 6.1, first derivation and its trap](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#61-the-gradient-of-a-quadratic-form): full | none: a gap in this kind | [Professor Dave, the gradient, 2:09 to 9:36](https://www.youtube.com/watch?v=AXH9Xm6Rbfc&t=129s): partial | full |

## Q3: least-squares regression

The question: the constant $c$ that minimises $\mathbb E[(X - c)^2]$ is $\mathbb E[X]$, and the function $f$ that minimises $\mathbb E[(X - f(Y))^2]$ is $\mathbb E[X \mid Y]$; on a fair die the best constant is 3.5 with error 2.9167, and fitting a line by least squares sets the derivative of the squared error to zero.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| N11 the squared-error minimiser over functions of $Y$ is $\mathbb E[X \mid Y]$ | N12, N13, N14, N15 | needed now | 15 | [conditional expectation plan 05, tasks 2.1 to 2.4](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/05-the-conditional-mean-is-the-best-guess.md#tasks): full | [conditional expectation, section 5](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/05-the-conditional-mean-is-the-best-guess.md#the-proof-worth-knowing): full; [little book, the least-squares regression identity, PDF p. 126](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=126): partial, no numbers, no failure, no check | none: a gap in this kind | [Deepia, Unadjusted Langevin Algorithm, 1:34 to 5:43](https://www.youtube.com/watch?v=m0OTso2Dc2U&t=94s): partial; [CS229, lecture 6, 32:00 to 36:00](https://www.youtube.com/watch?v=llnEgyyuYkQ&t=1920s): mention, the split without the minimiser | full |
| N12 the best constant guess is the mean, and its error is the variance | N5 | when it bites | 12 | [conditional expectation plan 01, tasks 2.1 to 2.5](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/01-expectation-and-the-best-constant-guess.md#tasks): full | [conditional expectation, section 1](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/01-expectation-and-the-best-constant-guess.md#the-proof-worth-knowing): full | none: a gap in this kind | none: a gap in this kind | full |
| N13 the least-squares slope: expand the squared error and set its derivative to zero | N5 | needed now | 5 | conditional expectation plan 05, task 2.3.4: partial, no failure | [conditional expectation, section 5, Example 3, step 3](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/05-the-conditional-mean-is-the-best-guess.md#worked-examples): full | none: a gap in this kind | none | full |
| N14 the least-squares line with an intercept: the normal equations $X^\top X\theta = X^\top y$ | N13 | when it bites | 12 | none: conditional expectation plans 05 and 07 fit a line only as a numerical check (mention) | [Introduction to Statistics, section 6.7, PDF pp. 128 to 131](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-statistics.pdf#page=128): partial, no failure, no check; [Chan, section 7.1.2, PDF pp. 412 to 416](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=413): partial, no worked numbers, no failure; [Anton, section 6.5, Example 1, PDF p. 404](../../../../../../goal-setting/learning/mathematics/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=404): partial, no failure, no check | [Least Squares Line, how the line is found](../../../../../../Obsidian/Mathematics/Statistics/Least%20Squares%20Line.md#how-the-line-is-found): full | Stanford CS229 lecture 2, 1:11:00 to 1:15:30 (found by the conditional-expectation video pulse, never placed): partial | full |
| N15 conditional expectation as a function of what was observed | N5 | when it bites | 15 | [conditional expectation plan 03](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/03-conditional-expectation-as-a-function.md): found, not read | [conditional expectation, section 3](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/03-conditional-expectation-as-a-function.md#worked-examples): full | none: a gap in this kind | [Nazer, Probability 4.7, 1:14 to 3:45](https://www.youtube.com/watch?v=86m-cizvK14&t=74s): partial | full |

## Q4: the score of the forward draw at the toy

The question: put Q1's Gaussian into Q2's score. At $t = 50$, $x_0 = (0, 0.5)$, $\varepsilon = (0.3, -1.2)$: $\nabla \log q(x_{50} \mid x_0) = -(x_{50} - 0.6916\,x_0)/0.5217 = -\varepsilon/0.7223 = (-0.4153, 1.6613)$ (`p1.example_score`). Leans on N1 and N6 and everything under them.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| N10 the conditional score of the forward draw is $-\varepsilon/\sqrt{1-\bar\alpha_t}$ | N1, N6 | needed now | 8 | [PoE plan 11, noise prediction is the score](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/11-noise-prediction-is-the-score.md): found, not read | [step 5 of the step file](../01-one-expert-from-scratch.md#step-5-the-noise-is-the-score-rescaled): full; [chapter 9, section 9.5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#95-eps---sigma_t-grad-log-q): found, not read | none: a gap in this kind | [Vizuara, lecture 6, 38:19 to 40:25](https://www.youtube.com/watch?v=6ELzfPXUqps&t=2299s): partial | full |

## Q5: what a squared-error fit of the noise returns

The question: training $\hat\varepsilon(x_t, t)$ by least squares on the drawn $\varepsilon$ returns $\mathbb E[\varepsilon \mid x_t]$, the average noise among all clean points that could have produced $x_t$, and never the one $\varepsilon$ that was drawn. Leans on N11 and N1.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---|---|---|---|---|---|
| N16 the denoising loss is a least-squares regression of $\varepsilon$ on $x_t$, so its optimum is $\mathbb E[\varepsilon \mid x_t]$ | N11, N1 | needed now | 5 | [PoE plan 20, score matching the training objective](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/20-score-matching-the-training-objective.md): found, not read | [PoE chapter 13, section 13.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/13-the-training-loss-two-ways.md#132-denoising-score-matching-and-its-optimum): full; [step 8 of the step file](../01-one-expert-from-scratch.md#step-8-the-denoising-loss): partial, the optimum's derivation carries no worked numbers | none: a gap in this kind | [Stanford CS236, lecture 16, 39:01 to 41:48](https://www.youtube.com/watch?v=VsllsC2JMGY&t=2341s): partial | full |

## Banked problems

Problems from the learner's books and the journeys' own textbooks, chosen per concept from the easiest that shows the mechanic to the hardest that bites. Depth: the forward draw and the Gaussian score are taught at the build depth (layer 1 builds both), so each gets 6, the bottom of the 6 to 10 range, because five problems the search found are already banked by quiz 1.2; least-squares regression is at the own depth and gets 5. Each problem is a node of its own: its passage is its solution, and it leans on the shared nodes above. Solutions S1 to S7 are written in [the quiz 1.1 solutions](quiz-1-1-solutions.md), and every number in them prints from `quiz-1-1-solutions-check.py` beside it. Each concept's table is the source block in the [worked-examples index](../../../../../../Obsidian/Mathematics/references/exercises/Worked%20Examples.md), with copies after the passages that teach it (see [Fills](#fills)).

**The forward draw's mean and variance (build depth)**

| Node | Problem | Source | Kind | Difficulty | Solution | Leans on | Verdict |
|---|---|---|---|---|---|---|---|
| B1.1 | $Y = aX + b$ for a standard Gaussian $X$: its CDF $\Phi((y-b)/a)$ and density | [Chan, follow-up to Example 4.26, PDF pp. 241 to 242](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=241) | worked example | easy | the book's, PDF p. 242 | N3 | ready |
| B1.2 | the mean and variance of a sum of independent variables, scalar and then vector | [Bishop, Exercises 1.10 (PDF p. 78) and 2.27 (PDF p. 151)](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=78) | exercise without a solution | easy | [S1](quiz-1-1-solutions.md#s1-the-mean-and-variance-of-a-sum-of-independent-variables-b12-bishop-110-and-227), written | N2, N5 | ready |
| B1.4 | $Y = \Sigma^{1/2}X + \mu$ has mean $\mu$ and covariance $\Sigma$ | [Chan, Theorem 5.19, PDF p. 315](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=315) | worked derivation | medium | the book's, PDF p. 315 | N3, N4, N5 | ready |
| B1.6 | $y = Ax + b + w$: the likelihood $p(y \mid x)$ and the mean and covariance of $p(y)$, then read with $A = \sqrt{\bar\alpha_{50}}\,I$, $b = 0$, $w \sim \mathcal N(0, 0.5217\,I)$ | [Mathematics for Machine Learning, Exercise 6.12 (a) and (b), PDF p. 229](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=229) | exercise without a solution | hard | [S2](quiz-1-1-solutions.md#s2-a-linear-map-plus-independent-noise-b16-mml-612), written | N1, N2, N3, N4 | ready |
| B1.7 | Chan's proof that $X_1 + X_2 \sim \mathcal N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$ sets $\sigma_1 = \sigma_2$; prove the general case | [Chan, Theorem 5.12, PDF pp. 299 to 300](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=299) | exercise without a solution (the book's proof is the special case) | hard | [S3](quiz-1-1-solutions.md#s3-two-gaussians-with-unequal-variances-b17-chan-theorem-512), written | N2, N3 | ready |
| B1.8 | Python: noise one cloud at three levels in one tensor and print measured against closed-form mean and variance; first without the square roots | [The Forward Process, try the jump yourself, steps 3 and 4](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md#try-the-jump-yourself) | programming exercise, Python, `noised.py` | medium | in the note, with the expected output | N1, N3 | ready |

**The score of a Gaussian (build depth)**

| Node | Problem | Source | Kind | Difficulty | Solution | Leans on | Verdict |
|---|---|---|---|---|---|---|---|
| B2.2 | $x^\top A x$ and its gradient for a symmetric and a non-symmetric $A$, then three 1D scores: $-2$, $-8$, $+0.25$ | [chapter 6, section 6.1, worked example](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#61-the-gradient-of-a-quadratic-form) | tutorial | easy | in the chapter, full | N6, N9 | ready |
| B2.3 | the mode of the 1D and the multivariate Gaussian is $\mu$ | [Bishop, Exercise 1.9, PDF p. 78](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=78) | exercise without a solution | easy | [S4](quiz-1-1-solutions.md#s4-the-mode-of-a-gaussian-is-its-mean-b23-bishop-19), written | N6 | ready |
| B2.4 | four changes to a standard normal (times a constant, plus a log constant, a shift, plus 137) and what each does to the score | [chapter 6, section 6.3, worked example](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#63-which-constants-and-shifts-leave-the-score-alone) | tutorial | medium | in the chapter, full | N6, N8 | ready |
| B2.5 | the score at $(3, 4)$ under $\Sigma = \operatorname{diag}(1, 4)$ is $(-3, -1)$, 34.6952° off the direction to the mean | [chapter 6, section 6.2, worked example](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#62-the-anisotropic-score-follows-the-inverse-covariance) | tutorial | medium | in the chapter, full | N6, N4 | ready |
| B2.6 | by the chain rule, the derivative of $\exp(-\tfrac12 (x-\mu)^\top S^{-1}(x-\mu))$ with the dimension of every partial, then divided by itself to give the score | [Mathematics for Machine Learning, Exercise 5.8 (a), PDF p. 177](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=177) | exercise without a solution | medium | [S5](quiz-1-1-solutions.md#s5-the-chain-rule-through-the-gaussians-exponential-b26-mml-58a), written | N6, N9, N7 | ready |
| B2.7 | the maximum-likelihood mean and variance of i.i.d. Gaussians by setting the derivatives of the log-likelihood to zero; the $\mu$-derivative is the $x$-score with its sign reversed | [Chan, Practice Exercise 8.5, PDF pp. 500 to 501](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=500) | exercise with the book's solution | hard | the book's, PDF p. 501 | N6, N8 | ready |

Ids B2.1 and B2.8 are retired: they were the little book's 1D score (PDF p. 114) and the step file's autograd check of the toy score, which quiz 1.2 banked first.

**Least-squares regression (own depth)**

| Node | Problem | Source | Kind | Difficulty | Solution | Leans on | Verdict |
|---|---|---|---|---|---|---|---|
| B3.1 | the best constant guess on a fair die is 3.5 with error 2.9167; guesses 3 and 4 cost 3.1667 | [conditional expectation, section 1, Example 1](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/01-expectation-and-the-best-constant-guess.md#worked-examples) | tutorial | easy | in the chapter, full | N12 | ready |
| B3.2 | the least-squares line through $(0, 1), (1, 3), (2, 4), (3, 4)$ is $y = 1.5 + x$ | [Anton, section 6.5, Example 1, PDF p. 404](../../../../../../goal-setting/learning/mathematics/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=404) | worked example | easy | the book's, PDF p. 404 | N14 | ready |
| B3.3 | starting salary on education score for six people by the normal equations: slope 5,370 dollars per score point | [Introduction to Statistics, Example 6.4, PDF pp. 130 to 131](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-statistics.pdf#page=130) | worked example | medium | the book's, PDF p. 131 | N14 | ready |
| B3.4 | Example 6.4 with the sixth person removed: re-estimate both coefficients and predict at score 2.8 | [Introduction to Statistics, Exercise 6.5, PDF p. 132](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-statistics.pdf#page=132) | exercise without a solution | medium | [S6](quiz-1-1-solutions.md#s6-example-64-without-its-sixth-person-b34-introduction-to-statistics-65), written | N14 | ready |
| B3.6 | Python: fit a line with `np.linalg.lstsq` and confirm the answer solves $X^\top X\theta = X^\top y$, run on B3.2's four points | [Chan, Theorem 7.1 and the Python listing, PDF pp. 414 to 416](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=414) | programming exercise, Python, `fit_line(x, y)` | medium | [S7](quiz-1-1-solutions.md#s7-a-line-fitted-by-lstsq-solves-the-normal-equations-b36-chan-theorem-71), written | N14 | ready |

Ids B1.3, B1.5 and B3.5 are retired: they were chapter 9's worked examples 9.1 and 9.2a and conditional expectation section 5's Example 3, which quiz 1.2 banked first (see [Problems searched for and dropped](#problems-searched-for-and-dropped)). The remaining ids are kept so the index rows and solution headings stay valid.

## Problems searched for and dropped

**Chan, Chapter 7, Exercise 10 (PDF p. 480).** Least squares on inputs corrupted by Gaussian noise turns into ridge regression. Dropped: ridge regression is not a node, and no passage in the four kinds teaches it to the full-steps standard, so banking it would block the quiz.

**Bishop, Exercises 1.25 and 1.26 (PDF p. 83).** The vector-target conditional mean. Dropped as a duplicate: quiz 1.2 banks Exercise 1.25 as its B22, and quiz 1.1 tests the scalar case through B3.5.

**Bishop, Exercise 2.17 (PDF p. 150).** The antisymmetric part of a precision matrix leaves the exponent. Dropped: the symmetric and antisymmetric split of a matrix is not taught by any node; B2.2's non-symmetric trap covers the same slip.

**Five problems quiz 1.2 banked first.** Chapter 9's worked examples 9.1 (the two-step noise variance, 0.28 both ways) and 9.2a (the four-step schedule, $x_2 = 1.9616$), the little book's 1D Gaussian score (PDF p. 114), the step file's autograd check of the toy score, and conditional expectation section 5's Example 3 (the best line against the conditional mean). The quiz 1.2 run wrote them into the worked-examples index while this run was searching, so quiz 1.1 leaves them to quiz 1.2 rather than asking them twice.

**Chan, Chapter 5, Exercise 10 (PDF p. 332).** A ±1 signal plus Gaussian noise. Left to quiz 1.2, whose conditional-mean problems use the same two-point toy.

## What the grades rest on

**Read in full.** The step file's steps 4, 5 and 8; chapter 9, sections 9.1 and 9.2; chapter 6, sections 6.1, 6.2 and 6.3; conditional expectation section 5, Example 3; the vault note The Forward Process; PoE plan 09, instructions 3.1 to 3.7; Chan PDF pp. 241 to 243, 298 to 300, 309 to 316, 332 to 333, 412 to 416, 475 to 476, 480 and 500 to 501; Introduction to Statistics PDF pp. 128 to 135; Mathematics for Machine Learning PDF pp. 177 and 227 to 230; the little book PDF pp. 114 and 126 to 127; Anton PDF pp. 401 to 404; Bishop's exercise statements 1.9, 1.10, 1.25, 1.26 and 2.27.

**Read in part.** Conditional expectation sections 1 and 5 (the outline, every bold step and failure line, and Example 3 in full); conditional expectation section 3 and PoE chapters 3, 5 and 13 (the outline with every worked-example and failure line); the vault notes Sampling A Diagonal Gaussian (the form and how sampling works), Diagonal Covariance (the form and its failure step) and Why The Normaliser Has No Slope In X (the opening and its numbers); PoE plan 08 and conditional expectation plans 01 and 05 (every task and instruction line).

**Only found, not read.** PoE plans 04, 06, 07, 11 and 20, conditional expectation plan 03, chapter 9 section 9.5, and the sampler-correctors journey's plans 03, 06 and 08. A grade in a cell that says "found, not read" is not a grade; no verdict depends on one, because each of those nodes has a full passage that was read.

**Written in this run, graded by its writer.** The vault note Least Squares Line (N14) and solutions S1 to S7. Each was graded full against the five parts by prep-cadence, which wrote it, and no second reviewer has read them.

**Videos.** Every video cell comes from a table already written (the step file's tables, the video lessons index, or the conditional-expectation journey's video pulse of 2026-10-19). No caption was fetched in this audit.

## Gaps

| Node | Kinds missing | Who fills it | State |
|---|---|---|---|
| N14 the least-squares line with an intercept: the normal equations | all four had no full passage: the books are partial, no plan or vault note taught it, the one video is unplaced | the vault note [Least Squares Line](../../../../../../Obsidian/Mathematics/Statistics/Least%20Squares%20Line.md) | filled, graded full |

**Kinds that are thin but block nothing.** Each of these nodes has a full passage in another kind, so no question waits on them: the Obsidian note for N5, N6, N7, N9, N10, N11, N12, N13, N15 and N16; the learning plan for N3; the video for N3 and N12.

## Fills

| Date | Node or problem | Passage | Path | Written by | Grade after the check |
|---|---|---|---|---|---|
| 2026-10-20 | B1.2, B1.6, B1.7, B2.3, B2.6, B3.4, B3.6 | seven solutions, S1 to S7, each with what and why, worked numbers, numbered steps, one failure with its number, and a check | [the quiz 1.1 solutions](quiz-1-1-solutions.md); numbers from [the check script](quiz-1-1-solutions-check.py) | prep-cadence | full against the five parts, self-graded; every number printed by the script, run on mscluster with the co3 environment; no `codex-author` review |
| 2026-10-20 | the three banked concepts | three source tables: forward-draw-mean-and-variance (rev 1, 6 rows), score-of-a-gaussian (rev 2, 6 rows), least-squares-regression (rev 1, 5 rows) | [the worked-examples index](../../../../../../Obsidian/Mathematics/references/exercises/Worked%20Examples.md) | prep-cadence | not a passage; contents list updated |
| 2026-10-20 | the three banked concepts | four copies of those tables, each placed after the passage that teaches its concept and checked equal to its source | chapter 9 before section 9.3; the vault note The Forward Process before "One small step"; chapter 6 before section 6.2; conditional expectation section 5 before "Why this matters" | prep-cadence | not a passage; copy equals source |
| 2026-10-20 | N14 | a vault note on the least-squares line with an intercept and the normal equations, with a byte-equal copy of the least-squares-regression table after its teaching passage and an entry in the statistics map | [Least Squares Line](../../../../../../Obsidian/Mathematics/Statistics/Least%20Squares%20Line.md); its script at `Obsidian/Mathematics/Statistics/sandbox/least_squares_line.py` | prep-cadence, under --build: `vault-writer` stopped after rendering the two figures in `Mathematics/Statistics/assets/least-squares-four-points-*.png` and wrote no note | full against the five parts, self-graded; the script ran twice with identical output; no `codex-author` or `vault-writer` review |
