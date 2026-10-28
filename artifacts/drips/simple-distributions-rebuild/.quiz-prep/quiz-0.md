# Coverage: Quiz 0, the college maths under the composition ladder
Standard: QUIZ_PREP_FORMAT v1

Preparation audit for quiz 0 of the [composition ladder](../PROMPTS.md), written to the format in `QUIZ_PREP_FORMAT.md` before the quiz page exists. The page is built afterwards by `quiz-cadence`, which reads this file, takes its planned questions and its bank questions, and holds back anything marked blocked. The audit edited no step file, nothing in `rebuild/` and nothing in `ladder/`. The passages in the Fills table were written by their owners and graded here.

## Table of contents

- [Position](#position)
- [Words this uses](#words-this-uses)
- [What the grades rest on](#what-the-grades-rest-on)
- [The questions](#the-questions)
- [Trees, one section per question](#trees-one-section-per-question)
- [Bank questions from the textbooks](#bank-questions-from-the-textbooks)
- [Gaps](#gaps)
- [Fills](#fills)

## Position

Navigation: 📋 [Table of contents](#table-of-contents) | [Words this uses](#words-this-uses) ➡️

| Item | Value |
|---|---|
| Quiz | Quiz 0, the college maths under the whole ladder: logs and exponents, the chain rule, the Gaussian density and completing the square, mean and variance under scaling and under sums of independent variables, Bayes' rule, the gradient of a log density |
| Goal and time | "derive the ε-score identity and train one 2D expert", 90 minutes |
| Scope root | `artifacts/drips/simple-distributions-rebuild/`, material from [layer 1's study line](../ladder/01-one-diffusion-expert.md) and [the step file](../01-one-expert-from-scratch.md) |
| Audited | 2026-10-20 |
| Questions | 46: 21 written (A1 to F3, in six topics) and 25 taken from the textbooks (KA1 to KF4) |
| Ready | 46 |
| Blocked by a gap | 0 |
| Unprepared by choice | 0 (the learner has tagged none) |
| Nodes | 43: 18 concept nodes (N01 to N18), none a gap, and 25 problem nodes, one per bank question |
| Short path | 92 minutes: the 7 nodes tiered needed now (N02, N05, N08, N13, N14, N17, N18), two minutes over the 90 |
| Full path | 266 minutes: all 18 concept nodes |

## Words this uses

Navigation: ⬅️ [Position](#position) | 📋 [Table of contents](#table-of-contents) | [What the grades rest on](#what-the-grades-rest-on) ➡️

**Covered** (node verdict): at least one kind holds a passage graded full. The format calls such a node a leaf once the learner holds it. No plan task in the journeys cited here is ticked, so whether the learner holds any node is the learner's to say, and this file does not claim it.

**Gap**: no passage in any kind is graded full. A question whose tree holds a gap is blocked.

**The step file**: [01-one-expert-from-scratch.md](../01-one-expert-from-scratch.md), the plan the learner walks for layer 1. Its cells sit in the learning-plan column.

**The foundations chapters**: the textbook written for the journey `deep-learning/diffusion-models/01-poe-derivation-foundations`, chapters 1 to 13. A chapter section is a textbook cell. Shelf pages (the algebra handbook, Chan's probability book, the calculus textbook) are textbook cells too, cited by PDF page.

**Tier**: set from the goal and the 90 minutes. A node is *needed now* when the ε-score identity, $\varepsilon = -\sqrt{1-\bar\alpha_t}\,\nabla\log q(x_t\mid x_0)$, cannot be derived without it: the log rules, the chain rule, the Gaussian density, the mean and variance of $aX + b$, sums of independent variances, the gradient of a log, and the Gaussian score. Everything else is *when it bites*: link it, read it if a miss points there. The seven needed-now nodes sum to 92 minutes, two over the 90, because the passage written for N13 runs about 18 minutes; trimming it would cost its failure section, so the overrun stays and is reported.

**Minutes** (the `Min` column): estimates. Each is the words of the passage graded full (code blocks, image links and video tables left out) at 120 words a minute, plus 5 minutes for the worked example by hand, rounded. A question's total is the sum of its tree's rows, so a node shared by several questions counts in each.

**Full**: all five parts of the format's standard are present (what and why in plain words, a worked example with every intermediate value, the method as steps, one failure with its number, a check). An ordered chain of lines, each with its value, counts as the steps. **Partial** names the missing part. **Mention** means named or used without being taught. **None** means no passage; the cell says why.

**Planned question**: the page does not exist yet, so each question here is the planned one: its mode (intuition, mechanic or consequence, per `quiz-cadence`'s selection rule), what it asks, the answer, and where every number in the answer comes from. `quiz-cadence` writes the wording; the tree was built for the root and the numbers given.

## What the grades rest on

Navigation: ⬅️ [Words this uses](#words-this-uses) | 📋 [Table of contents](#table-of-contents) | [The questions](#the-questions) ➡️

**Read whole.** Foundations chapter 1 sections 1.2 and 1.5; chapter 2 sections 2.2 and 2.3; chapter 3 sections 3.1 and 3.2; chapter 4 sections 4.1, 4.2 and 4.3; chapter 6 section 6.1; chapter 9 section 9.1 (with its two video tables); chapter 10 section 10.1. The step file's steps 4, 5 and 9 and step 10 from its definition to the end of its derivation, and step 1 to the end of its definition block. The conditional-expectation chapter 1, "The equations that matter" and Example 1. The sampler-correctors plan 05, task group 1. The foundations plans 04 (tasks 1 and 2, instructions 3) and 06 (instruction 7), and plan 09 instruction 3. The vault notes Solving For The Exponent With Logarithms (from "The question" to "Check"), The Log Of A Power And Of A Quotient, Sampling A Diagonal Gaussian (the opening, steps 6 and "How sampling works"), Why The Normaliser Has No Slope In X (to "Why it matters"), The Forward Process ("How the jump works"), Products And Quotients Of Diagonal Gaussians ("How the product works"), and the opening of Gaussian Integral. The video lessons index sections named in the video cells. The walk numbers: the 41 `p1.` keys of `rebuild/out/walk-numbers.json`, printed on 2026-10-20.

**Read after they were written, for the re-audit.** The three notes the fills produced: The Exponent Rules ("The method, step by step" and "Where it goes wrong", with the headings of the rest), The Front Factor Of A Gaussian Density ("The method" and "The slip, sigma squared in place of sigma", with its worked values counted by grep), and Scaling And Shifting A Random Variable (from "What scaling and shifting do" to "Where the rule goes wrong"). Each fill's script was written and run here before it was handed over, and each writer reports running it byte for byte with the same output.

**The bank.** Every bank problem and its solution page was read by the bank search (rendered or extracted with PyMuPDF), which lists the pages it read whole and the ones it only found in each index section's footer. Three were read again here: Chan PDF p. 107 (0.375 and 0.4), the calculus handbook PDF p. 182 ($P' = -(x-\mu)P/\sigma^2$) and Murphy PDF p. 71 (0.795 and 0.014). The three bank solutions written here (KC3, KD4, KF2) were computed by script.

**Read through PyMuPDF, one page each.** Algebra handbook PDF pages 74, 109 and 125. Chan PDF pages 105 to 106, 149 to 151 and 200 to 201. Calculus textbook PDF page 198. The little book of generative AI foundations PDF page 114.

**Read in part.** Foundations plans 01, 02, 03, 05 and 08: their task and instruction headings and every "Read first" line, which send the worked steps to the chapter sections above.

**Found, not read.** Probabilistic machine learning PDF page 99 (the Gaussian integral, linked from step 1); Chan PDF page 227 (the Gaussian density); the vault notes Diagonal Covariance and Multivariate Normal Distribution (headings only); the foundations plan 13. No cell is graded from these.

**Videos.** No video was watched and no caption was fetched in this audit. Every video cell is graded from the footer line of its table, which says whether its times were read from captions or from a chapter list. A video counts only with a timestamp.

## The questions

Navigation: ⬅️ [What the grades rest on](#what-the-grades-rest-on) | 📋 [Table of contents](#table-of-contents) | [Trees](#trees-one-section-per-question) ➡️

| Id | Mode | Root concept | Nodes | Gaps | Short path (min) | Full path (min) | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| [A1](#a1-why-the-log-of-a-product-is-a-sum) | intuition | why the log of a product is a sum | 2 | 0 | 16 | 39 | ready |
| [A2](#a2-the-moving-average-half-life) | mechanic | the moving average's half-life, 0.999ⁿ = 0.5 | 3 | 0 | 16 | 61 | ready |
| [A3](#a3-a-half-life-from-mixed-log-bases) | consequence | a half-life from mixed log bases | 3 | 0 | 16 | 61 | ready |
| [B1](#b1-why-the-slopes-multiply) | intuition | why the slopes multiply in the chain rule | 2 | 0 | 10 | 22 | ready |
| [B2](#b2-the-slope-of-the-gaussian-at-one-standard-deviation) | mechanic | the slope of the Gaussian at one standard deviation | 5 | 0 | 24 | 79 | ready |
| [B3](#b3-the-inner-slope-dropped) | consequence | the chain rule with the inner slope dropped | 2 | 0 | 10 | 22 | ready |
| [C1](#c1-why-the-front-factor-is-one-over-sigma-root-two-pi) | intuition | why the front factor is 1/(σ√(2π)) | 2 | 0 | 0 | 43 | ready |
| [C2](#c2-the-toy-gaussian-by-hand) | mechanic | the toy Gaussian by hand, in 1D and 2D | 4 | 0 | 14 | 71 | ready |
| [C3](#c3-completing-the-square) | mechanic | completing the square | 1 | 0 | 0 | 14 | ready |
| [C4](#c4-two-gaussians-multiplied) | consequence | two Gaussians multiplied: precisions add | 5 | 0 | 14 | 86 | ready |
| [D1](#d1-why-scaling-squares-the-variance) | intuition | why scaling squares the variance and a shift does not touch it | 2 | 0 | 18 | 32 | ready |
| [D2](#d2-a-scaled-and-shifted-die) | mechanic | a scaled and shifted die | 2 | 0 | 18 | 32 | ready |
| [D3](#d3-two-noise-steps-in-one) | mechanic | two independent noise steps as one | 3 | 0 | 31 | 45 | ready |
| [D4](#d4-adding-standard-deviations) | consequence | adding standard deviations instead of variances | 3 | 0 | 31 | 45 | ready |
| [E1](#e1-why-conditioning-divides-by-the-condition) | intuition | why conditioning divides by the condition | 1 | 0 | 0 | 14 | ready |
| [E2](#e2-the-librarian-and-the-farmers) | mechanic | Bayes' rule with the evidence built from its parts | 2 | 0 | 0 | 26 | ready |
| [E3](#e3-dividing-by-the-wrong-margin) | consequence | dividing by the wrong margin | 2 | 0 | 0 | 26 | ready |
| [E4](#e4-a-gaussian-prior-times-a-gaussian-likelihood) | composed | a Gaussian prior times a Gaussian likelihood | 7 | 0 | 14 | 112 | ready |
| [F1](#f1-why-the-normaliser-leaves-the-score) | intuition | why the normaliser leaves the score | 6 | 0 | 34 | 83 | ready |
| [F2](#f2-the-score-of-the-toy-gaussian) | mechanic | the score of the toy Gaussian, two ways | 9 | 0 | 61 | 130 | ready |
| [F3](#f3-sigma-in-place-of-sigma-squared) | consequence | σ in place of σ² in the score | 9 | 0 | 61 | 130 | ready |
| [KA1](#bank-questions-from-the-textbooks) | bank, easy | log form: $\log_4 64 = x$, $\ln e = x$ | 3 | 0 | 16 | 39 | ready |
| [KA2](#bank-questions-from-the-textbooks) | bank, easy | condense $6 + \ln(3e^5)$ and $\log_4(1/16) - 6^{\log_6 4}$ | 3 | 0 | 16 | 39 | ready |
| [KA3](#bank-questions-from-the-textbooks) | bank, medium | solve $2 + 4\cdot 3^{x-2} = 110$ | 4 | 0 | 16 | 61 | ready |
| [KA4](#bank-questions-from-the-textbooks) | bank, medium | when $75e^{-t/125}$ watts falls to 7 watts | 4 | 0 | 16 | 61 | ready |
| [KB1](#bank-questions-from-the-textbooks) | bank, easy | differentiate $e^{5x}$ | 3 | 0 | 10 | 22 | ready |
| [KB2](#bank-questions-from-the-textbooks) | bank, easy | differentiate $(x^2+3x)^{100}$ | 3 | 0 | 10 | 22 | ready |
| [KB3](#bank-questions-from-the-textbooks) | bank, medium | differentiate $\ln\sqrt{x^2+5}$ | 5 | 0 | 26 | 61 | ready |
| [KB4](#bank-questions-from-the-textbooks) | bank, hard | gradient of $x^2e^y$ at $(-2, 0)$, its length and direction | 4 | 0 | 10 | 36 | ready |
| [KC1](#bank-questions-from-the-textbooks) | bank, easy | write $x^2 - 4x + 5$ as $(x-2)^2 + 1$ | 2 | 0 | 0 | 14 | ready |
| [KC2](#bank-questions-from-the-textbooks) | bank, easy | complete the square in $3x^2 + 18x - 21 = 0$ (the same problem as C3) | 2 | 0 | 0 | 14 | ready |
| [KC3](#bank-questions-from-the-textbooks) | bank, medium | $\mathcal N(0, 0.25)$ at 0 and 0.5 in code and by hand, then the variance passed as the scale | 4 | 0 | 14 | 57 | ready |
| [KC4](#bank-questions-from-the-textbooks) | bank, hard | complete the square for $q(x_{t-1}\mid x_t, x_0)$ | 8 | 0 | 14 | 112 | ready |
| [KD1](#bank-questions-from-the-textbooks) | bank, easy | mean and variance of a four-valued count | 2 | 0 | 0 | 14 | ready |
| [KD2](#bank-questions-from-the-textbooks) | bank, easy | $E[(aX+b)^2]$ from $E[X] = 1$, $E[X^2] = 3$ | 3 | 0 | 18 | 32 | ready |
| [KD3](#bank-questions-from-the-textbooks) | bank, medium | prove the moment, scale and shift rules of variance | 3 | 0 | 18 | 32 | ready |
| [KD4](#bank-questions-from-the-textbooks) | bank, medium | three dice in code: 10.5 and 8.75, and why the printed listing gives 9 and 6 | 4 | 0 | 31 | 45 | ready |
| [KD5](#bank-questions-from-the-textbooks) | bank, hard | the sum of $N$ independent Gaussians by generating functions | 4 | 0 | 31 | 45 | ready |
| [KE1](#bank-questions-from-the-textbooks) | bank, easy | $P[3\mid\text{odd}]$ and $P[\text{odd}\mid 3]$ | 2 | 0 | 0 | 14 | ready |
| [KE2](#bank-questions-from-the-textbooks) | bank, easy | the chance of a win, then of opponent A given a win | 3 | 0 | 0 | 26 | ready |
| [KE3](#bank-questions-from-the-textbooks) | bank, medium | infection after a positive and a negative test | 3 | 0 | 0 | 26 | ready |
| [KE4](#bank-questions-from-the-textbooks) | bank, hard | the most probable mean under a Gaussian prior and likelihood | 13 | 0 | 48 | 172 | ready |
| [KF1](#bank-questions-from-the-textbooks) | bank, easy | differentiate $\ln(x^2+1)$ | 5 | 0 | 26 | 61 | ready |
| [KF2](#bank-questions-from-the-textbooks) | bank, easy | differentiate $\ln 5x$ and $\ln x^2$ | 5 | 0 | 26 | 61 | ready |
| [KF3](#bank-questions-from-the-textbooks) | bank, medium | differentiate the normal density; $P'/P$ is the score | 10 | 0 | 61 | 130 | ready |
| [KF4](#bank-questions-from-the-textbooks) | bank, medium | the maximum-likelihood mean of Gaussian data | 9 | 0 | 48 | 117 | ready |

## Trees, one section per question

Navigation: ⬅️ [The questions](#the-questions) | 📋 [Table of contents](#table-of-contents) | [Bank questions](#bank-questions-from-the-textbooks) ➡️

Links below are relative to this file, written out in full so each opens when clicked.

### A1: why the log of a product is a sum

Planned question A1 (intuition). Asks: the step file takes the log of a Gaussian and gets "minus the squared distance over 2σ², plus a constant". Which exponent rule makes $\log(e^a e^b) = a + b$, and why does that make the log of a product of densities a sum of their logs? Answer: the product of powers, $e^a e^b = e^{a+b}$; the log undoes $e$, so $\log(AB) = \log A + \log B$ ([foundations chapter 3.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/03-moving-into-logs.md#31-the-log-of-a-product-and-the-quotient-rule), steps 1 to 3). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N02** The log rules: the log of a product, a quotient and a power, and the log undoing $e$ | N01 | needed now | 16 | [foundations plan 04, instructions 3.1 to 3.6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/04-why-derivations-move-to-logs.md#3--logs-turn-products-into-sums-and-quotients-into-differences): partial. Sets the five targets (−1.3863, −2.9957, −4.3820, 1.6094, 0.0125) and the check script; the worked chain is in chapter 3. | [chapter 3, sections 3.1 and 3.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/03-moving-into-logs.md#31-the-log-of-a-product-and-the-quotient-rule): full. Six-step derivation from $e^ae^b = e^{a+b}$; worked on 0.25 and 0.05; failure: base 10 gives −0.6021 for log 0.25; check script.<br>[algebra handbook p. 110](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=110): found, the rule table the step file links; not graded. | [The Log Of A Power And Of A Quotient](../../../../../../Obsidian/Mathematics/Algebra/Logarithms/The%20Log%20Of%20A%20Power%20And%20Of%20A%20Quotient.md): partial. Derivation, worked on N(3, 4) at x = 7, four more rows, a script; no failure with its number. | mention. Plan 04's instruction 3.1 cites [a lecture, 8:47 to 9:20](https://www.youtube.com/watch?v=-RVM21Voo7Q&t=527s), where the product rule is used and not proved. The video lessons index has no row for the log rules: a gap in this kind only. | covered |
| **N01** The exponent rules: product of powers, power of a power, power of a product, negative and zero powers | nothing further | when it bites | 23 | none: no plan teaches them. | [algebra handbook p. 109](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=109): partial. The rule table with examples; no why, no failure, no check.<br>[chapter 3.1, step 2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/03-moving-into-logs.md#31-the-log-of-a-product-and-the-quotient-rule): mention ("powers of the same base add"). | [The Exponent Rules](../../../../../../Obsidian/Mathematics/Algebra/The%20Exponent%20Rules.md#what-the-rules-are-and-why-they-are-here): full, written for this quiz. The five rules derived by counting factors; the toy 0.7979, 0.4839, the 2D height 0.234199 and $0.999^{692.8} = 0.5$; seven numbered steps with a redo on new numbers; failure: adding exponents for a sum gives 1.213061 for 0.367879; the script as the check.<br>[The Log Of A Power And Of A Quotient](../../../../../../Obsidian/Mathematics/Algebra/Logarithms/The%20Log%20Of%20A%20Power%20And%20Of%20A%20Quotient.md): mention ("by the exponent rules", one line). | none: no row in the video lessons index. | covered |

Tier and minutes: short path 16 (N02), full path 39.

### A2: the moving average half-life

Planned question A2 (mechanic). Asks: the moving average keeps 0.999 of an old weight's share per step, so after $n$ steps the share is $0.999^n$. Solve $0.999^n = 0.5$. Answer: $n = \ln 0.5 / \ln 0.999 = -0.693147 / -0.0010005 = 692.8$ steps. Numbers from: the step file, [step 9](../01-one-expert-from-scratch.md#step-9-training-with-a-moving-average), and `p1.ema_half_life_steps` = 692.8. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N03** Solving for an unknown in the exponent with logs | N02 | when it bites | 22 | [the step file, step 9](../01-one-expert-from-scratch.md#step-9-training-with-a-moving-average): partial. The one line $n = \ln 0.5/\ln 0.999 = 692.8$ with why the logs and the ratio are there, and the failure "0.999n would grow with age"; no numbered steps, no check. | [algebra handbook p. 125](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=125): partial. Three steps (isolate, take logs, solve) on $2 + 4\cdot 3^{x-2} = 110$, answer 5; no failure, no check, whole-number answers only. | [Solving For The Exponent With Logarithms, the method](../../../../../../Obsidian/Mathematics/Algebra/Logarithms/Solving%20For%20The%20Exponent%20With%20Logarithms.md#the-method-take-logs-of-both-sides): full. Seven numbered steps on $0.9^n \le 0.1$ to 22; the forgotten flip gives $n \le 21.854$; a loop over $n$ as the check. | partial. [The video lessons index, solving an inequality with the unknown in the exponent](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#solving-an-inequality-with-the-unknown-in-the-exponent): TLMaths 0:06 to 2:23 solves $0.5^x < 10^{300}$ by natural logs; caption-read; no whole-number solve, no check. | covered |
| N02, N01 | see [A1](#a1-why-the-log-of-a-product-is-a-sum) | | | | | | | |

Tier and minutes: short path 16 (N02), full path 61.

### A3: a half-life from mixed log bases

Planned question A3 (consequence). Asks: a reader takes $\log_{10} 0.5$ on top and $\ln 0.999$ underneath and reports a half-life. What comes out, and how does it betray itself? Answer: $-0.301030 / -0.0010005 = 300.9$ steps, a plausible-looking number; checking it, $0.999^{300.9} = 0.74$, not 0.5. The right value is 692.8, where $0.999^{692.8} = 0.50$. Numbers from: step 9 of the step file and `p1.ema_half_life_steps`; the base-10 slip is the failure in [chapter 3.2, step 1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/03-moving-into-logs.md#32-logs-of-probabilities-five-numbers-by-hand). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| N03, N02, N01 | see [A2](#a2-the-moving-average-half-life) and [A1](#a1-why-the-log-of-a-product-is-a-sum) | | | | | | | |

Tier and minutes: short path 16, full path 61.

### B1: why the slopes multiply

Planned question B1 (intuition). Asks: in $\frac{d}{dx}g(f(x)) = g'(f(x))\,f'(x)$, why is the outer slope taken at $f(x)$ and why are the two slopes multiplied rather than added? Answer: a nudge $h$ moves $f$ by about $f'(x)h$, and the outer function sees its input move by that amount, so it moves by $g'(u)$ times it, with $u = f(x)$ ([chapter 4.3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/04-differentiating-by-hand.md#43-the-chain-rule-and-grad-log-f-as-grad-f-over-f), steps 1 to 4). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N05** The chain rule | N04 | needed now | 10 | [foundations plan 06, instruction 7.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/06-the-gradient-field-drawn.md#7--the-gradient-of-a-log-derived-rather-than-quoted): partial. Names the five lines to write; the worked chain is in chapter 4.3. | [chapter 4.3, the chain rule and worked example A](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/04-differentiating-by-hand.md#43-the-chain-rule-and-grad-log-f-as-grad-f-over-f): full. Four-step derivation from the tangent line; $(x^2+1)^3$ at 1 is 24; failure: the outer slope taken at $x$ gives 6; finite-difference check 24.0000000007.<br>[calculus textbook p. 198](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-textbook.pdf#page=198): partial. Rates multiply (20 miles a gallon times a quarter gallon a dollar is 5 miles a dollar); no failure, no check on the page. | none: no vault note teaches the chain rule. | none: no row in the video lessons index; a gap in this kind only. | covered |
| **N04** The derivative as a slope; the derivatives of $x^2$, $\log u$ and $e^u$ | nothing further | when it bites | 12 | [foundations plan 05, instruction 4](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/05-from-slope-to-gradient.md#4--the-derivative-as-a-slope-you-can-draw): partial. Targets and the figure script; the worked steps are in chapter 4.1. | [chapter 4.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/04-differentiating-by-hand.md#41-tangent-slopes): full. Secant slopes 5, 4.1, 4.01, 4.001; the log-density slope −1.5 at 1.5; failure: $h = 0$ gives 0/0; finite-difference check. The slopes of $\log u$ and $e^u$ are derived in 4.3. | none: no vault note on the derivative. | none. | covered |

Tier and minutes: short path 10 (N05), full path 22.

### B2: the slope of the Gaussian at one standard deviation

Planned question B2 (mechanic). Asks: for $p(x) = \mathcal N(x; 0, 0.25)$, find $p'(0.5)$ by the chain rule. Answer: inner $u = -x^2/(2 \times 0.25) = -0.5$ with $u' = -x/0.25 = -2$; outer $e^u = 0.606531$; so the bell's slope is $0.606531 \times (-2) = -1.213061$, and with the front factor 0.7979 the density's slope is $p'(0.5) = -0.9679$, which is $-2 \times 0.4839$ to rounding. Numbers from: `p1.normal_pdf_0_mu0_sd0.5` = 0.7979 and `p1.normal_pdf_0.5_mu0_sd0.5` = 0.4839 (step file, [step 1](../01-one-expert-from-scratch.md#step-1-a-gaussian-you-can-hold)). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N08** The 1D Gaussian density, evaluated by hand | N01, N07 | needed now | 14 | [the step file, step 1](../01-one-expert-from-scratch.md#step-1-a-gaussian-you-can-hold): partial. The toy's 0.7979 and 0.4839 with why each piece is there; no numbered method and no check in the step.<br>[foundations plan 03, instruction 5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/03-densities-and-the-gaussian.md#5--the-gaussian-by-hand-then-drawn): partial. Targets; the steps are in chapter 2.2. | [chapter 2.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md#22-density-values-by-hand): full. Three worked examples (3.9894; 0.2420; 0.7979, 0.3989, 0.1995); failure: the variance used as σ gives 0.0997; check script. | none graded: the Multivariate Normal Distribution note was found, not read. | partial. [The video lessons index, the Gaussian density and its formula](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#the-gaussian-density-and-its-formula): StatQuest and 3Blue1Brown, five rows; its footer says no video evaluates a height at a point. | covered |
| **N07** A density's area is 1, and the front factor $1/(\sigma\sqrt{2\pi})$ from the substitution $u = (x-\mu)/\sigma$ | N01 | when it bites | 20 | [the step file, step 1, derivation of the front factor](../01-one-expert-from-scratch.md#step-1-a-gaussian-you-can-hold): partial. Substitution, area $\sigma\sqrt{2\pi}$, failure 1.2533 at σ = 0.5 without the factor; no check that the area is 1.<br>[sampler-correctors plan 05, task group 1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/05-gaussian-integrals.md#1--gaussian-integrals-and-completing-the-square): partial. Uses $\int e^{-A(x-\mu)^2/2}dx = \sqrt{2\pi/A}$ as stated. | [chapter 2.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md#22-density-values-by-hand): partial. States the factor and checks the area 1.0000000000 numerically; does not derive it. | [The Front Factor Of A Gaussian Density](../../../../../../Obsidian/Mathematics/Calculus/Integration/The%20Front%20Factor%20Of%20A%20Gaussian%20Density.md#the-front-factor-from-the-bells-area): full, written for this quiz. Area 1 in plain words; σ = 0.5 (1.253314, 0.797885, 0.483941) and σ = 2 (5.013257, 0.199471); seven numbered steps; failure: σ² in place of σ gives 1.595769 and area 2.000000; a rectangle sum and a second route as the check.<br>[Gaussian Integral](../../../../../../Obsidian/Mathematics/Calculus/Integration/Gaussian%20Integral.md): partial. $\sqrt\pi$ shown numerically; its polar proof is still marked unfinished. | partial. [The video lessons index, the Gaussian integral](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#the-gaussian-integral): 3Blue1Brown 3:44 to 12:47 and Prime Newtons 1:33 to 14:02 prove $\sqrt\pi$; chapter boundaries; no σ worked. | covered |
| N05, N04 | see [B1](#b1-why-the-slopes-multiply) | | | | | | | |
| N01 | see [A1](#a1-why-the-log-of-a-product-is-a-sum) | | | | | | | |

Tier and minutes: short path 24 (N05, N08), full path 79.

### B3: the inner slope dropped

Planned question B3 (consequence). Asks: a reader differentiates $p(x) = 0.7979\,e^{-2x^2}$ as $0.7979\,e^{-2x^2}$, forgetting the inner slope. What do they get at $x = 0.5$, and what on the picture says it is wrong? Answer: 0.4839, the height itself, positive; to the right of the mean the bell falls, so the slope must be negative ($-0.9679$, from B2). Numbers from: `p1.normal_pdf_0.5_mu0_sd0.5`; the failure is the one in [chapter 4.3, worked example B, step 2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/04-differentiating-by-hand.md#43-the-chain-rule-and-grad-log-f-as-grad-f-over-f). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| N05, N04 | see [B1](#b1-why-the-slopes-multiply) | | | | | | | |

Tier and minutes: short path 10, full path 22.

### C1: why the front factor is one over sigma root two pi

Planned question C1 (intuition). Asks: why must the front factor be $1/(\sigma\sqrt{2\pi})$, shrinking as the bell widens, and what is the area under the bell at σ = 0.5 if it is left out? Answer: substituting $u = (x-\mu)/\sigma$ makes $dx = \sigma\,du$, so the unscaled bell's area is $\sigma\sqrt{2\pi}$, which at σ = 0.5 is 1.2533; the front factor is its reciprocal, 0.7979. Numbers from: the step file, [step 1](../01-one-expert-from-scratch.md#step-1-a-gaussian-you-can-hold), and `p1.normal_pdf_0_mu0_sd0.5`. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| N07, N01 | see [B2](#b2-the-slope-of-the-gaussian-at-one-standard-deviation) and [A1](#a1-why-the-log-of-a-product-is-a-sum) | | | | | | | |

Tier and minutes: short path 0 (no needed-now node), full path 43.

### C2: the toy Gaussian by hand

Planned question C2 (mechanic). Asks: evaluate $\mathcal N(x; 0, 0.25)$ at $x = 0$ and $x = 0.5$, then the front factor of the two-dimensional $\mathcal N(0, 0.25\,I)$. Answer: $1/(0.5\sqrt{2\pi}) = 0.7979$; times $e^{-1/2} = 0.606531$ gives 0.4839; the 2D front factor is $1/(2\pi \times 0.25) = 0.6366 = 0.7979^2$. Numbers from: `p1.normal_pdf_0_mu0_sd0.5`, `p1.normal_pdf_0.5_mu0_sd0.5`, and the step file's 2D front factor $1/(2\pi\sigma^2)$ in step 1. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N09** The isotropic 2D Gaussian as the product of two 1D densities, front factor $1/(2\pi\sigma^2)$ | N08 | when it bites | 14 | [the step file, step 1, the 2D front factor](../01-one-expert-from-scratch.md#step-1-a-gaussian-you-can-hold): partial. Why the two factors multiply and why σ is squared; no worked value, no check.<br>[foundations plan 03, instruction 5.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/03-densities-and-the-gaussian.md#5--the-gaussian-by-hand-then-drawn): partial. | [chapter 2.3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/02-densities-and-the-gaussian.md#23-the-gaussian-formula-from-1d-to-multivariate): full. Six-step collapse and split; 0.029275 by both routes; failure: Σ in place of Σ⁻¹ gives 17; the two routes as the check. | none graded: Diagonal Covariance was found, not read. | partial. 3Blue1Brown, 13:15 to 15:18, the product $g(x)g(y)$, row 4 of the [Gaussian density table](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#the-gaussian-density-and-its-formula); caption-read; no numbers. | covered |
| N08, N07 | see [B2](#b2-the-slope-of-the-gaussian-at-one-standard-deviation) | | | | | | | |
| N01 | see [A1](#a1-why-the-log-of-a-product-is-a-sum) | | | | | | | |

Tier and minutes: short path 14 (N08), full path 71.

### C3: completing the square

Planned question C3 (mechanic). Asks: complete the square in $3x^2 + 18x - 21 = 0$ and solve. Answer: divide by 3, $x^2 + 6x = 7$; add $(6/2)^2 = 9$ to both sides, $(x+3)^2 = 16$; $x = 1$ or $x = -7$. Numbers from: [algebra handbook p. 74](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=74), its own example. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N10** Completing the square: $Ax^2 - 2Bx + C = A(x - B/A)^2 + C - B^2/A$ | nothing further (expanding a square) | when it bites | 14 | [sampler-correctors plan 05, task group 1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/05-gaussian-integrals.md#1--gaussian-integrals-and-completing-the-square): full. Two expert energies to $A = 3.25$, $B = -0.75$, $C = 3.25$; mean −0.230769, variance 0.307692, constant 1.538462; failure: dropping the constant gives 1.390427 for 0.298540; numerical integration as the check. | [algebra handbook p. 74](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=74): partial. Six numbered steps on $3x^2 + 18x - 21 = 0$; no why, no failure, no check.<br>[chapter 10.1, step 4](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/10-completing-the-square.md#101-the-product-of-two-gaussians-precisions-add): partial. The identity stated with "check by expanding". | [Products And Quotients Of Diagonal Gaussians, how the product works](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Products%20And%20Quotients%20Of%20Diagonal%20Gaussians.md#how-the-product-works): partial. The completion is written in symbols; its numbers are at the level of the precision and the mean. | partial. [The video lessons index, completing the square](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#completing-the-square): Leonard, four rows; chapter boundaries; never inside a Gaussian exponent. | covered |

Tier and minutes: short path 0, full path 14.

### C4: two Gaussians multiplied

Planned question C4 (consequence). Asks: multiply $\mathcal N(x; -2, 1)$ by $\mathcal N(x; 2, 4)$. Two readers answer "mean 0" and "variance 5". What does each get wrong, and what is right? Answer: precisions add, $1 + 0.25 = 1.25$, so the variance is 0.8 (not $1 + 4 = 5$, which would be wider than both); the mean is precision-weighted, $0.8 \times (1 \times (-2) + 0.25 \times 2) = -1.2$ (not the plain average 0). Numbers from: [chapter 10.1, worked example 10.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/10-completing-the-square.md#101-the-product-of-two-gaussians-precisions-add), steps 1 and 2. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N11** The product of two Gaussian densities: precisions add, the mean is precision-weighted | N08, N10 | when it bites | 15 | [the step file, step 10, derivation of $c_1$ and $c_2$](../01-one-expert-from-scratch.md#step-10-the-reverse-step): partial. Precisions added and the mean as $\eta_t/\lambda_t$, with 0.0306, 0.04249 and 0.95472 at t = 50; no check in the passage. | [chapter 10.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/10-completing-the-square.md#101-the-product-of-two-gaussians-precisions-add): full. Eight-step derivation; the two-expert toy to mean (−1.2, 0) and variance 0.8; failures: variances added gives 5, means averaged gives 0; script check. | [Products And Quotients Of Diagonal Gaussians, how the product works](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Products%20And%20Quotients%20Of%20Diagonal%20Gaussians.md#how-the-product-works): full. Derivation through the logs, four worked rows, the negative precision −0.75 as the failure, the run as the check. | partial. [The video lessons index, the product of two Gaussians](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#the-product-of-two-gaussians-and-the-precision-weighted-mean): van Biezen works 68 and 75 to 70.33 with error 1.33; caption-read; no density product, no check. | covered |
| N10 | see [C3](#c3-completing-the-square) | | | | | | | |
| N08, N07 | see [B2](#b2-the-slope-of-the-gaussian-at-one-standard-deviation) | | | | | | | |
| N01 | see [A1](#a1-why-the-log-of-a-product-is-a-sum) | | | | | | | |

Tier and minutes: short path 14 (N08), full path 86.

### D1: why scaling squares the variance

Planned question D1 (intuition). Asks: why does multiplying a variable by $a$ multiply its variance by $a^2$ and its standard deviation by $|a|$, while adding $b$ leaves both alone? Answer: the variance averages squared distances from the mean; scaling multiplies every distance by $a$, so every squared distance by $a^2$; a shift moves the mean by the same $b$ as every value, so every distance is unchanged. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N13** The mean and variance of $aX + b$: $a\mu + b$ and $a^2\sigma^2$, standard deviation $\lvert a\rvert\sigma$ | N12 | needed now | 18 | [the step file, step 4, why the square roots are there](../01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule): partial. "Using $\beta_t$ as the noise coefficient would give variance $\beta_t^2$"; no worked example of a general $aX + b$, no steps, no check. | [Chan p. 151, Theorem 3.5](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=151): partial. The scale and shift rules with a picture and a proof exercise; no numbers, no failure, no check.<br>[conditional-expectation chapter 1, linearity (1.2)](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/01-expectation-and-the-best-constant-guess.md#the-equations-that-matter): partial. $E[aX + bY]$ only. | [Scaling And Shifting A Random Variable](../../../../../../Obsidian/Mathematics/Statistics/Scaling%20And%20Shifting%20A%20Random%20Variable.md#how-scaling-and-shifting-work): full, written for this quiz. The die's dots on a line in plain words; a face-by-face table to 35/3 = 11.6667 and sd 3.4157, and $-2X + 1$; six numbered steps; failures: $a$ for $a^2$ gives 5.8333, the shift added gives 12.6667, the ladder's slip 0.2722 for 0.5217; the script as the check.<br>[Sampling A Diagonal Gaussian, how sampling works](../../../../../../Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Sampling%20A%20Diagonal%20Gaussian.md#how-sampling-works): partial. The square stated, not derived, for a standard normal only.<br>[The Forward Process, how the jump works](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md#how-the-jump-works): partial. $\bar\alpha\,\mathrm{Var}[x_0]$ used, not derived. | none: the video lessons index says no listed creator works $\mathrm{Var}(aX + bY)$. | covered |
| **N12** The mean and variance from their definitions | nothing further | when it bites | 14 | none read: the conditional-expectation journey's plan 01 was found, not read. | [conditional-expectation chapter 1, the equations and Example 1](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/01-expectation-and-the-best-constant-guess.md#worked-examples): full. A fair die: mean 3.5, $E[X^2] = 91/6$, variance 35/12; failure: ignoring the weights gives 3.5 for the loaded die instead of 4.5; face-by-face check.<br>[Chan p. 150 to 151](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=150): partial. Definitions 3.6 and 3.7 and Example 3.16; no failure. | none: no vault note on expectation. | partial. [The video lessons index, the expected value as a weighted sum](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#the-expected-value-as-a-weighted-sum): StatQuest 5:09 to 8:02; chapter boundaries; no variance. | covered |

Tier and minutes: short path 18 (N13), full path 32.

### D2: a scaled and shifted die

Planned question D2 (mechanic). Asks: a fair die $X$ has mean 3.5 and variance 35/12. Find the mean, variance and standard deviation of $Y = 2X + 1$. Answer: mean $2 \times 3.5 + 1 = 8$; variance $4 \times 35/12 = 35/3 = 11.6667$; standard deviation 3.4157. Numbers from: [conditional-expectation chapter 1, Example 1](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/01-expectation-and-the-best-constant-guess.md#worked-examples). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| N13, N12 | see [D1](#d1-why-scaling-squares-the-variance) | | | | | | | |

Tier and minutes: short path 18, full path 32.

### D3: two noise steps in one

Planned question D3 (mechanic). Asks: two forward steps with $\alpha_1 = 0.9$, $\alpha_2 = 0.8$ leave the noise $\sqrt{\alpha_2(1-\alpha_1)}\,e_1 + \sqrt{1-\alpha_2}\,e_2$ with $e_1, e_2$ independent standard normals. Find its variance two ways. Answer: $0.8 \times 0.1 + 0.2 = 0.28$, and $1 - 0.9 \times 0.8 = 0.28$. Numbers from: [chapter 9.1, worked example 9.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#91-independent-gaussian-noises-add-their-variances). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N14** The variance of a sum of independent variables; a sum of independent Gaussians is Gaussian | N12, N13 | needed now | 13 | [foundations plan 09, instructions 3.3 and 3.4](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/09-the-forward-process.md#3--derive-the-jump-on-paper-before-anything-runs): partial. Asks for 0.28 both ways; the steps are in chapter 9.1.<br>[the step file, step 4, derivation of the one-shot formula](../01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule): partial. The two-step variance in symbols, with why the sum is there; no numbers for the sum, no check. | [chapter 9.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#91-independent-gaussian-noises-add-their-variances): full. Seven-step derivation (the cross term killed by independence, the shape by the generating function); 0.28 both ways; failure: standard deviations added give 0.730056, whose square 0.532982 carries the cross term 0.252982; check script. | [The Forward Process, how the jump works](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md#how-the-jump-works): partial. Means and variances worked for expert 1 at two levels and checked against a run; no failure in the section. | partial. CME296 lecture 1, [39:07 to 45:40](https://www.youtube.com/watch?v=tr-CUpw--ck&t=2347s), unrolls two steps and adds the independent variances, from the step file's step 4 table; caption-read; no numbers. 3Blue1Brown, Gaussian + Gaussian, in [the video lessons index](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#adding-two-independent-gaussians); chapter boundaries. | covered |
| N13, N12 | see [D1](#d1-why-scaling-squares-the-variance) | | | | | | | |

Tier and minutes: short path 31 (N13, N14), full path 45.

### D4: adding standard deviations

Planned question D4 (consequence). Asks: a reader adds the two coefficients of D3 before squaring. What variance comes out, where does the excess come from, and what does the same slip do to one forward step at t = 50 if $\beta_{50}$ is used as the noise coefficient? Answer: $(\sqrt{0.08} + \sqrt{0.2})^2 = 0.730056^2 = 0.532982$, which is 0.28 plus the cross term $2ab = 0.252982$ that independence removes. With $\beta_{50} = 0.03155$ as the coefficient the step's noise variance would be $0.03155^2 = 0.000995$ instead of 0.03155; the coefficient must be $\sqrt{\beta_{50}} = 0.1776$. Numbers from: [chapter 9.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#91-independent-gaussian-noises-add-their-variances), step 2's failure; $\beta_{50} = 0.03155$ from the step file's step 4 table and `p1.beta_50_sd` = 0.1776. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| N14, N13, N12 | see [D3](#d3-two-noise-steps-in-one) and [D1](#d1-why-scaling-squares-the-variance) | | | | | | | |

Tier and minutes: short path 31, full path 45.

### E1: why conditioning divides by the condition

Planned question E1 (intuition). Asks: why is $P(A \mid B)$ the joint divided by $P(B)$, and what has no answer when $P(B) = 0$? Answer: learning $B$ deletes every outcome outside $B$, and the survivors are rescaled by one common factor so they total 1, which forces the factor $1/P(B)$; with $P(B) = 0$ nothing survives to rescale. Numbers from: none needed; the reasoning is [chapter 1.2's derivation](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/01-joint-tables-and-independence.md#12-conditioning-is-deletion-then-rescaling). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N15** Conditional probability as deletion then rescaling | nothing further | when it bites | 14 | [foundations plan 01, instruction 4](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/01-probability-you-can-compute.md#4--conditioning-as-deletion-and-rescaling): partial. Sets the work; the steps are in chapter 1.2. | [chapter 1.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/01-joint-tables-and-independence.md#12-conditioning-is-deletion-then-rescaling): full. Five-step derivation; first die even given sum 7 is 3/6 = 0.5; failure: dividing by $P(A)$ gives 0.1667; the count ratio as the check. | none: no vault note on conditional probability. | none: the Bayes rows below are the nearest. | covered |

Tier and minutes: short path 0, full path 14.

### E2: the librarian and the farmers

Planned question E2 (mechanic). Asks: 10 librarians and 200 farmers; 40% of librarians and 10% of farmers fit a description. Someone fits it. What is the chance they are a librarian? Answer: from counts, $4/(4 + 20) = 1/6$; from the parts, $P(A) = 10/210 = 0.047619$, $P(B) = 0.019048 + 0.095238 = 0.114286$, and $0.4 \times 0.047619 / 0.114286 = 0.1667$. Numbers from: [chapter 1.5, worked example 1.5a](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/01-joint-tables-and-independence.md#15-bayes-in-two-lines-from-the-definition). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N16** Bayes' rule, with the evidence built by the law of total probability; posterior proportional to likelihood times prior | N15 | when it bites | 12 | [foundations plan 02, instruction 4](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/02-the-assumption-poe-rests-on.md#4--bayes-from-counts-to-symbols): partial. Sets the work; the steps are in chapter 1.5. | [chapter 1.5](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/01-joint-tables-and-independence.md#15-bayes-in-two-lines-from-the-definition): full. Four-step derivation with the evidence split; the librarian case both ways (1/6); failure: forgetting the 200 farmers reads off 40%; the two routes agree as the check.<br>[Chan p. 105 to 106](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=105): partial. Theorems 2.7 and 2.8 with proofs and Corollary 2.4; no numbers on these pages. | none: no vault note on Bayes' rule. | partial. [The video lessons index, Bayes' theorem](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#bayes-theorem): 3Blue1Brown 4:09 to 10:13, StatQuest 5:21 to 9:12; chapter boundaries; events only. | covered |
| N15 | see [E1](#e1-why-conditioning-divides-by-the-condition) | | | | | | | |

Tier and minutes: short path 0, full path 26.

### E3: dividing by the wrong margin

Planned question E3 (consequence). Asks: on two dice, $P(\text{first even}, \text{sum } 7) = 3/36$. A reader divides by $P(\text{first even}) = 0.5$ to get $P(\text{first even} \mid \text{sum } 7)$. What do they get, which probability is it really, and what is right? Answer: $0.0833/0.5 = 0.1667$, which is $P(\text{sum } 7 \mid \text{first even})$, the other direction; the right answer divides by $P(\text{sum } 7) = 6/36$ and is 0.5. Numbers from: [chapter 1.2, worked example 1.2a](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/01-joint-tables-and-independence.md#12-conditioning-is-deletion-then-rescaling), step 3, and chapter 1.5, worked example 1.5b. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| N15, N16 | see [E1](#e1-why-conditioning-divides-by-the-condition) and [E2](#e2-the-librarian-and-the-farmers) | | | | | | | |

Tier and minutes: short path 0, full path 26.

### E4: a Gaussian prior times a Gaussian likelihood

Planned question E4 (composed: Bayes with the product of Gaussians). Asks: a prior $\mathcal N(x; -2, 1)$ and a likelihood that, as a function of $x$, is $\mathcal N(x; 2, 4)$. The evidence does not depend on $x$. What are the posterior's mean and variance, and why may the evidence be dropped? Answer: posterior ∝ likelihood × prior, the evidence being one number for every $x$; the product has precision 1.25, variance 0.8 and mean −1.2. Numbers from: [chapter 10.1, worked example 10.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/10-completing-the-square.md#101-the-product-of-two-gaussians-precisions-add), first coordinate; the rule as the step file's step 10 states it ("the posterior is proportional to likelihood times prior", [Bishop p. 41](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=41), found, not read). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| N16, N15 | see [E2](#e2-the-librarian-and-the-farmers) and [E1](#e1-why-conditioning-divides-by-the-condition) | | | | | | | |
| N11, N10 | see [C4](#c4-two-gaussians-multiplied) and [C3](#c3-completing-the-square) | | | | | | | |
| N08, N07, N01 | see [B2](#b2-the-slope-of-the-gaussian-at-one-standard-deviation) and [A1](#a1-why-the-log-of-a-product-is-a-sum) | | | | | | | |

Tier and minutes: short path 14 (N08), full path 112.

### F1: why the normaliser leaves the score

Planned question F1 (intuition). Asks: multiply a density by 137. Its height grows 137 times; why does $\nabla \log p$ not move, and why does $\nabla p$ move? Answer: by the chain rule $\nabla \log p = \nabla p / p$, and the 137 appears once on top and once underneath, so it cancels; equivalently $\log(137p) = \log 137 + \log p$ and the constant has slope 0. Numbers from: [chapter 4.3, worked example B](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/04-differentiating-by-hand.md#43-the-chain-rule-and-grad-log-f-as-grad-f-over-f), steps 3 and 5: $(-3, -4)$ for both. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N17** $\nabla \log p = \nabla p / p$, and a constant factor drops out | N02, N05, N06 | needed now | 8 | [foundations plan 06, instructions 7.1 to 7.6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/06-the-gradient-field-drawn.md#7--the-gradient-of-a-log-derived-rather-than-quoted): partial. Targets $(-3, -4)$ and the 137 twin and a check script; the steps are in chapter 4.3. | [chapter 4.3, the slope of the log, the five lines and worked example B](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/04-differentiating-by-hand.md#43-the-chain-rule-and-grad-log-f-as-grad-f-over-f): full. $p(3,4) = 5.931153\times10^{-7}$, ratio $(-3, -4)$ for $p$ and $137p$; failures: the inner slope dropped, and "137 times larger"; finite-difference check. | [Why The Normaliser Has No Slope In X](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Why%20The%20Normaliser%20Has%20No%20Slope%20In%20X.md): partial. A constant's slope is 0, worked on N(3, 4) at β = 2 (quotient −1.0025), checked by `zslope.py`; no failure with its number. | partial. CME296 lecture 2, [8:17 to 13:22](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=497s), from the step file's step 5 table: ∇log p = ∇p/p and the normaliser's gradient is zero; caption-read; no numbers. | covered |
| **N06** Partial derivatives and the gradient | N04 | when it bites | 14 | [foundations plan 05, instructions 5 and 6](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/05-from-slope-to-gradient.md#5--partial-derivatives-one-variable-at-a-time): partial. | [chapter 4.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/04-differentiating-by-hand.md#42-first-order-against-second-order-rise): full. $\nabla f(3, 4) = (6, 8)$, length 10; failure: $y$ not held fixed; check: rises $1.00001\times10^{-3}$ against $10^{-8}$. | none: no vault note on the gradient. | partial. [The video lessons index, the gradient](../../../../../../Obsidian/Mathematics/references/videos/Video%20Lessons.md#the-gradient): Professor Dave, the Organic Chemistry Tutor, Leonard; chapter boundaries. | covered |
| N05, N04 | see [B1](#b1-why-the-slopes-multiply) | | | | | | | |
| N02, N01 | see [A1](#a1-why-the-log-of-a-product-is-a-sum) | | | | | | | |

Tier and minutes: short path 34 (N02, N05, N17), full path 83.

### F2: the score of the toy Gaussian

Planned question F2 (mechanic). Asks: find the score of $\mathcal N(x; 0, 0.25)$ at $x = 0.5$ from $-(x - \mu)/\sigma^2$, then confirm it as $p'(x)/p(x)$ with B2's numbers, then give the score of the 2D $\mathcal N(0, 0.25\,I)$ at $(0.5, 0)$. Answer: $-0.5/0.25 = -2$; $-0.9679/0.4839 = -2.0$; $(-2, 0)$. Numbers from: `p1.normal_pdf_0.5_mu0_sd0.5` and B2. Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| **N18** The score of a Gaussian, $-(x - \mu)/\sigma^2$, one coordinate at a time for an isotropic one | N02, N06, N08, N17 | needed now | 13 | [the step file, step 5](../01-one-expert-from-scratch.md#step-5-the-noise-is-the-score-rescaled): full for the conditional Gaussian. The log, the gradient and the chain to $-\varepsilon/\sqrt{1-\bar\alpha_t}$; toy $(-0.4153, 1.6613)$; failure: the divisor removed gives $(-0.3, 1.2)$; check by autograd and `p1.example_score`.<br>[foundations plan 08, instructions 1 and 3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/08-score-of-a-gaussian.md#3--the-score-of-a-gaussian-derived-twice): partial. | [chapter 6.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#61-the-gradient-of-a-quadratic-form): full. Six-step derivation with $\log Z$ on its own line; scores −2, −8 and +0.25; failure: a minus sign on the third; check script and finite differences.<br>[little book p. 114](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=114): partial. The derivation in symbols; no numbers, no failure, no check. | [The Log Of A Power And Of A Quotient, the two-dimensional score](../../../../../../Obsidian/Mathematics/Algebra/Logarithms/The%20Log%20Of%20A%20Power%20And%20Of%20A%20Quotient.md): partial. $(-2, -4)$ at $(1, 2)$ for $\beta = 2$; no failure. | partial. CME296 lecture 2, [21:28 to 24:12](https://www.youtube.com/watch?v=_WaR2fjZpEQ&t=1288s), states $-(x-\mu)/\sigma^2$; Vizuara lecture 6, [32:45 to 38:19](https://www.youtube.com/watch?v=6ELzfPXUqps&t=1965s), derives the noised point's score; both caption-read, from the step file's step 5 tables; no numbers worked. | covered |
| N17, N06 | see [F1](#f1-why-the-normaliser-leaves-the-score) | | | | | | | |
| N08, N07 | see [B2](#b2-the-slope-of-the-gaussian-at-one-standard-deviation) | | | | | | | |
| N05, N04 | see [B1](#b1-why-the-slopes-multiply) | | | | | | | |
| N02, N01 | see [A1](#a1-why-the-log-of-a-product-is-a-sum) | | | | | | | |

Tier and minutes: short path 61 (N02, N05, N08, N17, N18), full path 130.

### F3: sigma in place of sigma squared

Planned question F3 (consequence). Asks: a reader writes the score of $\mathcal N(0, 0.25)$ at 0.5 as $-(x-\mu)/\sigma$, and another as the gradient of the squared distance alone. What does each get? Answer: $-0.5/0.5 = -1$, half the right length (−2), pointing the right way, so it looks plausible; the squared distance alone gives $2(x - \mu) = +1$, pointing away from the mean. Numbers from: the toy's $\mu = 0$, $\sigma = 0.5$ (step file, step 1); the squared-distance slip is the step file's [step 5 consequence](../01-one-expert-from-scratch.md#step-5-the-noise-is-the-score-rescaled). Verdict: **ready**.

| Node | Leans on | Tier | Min | Learning plan | Textbook | Obsidian note | Video | Verdict |
|---|---|---|---:|---|---|---|---|---|
| N18 and its tree | see [F2](#f2-the-score-of-the-toy-gaussian) | | | | | | | |

Tier and minutes: short path 61, full path 130.

## Bank questions from the textbooks

Navigation: ⬅️ [Trees](#trees-one-section-per-question) | 📋 [Table of contents](#table-of-contents) | [Gaps](#gaps) ➡️

Twenty-five problems from the shelf, four or five per concept, graded from the easiest that shows the mechanic to the hardest that bites. Each is a question of its own and a node of its own (the problem and its solution), and every concept it uses is a concept node above, so its tree is that node plus the concept nodes in the sixth column with everything they lean on. Each has a row in the [worked-examples index](../../../../../../Obsidian/Mathematics/references/exercises/Worked%20Examples.md), in the six sections that list "quiz 0", and a copy of each section sits after the passage that teaches it: [Solving For The Exponent With Logarithms](../../../../../../Obsidian/Mathematics/Algebra/Logarithms/Solving%20For%20The%20Exponent%20With%20Logarithms.md), [foundations chapter 4](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/04-differentiating-by-hand.md) (two sections), [The Front Factor Of A Gaussian Density](../../../../../../Obsidian/Mathematics/Calculus/Integration/The%20Front%20Factor%20Of%20A%20Gaussian%20Density.md), [Scaling And Shifting A Random Variable](../../../../../../Obsidian/Mathematics/Statistics/Scaling%20And%20Shifting%20A%20Random%20Variable.md) and [foundations chapter 1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/01-joint-tables-and-independence.md).

The `Min` column is the time to solve the problem by hand, an estimate by difficulty (5 easy, 10 medium, 15 or 20 hard). It is not part of the read-first paths.

| Problem node | What it asks | Where | Solution | Min | Concept nodes in its tree | Verdict |
|---|---|---|---|---:|---|---|
| **KA1** | log form: $\log_4 64 = x$, $\ln e = x$ | [Algebra handbook, examples 1 and 2, PDF p. 113](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=113) | in the book, every step: 3 and 1 | 5 | N01, N02 | ready |
| **KA2** | condense $6 + \ln(3e^5)$ and $\log_4(1/16) - 6^{\log_6 4}$ | [Algebra handbook, PDF p. 116](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=116) | in the book, every step: $11 + \ln 3$ and $-6$ | 5 | N01, N02 | ready |
| **KA3** | solve $2 + 4\cdot 3^{x-2} = 110$ | [Algebra handbook, PDF p. 125](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=125) | in the book, every step: $x = 5$ | 10 | N01, N02, N03 | ready |
| **KA4** | when $75e^{-t/125}$ watts falls to 7 watts | [Anton, Calculus, section 0.5, Example 3, PDF p. 82](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-textbook.pdf#page=82) | in the book, every step: 296.4 days | 10 | N01, N02, N03 | ready |
| **KB1** | differentiate $e^{5x}$ | [Calculus practice problems, #390, PDF p. 71](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-practice-problems.pdf#page=71) | in the book, PDF p. 276: $5e^{5x}$ | 5 | N04, N05 | ready |
| **KB2** | differentiate $(x^2+3x)^{100}$ | [Calculus practice problems, #352, PDF p. 66](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-practice-problems.pdf#page=66) | in the book, PDF p. 262, every step | 5 | N04, N05 | ready |
| **KB3** | differentiate $\ln\sqrt{x^2+5}$ | [Calculus practice problems, #379, PDF p. 70](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-practice-problems.pdf#page=70) | in the book, PDF p. 271, every step: $x/(x^2+5)$ | 10 | N01, N02, N04, N05 | ready |
| **KB4** | gradient of $x^2e^y$ at $(-2, 0)$, its length and direction | [Anton, Calculus, section 13.6, Example 4, PDF pp. 988 to 989](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-textbook.pdf#page=988) | in the book, every value: $(-4, 4)$, $4\sqrt2$ | 15 | N04, N05, N06 | ready |
| **KC1** | write $x^2 - 4x + 5$ as $(x-2)^2 + 1$ | [Anton, Calculus, Example 8, PDF p. 44](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-textbook.pdf#page=44) | in the book | 5 | N10 | ready |
| **KC2** | complete the square in $3x^2 + 18x - 21 = 0$ (the same problem as C3) | [Algebra handbook, PDF p. 74](../../../../../../goal-setting/learning/mathematics/textbooks/algebra/algebra-handbook.pdf#page=74) | in the book, six steps: $x = 1$ or $-7$ | 5 | N10 | ready |
| **KC3** | $\mathcal N(0, 0.25)$ at 0 and 0.5 in code and by hand, then the variance passed as the scale | [Chan, the Python listing, PDF p. 228](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=228) | [written below](#kc3-solution-the-toy-gaussian-in-code): 0.7979, 0.4839; slip 1.5958, 0.2160 | 10 | N01, N07, N08 | ready |
| **KC4** | complete the square for $q(x_{t-1}\mid x_t, x_0)$ | [The little book, Appendix B, PDF pp. 189 to 192](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=189) | in the book, every step | 20 | N01, N07, N08, N10, N11, N15, N16 | ready |
| **KD1** | mean and variance of a four-valued count | [Introduction to Statistics, Example 4.2, PDF p. 82](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-statistics.pdf#page=82) | in the book: 0.65 and 1.0275 (a printed $4^2$ means $2^2$) | 5 | N12 | ready |
| **KD2** | $E[(aX+b)^2]$ from $E[X] = 1$, $E[X^2] = 3$ | [Chan, Example 3.15, PDF p. 149](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=149) | in the book: $3a^2 + 2ab + b^2$ | 5 | N12, N13 | ready |
| **KD3** | prove the moment, scale and shift rules of variance | [Chan, Practice Exercise 3.6, PDF p. 151](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=151) | in the book, PDF pp. 151 to 152, every step | 10 | N12, N13 | ready |
| **KD4** | three dice in code: 10.5 and 8.75, and why the printed listing gives 9 and 6 | [Chan, the dice listing, PDF pp. 236 to 237](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=236) | [written below](#kd4-solution-three-dice-in-code) | 10 | N12, N13, N14 | ready |
| **KD5** | the sum of $N$ independent Gaussians by generating functions | [Chan, Theorem 6.6, PDF p. 345](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=345) | in the book; the Gaussian generating function is derived in chapter 9.1, step 5 | 15 | N12, N13, N14 | ready |
| **KE1** | $P[3\mid\text{odd}]$ and $P[\text{odd}\mid 3]$ | [Chan, Example 2.44, PDF p. 98](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=98) | in the book: 1/3 and 1 | 5 | N15 | ready |
| **KE2** | the chance of a win, then of opponent A given a win | [Chan, Example 2.47, PDF pp. 106 to 107](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=106) | in the book, every value: 0.375 and 0.4 | 5 | N15, N16 | ready |
| **KE3** | infection after a positive and a negative test | [Murphy, section 2.3.1, PDF pp. 70 to 71](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=70) | in the book, every value: 0.795 and 0.014 (its count check prints 865 for 875) | 10 | N15, N16 | ready |
| **KE4** | the most probable mean under a Gaussian prior and likelihood | [Chan, Example 8.19, PDF pp. 523 to 524](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=523) | in the book, every step of the derivative route | 20 | N01, N02, N04, N05, N06, N07, N08, N10, N11, N15, N16, N17 | ready |
| **KF1** | differentiate $\ln(x^2+1)$ | [Anton, Calculus, section 3.2, Example 2, PDF p. 217](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-textbook.pdf#page=217) | in the book: $2x/(x^2+1)$ | 5 | N01, N02, N04, N05 | ready |
| **KF2** | differentiate $\ln 5x$ and $\ln x^2$ | [Anton, Calculus, exercises 3.2 #1 and #9, PDF p. 219](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-textbook.pdf#page=219) | answers $1/x$, $2/x$ at PDF p. 1245; [steps written below](#kf2-solution-the-log-of-a-constant-times-x) | 5 | N01, N02, N04, N05 | ready |
| **KF3** | differentiate the normal density; $P'/P$ is the score | [Calculus handbook, PDF p. 182](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-handbook.pdf#page=182) | in the book, every step | 10 | N01, N02, N04, N05, N06, N07, N08, N17, N18 | ready |
| **KF4** | the maximum-likelihood mean of Gaussian data | [Chan, Practice Exercise 8.2, PDF p. 491](../../../../../../goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=491) | in the book, every step: the sample mean | 10 | N01, N02, N04, N05, N06, N07, N08, N17 | ready |

### KC3 solution: the toy Gaussian in code

What and why: `scipy.stats.norm.pdf(x, loc, scale)` evaluates a Gaussian density, and its `scale` is the standard deviation, so the toy $\mathcal N(0, 0.25)$ is `norm.pdf(x, 0, 0.5)`.

1. By hand at 0: $0.5 \times \sqrt{2\pi} = 0.5 \times 2.506628 = 1.253314$, so the height is $1/1.253314 = 0.7979$.
2. By hand at 0.5: the exponent is $-0.5^2/(2 \times 0.25) = -0.5$, $e^{-0.5} = 0.606531$, and $0.7979 \times 0.606531 = 0.4839$.
3. In code: `norm.pdf(0, 0, 0.5)` and `norm.pdf(0.5, 0, 0.5)` print 0.7979 and 0.4839 (run here with SciPy in the co3 environment), the walk numbers `p1.normal_pdf_0_mu0_sd0.5` and `p1.normal_pdf_0.5_mu0_sd0.5`.

How it goes wrong: `norm.pdf(x, 0, 0.25)` treats 0.25 as the standard deviation. It prints 1.5958 at 0 and $1.5958 \times e^{-2} = 0.2160$ at 0.5, a narrower and taller bell. Check: the two hand values equal the walk numbers, and the slip's peak is exactly twice the right one, since halving σ doubles the peak.

### KD4 solution: three dice in code

What and why: a die has mean 3.5 and variance 35/12 ([conditional-expectation chapter 1, Example 1](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/textbook/01-expectation-and-the-best-constant-guess.md#worked-examples)); three independent dice add their means and their variances.

1. Mean of the sum: $3 \times 3.5 = 10.5$.
2. Variance of the sum, by independence: $3 \times 35/12 = 8.75$.
3. In code, `np.random.randint(1, 7, size=(100000, 3)).sum(1)` gives a sample mean near 10.5 and a variance near 8.75; run here with NumPy's generator at seed 0 (`integers(1, 7)`), it printed 10.5 and 8.692.

How it goes wrong: the printed listing calls `np.random.randint(1,6)`, whose upper end is excluded, so each die is uniform on 1 to 5, with mean 3 and variance 2. Three of them give 9 and 6, which look like plausible dice numbers. Check: a direct count over the five faces, $(1+2+3+4+5)/5 = 3$ and $((-2)^2 + (-1)^2 + 0 + 1 + 4)/5 = 2$, computed by script here as 9.0 and 6.0 for three dice; the same seeded run with the upper end 6 printed 9.002 and 5.973.

### KF2 solution: the log of a constant times x

What and why: the log turns the constant 5 into an added $\ln 5$, which has slope 0, so it leaves the derivative.

1. Split: $\ln 5x = \ln 5 + \ln x$.
2. Differentiate term by term: $0 + 1/x = 1/x$.
3. The same by the chain rule: $\frac{1}{5x} \times 5 = 1/x$.
4. For $\ln x^2 = 2\ln x$, the derivative is $2/x$.

How it goes wrong: dropping the inner slope 5 in step 3 gives $1/(5x)$, which is 0.1 at $x = 2$ against the right 0.5. Check: a central difference of $\ln 5x$ at $x = 2$ with step $10^{-5}$ gives 0.500000, computed by script here.

**Not banked, with the reason.** Bishop's Exercises 1.7, 1.11 and 2.38, Mathematics for Machine Learning's Exercise 6.10 and Murphy's Exercise 4.3 print no solution, and none was written. Anton's exercises 0.5 #25, #27, #49 and 13.3 #25, #39 print final values only. Chan's Theorem 5.12 needs a convolution integral and Chan's Example 4.26 a change of variables in a density, which no node here teaches; quiz 1.1 banks both. Murphy's section 4.4.2.2 has no numeric question. Introduction to Statistics Example 3.26 rounds before dividing and prints three wrong answers.

## Gaps

Navigation: ⬅️ [Bank questions](#bank-questions-from-the-textbooks) | 📋 [Table of contents](#table-of-contents) | [Fills](#fills) ➡️

**Gaps that block a question.** None. The audit found three, and each is filled by a passage graded full on re-reading; the Fills table records them. Ranked by how many questions each blocked:

| Node | Kinds missing a full passage | Questions it blocked | Who filled | Fill done |
|---|---|---|---|---|
| N01 The exponent rules | all four | A1, A2, A3, B2, C1, C2, C4, E4, F1, F2, F3 (11) | `vault-writer` | yes: [The Exponent Rules](../../../../../../Obsidian/Mathematics/Algebra/The%20Exponent%20Rules.md), graded full |
| N07 The front factor from the substitution | all four | B2, C1, C2, C4, E4, F2, F3 (7) | `vault-writer` | yes: [The Front Factor Of A Gaussian Density](../../../../../../Obsidian/Mathematics/Calculus/Integration/The%20Front%20Factor%20Of%20A%20Gaussian%20Density.md), graded full |
| N13 The mean and variance of $aX + b$ | all four | D1, D2, D3, D4 (4) | `vault-writer` | yes: [Scaling And Shifting A Random Variable](../../../../../../Obsidian/Mathematics/Statistics/Scaling%20And%20Shifting%20A%20Random%20Variable.md), graded full |

**Gaps that block nothing.** Kinds with no full passage on a node another kind covers. Each is listed so it is not hidden.

**Videos.** No video cell is full. None for N01, N04, N05, N13 and N15; mention for N02; partial for the rest. The chain rule (N05), the log rules (N02), the exponent rules (N01) and the mean and variance of $aX + b$ (N13) have no row in the video lessons index. Filled by `video-pulse` when a creator on the list covers one; no video search was run here.

**Obsidian notes.** No vault note teaches N04, N05, N06, N12, N15 or N16. Each has a full chapter section.

**Learning plans.** The foundations plans set targets and send the worked steps to their chapter, so every plan cell is partial. The step file is full for N18 only.

**Owed by the fills, outside the vault.** The three writers were limited to the vault, so the learning tree does not yet link back to the new notes: foundations chapter 3 (section 3.1, step 2), the sampler-correctors plan 05 (equation 5.4) and the conditional-expectation chapter 1 (linearity, 1.2) each owe a link. None of the three new notes has run through `~/.vault-sandbox`, which does not exist on the cluster; each ran with the system Python 3.12.10 and carries an unfinished box saying so, with three pictures each waiting for `render-diagrams`.

**The short path runs over.** The seven needed-now nodes take 92 minutes against the 90 the learner has.

## Fills

Navigation: ⬅️ [Gaps](#gaps) | 📋 [Table of contents](#table-of-contents)

| Node | Passage written | Path | Date | Grade after re-audit |
|---|---|---|---|---|
| N01 The exponent rules | a new note, by `vault-writer`, from `exponents.py` run here | [The Exponent Rules](../../../../../../Obsidian/Mathematics/Algebra/The%20Exponent%20Rules.md#what-the-rules-are-and-why-they-are-here) | 2026-10-20 | note full; video none |
| N07 The front factor from the substitution | a new note beside Gaussian Integral, which now points to it, by `vault-writer`, from `front_factor.py` run here | [The Front Factor Of A Gaussian Density](../../../../../../Obsidian/Mathematics/Calculus/Integration/The%20Front%20Factor%20Of%20A%20Gaussian%20Density.md#the-front-factor-from-the-bells-area) | 2026-10-20 | note full; video partial (the $\sqrt\pi$ proofs) |
| N13 The mean and variance of $aX + b$ | a new note beside Variance, by `vault-writer`, from `scale_shift.py` run here | [Scaling And Shifting A Random Variable](../../../../../../Obsidian/Mathematics/Statistics/Scaling%20And%20Shifting%20A%20Random%20Variable.md#how-scaling-and-shifting-work) | 2026-10-20 | note full; video none |
| the bank | six sections in the worked-examples index (25 rows, quiz 0, KA1 to KF4), each at revision 1, and one copy of each after the passage that teaches it | [Worked Examples](../../../../../../Obsidian/Mathematics/references/exercises/Worked%20Examples.md) | 2026-10-20 | six copies, each equal to its source |
