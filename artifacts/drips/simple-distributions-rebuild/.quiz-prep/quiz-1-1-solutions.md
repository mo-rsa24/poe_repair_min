# Quiz 1.1 banked problems: the solutions written here
Standard: QUIZ_PREP_FORMAT v1

⬅️ [Quiz 1.1 preparation](quiz-1-1.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 1](../ladder/01-one-diffusion-expert.md)

These are the full-steps solutions for the quiz 1.1 banked problems whose book gives no solution. Every other banked problem cites the book's or the journey's own worked solution in [the coverage file's bank tables](quiz-1-1.md#banked-problems). The quiz page links here after an attempt and never before it.

**Where the numbers come from.** The schedule values are the `p1.` keys that `python -m rebuild.walk_numbers` prints (`rebuild/out/walk-numbers.json`): $\bar\alpha_{50} = 0.4783$ and $\sqrt{1-\bar\alpha_{50}} = 0.7223$, with $\sqrt{0.4783} = 0.6916$ and $1 - 0.4783 = 0.5217$ from steps 4 and 5 of [the step file](../01-one-expert-from-scratch.md). The clean toy point is $x_0 = (0, 0.5)$. The data spread $0.25$ is the variance of step 1's Gaussian $\mathcal N(0, 0.5^2)$, whose density is 0.7979 at 0 and 0.4839 at 0.5 (`p1.normal_pdf_0_mu0_sd0.5`, `p1.normal_pdf_0.5_mu0_sd0.5`). Book numbers are cited where they are used. Every number below is printed by `quiz-1-1-solutions-check.py` in this folder:

```bash
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python quiz-1-1-solutions-check.py
```

Simulated values use one million draws with seed 0, so they sit within about two standard errors (0.001 for these variances) of the closed form.

## Contents

- [S1. The mean and variance of a sum of independent variables (B1.2, Bishop 1.10 and 2.27)](#s1-the-mean-and-variance-of-a-sum-of-independent-variables-b12-bishop-110-and-227)
- [S2. A linear map plus independent noise (B1.6, MML 6.12)](#s2-a-linear-map-plus-independent-noise-b16-mml-612)
- [S3. Two Gaussians with unequal variances (B1.7, Chan Theorem 5.12)](#s3-two-gaussians-with-unequal-variances-b17-chan-theorem-512)
- [S4. The mode of a Gaussian is its mean (B2.3, Bishop 1.9)](#s4-the-mode-of-a-gaussian-is-its-mean-b23-bishop-19)
- [S5. The chain rule through the Gaussian's exponential (B2.6, MML 5.8a)](#s5-the-chain-rule-through-the-gaussians-exponential-b26-mml-58a)
- [S6. Example 6.4 without its sixth person (B3.4, Introduction to Statistics 6.5)](#s6-example-64-without-its-sixth-person-b34-introduction-to-statistics-65)
- [S7. A line fitted by lstsq solves the normal equations (B3.6, Chan Theorem 7.1)](#s7-a-line-fitted-by-lstsq-solves-the-normal-equations-b36-chan-theorem-71)

## S1. The mean and variance of a sum of independent variables (B1.2, Bishop 1.10 and 2.27)

**What and why.** The forward draw adds a noise term to a scaled clean point, and the two are drawn independently. The mean of the sum is the sum of the means, and, because they are independent, the variance of the sum is the sum of the variances. Bishop asks it for two scalars (Exercise 1.10, PDF p. 78) and then for two random vectors, where variance becomes a covariance matrix (Exercise 2.27, PDF p. 151).

**Worked numbers.** Take a clean coordinate $x \sim \mathcal N(0, 0.25)$ scaled by 0.6916, and noise $0.7223\,\varepsilon$ with $\varepsilon \sim \mathcal N(0, 1)$ drawn independently. $\operatorname{Var}[0.6916\,x] = 0.4783 \times 0.25 = 0.1196$ and $\operatorname{Var}[0.7223\,\varepsilon] = 0.5217$, so the sum has variance $0.1196 + 0.5217 = 0.6413$.

**Steps.**
1. Mean, by linearity of expectation: $\mathbb E[x + z] = \mathbb E[x] + \mathbb E[z]$. Here $0 + 0 = 0$.
2. Centre each term: write $u = x - \mathbb E[x]$ and $v = z - \mathbb E[z]$, so $x + z - \mathbb E[x + z] = u + v$.
3. Square and take expectations: $\operatorname{var}[x + z] = \mathbb E[u^2] + 2\,\mathbb E[uv] + \mathbb E[v^2]$.
4. Independence makes the middle term a product of two zero means: $\mathbb E[uv] = \mathbb E[u]\,\mathbb E[v] = 0 \times 0 = 0$. So $\operatorname{var}[x + z] = \operatorname{var}[x] + \operatorname{var}[z]$, which is $0.1196 + 0.5217 = 0.6413$.
5. For vectors, replace the square by the outer product: $\operatorname{cov}[x + z] = \mathbb E[(u + v)(u + v)^\top] = \Sigma_x + \mathbb E[u v^\top] + \mathbb E[v u^\top] + \Sigma_z$.
6. Independence zeroes each cross block, entry by entry, as in step 4, so $\operatorname{cov}[x + z] = \Sigma_x + \Sigma_z$. With $\Sigma_x = 0.1196\,I$ and $\Sigma_z = 0.5217\,I$ in two dimensions this is $0.6413\,I$. Read on one coordinate, step 6 is step 4, which is the agreement with Exercise 1.10 that Exercise 2.27 asks for.

**Goes wrong.** Adding the standard deviations instead of the variances: $0.3458 + 0.7223 = 1.0681$, whose square 1.1408 is 78% above the true 0.6413. The same mistake appears when the two terms are not independent: with $z = x$, $\operatorname{Var}[2x] = 1.0$ where the independent rule gives $0.25 + 0.25 = 0.5$, because the cross term $2\,\mathbb E[uv] = 0.5$ no longer vanishes.

**Check.** One million draws of $0.6916\,x + 0.7223\,\varepsilon$ have sample variance 0.6403, against 0.6413. The second route is [chapter 9, section 9.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#91-independent-gaussian-noises-add-their-variances), steps 2 to 4, which runs the same argument with $a\,e_1 + b\,e_2$.

## S2. A linear map plus independent noise (B1.6, MML 6.12)

**What and why.** Mathematics for Machine Learning's Exercise 6.12 (PDF p. 229) takes $x \sim \mathcal N(\mu_x, \Sigma_x)$, a matrix $A$, a shift $b$ and independent noise $w \sim \mathcal N(0, Q)$, and asks (a) for $p(y \mid x)$ and (b) for the mean and covariance of $p(y)$, where $y = Ax + b + w$. The forward process is this exercise with $A = \sqrt{\bar\alpha_t}\,I$, $b = 0$ and $Q = (1-\bar\alpha_t) I$: part (a) is $q(x_t \mid x_0)$, and part (b) is the whole noisy cloud at step $t$.

**Worked numbers.** At $t = 50$: $A = 0.6916\,I$, $Q = 0.5217\,I$. (a) With $x = x_0 = (0, 0.5)$, $p(y \mid x_0) = \mathcal N((0, 0.3458), 0.5217\,I)$. (b) With a clean cloud $x \sim \mathcal N((0, 0.5), 0.25\,I)$, $\mu_y = (0, 0.3458)$ and $\Sigma_y = 0.4783 \times 0.25\,I + 0.5217\,I = 0.6413\,I$.

**Steps.**
1. Part (a). Fix $x$. Then $Ax + b$ is a constant vector and $y$ is that constant plus $w$.
2. A constant plus a Gaussian is a Gaussian centred at the constant with the same covariance, so $p(y \mid x) = \mathcal N(y \mid Ax + b,\ Q)$. On the numbers: centre $0.6916 \times (0, 0.5) = (0, 0.3458)$, covariance $0.5217\,I$.
3. Part (b), the mean. Linearity of expectation and $\mathbb E[w] = 0$ give $\mu_y = A\mu_x + b$: $0.6916 \times (0, 0.5) + 0 = (0, 0.3458)$.
4. The covariance of $Ax$ alone. With $u = x - \mu_x$, $\operatorname{cov}[Ax] = \mathbb E[Au\,(Au)^\top] = A\,\mathbb E[uu^\top]\,A^\top = A\Sigma_x A^\top$ (Chan, Theorem 5.14, PDF p. 310). On the numbers: $0.6916^2 \times 0.25 = 0.4783 \times 0.25 = 0.1196$ on each coordinate.
5. Add the noise. $x$ and $w$ are independent, so the covariances add as in S1: $\Sigma_y = A\Sigma_x A^\top + Q = 0.1196\,I + 0.5217\,I = 0.6413\,I$.
6. Shape. $y$ is a linear function of the stacked Gaussian vector $(x, w)$, whose parts are independent Gaussians, so $y$ is Gaussian, and $p(y) = \mathcal N(A\mu_x + b,\ A\Sigma_xA^\top + Q)$. The one-dimensional reason linear combinations of independent Gaussians stay Gaussian is section 9.1's generating-function step.

**Goes wrong.** Dropping the $A^\top$ in step 4 scales the clean variance by $\sqrt{\bar\alpha}$ instead of $\bar\alpha$: $0.6916 \times 0.25 + 0.5217 = 0.6946$, against the true 0.6413.

**Check.** One million draws of $y$ have mean $(-0.0013, 0.3463)$ and variances $(0.6416, 0.6404)$, against $(0, 0.3458)$ and 0.6413. Part (a) is Q1's answer, and part (b) with $\Sigma_x = 0$ returns part (a).

## S3. Two Gaussians with unequal variances (B1.7, Chan Theorem 5.12)

**What and why.** Chan's Theorem 5.12 (PDF p. 299) states that independent $X_1 \sim \mathcal N(\mu_1, \sigma_1^2)$ and $X_2 \sim \mathcal N(\mu_2, \sigma_2^2)$ sum to $\mathcal N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$, but the proof that follows writes both densities with one $\sigma^2$ and ends on variance $2\sigma^2$ (PDF p. 300). The forward draw is the unequal case: a scaled clean coordinate plus noise of a different size. This solution proves the general statement through generating functions, the route of [chapter 9, section 9.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#91-independent-gaussian-noises-add-their-variances).

**Worked numbers.** $X_1 = 0.6916 \times x$ for the clean coordinate $x \sim \mathcal N(0.5, 0.25)$, so $X_1 \sim \mathcal N(0.3458, 0.1196)$. $X_2 = 0.7223\,\varepsilon \sim \mathcal N(0, 0.5217)$. The sum is $\mathcal N(0.3458, 0.6413)$.

**Steps.**
1. Write each as a shifted, scaled standard normal: $X_k = \mu_k + \sigma_k e_k$ with $e_1, e_2$ independent $\mathcal N(0, 1)$.
2. The generating function of one standard normal is $\mathbb E[e^{\lambda e}] = e^{\lambda^2/2}$, by completing the square in the exponent (section 9.1, step 5).
3. Pull the constant out: $\mathbb E[e^{\lambda X_k}] = e^{\lambda\mu_k}\,\mathbb E[e^{\lambda\sigma_k e_k}] = e^{\lambda\mu_k + \lambda^2\sigma_k^2/2}$.
4. Independence factors the generating function of the sum: $\mathbb E[e^{\lambda(X_1 + X_2)}] = \mathbb E[e^{\lambda X_1}]\,\mathbb E[e^{\lambda X_2}]$.
5. Multiply and collect the exponents: $e^{\lambda(\mu_1 + \mu_2) + \lambda^2(\sigma_1^2 + \sigma_2^2)/2}$.
6. That is the generating function of $\mathcal N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$, and a generating function that exists near $\lambda = 0$ fixes the distribution. On the numbers: $\mathcal N(0.3458 + 0, 0.1196 + 0.5217) = \mathcal N(0.3458, 0.6413)$.

**Goes wrong.** Reading the book's $2\sigma^2$ as the general answer: with $\sigma^2 = \operatorname{Var}X_1$ it gives 0.2392, and with $\sigma^2 = \operatorname{Var}X_2$ it gives 1.0434; neither is the true 0.6413.

**Check.** At $\lambda = 1$ the product of the two generating functions and the closed form both print 1.9473. One million draws of $X_1 + X_2$ have mean 0.3447 and variance 0.6392, against 0.3458 and 0.6413.

## S4. The mode of a Gaussian is its mean (B2.3, Bishop 1.9)

**What and why.** Bishop's Exercise 1.9 (PDF p. 78) asks to show that the maximum of the Gaussian density is at $\mu$, in one dimension and in $d$. The score answers it in one line: it is zero exactly at $\mu$ and points back toward $\mu$ from everywhere else. A log does not move a maximum, because log is strictly increasing ([PoE chapter 3, section 3.3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/03-moving-into-logs.md#33-the-argmax-survives-because-log-is-strictly-increasing)).

**Worked numbers.** Step 1's Gaussian $\mathcal N(0, 0.5^2)$ has density 0.7979 at 0 and 0.4839 at 0.5. Its score at 0.5 is $-(0.5 - 0)/0.25 = -2$, pointing back toward 0, and its second derivative of $\log p$ is $-1/0.25 = -4$ everywhere. For the forward-draw Gaussian $q(x_{50} \mid x_0)$ the mode is $0.6916\,x_0 = (0, 0.3458)$.

**Steps.**
1. Take logs: $\log p(x) = -(x - \mu)^2/(2\sigma^2) - \tfrac12\log(2\pi\sigma^2)$.
2. Differentiate in $x$; the constant has no slope: $\frac{d}{dx}\log p = -(x - \mu)/\sigma^2$.
3. Set it to zero: $-(x - \mu)/\sigma^2 = 0$ holds only at $x = \mu$.
4. Differentiate again: $-1/\sigma^2 < 0$, so the stationary point is a maximum, and it is the only one. On the numbers, $-4$.
5. In $d$ dimensions the score is $-\Sigma^{-1}(x - \mu)$ ([chapter 6, section 6.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#61-the-gradient-of-a-quadratic-form)). $\Sigma^{-1}$ is invertible, so the score is the zero vector only when $x - \mu = 0$.
6. The matrix of second derivatives is $-\Sigma^{-1}$, which is negative definite because $\Sigma$ is a covariance of full rank, so $\mu$ is the maximum.

**Goes wrong.** Dropping the minus sign in step 2 still puts the stationary point at $\mu$, but the score at 0.5 becomes $+2$, pointing away from the mean, and the second derivative becomes $+4$, which would call the peak a minimum. Reading "score zero" as "peak" is also wrong outside the single Gaussian: a two-bump mixture has zero score at the trough between its bumps ([PoE chapter 5, section 5.3](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/05-predicting-a-score-field.md#53-peak-trough-or-saddle-and-what-the-score-cannot-tell-you)).

**Check.** A central finite difference of $\log p$ with $h = 10^{-6}$ prints $-2.0$ at 0.5 and $0.0$ at 0, and the density is larger at 0 (0.7979) than at 0.5 (0.4839).

## S5. The chain rule through the Gaussian's exponential (B2.6, MML 5.8a)

**What and why.** Mathematics for Machine Learning's Exercise 5.8 (a) (PDF p. 177) asks for $df/dx$ where $f(z) = \exp(-\tfrac12 z)$, $z = y^\top S^{-1} y$ and $y = x - \mu$, with the dimension of every partial. $f$ is the Gaussian density without its constant, so dividing its gradient by $f$ gives the Gaussian score, $-S^{-1}(x - \mu)$. This is the same score Q2 derives, reached through the density instead of the log.

**Worked numbers.** $\mu = 0$, $S = \operatorname{diag}(1, 4)$, $x = (3, 4)$, the point of [chapter 6, section 6.2](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/06-the-gaussian-score.md#62-the-anisotropic-score-follows-the-inverse-covariance). Then $y = (3, 4)$, $S^{-1} = \operatorname{diag}(1, 0.25)$, $z = 9 + 4 = 13$, $f = e^{-6.5} = 0.0015034$, $df/dx = (-0.0045103, -0.0015034)$, and $(df/dx)/f = (-3, -1)$.

**Steps.**
1. Name the three links and their sizes: $f$ is a scalar, $z$ is a scalar, $y$ and $x$ are $D$-vectors.
2. Outer link, $1 \times 1$: $\partial f/\partial z = -\tfrac12 e^{-z/2}$. On the numbers, $-\tfrac12 \times 0.0015034$.
3. Middle link, $1 \times D$: for symmetric $S^{-1}$, $\partial z/\partial y = 2\,y^\top S^{-1}$ (the gradient of a quadratic form, chapter 6, section 6.1). On the numbers, $2\,(3, 1) = (6, 2)$.
4. Inner link, $D \times D$: $\partial y/\partial x = I$, because $y$ moves one for one with $x$.
5. Multiply in order, $1 \times 1$ times $1 \times D$ times $D \times D$, giving $1 \times D$: $df/dx = -\tfrac12 e^{-z/2} \cdot 2\,y^\top S^{-1} \cdot I = -e^{-z/2}(x - \mu)^\top S^{-1}$. On the numbers, $-0.0015034 \times (3, 1) = (-0.0045103, -0.0015034)$.
6. Divide by $f = e^{-z/2}$ and transpose to a column: $\nabla \log f = -S^{-1}(x - \mu) = (-3, -1)$, the score chapter 6.2 prints.

**Goes wrong.** Dropping the 2 from step 3 halves everything: the score comes out $(-1.5, -0.5)$, pointing the right way at half the length. The same slip comes from writing $\partial z/\partial y = y^\top S^{-1}$, forgetting that $y$ appears twice in $y^\top S^{-1} y$.

**Check.** Central finite differences of $f$ at $(3, 4)$ with $h = 10^{-6}$ print $(-0.0045103, -0.0015034)$, equal to step 5 to seven decimals.

## S6. Example 6.4 without its sixth person (B3.4, Introduction to Statistics 6.5)

**What and why.** Introduction to Statistics fits starting salary $S$ against education score $Ed$ for six people by the normal equations (Example 6.4, PDF pp. 130 to 131), and Exercise 6.5 (PDF p. 132) asks to drop the sixth person, re-estimate $\hat\beta_0$ and $\hat\beta_1$, and predict the salary at score 2.8. Refitting after removing one point shows how much a least-squares line leans on each point.

**Worked numbers.** The five people left: $S = 20000, 24500, 23000, 25000, 20000$ dollars and $Ed = 2.8, 3.4, 3.2, 3.8, 3.2$. Then $\sum Ed = 16.4$, $\sum S = 112500$, $\sum S\cdot Ed = 371900$, $\sum Ed^2 = 54.32$, so $S_{xy} = 2900$, $S_{xx} = 0.528$, slope $\hat\beta_1 = 5492.42$ dollars per score point, intercept $\hat\beta_0 = 4484.85$ dollars, and the prediction at 2.8 is 19,863.64 dollars.

**Steps.**
1. Add up the four sums over the five people: 16.4, 112500, 371900 and 54.32.
2. Centre the cross sum: $S_{xy} = \sum S\cdot Ed - (\sum Ed)(\sum S)/n = 371900 - 16.4 \times 112500/5 = 371900 - 369000 = 2900$.
3. Centre the square sum: $S_{xx} = \sum Ed^2 - (\sum Ed)^2/n = 54.32 - 268.96/5 = 54.32 - 53.792 = 0.528$.
4. Slope, from the book's equation 6.17: $\hat\beta_1 = S_{xy}/S_{xx} = 2900/0.528 = 5492.42$.
5. Intercept, from equation 6.16: $\hat\beta_0 = \bar S - \hat\beta_1\,\overline{Ed} = 22500 - 5492.42 \times 3.28 = 4484.85$.
6. Predict: $\hat S(2.8) = 4484.85 + 5492.42 \times 2.8 = 19863.64$. With all six people the same steps give slope 5370.37, intercept 4777.78 and 19,814.81 at 2.8, so removing the sixth person raises the slope by 122 dollars per point and the prediction at 2.8 by 49 dollars.

**Goes wrong.** Fitting without the intercept, slope $= \sum S\cdot Ed / \sum Ed^2 = 371900/54.32 = 6846.47$, predicts 19,170.10 at 2.8: about 694 dollars low, because the line is forced through the origin. A smaller slip sits in the book itself: its intercept 4,779 comes from the rounded slope 5,370 ($22500 - 5370 \times 3.3 = 4779$); the unrounded slope gives 4,777.78.

**Check.** `np.polyfit(Ed, S, 1)` on the five people prints slope 5492.42 and intercept 4484.85, and on all six prints 5370.37 and 4777.78.

## S7. A line fitted by lstsq solves the normal equations (B3.6, Chan Theorem 7.1)

**What and why.** Chan's Theorem 7.1 (PDF p. 414) says the least-squares line satisfies the normal equation $X^\top X\theta = X^\top y$, and the Python listing on PDF p. 416 fits a line with `np.linalg.lstsq` on a design matrix whose columns are $x$ and a column of ones. The exercise is to write `fit_line(x, y)` that way and confirm on numbers that its answer solves the normal equations, using Anton's four points (Example 1, PDF p. 404), whose line is $y = 1.5 + x$.

**Worked numbers.** Points $(0, 1), (1, 3), (2, 4), (3, 4)$. With columns $(x, 1)$: $X^\top X = \begin{pmatrix} 14 & 6 \\ 6 & 4 \end{pmatrix}$, $X^\top y = (23, 12)$, and $\theta = (1, 1.5)$, slope first. Residuals $(-0.5, 0.5, 0.5, -0.5)$, residual sum of squares 1.0.

**Steps.**
1. Stack the design matrix: `X = np.column_stack((x, np.ones(len(x))))`, so column 1 multiplies the slope and column 2 the intercept.
2. Solve: `theta = np.linalg.lstsq(X, y, rcond=None)[0]`, which prints `[1. 1.5]`.
3. Form the normal equations by hand: $\sum x^2 = 14$, $\sum x = 6$, $n = 4$, $\sum xy = 23$, $\sum y = 12$.
4. Substitute: $14 \times 1 + 6 \times 1.5 = 23$ and $6 \times 1 + 4 \times 1.5 = 12$. Both hold.
5. Read the residuals $y - X\theta = (-0.5, 0.5, 0.5, -0.5)$: they sum to 0 (the intercept's equation) and $\sum x_i r_i = 0 + 0.5 + 1 - 1.5 = 0$ (the slope's equation). That is what $X^\top(y - X\theta) = 0$ says.

**Goes wrong.** Leaving out the column of ones fits a line through the origin: slope $23/14 = 1.6429$ and residual sum of squares 4.2143, against 1.0. Reading Chan's $\theta$ in the other order, as (intercept, slope), predicts 5.5 at $x = 3$ where the fitted line gives 4.5.

**Check.** The script prints $X^\top X\theta = (23, 12)$, equal to $X^\top y$, and $X^\top r = (0, 0)$. Anton's printed answer, $y = 1.5 + x$, is the second route.
