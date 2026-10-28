# Quiz 1.2 banked problems: the solutions written here
Standard: QUIZ_PREP_FORMAT v1

⬅️ [Quiz 1.2 preparation](quiz-1-2.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 1](../ladder/01-one-diffusion-expert.md)

These are the full-steps solutions for the banked problems whose book gives no solution. Every other banked problem cites the book's or the journey's own worked solution in [the coverage file's bank table](quiz-1-2.md#banked-problems). The quiz page links here after an attempt and never before it.

**Where the numbers come from.** The schedule values are the `p1.` keys that `python -m rebuild.walk_numbers` prints (`rebuild/out/walk-numbers.json`): $\bar\alpha$ at $t = 0, 25, 50, 75, 99$ is 0.9994, 0.8356, 0.4783, 0.1335, 0.0, and $\sqrt{1-\bar\alpha}$ is 0.0251, 0.4054, 0.7223, 0.9309, 1.0. The toy is $x_0 = (0, 0.5)$, $\varepsilon = (0.3, -1.2)$, $t = 50$, $x_{50} = (0.2167, -0.521)$ (`p1.example_x_t`). The single-step value $\beta_{50} = 0.03155$, and $\beta_0 = 0.000631$, come from the schedule table in [step 4 of the step file](../01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule). Products are rounded to 4 decimals at the end, so a fourth digit can differ by one from a value computed without rounding.

## Contents

- [S1. The toy forward draw (B1)](#s1-the-toy-forward-draw-b1)
- [S2. One step keeps the variance at 1 (B3, UDL 18.1)](#s2-one-step-keeps-the-variance-at-1-b3-udl-181)
- [S3. Three steps in one jump (B5, UDL 18.3)](#s3-three-steps-in-one-jump-b5-udl-183)
- [S4. The ratio of consecutive kept shares (B7, UDL 18.9)](#s4-the-ratio-of-consecutive-kept-shares-b7-udl-189)
- [S5. The score through the covariance (B13)](#s5-the-score-through-the-covariance-b13)
- [S6. The unshrunk corruption against the forward process (B15)](#s6-the-unshrunk-corruption-against-the-forward-process-b15)
- [S7. The best straight-line denoiser (B18, Chan Exercise 5.16)](#s7-the-best-straight-line-denoiser-b18-chan-exercise-516)
- [S8. The conditional mean for a vector target (B21, Bishop 1.25)](#s8-the-conditional-mean-for-a-vector-target-b21-bishop-125)
- [S9. The noise a network learns on Gaussian data (B26)](#s9-the-noise-a-network-learns-on-gaussian-data-b26)
- [S10. The noise a network learns on two clean points (B27)](#s10-the-noise-a-network-learns-on-two-clean-points-b27)
- [S11. One draw's loss, and what the half does (B30)](#s11-one-draws-loss-and-what-the-half-does-b30)
- [S12. A batch mean stands in for the expectation (B31)](#s12-a-batch-mean-stands-in-for-the-expectation-b31)
- [S13. The first step's loss in noise form (B33, UDL 18.11)](#s13-the-first-steps-loss-in-noise-form-b33-udl-1811)
- [S14. The floor under the loss (B34)](#s14-the-floor-under-the-loss-b34)

## S1. The toy forward draw (B1)

**What and why.** The forward process makes a noisy point from a clean one in a single draw: shrink the clean point by $\sqrt{\bar\alpha_t}$ and add noise scaled by $\sqrt{1-\bar\alpha_t}$. Every later step of layer 1 evaluates something at this noisy point.

**Steps.**
1. Read the schedule at $t = 50$: $\bar\alpha_{50} = 0.4783$, $\sqrt{1-\bar\alpha_{50}} = 0.7223$.
2. Take the square root of the kept share: $\sqrt{0.4783} = 0.6916$.
3. Shrink the clean point: $0.6916 \times (0, 0.5) = (0, 0.3458)$.
4. Scale the noise: $0.7223 \times (0.3, -1.2) = (0.2167, -0.8668)$.
5. Add: $(0 + 0.2167,\ 0.3458 - 0.8668) = (0.2167, -0.5210)$.

**Goes wrong.** Using the kept share 0.4783 instead of its root gives $(0.2167, 0.2392 - 0.8668) = (0.2167, -0.6276)$, off by 0.1066 in the second coordinate.

**Check.** `p1.example_x_t` prints `[0.2167, -0.521]`.

## S2. One step keeps the variance at 1 (B3, UDL 18.1)

**What and why.** Each forward step shrinks the previous point and adds fresh noise. The two scales are chosen so a point with variance 1 keeps variance 1, which is why the noisy cloud ends as $\mathcal N(0, I)$ rather than blowing up.

**Steps.**
1. Write the step: $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t$, with $\varepsilon_t$ independent of $x_{t-1}$.
2. Scaling multiplies a variance by the square of the scale: $\mathrm{Var}[\sqrt{1-\beta_t}\,x_{t-1}] = (1-\beta_t)\,I$ and $\mathrm{Var}[\sqrt{\beta_t}\,\varepsilon_t] = \beta_t I$.
3. Independent terms add their variances: $\mathrm{Var}[x_t] = (1-\beta_t) I + \beta_t I = I$.
4. On numbers, at $t = 50$: $(1 - 0.03155) + 0.03155 = 0.96845 + 0.03155 = 1$.

**Goes wrong.** Adding the standard deviations instead: $\sqrt{0.96845} + \sqrt{0.03155} = 0.9841 + 0.1776 = 1.1617$, whose square 1.3496 would grow the variance by about a third at every step.

**Check.** The kept-plus-added identity holds for any $\beta$: $(1-\beta) + \beta = 1$. The second route is section 9.1 of the [forward-process chapter](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#91-independent-gaussian-noises-add-their-variances) with $a = \sqrt{1-\beta}$, $b = \sqrt\beta$, $a^2 + b^2 = 1$.

## S3. Three steps in one jump (B5, UDL 18.3)

**What and why.** Three small steps collapse into one jump with kept share $(1-\beta_3)(1-\beta_2)(1-\beta_1)$, because the independent noises add their variances. This is the induction step of the one-shot form, done once more by hand.

**Steps.** Take the chapter's toy shares $1-\beta = 0.9, 0.8, 0.95$.
1. After two steps (UDL eq. 18.6): $z_2 = \sqrt{0.72}\,x + \sqrt{0.28}\,\varepsilon$.
2. Step three: $z_3 = \sqrt{0.95}\,z_2 + \sqrt{0.05}\,\varepsilon_3 = \sqrt{0.95 \times 0.72}\,x + \sqrt{0.95}\sqrt{0.28}\,\varepsilon + \sqrt{0.05}\,\varepsilon_3$.
3. Signal share: $0.95 \times 0.72 = 0.684$.
4. Noise variance: $0.95 \times 0.28 + 0.05 = 0.266 + 0.05 = 0.316$.
5. Collapse the two independent noises into one: $z_3 = \sqrt{0.684}\,x + \sqrt{0.316}\,\varepsilon'$, and $0.316 = 1 - 0.684$, which is eq. 18.44.

**Goes wrong.** Forgetting to shrink the old noise (adding 0.28 + 0.05 = 0.33) gives a noise variance 0.014 too large, and signal plus noise shares sum to 1.014.

**Check.** Worked example 9.2a of the [forward-process chapter](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#92-reading-the-image-ladder-and-the-one-jump-form) carries the same recursion to 0.316 and on to 0.5212.

## S4. The ratio of consecutive kept shares (B7, UDL 18.9)

**What and why.** The kept share after $t$ steps is a running product, so dividing two neighbours leaves only the newest factor. The reverse step uses this to turn $\bar\alpha$ values back into one-step shares.

**Steps.**
1. Write both products: $\bar\alpha_t = \prod_{s \le t}(1-\beta_s)$ and $\bar\alpha_{t-1} = \prod_{s \le t-1}(1-\beta_s)$.
2. Divide: every factor up to $t-1$ cancels, leaving $\bar\alpha_t / \bar\alpha_{t-1} = 1 - \beta_t$.
3. Take square roots: $\sqrt{\bar\alpha_t / \bar\alpha_{t-1}} = \sqrt{1-\beta_t}$.
4. On numbers at $t = 50$: $1 - \beta_{50} = 0.96845$, so $\bar\alpha_{49} = 0.4783 / 0.96845 = 0.4939$, and $\sqrt{0.96845} = 0.9841$.

**Goes wrong.** Reading the ratio as the cumulative share itself: $\sqrt{0.4783} = 0.6916$ instead of 0.9841, which would treat one step as wiping out half the signal.

**Check.** The step file says one step at $t = 50$ "keeps 0.9685" while the product is 0.4783 ([step 4](../01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule)); $0.4939 \times 0.96845 = 0.4783$.

## S5. The score through the covariance (B13)

**What and why.** For a Gaussian with covariance $\Sigma$ the score is $-\Sigma^{-1}(x-\mu)$ (Little Book, PDF p. 115). The forward process has $\Sigma = (1-\bar\alpha_t) I$, so the inverse is a division by $1-\bar\alpha_t$, and this route must agree with $-\varepsilon/\sqrt{1-\bar\alpha_t}$.

**Steps.**
1. Centre: $\mu = \sqrt{\bar\alpha_{50}}\,x_0 = (0, 0.3458)$.
2. Displacement: $x_{50} - \mu = (0.2167, -0.521 - 0.3458) = (0.2167, -0.8668)$.
3. $\Sigma^{-1} = I / 0.5217$, since $1 - 0.4783 = 0.5217$.
4. Score: $-(0.2167, -0.8668)/0.5217 = (-0.4154, 1.6615)$.

**Goes wrong.** Dividing the noise itself by the variance, $-\varepsilon/0.5217 = (-0.575, 2.300)$, is too long by the factor $1/0.7223 = 1.384$.

**Check.** The second route, $-\varepsilon/0.7223 = (-0.4153, 1.6613)$, equals `p1.example_score`; the last digit of step 4 differs only because the displacement was rounded before dividing.

## S6. The unshrunk corruption against the forward process (B15)

**What and why.** The Little Book (PDF p. 124) corrupts with $\tilde x = x + \varepsilon$, $\varepsilon \sim \mathcal N(0, \sigma^2 I)$, and gets $\nabla\log p(\tilde x\mid x) = (x - \tilde x)/\sigma^2$. The forward process also shrinks the clean point. The two scores have the same shape: displacement from the centre over the variance.

**Steps.**
1. Book's case: centre $x$, variance $\sigma^2$, score $-(\tilde x - x)/\sigma^2$. With $\tilde x - x = \sigma z$, $z \sim \mathcal N(0, I)$, this is $-z/\sigma$.
2. Forward process: centre $\sqrt{\bar\alpha_t}\,x_0$, variance $1-\bar\alpha_t$, so the same steps give $-\varepsilon/\sqrt{1-\bar\alpha_t}$.
3. Match: $\sigma \leftrightarrow \sqrt{1-\bar\alpha_t}$ and centre $x \leftrightarrow \sqrt{\bar\alpha_t}\,x_0$.
4. On the toy: $\sigma = 0.7223$ gives $-(0.3, -1.2)/0.7223 = (-0.4153, 1.6613)$.

**Goes wrong.** Using the book's centre unshrunk, $x_0 = (0, 0.5)$, gives displacement $(0.2167, -1.021)$ and score $(-0.4154, 1.9571)$, wrong in the second coordinate by 0.2958.

**Check.** `p1.example_score` prints `[-0.4153, 1.6613]`.

## S7. The best straight-line denoiser (B18, Chan Exercise 5.16)

**What and why.** With $Y = X + N$, the best guess of $X$ of the form $aY$ is the denoiser restricted to straight lines. For Gaussian $X$ the best line is also the conditional mean, and it matches the slope of 13.2's denoising-score-matching example.

**Steps.** Take $\sigma_X^2 = 1$ and $\sigma_N^2 = 0.25$ (13.2's $\sigma = 0.5$).
1. $\mathrm{Cov}(X, Y) = \mathrm{Var}X = 1$ and $\mathrm{Var}Y = 1 + 0.25 = 1.25$, so $\rho = 1/\sqrt{1 \times 1.25} = 0.8944$.
2. Expand $\mathbb E[(X - aY)^2] = 1 - 2a + 1.25a^2$.
3. Set the derivative to zero: $-2 + 2.5a = 0$, so $a = 0.8$.
4. The error at $a = 0.8$: $1 - 1.6 + 0.8 = 0.2$.
5. In general $a = \sigma_X^2/(\sigma_X^2 + \sigma_N^2)$ and the error is $\sigma_X^2\sigma_N^2/(\sigma_X^2+\sigma_N^2) = \sigma_X^2/(1+\eta)$ with $\eta = \sigma_X^2/\sigma_N^2 = 4$: $1/5 = 0.2$.

**Goes wrong.** Taking $a = 1$ (trust the reading) gives error $1 - 2 + 1.25 = 0.25$, the full noise variance.

**Check.** Second route through the score: $Y \sim \mathcal N(0, 1.25)$ has score $-Y/1.25 = -0.8Y$ (13.2's slope), and Tweedie's $Y + 0.25 \times (-0.8Y) = 0.8Y$ gives the same $a$.

## S8. The conditional mean for a vector target (B21, Bishop 1.25)

**What and why.** The denoising loss sums squared errors over coordinates. A squared length splits into one squared error per coordinate, so the best vector guess is the vector of per-coordinate conditional means.

**Steps.**
1. $\lVert y(x) - t\rVert^2 = \sum_k (y_k(x) - t_k)^2$, so $\mathbb E\lVert y - t\rVert^2 = \sum_k \mathbb E(y_k - t_k)^2$.
2. Each term depends only on $y_k$, so each is minimised on its own by $y_k(x) = \mathbb E[t_k\mid x]$ (Bishop eq. 1.89, PDF p. 65).
3. Hence $y(x) = \mathbb E[t\mid x]$, and the leftover is $\sum_k \mathbb E[\mathrm{Var}(t_k\mid x)]$, eq. 1.90 per coordinate.
4. On numbers: two independent fair dice, observing each one's parity. Per die the conditional mean is 3 on odd and 4 on even, with floor $8/3$ ([the die example](../../../../../../goal-setting/learning/mathematics/statistics/conditional-expectation/plans/05-the-conditional-mean-is-the-best-guess.md)). The vector floor is $8/3 + 8/3 = 16/3 = 5.3333$.

**Goes wrong.** Guessing the constant $(3.5, 3.5)$ costs $35/12$ per die, $35/6 = 5.8333$ in all, 0.5 above the floor.

**Check.** Each coordinate's total, $8/3 + 1/4 = 35/12$, is the die's variance, so $2 \times 35/12 = 35/6$ by the direct route as well.

## S9. The noise a network learns on Gaussian data (B26)

**What and why.** If the clean data is $\mathcal N(0, I)$, the noisy marginal stays $\mathcal N(0, I)$ at every step, since $\bar\alpha_t + (1-\bar\alpha_t) = 1$. The marginal score is then $-x_t$, so the trained network outputs $\sqrt{1-\bar\alpha_t}\,x_t$, not the noise that was drawn.

**Steps.**
1. $q_t = \mathcal N(0, \bar\alpha_t I + (1-\bar\alpha_t) I) = \mathcal N(0, I)$, so $\nabla\log q_t(x_t) = -x_t$.
2. Marginal identity: $\mathbb E[\varepsilon\mid x_t] = -0.7223 \times (-x_t) = 0.7223\,x_t$ at $t = 50$.
3. Second route, jointly Gaussian (Chan Theorem 8.3, PDF p. 545): $\mathrm{Cov}(\varepsilon, x_t) = 0.7223\,I$, $\mathrm{Var}\,x_t = I$, so $\mathbb E[\varepsilon\mid x_t] = 0.7223\,x_t$.
4. At the toy point: $0.7223 \times (0.2167, -0.521) = (0.1565, -0.3763)$.

**Goes wrong.** Answering with the sampled $\varepsilon = (0.3, -1.2)$: that draw is one of many that reach $x_{50}$, and a network that sees only $x_{50}$ cannot recover it.

**Check.** The leftover variance per coordinate is $1 - 0.7223^2 = 1 - 0.5217 = 0.4783 = \bar\alpha_{50}$, which S14 uses.

## S10. The noise a network learns on two clean points (B27)

**What and why.** With two clean points $u_1 = (0, 0.5)$ and $u_2 = (0, -0.5)$, equally likely, the toy noisy point $x_{50}$ could have come from either. The sampled noise belongs to $u_1$, but the network averages over both by their posterior weights.

**Steps** (numbers from `verify_q12.out` in the fill stage, reproduced by the chapter 13 section the fill adds):
1. Shrunk centres: $0.6916\,u_i = (0, 0.3458)$ and $(0, -0.3458)$.
2. Displacements: $(0.2167, -0.8668)$ and $(0.2167, -0.1752)$; squared lengths 0.7983 and 0.0777.
3. Kernels $q(x_{50}\mid u_i) = e^{-|d|^2/(2 \times 0.5217)}/(2\pi \times 0.5217)$: 0.1419 and 0.2832.
4. Bayes: prior times kernel, 0.0710 and 0.1416, sum $q_t(x_{50}) = 0.2126$, posterior 0.3339 and 0.6661.
5. Conditional scores $-d_i/0.5217$: $(-0.4154, 1.6615)$ and $(-0.4154, 0.3358)$.
6. Posterior average: marginal score $(-0.4154, 0.7784)$.
7. $\mathbb E[\varepsilon\mid x_{50}] = -0.7223 \times (-0.4154, 0.7784) = (0.3, -0.5623)$.

**Goes wrong.** Equal weights give $(-0.4154, 0.9987)$ and noise $(0.3, -0.7213)$; the sampled noise $(0.3, -1.2)$ ignores that $u_2$ is twice as plausible.

**Check.** The finite-difference gradient of $\log q_t$ at $x_{50}$ is $(-0.4154, 0.7784)$, and the noise each clean point implies, $(0.3, -1.2001)$ and $(0.3, -0.2426)$, averages with weights 0.3339 and 0.6661 to $(0.3, -0.5623)$.

## S11. One draw's loss, and what the half does (B30)

**What and why.** The loss for one draw is half the squared length of the miss. With two coordinates, half the sum is the mean over coordinates.

**Steps.** Use the untrained rule $\hat\varepsilon = 2x_t$ (`p1.init_eps_at_(0.3,-0.1)` prints $(0.6, -0.2)$, twice its input).
1. $\hat\varepsilon = 2 \times (0.2167, -0.521) = (0.4334, -1.042)$.
2. Miss: $\hat\varepsilon - \varepsilon = (0.1334, 0.158)$.
3. Squares: 0.0178 and 0.0250.
4. Half the sum: $(0.0178 + 0.0250)/2 = 0.0214$.

**Goes wrong.** Dropping the half reports 0.0428, double, with the same best predictor.

**Check.** The mean of the two squares is the same number by its definition: $(0.0178 + 0.0250)/2 = 0.0214$.

## S12. A batch mean stands in for the expectation (B31)

**What and why.** The loss is an expectation over clean point, step and noise. Training estimates it by the mean of per-draw losses over a batch (1000 draws in step 9; three here).

**Steps.** Rule $\hat\varepsilon = 2x_t$.
1. Draw 1: $u_1$, $t = 50$, $\varepsilon = (0.3, -1.2)$: loss 0.0214 (S11).
2. Draw 2: $u_2 = (0, -0.5)$, $t = 25$, $\varepsilon = (-0.5, 0.4)$: $x_t = 0.9141\,u_2 + 0.4054\,\varepsilon = (-0.2027, -0.2949)$; miss $(0.0946, -0.9898)$; loss 0.4943.
3. Draw 3: $u_1$, $t = 75$, $\varepsilon = (1.0, 0.2)$: $x_t = 0.3654\,u_1 + 0.9309\,\varepsilon = (0.9309, 0.3689)$; miss $(0.8618, 0.5377)$; loss 0.5159.
4. Batch mean: $(0.0214 + 0.4943 + 0.5159)/3 = 0.3439$.

**Goes wrong.** Reporting draw 1 alone (0.0214) says the untrained rule is nearly perfect; the batch says otherwise.

**Check.** On 200,000 draws of `bar(0.2)` the same rule's loss is 0.79 (`p1.init_loss_on_bar`), the expectation this mean estimates on that dataset.

## S13. The first step's loss in noise form (B33, UDL 18.11)

**What and why.** UDL rewrites the first step's reconstruction error in terms of the noise predictor; the factor in front is a ratio of noise to signal at step 1.

**Steps.** At the first step $\bar\alpha_1 = 1 - \beta_1$, so $\sqrt{1-\bar\alpha_1} = \sqrt{\beta_1}$.
1. From eq. 18.31: $x = z_1/\sqrt{1-\beta_1} - \sqrt{\beta_1}\,\varepsilon/\sqrt{1-\beta_1}$.
2. From eq. 18.35: $f_1 = z_1/\sqrt{1-\beta_1} - \beta_1 g_1/(\sqrt{\beta_1}\sqrt{1-\beta_1})$.
3. Subtract: $x - f_1 = \frac{\beta_1}{\sqrt{1-\bar\alpha_1}\sqrt{1-\beta_1}}(g_1 - \varepsilon)$, since $\beta_1/\sqrt{\beta_1} = \sqrt{\beta_1}$. This is eq. 18.37.
4. On numbers, with the code's first step $\beta_0 = 0.000631$: the factor is $\sqrt{0.000631/0.999369} = 0.0251$.

**Goes wrong.** Keeping $\sqrt{\bar\alpha_1}$ where $\sqrt{1-\beta_1}$ belongs is harmless only here (they are equal at step 1); at $t = 50$ the two differ, 0.6916 against 0.9841 (S4).

**Check.** $0.0251$ equals $\sqrt{1-\bar\alpha_0}/\sqrt{\bar\alpha_0} = 0.0251/0.9997$, the noise-to-signal ratio at the first noisy state.

## S14. The floor under the loss (B34)

**What and why.** Even the best predictor misses, because many noise draws reach the same noisy point. Its loss is the average conditional variance of the noise (Bishop eq. 1.90, PDF p. 66; Chan, PDF pp. 541 to 542), so training stops above zero.

**Steps.**
1. Split the loss: $\mathbb E\lVert\hat\varepsilon - \varepsilon\rVert^2 = \mathbb E\lVert\hat\varepsilon - \mathbb E[\varepsilon\mid x_t]\rVert^2 + \mathbb E\lVert\mathbb E[\varepsilon\mid x_t] - \varepsilon\rVert^2$; the second term does not depend on the network.
2. For $\mathcal N(0, I)$ data (S9) the leftover per coordinate is $1 - (1-\bar\alpha_t) = \bar\alpha_t$; with the half and two coordinates the floor is $\bar\alpha_t$.
3. At $t = 0, 25, 50, 75, 99$: 0.9994, 0.8356, 0.4783, 0.1335, 0.0.
4. For the two clean points at $x_{50}$ (S10): the first coordinate's implied noise is 0.3 either way (variance 0); the second is $-1.2001$ or $-0.2426$ with weights 0.3339 and 0.6661, variance $0.3339 \times 0.6661 \times 0.9575^2 = 0.2039$; floor at this point $0.2039/2 = 0.1019$.

**Goes wrong.** Expecting the loss to reach 0. The trained models in `rebuild/out/phase1` end near 0.2 (step 8 of the step file) because nearly clean steps leave the noise almost unrecoverable: at $t = 0$ the floor for Gaussian data is 0.9994.

**Check.** S9's leftover $1 - 0.7223^2 = 0.4783$ gives the $t = 50$ row by a second route.
