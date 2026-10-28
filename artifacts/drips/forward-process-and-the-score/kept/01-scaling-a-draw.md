Source: transcript 9484913d-a247-4bdb-bb44-0f11895d96cc.jsonl, the rung: message 4f7410b9-97fe-4f37-b948-ba514fe53e14 at 2026-10-20T15:41:00.256Z; the example: message e5e8e602-f41c-4fb6-aaa5-1188ee2a9506 at 2026-10-21T02:31:56.429Z; the t = 75 correction: message 7eef2c2b-cad5-4930-90f2-d97a42c5d906 at 2026-10-21T02:35:17.624Z

<!-- the rung: message 4f7410b9-97fe-4f37-b948-ba514fe53e14 -->
```
forward process → score > the forward process > scaling a draw
depth 1 · piece 1 of 4 · 0 visited, 3 left here
```

**Why this rung comes first**

The forward draw multiplies the noise by `√(1 − ᾱ_t)`, not by `(1 − ᾱ_t)`. That square root comes entirely from the rule in this rung.

**What variance measures**

$$\operatorname{Var}(X)=\mathbb E\big[(X-\mu)^2\big]$$

Read it term by term. $X-\mu$ is the miss, meaning how far one draw lands from the mean $\mu$. Squaring the miss makes left and right misses both count as positive. $\mathbb E$ averages over all draws. So [variance](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/03-mean-variance-centering.md) is the average squared miss. Its square root, the standard deviation, is the typical miss in the same units as $X$.

**The draw we scale**

$\varepsilon$ is a standard normal draw. Its mean is 0 and its variance is 1, which means its typical miss is 1.

**The rule, derived**

Let $Y=c\,\varepsilon$ for a fixed number $c$.

$$\mathbb E[Y]=c\cdot\mathbb E[\varepsilon]=c\cdot 0=0$$

The mean stays at 0, so a miss of $Y$ is just $Y$ itself.

$$
\begin{aligned}
\operatorname{Var}(c\varepsilon)
&=\mathbb E\big[(c\varepsilon-0)^2\big] &&\text{definition, with mean }0\\
&=\mathbb E\big[c^2\varepsilon^2\big] &&(ab)^2=a^2b^2\\
&=c^2\,\mathbb E\big[\varepsilon^2\big] &&\text{a constant factor comes out of an average}\\
&=c^2\cdot 1 &&\mathbb E[\varepsilon^2]=\operatorname{Var}(\varepsilon)=1
\end{aligned}
$$

The third line is ordinary averaging: $(c^2a+c^2b)/2=c^2(a+b)/2$. It is the same move the [expectation chapter](/home-mscluster/mmolefe/goal-setting/learning/mathematics/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=200) states for a scaled variable.

$$\boxed{\operatorname{Var}(c\,\varepsilon)=c^2,\qquad \text{typical miss}=|c|}$$

**Multiplying a draw by $c$ multiplies its variance by $c^2$.** So if you want a given variance, you multiply by its square root.

**Worked: a coin first, so you can see the average**

Take a draw that is $+1$ or $-1$, each half the time. Its mean is 0 and its variance is $(1^2+1^2)/2=1$, the same as $\varepsilon$. The rule never used the Gaussian shape.

With $c=0.7223$ the draw becomes $\pm0.7223$. Each square is $0.7223^2=0.5217$, so the average square is $0.5217=c^2$. ✓

**Worked: the noise term at step 50**

The schedule gives $\bar\alpha_{50}=0.4783$. The noise share is $1-0.4783=0.5217$, and the multiplier is $c=\sqrt{0.5217}=0.7223$.

$$\operatorname{Var}(0.7223\,\varepsilon)=0.7223^2=0.5217=1-\bar\alpha_{50}\ ✓$$

Drop the root and multiply by $0.5217$ instead, and the variance becomes $0.5217^2=0.2722$. That is the wrong noise share.

**Worked: the first step, and two coordinates**

At code step 0, one step adds noise share $\beta_0=0.000631$. The multiplier is $\sqrt{0.000631}=0.02512$, and the variance comes back as $0.02512^2=0.000631$ ✓.

In 2D each coordinate gets its own independent $\varepsilon$, scaled by the same $c$. That makes the covariance $c^2 I$. On the toy, $\varepsilon=(0.3,-1.2)$ scales to $0.7223\,\varepsilon=(0.2167,-0.8668)$. That vector is the noise half of the `x_50` you'll rebuild in piece 4.

```
next: independent variances add, the reason two noise steps merge into one
expand: nothing below this; it bottoms out at averaging squares. `example` instead
example: ε scaled at t = 25 and t = 75, variances checked against 0.1644 and 0.8665
run: 100,000 numpy draws × 0.7223, sample variance beside 0.5217 (about 10 s)
thread: why the schedule stores variance shares while the formula needs standard deviations
also: latex, sketch, why, map, skip, spell-it-out, controls
```

<!-- the example: message e5e8e602-f41c-4fb6-aaa5-1188ee2a9506 -->
**Example: rung 1 on the real schedule**

**1D, step 25.** The schedule gives $\bar\alpha_{25}=0.8356$, so the noise share is $1-0.8356=0.1644$.

$$c=\sqrt{0.1644}=0.4054,\qquad \operatorname{Var}(0.4054\,\varepsilon)=0.4054^2=0.1644\ \checkmark$$

**1D, step 75, with four values you can square by hand.** Take a draw that is $\pm\sqrt{1.5}=\pm1.2247$ or $\pm\sqrt{0.5}=\pm0.7071$, each a quarter of the time. Its mean is 0 and its mean square is $(1.5+0.5)/2=1$. Multiply it by $c=\sqrt{1-0.1335}=0.9309$:

$$\pm1.2247\to\pm1.1401,\quad \pm0.7071\to\pm0.6582$$

$$\text{mean square}=\frac{1.1401^2+0.6582^2}{2}=\frac{1.2999+0.4333}{2}=0.8666\approx0.9309^2=0.8665\ \checkmark$$

**1D, the wrong unit at step 25.** Multiplying by the share $0.1644$ instead of its root gives variance $0.1644^2=0.0270$. That is six times too little noise.

**2D, step 25.** $\varepsilon=(0.3,-1.2)$ becomes $0.4054\,\varepsilon=(0.1216,-0.4865)$, with covariance $0.1644\,I$.

**2D, step 75.** The same draw becomes $0.9309\,\varepsilon=(0.2793,-1.1171)$, with covariance $0.8665\,I$. It is the same direction as at step 25, just longer.

**The picture: the typical miss runs ahead of the variance**

```
 t   typical miss c (█ = 0.05)    variance c² (█ = 0.05)
  0  ▏                     0.0251   ▏                     0.0006
 25  ████████              0.4054   ███                   0.1644
 50  ██████████████        0.7223   ██████████            0.5217
 75  ███████████████████   0.9309   █████████████████     0.8665
 99  ████████████████████  1.0000   ████████████████████  1.0000
```

Read each row left to right. At step 25 the noise is already `0.4054` wide but holds only `0.1644` of the variance. **For $c$ below 1, $c^2$ is smaller than $c$, so the noise looks big long before it owns most of the variance.** The two columns only meet at step 99.

<!-- the t = 75 correction: message 7eef2c2b-cad5-4930-90f2-d97a42c5d906 -->
```
forward process → score > the forward process > independent variances add
depth 1 · piece 2 of 4 · 1 visited, 2 left here
```

**A small correction from the vault writer.** My step 75 example rounded `c` to `0.9309` before squaring. With the exact `c`, the toy draw becomes `(0.2793, −1.1170)` instead of `−1.1171`, and the mean square is `0.8665` instead of `0.8666`. The [Variance note](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Variance.md) keeps the exact numbers. Rung 2 was filed in [Sums Of Independent Gaussians](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Sums%20Of%20Independent%20Gaussians.md).

```
next: two steps then induction; the noise from steps 0 to t merges into variance 1 − ᾱ_t
example: three noises 0.0251ε, 0.4054ε, 0.7223ε summed, variance and typical miss checked
break: Y = X on 100,000 numpy draws; you predict the variance before it prints
also: integrate (when f3ca2449 returns), variance or forward, draw by script, map, spell-it-out, controls
```
