# 01: Scaling a draw scales its variance by c²

*Piece 1 of 18 in [the forward process and the score](00-INDEX.md). After: [02: Independent variances add](02-independent-variances-add.md).*

Filed in the vault: [Variance](../../../../../Obsidian/Mathematics/Statistics/Variance.md). Scene: [Derive 2 on Dissolving the Dog](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-2).

## Why this comes first

The forward draw multiplies the noise by $\sqrt{1-\bar\alpha_t}$, not by $(1-\bar\alpha_t)$. That square root comes entirely from the rule in this piece.

## What variance measures

$$\operatorname{Var}(X)=\mathbb E\big[(X-\mu)^2\big]$$

Read it term by term. $X-\mu$ is the miss: how far one draw lands from the mean $\mu$. Squaring it makes left and right misses both count, and positive. $\mathbb E$ averages over all draws. So [variance](../../../../../goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/03-mean-variance-centering.md) is the average squared miss. Its square root, the standard deviation, is the typical miss in the same units as $X$.

## The draw we scale

$\varepsilon$ is a standard normal draw. Its mean is 0 and its variance is 1, so its typical miss is 1.

## The rule, derived

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

> [!aside]
> The third line is ordinary averaging: $(c^2a+c^2b)/2=c^2(a+b)/2$. The [expectation chapter](../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=200) states the same move for a scaled variable.

<a id="eq-scaling"></a>**(1) Scaling a draw**

$$\boxed{\operatorname{Var}(c\,\varepsilon)=c^2,\qquad \text{typical miss}=|c|}$$

**Multiplying a draw by $c$ multiplies its variance by $c^2$.** So if you want a given variance, you multiply by its square root.

## Worked examples

> [!example]
> **A coin first, so the average is visible.** Take a draw that is $+1$ or $-1$, each half the time. Its mean is 0 and its variance is $(1^2+1^2)/2=1$, the same as $\varepsilon$. The rule never used the Gaussian shape.
>
> With $c=0.7223$ the draw becomes $\pm0.7223$. Each square is $0.7223^2=0.5217$, so the average square is $0.5217=c^2$. ✓

> [!example]
> **The noise term at step 50.** The schedule gives $\bar\alpha_{50}=0.4783$. The noise share is $1-0.4783=0.5217$, and the multiplier is $c=\sqrt{0.5217}=0.7223$.
>
> $$\operatorname{Var}(0.7223\,\varepsilon)=0.7223^2=0.5217=1-\bar\alpha_{50}\ ✓$$
>
> Drop the root and multiply by $0.5217$ instead, and the variance becomes $0.5217^2=0.2722$. That is the wrong noise share.

> [!example]
> **The first step, and two coordinates.** At code step 0, one step adds noise share $\beta_0=0.000631$. The multiplier is $\sqrt{0.000631}=0.02512$, and the variance comes back as $0.02512^2=0.000631$ ✓.
>
> In 2D each coordinate gets its own independent $\varepsilon$, scaled by the same $c$. That makes the covariance $c^2 I$. On the toy, $\varepsilon=(0.3,-1.2)$ scales to $0.7223\,\varepsilon=(0.2167,-0.8668)$. That vector is the noise half of the $x_{50}$ rebuilt in piece 4.

## The same rule at steps 25 and 75

> [!example]
> **1D, step 25.** The schedule gives $\bar\alpha_{25}=0.8356$, so the noise share is $1-0.8356=0.1644$.
>
> $$c=\sqrt{0.1644}=0.4054,\qquad \operatorname{Var}(0.4054\,\varepsilon)=0.4054^2=0.1644\ ✓$$

> [!example]
> **1D, step 75, with four values to square by hand.** Take a draw that is $\pm\sqrt{1.5}=\pm1.2247$ or $\pm\sqrt{0.5}=\pm0.7071$, each a quarter of the time. Its mean is 0 and its mean square is $(1.5+0.5)/2=1$. Multiply it by $c=\sqrt{1-0.1335}=0.9309$:
>
> $$\pm1.2247\to\pm1.1401,\quad \pm0.7071\to\pm0.6582$$
>
> $$\text{mean square}=\frac{1.1401^2+0.6582^2}{2}=\frac{1.2999+0.4333}{2}=0.8666\approx0.9309^2=0.8665\ ✓$$

> [!example]
> **1D, the wrong unit at step 25.** Multiplying by the share $0.1644$ instead of its root gives variance $0.1644^2=0.0270$. That is six times too little noise.

> [!example]
> **2D, steps 25 and 75.** $\varepsilon=(0.3,-1.2)$ becomes $0.4054\,\varepsilon=(0.1216,-0.4865)$ at step 25, with covariance $0.1644\,I$. At step 75 it becomes $0.9309\,\varepsilon=(0.2793,-1.1171)$, with covariance $0.8665\,I$. Same direction, longer.

> [!aside]
> The vault writer recomputed the step 75 example with the exact $c=\sqrt{0.8665}$ instead of $0.9309$ rounded first: the toy draw is $(0.2793,-1.1170)$ and the mean square is $0.8665$. The difference is in the last decimal place only.

## The typical miss runs ahead of the variance

```text
 t   typical miss c (█ = 0.05)    variance c² (█ = 0.05)
  0  ▏                     0.0251   ▏                     0.0006
 25  ████████              0.4054   ███                   0.1644
 50  ██████████████        0.7223   ██████████            0.5217
 75  ███████████████████   0.9309   █████████████████     0.8665
 99  ████████████████████  1.0000   ████████████████████  1.0000
```

Read each row left to right. At step 25 the noise is already `0.4054` wide but holds only `0.1644` of the variance. **For $c$ below 1, $c^2$ is smaller than $c$, so the noise looks big long before it owns most of the variance.** The two columns meet only at step 99. The scene's "Dog fades, noise grows" chart draws the same gap as dotted share lines beside the solid multipliers.

## Illustration

<figure class="slot">
<img src="figures/01-scaling-a-draw.png" alt="A standard bell curve and its copy scaled by 0.7223, with their spreads marked">
<figcaption>
A standard bell and its copy scaled by 0.7223: the spread shrinks by 0.7223, the variance by 0.5217.
<span class="slot-note">Waiting on <code>figures/01-scaling-a-draw.png</code>: paste the prompt below into ChatGPT, then save the image here.</span>
</figcaption>
</figure>

**Prompt** (class: maths-visual, save as `figures/01-scaling-a-draw.png`)

```text
Style: warm off-white paper background. One subject per image, drawn as clean geometry in fine ink-like lines. A restrained palette of exactly two accent colors carrying meaning, never decoration. All labels in a clean serif type, sparse, lowercase. No pills, no cards, no flowchart boxes, no UI chrome. Arrows only where the arrows are the subject.

Scene: two bell curves on one horizontal axis, both centred at zero. The first, in the first accent colour, is a standard normal with a thin bracket under it spanning one standard deviation either side, labelled "spread 1". The second, in the second accent colour, is the same curve squeezed horizontally to 0.7223 of the width and taller so its area matches, with its own bracket labelled "spread 0.7223". Between them a small serif note reads "variance 0.5217 = 0.7223²". No grid, no other labels.
```
