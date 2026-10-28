Source: transcript 9484913d-a247-4bdb-bb44-0f11895d96cc.jsonl, the rung: message 07b1a108-209b-485a-89d9-d784651d1e1b at 2026-10-21T04:24:45.361Z

```
forward process → score > the forward process > the one-shot draw and its density
depth 1 · piece 4 of 4 · 3 visited, 0 left here
```

**Why this rung is next**

Rung 3 left the noise as a stack of independent pieces whose [variance](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/03-mean-variance-centering.md) totals $1-\bar\alpha_t$. Training draws it as one piece. This rung writes it that way and then gives the density $q(x_t\mid x_0)$, which piece 3 will differentiate.

**The stack is one draw**

**A sum of independent [Gaussians](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md#notes-in-this-folder) is itself a Gaussian.** That fact is filed in [Sums Of Independent Gaussians](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Sums%20Of%20Independent%20Gaussians.md). So the whole noise stack is one Gaussian with mean 0 and variance $1-\bar\alpha_t$. Rung 1, run backwards, writes that as one standard normal draw times its [standard deviation](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Variance.md#what-variance-measures):

$$\boxed{x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon}$$

**What it is, given the clean point**

The bar in $q(x_t\mid x_0)$ means [given](/home-mscluster/mmolefe/goal-setting/learning/mathematics/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=287) $x_0$: the clean point is held fixed and only $\varepsilon$ is random. Adding the fixed $\sqrt{\bar\alpha_t}\,x_0$ moves the centre and leaves the spread alone. Each coordinate gets its own independent draw, so in 2D the covariance is $(1-\bar\alpha_t)I$:

$$q(x_t\mid x_0)=\mathcal N\!\big(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)$$

**The density, term by term**

A 2D [Gaussian](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md#notes-in-this-folder) with centre $\mu$ and the same variance $\sigma^2$ on both coordinates has density

$$\mathcal N(x;\mu,\sigma^2 I)=\underbrace{\frac{1}{2\pi\sigma^2}}_{\text{total volume }=1}\;\exp\!\Big(-\underbrace{\frac{|x-\mu|^2}{2\sigma^2}}_{\text{squared distance, in widths}}\Big)$$

$|x-\mu|^2$ is the squared distance from the centre. Dividing by $2\sigma^2$ measures that distance against the width. $\exp(-\cdot)$ makes the height fall as you move away from the centre. $1/(2\pi\sigma^2)$ scales the bump so it holds [total probability 1](/home-mscluster/mmolefe/Obsidian/Mathematics/Calculus/Integration/The%20Front%20Factor%20Of%20A%20Gaussian%20Density.md). With $\mu=\sqrt{\bar\alpha_t}\,x_0$ and $\sigma^2=1-\bar\alpha_t$:

$$\boxed{q(x_t\mid x_0)=\frac{1}{2\pi(1-\bar\alpha_t)}\exp\!\Big(-\frac{|x_t-\sqrt{\bar\alpha_t}\,x_0|^2}{2(1-\bar\alpha_t)}\Big)}$$

**Worked: 1D, clean point 0.5 at step 50**

The centre is $0.6916\times0.5=0.3458$ and the [standard deviation](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Variance.md#what-variance-measures) is $0.7223$. The peak height is $1/(\sqrt{2\pi}\times0.7223)=0.5523$.

**Worked: the toy at step 50**

$x_0=(0,0.5)$ and $\varepsilon=(0.3,-1.2)$. The centre is $(0,\,0.3458)$ and the noise is $0.7223\,\varepsilon=(0.2167,-0.8668)$:

$$x_{50}=(0.2167,\ -0.5210)$$

That matches `p1.example_x_t`. Now evaluate the density there:

$$
\begin{aligned}
|x_{50}-\text{centre}|^2&=|0.7223\,\varepsilon|^2=0.5217\times(0.09+1.44)=0.7983\\
\text{exponent}&=-\frac{0.7983}{2\times0.5217}=-0.765\\
q&=\frac{1}{2\pi\times0.5217}\,e^{-0.765}=0.3050\times0.4653=0.1419
\end{aligned}
$$

**Worked: the same draw at step 25**

$\bar\alpha_{25}=0.8356$, so $x_{25}=(0.1216,\ -0.0295)$ and $|x_{25}-\text{centre}|^2=0.2515$.

$$\text{exponent}=-\frac{0.2515}{2\times0.1644}=-0.765,\qquad q=0.9682\times0.4653=0.4505$$

**At the point it drew, the exponent is always $-|\varepsilon|^2/2=-0.765$, whatever the step.** The displacement is $\sqrt{1-\bar\alpha_t}\,\varepsilon$, so squaring it gives $(1-\bar\alpha_t)|\varepsilon|^2$, and that $1-\bar\alpha_t$ cancels the one in the denominator. Piece 3 differentiates exactly this exponent.

**Broken: drop the root from the centre**

Use $\bar\alpha_t\,x_0=(0,\,0.2391)$ as the centre. Then $x_{50}$ sits at $(0.2167,\,-0.7601)$ from it, which is no longer $0.7223\,\varepsilon$, and the cancellation above fails.

The [forward process](/home-mscluster/mmolefe/Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md) piece is done, 4 rungs covered. Next is the cosine schedule, 4 rungs.

```
next: the cosine schedule, its boundary values C_k computed by hand at k = 0, 1 and 51
example: the toy's density on a 3 by 3 grid around x_50, heights you can sketch
expand: why a sum of independent Gaussians is a Gaussian (1 child, the product of two bell curves)
thread --worked-examples: q(x_t | x_0) at three steps and three points, solutions held back
also: draw by script, launch, integrate (when f3ca2449 returns), variance or forward, map, spell-it-out, controls
```
