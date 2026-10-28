Source: transcript 9484913d-a247-4bdb-bb44-0f11895d96cc.jsonl, the rung: message 33967dd8-a331-4f5c-8896-0c3e63d18243 at 2026-10-21T03:57:16.465Z

```
forward process → score > the forward process > two steps, then every step
depth 1 · piece 3 of 4 · 2 visited, 1 left here
```

**Why this rung is next**

The [forward process](/home-mscluster/mmolefe/Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md) adds noise in 100 small steps, but the training code jumps straight to step `t`. This rung shows the hundred steps and the one jump give the same thing. Rungs 1 and 2 do all the work.

**One name, to avoid a clash**

The step file calls the clean point $x_0$, and also calls the state after code step 0 $x_0$. In this rung $s_t$ means the state after code step $t$, and $x_0$ means only the clean point.

**One step**

$$s_t=\sqrt{1-\beta_t}\;s_{t-1}+\sqrt{\beta_t}\;\varepsilon_t$$

Read it term by term. $\sqrt{1-\beta_t}\,s_{t-1}$ shrinks everything already present, signal and old noise alike. $\sqrt{\beta_t}\,\varepsilon_t$ adds fresh noise with [variance](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/03-mean-variance-centering.md) $\beta_t$ (rung 1). The draw $\varepsilon_t$ is new at this step, so it is independent of everything before it.

**Two steps, derived**

Start from any state $z$ and take steps $j$ and $j+1$:

$$
\begin{aligned}
s&=\sqrt{1-\beta_{j+1}}\Big[\sqrt{1-\beta_j}\,z+\sqrt{\beta_j}\,\varepsilon_j\Big]+\sqrt{\beta_{j+1}}\,\varepsilon_{j+1} &&\text{substitute step }j\\
&=\underbrace{\sqrt{(1-\beta_{j+1})(1-\beta_j)}}_{\text{signal multiplier}}\,z+\underbrace{\sqrt{1-\beta_{j+1}}\sqrt{\beta_j}\,\varepsilon_j}_{\text{old noise, shrunk}}+\underbrace{\sqrt{\beta_{j+1}}\,\varepsilon_{j+1}}_{\text{fresh noise}} &&\text{multiply out}
\end{aligned}
$$

The two noise pieces are independent, so their variances add (rung 2). Each variance is its multiplier squared (rung 1):

$$
\begin{aligned}
\text{noise variance}&=(1-\beta_{j+1})\,\beta_j+\beta_{j+1}\\
&=\beta_j+\beta_{j+1}-\beta_j\beta_{j+1} &&\text{multiply out}\\
&=1-(1-\beta_j)(1-\beta_{j+1}) &&\text{since }(1-\beta_j)(1-\beta_{j+1})=1-\beta_j-\beta_{j+1}+\beta_j\beta_{j+1}
\end{aligned}
$$

**The noise variance is one minus the signal share.** The signal share is $(1-\beta_j)(1-\beta_{j+1})$, the square of the signal multiplier.

**Every step, by induction**

The claim is that after code step $t$:

$$s_t=\sqrt{\bar\alpha_t}\,x_0+\text{noise of variance }1-\bar\alpha_t,\qquad \bar\alpha_t=\prod_{j=0}^{t}(1-\beta_j)$$

**First case.** At $t=0$ there is one step from the clean point. The signal multiplier is $\sqrt{1-\beta_0}$, so $\bar\alpha_0=1-\beta_0$, and the noise variance is $\beta_0=1-\bar\alpha_0$. ✓

**From $t-1$ to $t$.** Assume it holds at $t-1$, then apply one step:

$$
\begin{aligned}
\text{signal: }&\sqrt{1-\beta_t}\cdot\sqrt{\bar\alpha_{t-1}}=\sqrt{(1-\beta_t)\,\bar\alpha_{t-1}}=\sqrt{\bar\alpha_t}\\
\text{noise: }&(1-\beta_t)(1-\bar\alpha_{t-1})+\beta_t=1-(1-\beta_t)\,\bar\alpha_{t-1}=1-\bar\alpha_t
\end{aligned}
$$

The noise line is the two-step [algebra](/home-mscluster/mmolefe/Obsidian/Mathematics/Algebra/Algebra.md) again, with $\bar\alpha_{t-1}$ in place of $1-\beta_j$.

$$\boxed{s_t=\sqrt{\bar\alpha_t}\,x_0+\text{noise of variance }1-\bar\alpha_t}$$

**Worked: the first two steps of the schedule**

$\beta_0=0.0006313$ and $\beta_1=0.0011169$.

$$\text{noise}=(1-0.0011169)(0.0006313)+0.0011169=0.0006306+0.0011169=0.0017475$$

$$\bar\alpha_1=(1-0.0006313)(1-0.0011169)=0.9982525,\qquad 1-\bar\alpha_1=0.0017475\ \checkmark$$

The signal multiplier is $\sqrt{0.9982525}=0.9991$, so after two steps the point is still almost clean.

**Worked: step 50**

$\bar\alpha_{49}=0.4938$, so the old noise has variance $0.5062$. Step 50 has $\beta_{50}=0.03155$.

$$\text{old noise, shrunk}=0.96845\times0.5062=0.4902,\qquad \text{plus fresh}=0.4902+0.0315=0.5217=1-\bar\alpha_{50}\ \checkmark$$

$$\text{signal}=\sqrt{0.96845\times0.4938}=\sqrt{0.4783}=0.6916\ \checkmark$$

The typical misses are $0.7001$ for the old noise and $0.1776$ for the fresh noise. Combined they give $\sqrt{0.5217}=0.7223$, while adding the misses directly would wrongly give $0.8777$.

**Broken: forget to shrink the old noise**

Add the fresh noise without shrinking the old: $0.5062+0.0315=0.5377$. The total variance becomes $0.4783+0.5377=1.0160$, and it climbs further above 1 with every step. The $\sqrt{1-\beta_t}$ in front of $s_{t-1}$ is what keeps the total at 1.

```
next: the one-shot draw x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε, and its density q(x_t | x_0)
example: steps 98 and 99, where the 0.999 clip fires, checked against ᾱ_99
break: 100 steps in numpy without the shrink; you predict the final variance before it prints
thread --derive: the induction step, back to multiplying out two brackets
also: launch (the derive thread), integrate (when f3ca2449 returns), variance or forward, map, spell-it-out, controls
```
