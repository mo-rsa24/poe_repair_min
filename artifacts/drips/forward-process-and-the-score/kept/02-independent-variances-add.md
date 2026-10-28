Source: transcript 9484913d-a247-4bdb-bb44-0f11895d96cc.jsonl, the rung: message e5e8e602-f41c-4fb6-aaa5-1188ee2a9506 at 2026-10-21T02:31:56.429Z

```
forward process → score > the forward process > independent variances add
depth 1 · piece 2 of 4 · 1 visited, 2 left here
```

**Why this rung is next**

Every forward step adds fresh noise on top of the noise already there. To merge all of it into one term, you need the [variance of a sum](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Sums%20Of%20Independent%20Gaussians.md#the-rule-taught-in-applied-in).

**Independent, in plain words**

Two draws $X$ and $Y$ are independent when knowing one tells you nothing about the other. Each has mean 0 here.

**The rule, derived**

$$
\begin{aligned}
\operatorname{Var}(X+Y)
&=\mathbb E\big[(X+Y)^2\big] &&\text{mean of }X+Y\text{ is }0\\
&=\mathbb E\big[X^2+2XY+Y^2\big] &&\text{multiply out }(a+b)^2\\
&=\mathbb E[X^2]+2\,\mathbb E[XY]+\mathbb E[Y^2] &&\text{an average of a sum is the sum of averages}\\
&=\operatorname{Var}(X)+2\cdot 0+\operatorname{Var}(Y) &&\text{independent: }\mathbb E[XY]=\mathbb E[X]\,\mathbb E[Y]=0\cdot0
\end{aligned}
$$

$$\boxed{\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)\quad\text{for independent }X,Y}$$

**For independent draws the [cross term](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Sums%20Of%20Independent%20Gaussians.md#why-independence-kills-the-cross-term) $2XY$ averages to zero, so the variances simply add.** The [matrix cookbook](/home-mscluster/mmolefe/goal-setting/learning/mathematics/textbooks/linear-algebra/matrix-cookbook.pdf#page=41) states the same rule for [Gaussian](/home-mscluster/mmolefe/Obsidian/Mathematics/Statistics/Distributions/Multivariate%20Normal%20Distribution/Multivariate%20Normal%20Distribution.md#notes-in-this-folder) vectors.

**Worked: two coins**

Take two independent $\pm1$ coins. The four outcomes are equally likely:

$$X+Y\in\{2,\,0,\,0,\,-2\},\qquad (X+Y)^2\in\{4,\,0,\,0,\,4\},\qquad \text{mean}=2=1+1\ \checkmark$$

The cross products $XY$ are $1,-1,-1,1$, and their average is 0.

**Worked: what breaks without [independence](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/plans/01-probability-you-can-compute.md#2-%F0%9F%94%A7-build-the-two-joint-tables-that-decide-independence)**

Let $Y=X$. Now $X+Y=2X$, and rung 1 gives $\operatorname{Var}(2X)=4$, not $2$. The cross term $XY=X^2$ averages to 1 instead of 0, which supplies the missing 2.

**Worked: two noise terms from the schedule**

$$\operatorname{Var}(0.4054\,\varepsilon_a+0.7223\,\varepsilon_b)=0.1644+0.5217=0.6861$$

The typical miss is $\sqrt{0.6861}=0.8283$. Adding the typical misses directly would give $0.4054+0.7223=1.1277$, which is wrong.

```
next: two steps then induction; the noise from steps 0 to t merges into variance 1 − ᾱ_t
expand: nothing below this; it bottoms out at multiplying out (a + b)². `example` instead
example: three noises 0.0251ε, 0.4054ε, 0.7223ε summed, variance and typical miss checked
break: Y = X on 100,000 numpy draws; you predict the variance before it prints
thread --derive: Var(X + Y), back to the definition of an average
also: integrate (when f3ca2449 returns), variance or forward, draw by script, map, spell-it-out, controls
```
