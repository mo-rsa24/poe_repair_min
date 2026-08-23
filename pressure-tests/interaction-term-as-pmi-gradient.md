# Pressure test: the dropped term, in density form and in score form

A `/pressure-test` run on three claims drafted for section 5 of the ICLR submission
([`paper/iclr/iclr2027_conference.tex`](../paper/iclr/iclr2027_conference.tex)), plus the
novelty question behind them.

It answers, ahead of time, the question piece 5 of the
[Product-of-Experts factorization thread](../drips/poe-two-concept-factorization/00-INDEX.md)
was going to ask: what exactly is the term dropped when the two concepts are not conditionally
independent, written both as a density gap and as a score correction.

Verdicts, per claim: Claim 1 correct with a caveat that changes the statement, Claim 2
correct but already published, Claim 3 wrong in direction.

> [!aside]
> Notation follows the thread: $x$ the clean image, $c_1$ and $c_2$ the two concepts,
> and now $x_t$ the noisy state at step $t$, $\sigma_t$ its noise scale,
> $\varepsilon_\theta$ the noise-prediction network. $r_t$ is the per-step residual between
> the joint-prompt prediction and the product-of-experts prediction.

## The three claims, distilled

**Claim 1** says the independence assumption Liu et al. state over the clean image does not
survive transport to the noisy state, and the leftover is exactly a covariance under the
denoising posterior:

<a id="eq-claimed-cov"></a>**(1) The covariance identity, as claimed**

$$p(c_1,c_2 \mid x_t) - p(c_1 \mid x_t)\,p(c_2 \mid x_t)
= \operatorname{Cov}_{x_0 \sim p(x_0 \mid x_t)}\big[\, p(c_1 \mid x_0),\; p(c_2 \mid x_0) \,\big]$$

**Claim 2** says the residual between joint-prompt and product-of-experts noise prediction is
a gradient of pointwise mutual information:

<a id="eq-pmi-gradient"></a>**(2) The interaction term in score form**

$$r_t = -\sigma_t \,\nabla_{x_t} \log
\left[ \frac{p(c_1,c_2 \mid x_t)}{p(c_1 \mid x_t)\, p(c_2 \mid x_t)} \right]$$

**Claim 3** says the covariance in [(1)](#eq-claimed-cov) shrinks as the posterior
$p(x_0 \mid x_t)$ narrows, so $r_t$ vanishes at $t \to 0$ and the correction matters most
early in sampling, at high noise.

## The load-bearing claim

**Load-bearing: Claim 2.** If $r_t$ is not the PMI gradient, the paper's central sentence
("the failure is exactly a missing interaction term") loses its formal content and the LoRA
becomes an unmotivated residual-fitter. Claim 1 is a supporting lemma about *where* the term
comes from; Claim 3 is a testable consequence.

### Unstated premise A: the joint prompt is not the conjunction event

Fatal if unexamined. The math is about conditioning on $\{c_1 \wedge c_2\}$. The measurement
conditions SDXL on the string `"a cat and a dog"`. Those are different conditioning variables.
Nothing guarantees

$$p(x_t \mid \texttt{"a cat and a dog"}) = p(x_t \mid c_1, c_2)$$

This is the gap a reviewer will find first, and it is not addressed by anything in claims 1
to 3.

### Unstated premise B: one consistent distribution across all three conditionings

Claim 2 is a Bayes identity over true densities. The measured $r_t$ is a difference of three
separate forward passes of an imperfectly-trained network. So

$$r_t^{\text{measured}} = \underbrace{\text{PMI gradient}}_{\text{what (2) describes}}
+ \underbrace{\text{cross-conditioning inconsistency of the network}}_{\text{unmeasured}}$$

and no experiment in the current design separates them.

### Unstated premise C: the product-of-experts prediction is the $w=1$ one

The identity holds for $\varepsilon_1 + \varepsilon_2 - \varepsilon_0$. Every run is at
$w = 7.5$. The secondary question is not a footnote, it is a precondition on Claim 2.

## Where this sits in the literature

This is the part that changes what can be claimed.

**Claim 2 is already published, as an equation, at the noisy state.** GCDM, in *Enhanced
Controllability of Diffusion Models via Feature Disentanglement and Realism-Enhanced Sampling
Methods* (arXiv 2302.14368, ECCV 2024), derives $\nabla_{x_t} \log p(x_t, z_c, z_s)$
explicitly **without** assuming conditional independence, and their equations 28 to 29 factor
it as the product-of-experts score times exactly the ratio

$$\frac{p(z_c, z_s \mid x_t)}{p(z_c \mid x_t)\, p(z_s \mid x_t)}$$

Their stated motivation is the same: two conditions that both carry object information are not
independent, and composable diffusion fails on them. They do not use the phrase "mutual
information" and they do not derive Claim 1's covariance, but the score-space identity is
theirs.

> [!aside]
> Confidence: high. This came from the paper's own appendix text (their Appendix 0.B.3), not
> from memory.

**The PMI reading of the ratio is also in the compositional diffusion literature.** CompLift
(Yu & Gao, ICML 2025, arXiv 2505.13740) builds compositional generation around the lift score
$\log p(x \mid c) / p(x)$, estimated from the denoising-error difference

$$\operatorname{lift}(x \mid c) \approx
\mathbb{E}_{t,\varepsilon}\Big[\; \lVert \varepsilon - \varepsilon_\theta(x_t, \varnothing) \rVert^2
- \lVert \varepsilon - \varepsilon_\theta(x_t, c) \rVert^2 \;\Big]$$

and cites Kong et al. (*Interpretable Diffusion via Information Decomposition*, arXiv
2310.07972, ICLR 2024) for reading lift scores as pointwise mutual information. Kong et al.
write exact expressions for mutual and conditional mutual information in terms of the
denoising model, including pointwise estimates.

**Two caveats in your favour:** CompLift's lift is between the *image and one concept*, not
between the *two concepts given the state*, and it is used as a post-hoc
rejection-and-resample criterion, not a gradient added into the sampler. So the PMI machinery
is established, but not aimed at your object.

**A rival explanation for the failure mode is published and it contradicts the framing.** Du
et al., *Reduce, Reuse, Recycle* (arXiv 2302.11552, ICML 2023), conclude that **the sampler,
not the model, is responsible** for compositional generation failure: reverse diffusion does
not sample the product distribution because the score of the diffused product is not the sum
of the diffused scores, and their fix is MCMC-corrected sampling with an energy
parameterization. Skreta et al.'s Feynman-Kac Correctors (arXiv 2503.02819) and SuperDiff
(arXiv 2412.17762) continue this line, correcting the sampler for products, tempering, and
logical AND from continuity-equation and sequential-Monte-Carlo arguments rather than by
learning a term.

**Nobody found writing Claim 1's covariance form.** GCDM gestures at the clean-versus-noisy
question through a manifold argument in a figure, not an identity. That specific decomposition
looks like yours.

## What is likely right

Claim 2 is **correct**, and cleanly so. It needs **no** independence assumption at all, at
either $x_0$ or $x_t$. It is a pure identity.

> [!example]
> **Verifying [(2)](#eq-pmi-gradient) by cancellation.** Substitute Bayes,
> $\log p(x_t \mid c) = \log p(c \mid x_t) + \log p(x_t) - \log p(c)$, into each of the four
> score terms in $\varepsilon_{\text{joint}} - (\varepsilon_1 + \varepsilon_2 - \varepsilon_0)$:
>
> $$\log p(x_t \mid c_1,c_2) - \log p(x_t \mid c_1) - \log p(x_t \mid c_2) + \log p(x_t)$$
>
> The $\log p(x_t)$ contributions carry coefficients $+1, -1, -1, +1$ and cancel exactly. The
> constants $\log p(c)$ vanish under $\nabla_{x_t}$. What remains is
>
> $$\log p(c_1,c_2 \mid x_t) - \log p(c_1 \mid x_t) - \log p(c_2 \mid x_t) + \text{const}$$
>
> which is the log of the ratio in [(2)](#eq-pmi-gradient).

Being right and being yours are different questions, and here they separate.

The conditional-independence step in Claim 1, $c \perp x_t \mid x_0$, is safe. It holds by
construction: the forward kernel $q(x_t \mid x_0)$ does not depend on $c$, so the graphical
model is $c \to x_0 \to x_t$ and the tower property applies. This is the same identity
underneath diffusion classifiers and Kong et al.'s estimator. It smuggles nothing as
probability. It does smuggle something as a claim about SDXL, per unstated premise B.

## What is likely wrong

### Claim 1 is incomplete, and the missing piece is the one the paper is about

Apply the tower property to both sides and add and subtract
$\mathbb{E}[p(c_1 \mid x_0)\,p(c_2 \mid x_0)]$:

<a id="eq-two-term"></a>**(3) The exact decomposition, both terms**

$$
p(c_1,c_2 \mid x_t) - p(c_1 \mid x_t)\,p(c_2 \mid x_t)
= \underbrace{\mathbb{E}_{x_0 \mid x_t}\Big[\, p(c_1,c_2 \mid x_0) - p(c_1 \mid x_0)\,p(c_2 \mid x_0) \,\Big]}_{\text{clean-image dependence, averaged}}
+ \underbrace{\operatorname{Cov}_{x_0 \mid x_t}\Big[\, p(c_1 \mid x_0),\; p(c_2 \mid x_0) \,\Big]}_{\text{your term}}
$$

The claimed identity [(1)](#eq-claimed-cov) is the second term alone. It is the whole story
**only if the first term is zero**, which is precisely Liu et al.'s clean-image assumption, the
one the paper exists to attack. As written, Claim 1 assumes the thing it denies.

The fix is not to weaken it. State [(3)](#eq-two-term) and make the rhetorical move explicit:
*even granting Liu et al. their assumption at the clean image, dependence reappears at the
noisy state through the covariance.* That is a stronger argument than the one you have, and it
is honest.

### Claim 3 is wrong in direction, and the reasoning has two separate faults

**The first fault.** The covariance measures the size of the numerator gap, but $r_t$ depends
on the **gradient of the log ratio**, which is a different object. At $t \to 0$ the gradient
tends to

$$\nabla \log \left[ \frac{p(c_1,c_2 \mid x_0)}{p(c_1 \mid x_0)\, p(c_2 \mid x_0)} \right]$$

which is finite and generically nonzero. $r_t$ still vanishes, but only because of the
$\sigma_t$ prefactor. Right answer, wrong mechanism.

**The second fault is the one that bites.** Run the same analysis at the other end.

> [!example]
> **The counterexample at $t \to T$.** As $t \to T$, $x_t$ is pure noise, so
> $p(c \mid x_t) \to p(c)$ for **every** conditioning. The log ratio tends to the constant
>
> $$\log \frac{p(c_1,c_2)}{p(c_1)\,p(c_2)}$$
>
> and its gradient tends to zero. The covariance is *largest* here, since the posterior is the
> full data distribution, while $r_t$ goes to zero. That is the direct counterexample to the
> reasoning.
>
> Scaling it out: the posterior mean moves with the state at rate $\alpha_t / \sigma_t^2$, so
>
> $$\sigma_t \cdot \nabla \;\sim\; \frac{\alpha_t}{\sigma_t} = \sqrt{\mathrm{SNR}} \;\to\; 0$$

So $\lVert r_t \rVert$ vanishes at **both** ends and peaks at an interior log-SNR. Not "early
in sampling."

<figure class="slot">
<img src="figures/rt-interior-peak.png" alt="The residual norm against log-SNR, vanishing at both ends with an interior peak">
<figcaption>
The corrected prediction: the residual norm vanishes at both ends of the noise schedule and peaks at an interior log-SNR.
<span class="slot-note">Waiting on <code>figures/rt-interior-peak.png</code>: generate from the prompt below (class: maths-visual) and save it here.</span>
</figcaption>
</figure>

**Prompt** (class: maths-visual, save as `figures/rt-interior-peak.png`)

```text
A dark background. A single smooth curve sweeping left to right across the
frame, rising from zero at the far left, reaching one broad rounded maximum
near the middle, and falling back to zero at the far right. The curve is
drawn in one restrained accent colour, thin and clean. At the far left and
far right, where the curve meets zero, two short faint tick marks in a second
accent colour mark the two vanishing limits. Beneath the curve, very faint,
two contrasting dashed guide curves: one decaying monotonically from left to
right, clearly not matching the solid curve, shown only to be visibly
different in shape. No axes boxes, no gridlines, no legend. A small
serif-set label at the left tick reading sigma-t to zero and at the right
tick reading square-root-SNR to zero. Clean serif mathematical type. No
cards, no pills, no flowchart boxes.
```

### A measurement confound follows immediately

In $\varepsilon$ units, $r_t \to 0$ at low $t$ for pure $\sigma_t$ scaling reasons, whatever
the composition story. An observed "the correction is bigger early" curve is therefore *not*
evidence for the mechanism, because a model with no interaction structure at all predicts the
same decay. Normalize before claiming anything from it: divide by
$\lVert \varepsilon_{\text{joint}} - \varepsilon_{\text{uncond}} \rVert$ at the same step, or
convert to $x_0$-space, and plot against log-SNR rather than step index.

<figure class="slot">
<img src="figures/posterior-narrowing-covariance.png" alt="The denoising posterior narrowing as t decreases, with two concept-probability functions over it">
<figcaption>
Where the covariance term comes from: two concept-probability functions read over a denoising posterior that narrows as the noise falls.
<span class="slot-note">Waiting on <code>figures/posterior-narrowing-covariance.png</code>: generate from the prompt below (class: maths-visual) and save it here.</span>
</figcaption>
</figure>

**Prompt** (class: maths-visual, save as `figures/posterior-narrowing-covariance.png`)

```text
A dark background. Three wide bell-shaped density curves stacked from back to
front in receding perspective, each narrower and taller than the one behind
it, all drawn in one restrained accent colour, representing one distribution
contracting toward a point. Across the widest back curve only, two smooth
scalar functions are laid over its support in a second accent colour, one
rising and one falling, their overlap region lightly filled to suggest a
covariance between them. On the narrowest front curve the same two functions
have collapsed to a single point with no spread and no fill. No axes, no
gridlines, no legend. One small serif-set label near the back curve reading
p(x0 given xt). Clean serif mathematical type. No cards, no pills, no
flowchart boxes.
```

### On the secondary question: yes, defensible, and stronger than put

Classifier-free guidance at weight $w$ targets $p(x)\,p(c \mid x)^w$, a tempered distribution,
not $p(x \mid c)$. There is no probabilistic derivation licensing $w > 1$ even for a single
condition. Composed at $w = 7.5$ per expert, the sampler targets

$$p(x) \prod_i p(c_i \mid x_t)^{w_i}$$

which is neither the joint nor the product the derivation describes. Do not present this as
your observation: Feynman-Kac Correctors formalizes exactly the "annealed, geometric-averaged,
or product distributions" gap and states that score mixing heuristics do not approximate the
intermediate distributions. Cite it and move on in two sentences.

### The positioning threat, larger than any of the three claims

Du et al. 2023 say the failure mode is caused by the sampler and is fixable without learning
anything. If an annealed MCMC or SMC corrector removes chimeras at fixed $w$, the missing-term
story is not just unmotivated, it is refuted. This is a confound in the experiment design, not
only a related-work paragraph.

## Verdict

**Promising but needs rework**, with the rework confined to section 5's claims. The empirical
contribution (measure $r_t$, fit it with a low-rank adapter that never sees the joint prompt,
show transfer to unseen pairs) is untouched by everything above and remains the paper's real
contribution.

On the direct question, novel enough to present as a contribution:

**Claim 2: no. Present as a restatement and cite GCDM.** Their equations 28 to 29 are
[(2)](#eq-pmi-gradient) in score space at the noisy state. Converting to $\varepsilon$-space
via $-\sigma_t$ and naming the ratio "pointwise mutual information" is presentation, not
contribution. Claiming it will be caught, and it is the kind of catch that costs a reviewer's
trust in the rest.

**Claim 1: yes, as a lemma, in the corrected two-term form [(3)](#eq-two-term).** The
covariance decomposition appears to be yours, and it does work GCDM does not: it says *why*
the dependence appears at $x_t$ and what controls its size. Frame it as "even under Liu et
al.'s assumption, this term survives." One proposition, not a section.

**Claim 3: no, it is wrong. Replace it with the corrected prediction, which is worth more.**
"$\lVert r_t \rVert$ peaks at an interior log-SNR and vanishes at both ends" is sharp,
falsifiable, non-obvious, and the log-SNR curve to test it against already exists. A
monotone-decaying prediction is neither surprising nor separable from $\sigma_t$ scaling; an
interior peak at a predicted location is both.

## What to do next

**First, the cheap experiment that decides whether the paper survives.** Run the
chimera-producing pairs through an annealed MCMC or SMC corrector at fixed $w$ (Du et al.'s
released code at `github.com/yilundu/reduce_reuse_recycle` is the direct route). If chimeras
persist, the published rival explanation is ruled out and the missing-term story stands. If
they vanish, better to know now, before section 5 is written around it.

**Second, close the conjunction gap (premise A).** State plainly in the paper that $r_t$ is
measured with a joint prompt string standing in for the conjunction event, and give one
measurement bounding the substitution: compare $r_t$ from `"a cat and a dog"` against $r_t$
from a different surface form of the same conjunction, at the same seeds and steps. If the two
differ by an amount comparable to $\lVert r_t \rVert$ itself, the identity is not measuring
what the math describes, and better to find that yourself.

**Third, re-plot before re-claiming.** $\lVert r_t \rVert$ normalized by
$\lVert \varepsilon_{\text{joint}} - \varepsilon_{\text{uncond}} \rVert$, against log-SNR, not
step index. The corrected Claim 3 predicts an interior maximum. Record the predicted location
before looking.

**Fourth, rewrite section 5 in this order:** state Liu et al.'s assumption at $x_0$; cite GCDM
for the exact ratio at $x_t$ and for the observation that it does not decompose; contribute the
two-term covariance decomposition [(3)](#eq-two-term) as your proposition; derive the
interior-peak prediction; cite Feynman-Kac Correctors for the guidance-weight mismatch in two
sentences. That ordering makes the maths section a short, correct, well-cited setup for the
empirical work rather than a novelty claim that will not hold.

## Sources

- [Enhanced Controllability of Diffusion Models via Feature Disentanglement (GCDM), arXiv 2302.14368](https://arxiv.org/html/2302.14368v5)
- [Improving Compositional Generation with Diffusion Models Using Lift Scores, arXiv 2505.13740](https://arxiv.org/html/2505.13740v1)
- [CompLift project page](https://chenningyu.com/complift/)
- [Interpretable Diffusion via Information Decomposition, arXiv 2310.07972](https://arxiv.org/abs/2310.07972)
- [Reduce, Reuse, Recycle, arXiv 2302.11552](https://arxiv.org/abs/2302.11552)
- [Feynman-Kac Correctors in Diffusion, arXiv 2503.02819](https://arxiv.org/pdf/2503.02819)
- [SuperDiff, arXiv 2412.17762](https://arxiv.org/abs/2412.17762)
- [Product of Experts for Visual Generation, arXiv 2506.08894](https://arxiv.org/html/2506.08894v1)
