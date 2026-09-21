# Is the dropped interaction term really a pointwise-mutual-information gradient, and is that new?

Section 5 of the ICLR submission argues that product-of-experts composition drops a specific
interaction term. This note is a `/pressure-test` run against three claims drafted for that
section, before the section is written: that the term is a covariance under the denoising
posterior, that it equals the gradient of a pointwise-mutual-information ratio, and that it
should be largest early in sampling. The verdict is mixed: one claim holds as a pure identity but
is already published elsewhere, one claim is incomplete in a way that matters, and one claim is
wrong in direction.

## What is in here

One written pressure-test (`interaction-term-as-pmi-gradient.md`, rendered to
`interaction-term-as-pmi-gradient.html` by `build.sh` using the local `katex/` math-rendering
bundle and `tufte.css`, both vendored styling rather than content). No rendered figures: the note
names two image prompts (`figures/rt-interior-peak.png`, `figures/posterior-narrowing-covariance.png`)
that were never generated, so nothing here is shown as a picture.

## The pieces

**Claim 1, the covariance identity.** The independence gap at the noisy state $x_t$ equals a
covariance of the two concept-probability functions under the denoising posterior $p(x_0 \mid
x_t)$. Verdict: correct but incomplete as stated, because it silently assumes the very
clean-image independence the paper argues against; the fix is to state the full two-term
decomposition and frame the covariance as what survives even granting that assumption.

**Claim 2, the load-bearing one.** The residual $r_t$ between the joint-prompt and
product-of-experts noise predictions equals $-\sigma_t$ times the gradient of the log of the
pointwise-mutual-information ratio $p(c_1,c_2\mid x_t) / [p(c_1\mid x_t)\,p(c_2\mid x_t)]$.
Verdict: correct, verified in the note by direct cancellation of the Bayes expansion, and
already published: GCDM (arXiv 2302.14368) derives the same score-space ratio at the noisy state
in its Appendix 0.B.3. Presenting it as this paper's contribution would read as a claim a
reviewer catches.

**Claim 3, the timing claim.** That the correction should be largest early in sampling (high
noise). Verdict: wrong in direction. The note derives that $\|r_t\|$ vanishes at both ends of the
noise schedule and peaks at an interior log-SNR, with a worked counterexample at $t \to T$ where
the covariance is largest but the gradient (and hence $r_t$) still goes to zero.

**Three unstated premises named as risks**, independent of which claim they attach to: that a
literal joint-prompt string stands in for the true conjunction event; that one consistent
distribution is assumed across three separately-run forward passes of an imperfectly trained
network; and that every measurement runs at classifier-free guidance weight 7.5, not the weight
1 the identity is derived for.

**A ranked next-step list**, starting with running the paper's chimera-producing pairs through a
published sampler-only corrector (Du et al.'s MCMC sampler) to check whether the missing-term
story survives a rival explanation that blames the sampler rather than the model.

**Held since 2026-09-05.** Nothing in the plan tree, the paper draft, or the drip thread it names
(`drips/poe-two-concept-factorization/00-INDEX.md`, which does not currently exist under that
path) points back at this note by path. It is kept because it is the only place the section-5
claims about the interaction term have been checked against the literature before the section is
written, and because Claim 2's citation to GCDM changes what section 5 may claim as novel.
