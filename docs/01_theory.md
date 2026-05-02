# Theory — Why PoE fails, what cannot be recovered, what can

This file is the mathematical core. It derives — from generative modelling
and probability theory alone — what PoE composition omits, says precisely
what part of that omission is non-identifiable from singleton image-
conditional information, and identifies the two channels through which the
rest is reachable. The two inference-time methods (M2, C-PoE) are derived as
specific instances of those channels.

The notation follows Liu et al. (2022, *Compositional Visual Generation
with Composable Diffusion Models*) and the standard score-based diffusion
conventions (Song et al. 2021; Ho et al. 2020).

## TL;DR

For two captions $c_1, c_2$, Bayes gives an exact factorisation: the joint
score equals the PoE sum *plus* a correction proportional to
$\nabla_{x_t} \log R_t(x_t)$, where $R_t$ is the conditional co-occurrence
ratio of $c_1$ and $c_2$ given $x_t$. The data-side of $R_t$ is
non-identifiable from singleton image marginals (Sklar 1959; Hoeffding 1940).
Two information channels remain open: the *text encoder* trained on real
joint captions, and the *UNet* trained on real joint images. We test one
inference-time intervention per channel — **M2** (synthesise $\hat e_J$
offline-trained with $e_J$ as supervision; never call SDXL on $e_J$ at
sampling) and **C-PoE** (modulate the PoE pull by $\max(0, \cos \theta)^\gamma$
where $\theta$ is the angle between the two singleton classifier directions).
Neither claims to recover $R_t$. Both keep $e_J$ out of sampling.

---

## 1. Setup

We work in latent space $\mathcal X \subseteq \mathbb R^n$ with caption space
$\mathcal C$. The training distribution $p(x, c)$ is fixed; both the text
encoder and the UNet were fitted on it.

**Diffusion preliminaries.** A noise schedule $\{\alpha_t, \sigma_t\}_{t \in [0, T]}$
defines the forward process

$$
x_t \;=\; \alpha_t\, x_0 \;+\; \sigma_t\, \eta, \qquad \eta \sim \mathcal N(0, I).
$$

The text encoder $E : \mathcal C \to \mathbb R^d$ is deterministic. The
noise predictor $\varepsilon_\theta(x_t, t, e)$ takes $e = E(c)$ and
approximates $\mathbb E[\eta \mid x_t, t, c]$. Tweedie's identity gives the
score–noise relation

$$
\nabla_{x_t} \log p_t(x_t \mid c) \;=\; -\sigma_t^{-1}\, \varepsilon^*(x_t, t, c),
\qquad
\varepsilon^*(x_t, t, c) \;:=\; \mathbb E[\eta \mid x_t, t, c]. \tag{1.1}
$$

**Compositional setup.** Given two captions $c_1, c_2$, we want to sample
from $p(x \mid c_1, c_2)$. Write $c_J$ for the joint caption "$c_1$ and
$c_2$" and let $e_A = E(c_1)$, $e_B = E(c_2)$, $e_\emptyset = E(\emptyset)$,
$e_J = E(c_J)$.

**Product-of-Experts.** Composing two conditions as a product of
conditional densities yields the score-additive sampler

$$
\varepsilon^{\text{PoE}}_t(x_t)
\;:=\;
\varepsilon_\theta(x_t, t, e_A) + \varepsilon_\theta(x_t, t, e_B) - \varepsilon_\theta(x_t, t, e_\emptyset). \tag{1.2}
$$

Three UNet forwards on singletons. No joint conditioning. This is the "AND"
composition in Liu et al.'s vocabulary.

---

## 2. The exact gap (Bayes, no model assumptions)

Bayes gives

$$
p(x \mid c_1, c_2) \;=\; \frac{p(c_1, c_2 \mid x)\, p(x)}{p(c_1, c_2)}.
$$

Multiplying numerator and denominator by $p(c_1 \mid x)\, p(c_2 \mid x)$ and
substituting $p(c_i \mid x) = p(x \mid c_i)\, p(c_i) / p(x)$:

$$
p(x \mid c_1, c_2)
\;=\;
\underbrace{\frac{p(x \mid c_1)\, p(x \mid c_2)}{p(x)}}_{\text{PoE form}}
\;\cdot\;
\underbrace{\frac{p(c_1, c_2 \mid x)}{p(c_1 \mid x)\, p(c_2 \mid x)}}_{=:\, R(x)}
\;\cdot\;
\underbrace{\frac{p(c_1)\, p(c_2)}{p(c_1, c_2)}}_{\text{const in } x}. \tag{2.1}
$$

The third factor is constant in $x$ and folds into the normaliser. The
second factor

$$
\boxed{\; R(x) \;=\; \frac{p(c_1, c_2 \mid x)}{p(c_1 \mid x)\, p(c_2 \mid x)} \;=\; \exp\!\big(\,\mathrm{PMI}(c_1; c_2 \mid x)\,\big) \;} \tag{2.2}
$$

is the **interaction term** — the conditional pointwise mutual information of
$c_1$ and $c_2$ given $x$, in exponential form. PoE is exact iff $R(x) \equiv 1$,
i.e. iff the two captions are conditionally independent given $x$. No
modelling assumption was made; (2.1) is an algebraic identity.

In score form, taking $\nabla_{x_t} \log$ of the diffused version of (2.1)
and using (1.1):

$$
\boxed{\;
\varepsilon^*(x_t, t, c_1, c_2)
\;=\;
\varepsilon^{\text{PoE},*}_t(x_t)
\;-\; \sigma_t\, \nabla_{x_t} \log R_t(x_t),
\;} \tag{2.3}
$$

where $R_t(x_t) := \mathbb E[R(x_0) \mid x_t]$ is the diffused interaction
factor. Define the **interaction gradient**

$$
g^{\mathrm{IT}}_t(x_t) \;:=\; -\sigma_t\, \nabla_{x_t} \log R_t(x_t). \tag{2.4}
$$

Plain language: at the level of true scores, PoE is right except for one
term — a noise-space gradient of the conditional PMI. Everything that
follows is about this term.

---

## 3. Identifiability — what cannot be done

The interaction term $R(x)$ is a property of the joint conditional
$p(c_1, c_2 \mid x)$. Knowing the singleton conditionals $p(c_i \mid x)$ is
not enough to pin it down:

> **Proposition (non-identifiability of the joint from singleton marginals).**
> There exist distributions $p_1, p_2$ on $(x, c_1, c_2)$ with
> $p_1(x \mid c_i) = p_2(x \mid c_i)$ for $i = 1, 2$ and all $x$, but
> $p_1(x \mid c_1, c_2) \neq p_2(x \mid c_1, c_2)$.

This is the standard non-uniqueness of joint distributions given marginals
(Sklar 1959; Hoeffding 1940; Nelsen 2006). A constructive witness: any
non-trivial copula on the pair of conditional indicators
$\mathbf 1\{C_i = c_i\} \mid x$ that preserves the conditional marginals
yields a different joint with a different $R(x)$.

**Operational consequence.** No function of $\{p_t(x_t \mid c_1)\}$ and
$\{p_t(x_t \mid c_2)\}$ alone can reconstruct $R_t(x_t)$. Therefore
$g^{\mathrm{IT}}_t$ is **non-identifiable from singleton marginals alone**.
We do not claim to recover it. Anything this project recovers must come
through a channel *not closed* by this proposition.

---

## 4. The two open channels

The proposition speaks about *image-conditional* information at inference.
It is silent about two other sources of joint information present in the
system as deployed:

**(C1) Text encoder pre-training.** $E$ was fitted on $p(x, c)$. Joint
captions of the form "$c_1$ and $c_2$" appear in training with real-world
frequencies, so $E$ has absorbed compositional structure into its outputs —
including the joint embedding $e_J$ itself. The proposition does not cover
this: $E$ is a fixed function on captions, not a sampler conditioned on
images.

**(C2) UNet pre-training.** $\varepsilon_\theta$ was also fitted on $p(x, c)$.
Its response to *any* embedding — singleton, joint, or anything in between —
is a learned function of the data. Internal activations and output
directions under singleton conditioning carry training-time traces that are
not the same as singleton image-conditional information.

A method is **legal under the marginal-only constraint** if at sampling it
uses only:

- $x_t, t$ at the current step,
- the singleton embeddings $e_A, e_B, e_\emptyset$ (text-only — these may be
  precomputed),
- finitely many UNet forwards $\varepsilon_\theta(x_t, t, e)$ for
  $e \in \{e_A, e_B, e_\emptyset\}$ and small perturbations,
- pre-trained auxiliary models that do not require $e_J$ at sampling,
- *offline-trained* components whose training-time supervision uses $e_J$.

It is **not legal** if it requires $\varepsilon_\theta(\cdot, \cdot, e_J)$
during sampling. We never call the UNet on $e_J$ at inference.

---

## 5. What the network actually shows us

The network can observe a noise-space residual

$$
\Delta^\epsilon_{12,t}(x_t)
\;:=\; \varepsilon_\theta(x_t, t, e_J) \;-\; \varepsilon^{\text{PoE}}_t(x_t). \tag{5.1}
$$

This is the model-side analogue of $g^{\mathrm{IT}}_t$. To connect it to the
identity (2.3):

$$
\Delta^\epsilon_{12,t}(x_t) =
\underbrace{\big[\varepsilon_\theta(x_t, t, e_J) - \varepsilon^*(x_t, t, c_1, c_2)\big]}_{\text{joint approximation error}}
\;+\; g^{\mathrm{IT}}_t(x_t)
\;+\;
\underbrace{\big[\varepsilon^{\text{PoE},*}_t - \varepsilon^{\text{PoE}}_t\big]}_{\text{singleton approximation error}}. \tag{5.2}
$$

If $\varepsilon_\theta$ approximates $\varepsilon^*$ well on the captions in
its training support — which singletons $c_1, c_2$ and the joint $c_J$ all
are — then the approximation errors are small and $\Delta^\epsilon_{12,t}
\approx g^{\mathrm{IT}}_t$. We do not measure this in v1; it is the
mechanism justification for why the methods below have a non-trivial target
to hit.

---

## 6. Two inference-time interventions

### 6.1 M2 — embedding synthesis (channel C1)

The text encoder $E$ is fixed. Define

$$
d_J \;:=\; e_J - e_\emptyset, \qquad
\mathcal S \;:=\; \mathrm{span}\{\, e_A - e_\emptyset,\; e_B - e_\emptyset \,\}, \qquad
\rho \;:=\; \frac{\|d_J - \Pi_{\mathcal S}(d_J)\|}{\|d_J\|}, \tag{6.1}
$$

where $\Pi_{\mathcal S}$ is orthogonal projection onto $\mathcal S$. The
quantity $\rho \in [0, 1]$ is the fraction of $d_J$ that lives outside the
affine span of the singleton directions. **Empirical claim (H1):** $\rho > 0$
on collision pairs (cat × dog) in at least one encoder mode (pooled,
sequence-mean, sequence-Frobenius).

If H1 holds, then no affine combination of $\{e_A, e_B, e_\emptyset\}$
equals $e_J$. The only way to feed the UNet anything close to the joint
conditioning at inference, without ever calling
$\varepsilon_\theta(\cdot, \cdot, e_J)$ at sampling, is to learn an
approximation $\hat e_J$ from singletons offline:

$$
\hat e_J \;=\; f_\phi(e_A, e_B, e_\emptyset),
\qquad
\phi \;\in\; \arg\min_\phi\; \mathbb E_{(c_1, c_2) \sim \mathcal D_{\text{text}}}\; \mathcal L\big( f_\phi(E(c_1), E(c_2), e_\emptyset),\; E(c_J) \big). \tag{6.2}
$$

Training is on text alone — $\mathcal D_{\text{text}}$ is a corpus of
caption pairs and $E(c_J)$ is the supervisory target. $E(c_J)$ is used
during training, never at sampling. The simplest loss is
$\|\hat e_J - e_J\|_2^2$ in embedding space (we use a cosine + MSE combination
on both sequence and pooled outputs).

**M2-replace** (the v1 deployment): replace the joint conditioning at
inference with $\hat e_J$ entirely — single CFG branch on $(\hat e_J, \emptyset)$.
This tests the strong reading: *is PoE itself the bottleneck on collision
pairs, given a faithful joint conditioning?*

### 6.2 C-PoE — conflict-aware composition (channel C2)

We do not have access to $g^{\mathrm{IT}}_t$ from singletons alone. We do
have access to the UNet's outputs under singleton conditioning, and those
outputs have an interesting structure.

**Implicit-classifier identity** (Ho & Salimans 2022). From Bayes,

$$
\nabla_{x_t} \log p_t(c \mid x_t)
\;=\; -\sigma_t^{-1} \big( \varepsilon^*(x_t, t, c) - \varepsilon^*(x_t, t) \big). \tag{6.3}
$$

Define the singleton **classifier directions**

$$
u_i(x_t) \;:=\; \varepsilon_\theta(x_t, t, e_i) - \varepsilon_\theta(x_t, t, e_\emptyset),
\qquad i \in \{A, B\}. \tag{6.4}
$$

By (6.3) and the score–noise relation,
$-u_i(x_t) \propto \nabla_{x_t} \log p_t(c_i \mid x_t)$ up to model error.
The PoE step (1.2) is then

$$
\varepsilon^{\text{PoE}}_t \;=\; \varepsilon_\theta(x_t, t, e_\emptyset) + u_A(x_t) + u_B(x_t).
$$

PoE adds the two classifier-direction nudges with equal weight on top of
the unconditional. That is the "AND" composition.

**The structure of the sum.** Decompose
$\langle u_A, u_B \rangle = \|u_A\|\,\|u_B\|\, \cos \theta(x_t)$. Three regimes:

| Regime | $\theta$ | What it means | PoE behaviour |
|---|---|---|---|
| Cooperative | $0 \leq \theta < 90°$ | Nudges share a half-space | $\|u_A + u_B\| \geq \max(\|u_A\|, \|u_B\|)$; adding helps. |
| Orthogonal | $\theta \approx 90°$ | Independent directions | $\|u_A + u_B\|^2 = \|u_A\|^2 + \|u_B\|^2$; adding is neutral. |
| Conflicting | $\theta > 90°$ | Pushing $\log p(c_1 \mid x_t)$ up requires moving $x_t$ in a direction that pushes $\log p(c_2 \mid x_t)$ down | $\|u_A + u_B\| < \max(\|u_A\|, \|u_B\|)$; adding cancels. |

In the conflicting regime, PoE produces a destructively-cancelled, weak
pull. The key fact is geometric: a vector $v$ that strictly improves both
$\log p(c_1 \mid x_t)$ and $\log p(c_2 \mid x_t)$ to first order exists if
and only if the two classifier directions span an open half-space, which
holds iff $\theta < 90°$. At $\theta \geq 90°$ the conditions are *locally
Pareto-incompatible at $x_t$*.

**The intervention.** Replace (1.2) with

$$
\varepsilon^{\text{C-PoE}}_t(x_t) \;=\; \varepsilon_\theta(x_t, t, e_\emptyset) \;+\; \alpha(\theta(x_t)) \cdot \big(u_A(x_t) + u_B(x_t)\big), \tag{6.5}
$$

with

$$
\alpha(\theta) \;=\; \max\!\big(0,\, \cos \theta\big)^\gamma. \tag{6.6}
$$

At $\gamma = 0$ this is plain PoE; for $\gamma > 0$ it suppresses the
destructive cancellation and waits for $x_t$ to drift into a region where
both classifiers agree before pushing. Default $\gamma = 2$.

Plain language: when the two conditions disagree about which way to nudge
$x_t$, ease off the gas and let denoising find a state where they agree.
When they agree, push as usual.

**This is not a recovery of $R_t$.** It is a *conflict-damping* method
whose link to the interaction gradient is empirical (does it improve cat ×
dog?), not derivable. The under-conditioning failure mode — both nudges
genuinely needed to push, $\alpha$ suppresses them, output ends up vague —
is named in [`03_caveats.md`](03_caveats.md).

### 6.3 M2 + C-PoE — combined

The two methods act on different substrates: M2 changes what the UNet
*sees* (the joint conditioning slot), C-PoE changes how the *singleton*
PoE sum is *weighted*. Combined per step:

$$
\varepsilon^{\text{M2+C-PoE}}_t \;=\; \varepsilon_\theta(x_t, t, e_\emptyset) + \alpha(\theta) \cdot (u_A + u_B) + \lambda_J \cdot u_J,
$$

with $u_J = \varepsilon_\theta(x_t, t, \hat e_J) - \varepsilon_\theta(x_t, t, e_\emptyset)$.

At $\lambda_J = 0$ this degenerates to C-PoE; at $\gamma = 0$ it becomes a
PoE-residual-corrected method. We use $\gamma = 2, \lambda_J = 1$.

---

## 7. What this section did not say

- It did not claim $\Delta^\epsilon_{12,t} = g^{\mathrm{IT}}_t$. (5.2)'s
  approximation errors are real.
- It did not claim the angle $\theta$ tracks the magnitude of
  $g^{\mathrm{IT}}_t$. The cooperative-but-large-interaction case (e.g. "cat"
  and "Persian cat") would be a counterexample.
- It did not claim M2 + C-PoE exhausts the recoverable signal. Other
  channels exist (e.g. CLIP-image classifier on Tweedie posterior); we
  chose two because each has a clean derivation.
- It did not claim C-PoE recovers $R_t(x_t)$. C-PoE is a conflict-damping
  method.
- It did not claim collision pairs in general are solved. v1's evidence is
  N=1 in the collision regime; see [`03_caveats.md`](03_caveats.md).
