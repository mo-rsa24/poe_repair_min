# Methods Design — Why each method takes the form it does

The math motivating both methods is in [`01_theory.md`](01_theory.md) §6;
this file covers the concrete design choices that math leaves under-specified.

The two methods are:

- **M2 — embedding synthesis** for channel C1 ([`01_theory.md`](01_theory.md) §6.1).
- **C-PoE — conflict-aware composition** for channel C2 ([`01_theory.md`](01_theory.md) §6.2).
- **M2 + C-PoE — combined**.

---

## 1. M2 — synthesizer training and deployment

### 1.1 Training objective

The synthesizer $f_\phi(e_A, e_B, e_\emptyset) \to \hat e_J$ is trained on
text alone (no images), with $E(c_J)$ as the supervisory target. Loss:

$$
\mathcal L = w_{\text{seq}}\big[w_{\cos}(1 - \cos(\hat e_J^{\text{seq}}, e_J^{\text{seq}})) + w_{\text{mse}}\,\|\hat e_J^{\text{seq}} - e_J^{\text{seq}}\|^2\big]
+ w_{\text{pool}}[\dots\text{same on pooled}\dots]
$$

Training data: caption-pair list of size ~70k assembled from a curated noun
list, oversampled with same-supercategory pairs (cat/dog/lion/tiger…) so the
synthesizer is aligned with the deployment regime. See
[`poe_repair/embeddings/dataset.py`](../poe_repair/embeddings/dataset.py) and
[`poe_repair/embeddings/holdout_pairs.py`](../poe_repair/embeddings/holdout_pairs.py).

### 1.2 Deployment — `replace` mode

At inference, **M2-replace** runs a single guided UNet branch conditioned
on $\hat e_J$ — the PoE sum is dropped entirely. This tests the strong
reading: *is PoE itself the bottleneck, given a faithful joint conditioning?*

Implementation: [`poe_repair/methods/_sampling.py:run_m2_replace`](../poe_repair/methods/_sampling.py).

```
for each timestep t:
    eps_J_raw, eps_uncond = unet([latent, latent], cond=[ê_J, e_∅])
    eps_J = eps_uncond + s * (eps_J_raw - eps_uncond)        # plain CFG
    x0 = tweedie_mean(latents, alpha_bar_t, eps_J)
    latents = ddim_step(x0, eps_uncond)
```

That's it. Two UNet branches per step. Standard CFG with $s = 7.5$. No PoE.
No conflict gate. The mono prompts only enter the picture as inputs to the
synthesizer, which produces a single synthesized joint embedding; the
diffusion process is conditioned only on $\hat e_J$ and $e_\emptyset$.

### 1.3 Synthesizer architecture default

Default is a small residual MLP on the concatenation $[e_A; e_B; e_\emptyset]$
predicting $\hat e_J - e_\emptyset$ as a residual. See
[`poe_repair/embeddings/synthesizer.py:ResidualMLPSynthesizer`](../poe_repair/embeddings/synthesizer.py).

---

## 2. C-PoE — conflict-aware composition

### 2.1 The intervention

From [`01_theory.md`](01_theory.md) §6.2 (Eq. 6.5–6.6):

$$
\varepsilon^{\text{C-PoE}}_t = \varepsilon_\theta(x_t, t, e_\emptyset)
\;+\; \alpha(\theta(x_t)) \cdot \big(u_A(x_t) + u_B(x_t)\big),
$$

with $u_i = \varepsilon_\theta(x_t, t, e_i) - \varepsilon_\theta(x_t, t, e_\emptyset)$,
$\theta(x_t) = \angle(u_A, u_B)$, and $\alpha(\theta) = \max(0, \cos\theta)^\gamma$.

Implementation: [`poe_repair/methods/_sampling.py:run_c_poe`](../poe_repair/methods/_sampling.py).

### 2.2 Why this and not other modulations

- **Recovers PoE in the cooperative regime.** At $\theta = 0$, $\alpha = 1$
  and the sampler is standard PoE. No regression on cooperative pairs.
- **Continuous and differentiable.** No threshold, no jump. Easy to compose
  with DDIM.
- **Single sharpness knob.** $\gamma$ controls how aggressively the conflict
  regime is suppressed. $\gamma = 0$ recovers PoE; $\gamma \to \infty$ snaps
  to a hard gate. **Default $\gamma = 2$.**

### 2.3 Cost

Zero new UNet forwards beyond the three vanilla PoE already runs (the angle
$\theta$ is computed from the existing $\varepsilon_A, \varepsilon_B,
\varepsilon_\emptyset$). The cheapest possible inference-time intervention.

### 2.4 What C-PoE does not do

- It does not estimate $\log R_t(x_t)$. It modulates the PoE pull based on a
  measurable proxy ($\theta$) for where the PoE forward is most wrong.
- It does not change the conditional set: only $e_A, e_B, e_\emptyset$
  appear, no $e_J$.
- It does not produce a new direction in latent space, only a new magnitude.

---

## 3. M2 + C-PoE — combined

The combined sampler runs all four UNet branches per step
$\{A, B, \hat e_J, \emptyset\}$:

```
u_A = eps_A - eps_uncond
u_B = eps_B - eps_uncond
u_J = eps_synth - eps_uncond
alpha = max(0, cos<u_A, u_B>)^gamma
eps_t = eps_uncond + alpha*(u_A + u_B) + lambda_J * u_J
```

M2 changes what the UNet sees in the joint slot; C-PoE changes how the
singleton PoE sum is weighted. They compose multiplicatively when run
together: a richer $\hat e_J$ supplies better joint conditioning, while
C-PoE prevents the singleton sum from cancelling at conflicting $x_t$.

Defaults: $\gamma = 2, \lambda_J = 1$. Implementation:
[`poe_repair/methods/_sampling.py:run_m2_c_poe`](../poe_repair/methods/_sampling.py).

---

## 4. What this design does not include

- **No joint-prompt UNet forwards at sampling.** Both methods preserve the
  marginal-only premise.
- **No mask / box supervision.** Spatial supervision would defeat the
  premise.
- **No model fine-tuning.** Both methods are inference-time; only the M2
  synthesizer is trained, and only on text.
- **No M2-correction mode.** v1 ships M2-replace only. Adding the
  PoE-residual variant later is a one-function addition to `_sampling.py`
  if needed.
- **No AD-PoE attention-overlap heuristic.** Demoted to "not run".
