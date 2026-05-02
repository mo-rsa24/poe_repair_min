# PoE Repair — Overview

This is a minimal, single-purpose project. It tests one claim with two pairs
at one seed:

> *Vanilla Product-of-Experts (PoE) composition fails on `("a cat", "a dog")`
> at seed 42. M2 (synthesised joint embedding ê_J) and C-PoE (conflict-angle
> damping) — both inference-time, marginal-only — produce a coherent two-subject
> scene from the same x_T. On the cooperative pair `("a butterfly", "a flower
> meadow")`, where PoE already works, the methods must not destroy it.*

The math behind why the methods take the form they do is in
[`01_theory.md`](01_theory.md). This is *not* a benchmark paper.

## 1. The thesis in one paragraph

For two captions $c_1, c_2$, the joint score factors exactly as

$$
\nabla_{x_t} \log p_t(x_t \mid c_1, c_2)
= \nabla_{x_t} \log \frac{p_t(x_t \mid c_1)\, p_t(x_t \mid c_2)}{p_t(x_t)}
+ \nabla_{x_t} \log R_t(x_t),
\qquad
R_t(x_t) = \frac{p_t(c_1, c_2 \mid x_t)}{p_t(c_1 \mid x_t)\, p_t(c_2 \mid x_t)}.
$$

PoE composition uses only the first term. The second is the **interaction
term** $g^{\mathrm{IT}}_t = -\sigma_t \nabla \log R_t$. On same-supercategory
collisions (cat × dog), $R_t \neq 1$ and PoE's omission is visible. The
*data-side* of $R_t$ is provably non-recoverable from singleton image-conditional
information alone — the M1 identifiability bound ([`01_theory.md`](01_theory.md) §3).

Two channels remain open:

- **C1 (text encoder).** $E$ was pre-trained on captions including real
  joint phrases like "a cat and a dog"; its output $e_J$ carries joint
  structure that is **not** in the affine span of the singleton embeddings.
  We never call SDXL on the real $e_J$ at sampling — instead, we *synthesise*
  $\hat e_J$ from $(e_A, e_B, e_\emptyset)$ via a small text-only network
  trained offline. This is **M2**.

- **C2 (UNet).** From the three forwards PoE already runs we can compute the
  angle $\theta$ between $u_A = \varepsilon_A - \varepsilon_\emptyset$ and
  $u_B = \varepsilon_B - \varepsilon_\emptyset$. When $\theta > 90°$, the two
  pulls actively cancel — this is the regime where PoE is structurally
  incapable of satisfying both conditions in one step. **C-PoE** multiplies
  the PoE pull by $\max(0, \cos \theta)^\gamma$.

Both methods are derived in [`01_theory.md`](01_theory.md) §6, not assumed.

## 2. Scope (v1)

| Pair | Regime | Prediction |
|---|---|---|
| `("a cat", "a dog")` at seed 42 | collision | PoE fails. M2 and C-PoE produce coherent two-subject scenes. M2+C-PoE is the cleanest. |
| `("a butterfly", "a flower meadow")` at seed 42 | cooperative | PoE works. Methods do not regress it. |

This is N=2 evidence — explicitly a *control overfit* to test mechanism, not
a generalisation claim. See [`03_caveats.md`](03_caveats.md) §"Single-seed scope".

## 3. Where to go next

| You want… | Read… |
|---|---|
| The math derivation | [`01_theory.md`](01_theory.md) |
| What M2 / C-PoE / M2+C-PoE actually do | [`02_methods.md`](02_methods.md) |
| What this work will not claim | [`03_caveats.md`](03_caveats.md) |
| To reproduce cat+dog seed 42 end-to-end | [`04_walkthrough.md`](04_walkthrough.md) |

## 4. The seven symbols you need

| Symbol | Meaning |
|---|---|
| $x_t$ | Latent at diffusion timestep $t$. |
| $\varepsilon_\theta(x_t, t, e)$ | UNet noise prediction conditioned on embedding $e$. |
| $e_A, e_B, e_\emptyset, e_J$ | Text-encoder embeddings of prompts A, B, empty, joint "$c_1$ and $c_2$". |
| $\varepsilon^{\text{PoE}}_t$ | $\varepsilon_\theta(x_t, t, e_A) + \varepsilon_\theta(x_t, t, e_B) - \varepsilon_\theta(x_t, t, e_\emptyset)$. |
| $u_i(x_t)$ | $\varepsilon_\theta(x_t, t, e_i) - \varepsilon_\theta(x_t, t, e_\emptyset)$, $i \in \{A, B\}$. Implicit-classifier direction. |
| $\theta(x_t)$ | $\angle(u_A(x_t), u_B(x_t))$. Conflict angle. |
| $\hat e_J$ | $f_\phi(e_A, e_B, e_\emptyset)$, the synthesizer's offline-trained output. |

## 5. Where the code lives

- **Package:** [`poe_repair/`](../poe_repair/).
- **Run:** `bash scripts/run_all.sh` (after training the synthesizer once via
  `bash scripts/train_synthesizer.sh`).
- **Output:** `outputs/grid/cat_dog_butterfly_seed42.png` is the deliverable.

The code is < 2k lines; every file is reachable from one of the four sampler
runners. Anything not reachable was excluded by design.
