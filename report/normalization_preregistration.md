# Pre-registered: how correction size is measured

**Committed 2026-08-05, before any cross-type plot exists.**

Plan `plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/plans/instrument-02-fix-the-size-measure-before-any-result.md`.

## The committed choice

**relative_norm**

$$\text{size}(t) \;=\; \frac{\lVert r_t \rVert}{\lVert \tilde{\varepsilon}_{\text{PoE}} \rVert}$$

Per step, then the **median over steps** for a per-cell number, then the
**median over seeds** for a per-pair number. Medians, not means: the early
steps carry heavy fp16 cancellation noise and one bad step should not move a
pair's number.

All tensors upcast to fp32 before any norm. Guided quantities throughout,
computed via the sampler's own `guided_eps` and `poe_eps`.

Anywhere a script names this measure, the string is `relative_norm`.

## Why, and what was rejected

The plan named two candidates. One of them turned out not to be a measure.

### Rejected: "fraction of the PoE to Mono distance"

Measured on three pairs, it is **identically 1.000000**, to six decimal places,
every step, every pair.

The reason is definitional, not numerical. r_t *is* the PoE-to-Mono gap:

$$r_t \;=\; \tilde{\varepsilon}_{\text{Mono}} - \tilde{\varepsilon}_{\text{PoE}}$$

so dividing $\lVert r_t \rVert$ by $\lVert \tilde{\varepsilon}_{\text{Mono}} -
\tilde{\varepsilon}_{\text{PoE}} \rVert$ divides a quantity by itself. It
cannot vary, cannot distinguish pairs, and cannot fail. Had it been adopted,
every composition type would have scored 1.0 and the plan-07 scatter would have
been a flat line presented as a finding.

Recorded here so the wording is not revived later in a form that looks
different but computes the same thing.

### Rejected: r_t against the latent step size

The sensible repair of the above is to compare r_t against something it is not
defined from: the distance the latent actually moves per step,
$\lVert x_{t+1} - x_t \rVert$. Measured:

| pair | relative_norm | vs step size | relative_norm IQR | step-size IQR |
|---|---|---|---|---|
| a_wolf__x__a_husky | 0.1008 | 4.6233 | 0.0234 | 1.7721 |
| a_leopard__x__a_jaguar | 0.1155 | 4.5216 | 0.0377 | 0.9592 |
| a_cat__x__a_dog | 0.1827 | 8.7732 | 0.1146 | 3.6735 |

Rejected for one specific reason: **the denominator depends on the sampler
schedule.** Plan 08 varies the sampler, so a measure that moves when the step
size moves would make cross-sampler comparison meaningless, and the change
would be invisible in the plot. Its within-pair spread is also ~50x larger,
which is a symptom of the same thing.

### Adopted: relative_norm

The denominator is the prediction being corrected. It answers the question the
paper actually asks, "how large is the correction relative to what it is
correcting", it is bounded, it does not depend on the sampler schedule, and its
spread within a pair is small enough that the between-pair differences above
(0.101, 0.116, 0.183) are visible against it.

## The three pairs this was computed on

Chosen to span the pool rather than to look tidy: one training pair
(`a_wolf__x__a_husky`), one unseen transfer pair (`a_leopard__x__a_jaguar`),
and the known-failure reference (`a_cat__x__a_dog`).

The reference pair reads highest at 0.183, roughly 1.7x the other two. That is
the expected direction, since it is the pair PoE fails on most reliably, but
three pairs is not evidence of a trend and this memo does not claim one. Plan
07 tests it properly.

## What this commits us to

- Any cross-pair or cross-composition-type plot uses `relative_norm`.
- If a second normalization is later shown as well, both are reported and
  neither is adopted retrospectively (master plan Goal 5).
- Changing this choice after seeing a scatter requires a dated amendment
  below, stating what was seen first.

## Amendments

None.

## Reproduce

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY scripts/normalization_candidates.py
```
