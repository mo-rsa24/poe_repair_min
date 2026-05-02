# poe_repair — Minimal proof-of-mechanism

PoE composition fails on `("a cat", "a dog")` at SDXL seed 42. M2 (synthesised
joint embedding) and C-PoE (conflict-angle damping) — both inference-time,
marginal-only — repair it. `("a butterfly", "a flower meadow")` is the
cooperative-pair sanity that PoE already works and the methods do not regress.

The image is the result. The math is the explanation. See
[`docs/00_overview.md`](docs/00_overview.md) for the entry point.

## Install

```bash
pip install -e .
```

## Run

One-time (~2-3 hours on a single GPU):

```bash
bash scripts/train_synthesizer.sh
```

Repeated:

```bash
bash scripts/run_all.sh
```

Output: `outputs/grid/cat_dog_butterfly_seed42.png`.

## Docs

- [`docs/00_overview.md`](docs/00_overview.md) — thesis + scope.
- [`docs/01_theory.md`](docs/01_theory.md) — math derivation (M1 bound, two channels, M2/C-PoE).
- [`docs/02_methods.md`](docs/02_methods.md) — concrete design choices.
- [`docs/03_caveats.md`](docs/03_caveats.md) — what this work will not claim.
- [`docs/04_walkthrough.md`](docs/04_walkthrough.md) — reproduce cat × dog seed 42.

## Provenance

Extracted from `Playground/PhD/neurips2026/prototypes/spatial_interaction_sdxl/`
on 2026-05-01. The originating repo retains the broader benchmark scaffolding
(D1–D4 diagnostics, AD-PoE ablation, multi-group taxonomy, F1–F7 figures).
This project keeps only what is needed to test mechanism on two pairs at
seed 42.
