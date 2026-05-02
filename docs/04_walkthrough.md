# Walkthrough — End-to-end on cat × dog seed 42

This file walks you through reproducing the v1 result from a fresh checkout.

## 0. Setup

```bash
# From the project root.
pip install -e .
```

Confirm:
- SDXL weights cache is reachable (the runtime calls
  `stabilityai/stable-diffusion-xl-base-1.0`; first run will download ~7 GB).
- A single GPU with ~30 GB free (the four samplers fit in ~18 GB; the
  synthesizer trainer fits alongside).
- Pilot data in place at `data/pilot/seed_42/`:

```bash
$ ls data/pilot/seed_42/
a_butterfly__x__a_flower_meadow/
a_cat__x__a_dog/
```

Each cell needs `grid_assets.json`, `grid_assets/trajectory_flat_*.npy`,
`solo_a.png`, `solo_b.png`, `monolithic.png`, `poe.png`. These are the
upstream pilot artefacts from `neurips2026/`.

## 1. Train the synthesizer (one-time, ~2-3 hours)

The synthesizer ($f_\phi: (e_A, e_B, e_\emptyset) \to \hat e_J$) is text-only;
no images involved.

```bash
bash scripts/train_synthesizer.sh
```

Expected behaviour:
- Held-out `seq_cos` validation cosine rises above ~0.85 within ~20k steps.
- Plateaus above ~0.90 by 100k steps.
- Best checkpoint at `checkpoints/synthesizer/residual_mlp/best.pt`.

If `seq_cos` plateaus below ~0.80, M2 has limited recovery ceiling — see
[`03_caveats.md`](03_caveats.md) §3 (regression-to-mean).

## 2. Run the four samplers + grid

```bash
bash scripts/run_all.sh
```

This runs in series:
1. **PoE baseline** (`poe_repair.methods.poe_baseline`) → `outputs/poe/pairs/<slug>/seed_42/poe.png`.
2. **C-PoE** (`poe_repair.methods.c_poe`) → `outputs/c_poe/pairs/<slug>/seed_42/c_poe.png`.
3. **M2-replace** (`poe_repair.methods.m2_replace`) → `outputs/m2_replace/pairs/<slug>/seed_42/m2_replace.png`.
4. **M2 + C-PoE** (`poe_repair.methods.m2_c_poe`) → `outputs/m2_c_poe/pairs/<slug>/seed_42/m2_c_poe.png`.
5. **Grid** (`poe_repair.figures.grid`) → `outputs/grid/cat_dog_butterfly_seed42.png`.

Each sampler takes ~25-30 s on an A100/L40-class GPU; total wall-clock for
the four samplers + grid is ~2-3 min on a single GPU after model load.

Override defaults via env vars: `GAMMA`, `LAMBDA_J`, `GRID_OUT`.

## 3. Read the grid

The deliverable is `outputs/grid/cat_dog_butterfly_seed42.png` — a 2-row
× 7-column figure.

| Row | Columns expected to show |
|---|---|
| `a_cat__x__a_dog` (collision) | A: cat. B: dog. A∧B: *single-prompt* "a cat and a dog" — sometimes works, sometimes hybrid. PoE: failure mode (chimera, dominance, or destructive cancellation). M2 / C-PoE / M2+C-PoE: progressively cleaner two-subject scenes. |
| `a_butterfly__x__a_flower_meadow` (cooperative) | A, B, A∧B all coherent. PoE coherent. Methods do not regress. |

If the cat×dog row has all three methods producing coherent two-subject
images while the cooperative row is unchanged, **v1 succeeds**.

If M2 produces a washed-out cat×dog image, see
[`03_caveats.md`](03_caveats.md) §3.

If C-PoE produces a vague cat×dog image, see
[`03_caveats.md`](03_caveats.md) §2.

## 4. Smoke tests (when porting / debugging)

**C-PoE bytewise reproducibility.** If you re-port the pipeline and want to
confirm the SDXL stack and `_sampling.run_c_poe` are bit-identical to a
known-good baseline:

```bash
python -m poe_repair.methods.c_poe --pair-filter a_cat__x__a_dog --seed-filter 42
sha256sum outputs/c_poe/pairs/a_cat__x__a_dog/seed_42/c_poe.png
```

C-PoE is the cheapest sampler that exercises full SDXL + the conflict-gate
math without depending on the synthesizer. If the hash matches a reference
hash you trust, the SDXL stack is clean.

M2 and M2+C-PoE are *not* bytewise reproducible across synthesizer training
runs (different RNG = different checkpoint = different $\hat e_J$). Compare
them qualitatively, not byte-for-byte.

## 5. What if I want to add a seed or a pair?

Edit `pairs.py`:

```python
PAIRS = [
    {"slug": "a_cat__x__a_dog", "prompt_a": "a cat", "prompt_b": "a dog", "regime": "collision"},
    {"slug": "a_butterfly__x__a_flower_meadow", "prompt_a": "a butterfly", "prompt_b": "a flower meadow", "regime": "cooperative"},
    # add more here
]
SEEDS = [42, 1, 2]   # add seeds here
```

Then drop the corresponding pilot cell into `data/pilot/seed_<n>/<slug>/`
(must contain `grid_assets.json` + `grid_assets/trajectory_flat_*.npy` +
the four reference PNGs). Re-run `bash scripts/run_all.sh`. The grid will
auto-extend.

This is a *data-only* extension — no code change. See
[`03_caveats.md`](03_caveats.md) §5 for what kind of additional evidence
strengthens what claim.
