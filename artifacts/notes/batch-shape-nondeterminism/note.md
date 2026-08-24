# Same UNet, same inputs: batch size changes the answer

## 2026-08-05

**Claim:** the same UNet, given identical inputs, returns different numbers when
run on 3 prompts versus 4; that difference compounds over denoising steps; and
holding the batch size fixed makes it bit-identical.

**Result:** figure built, all three parts confirmed by direct measurement.

**Files:** `measure.py` (reproduces everything), `measurements.json` (raw
numbers), `batch_shape_effect.png` (two-panel figure), this note.

**One-line finding:** the effect is real (2.0e-03 per call), it is caused by
batch size and not randomness (same shape twice is bit-identical), and it
compounds to 1.66 in the final latent, which is why the lambda=0 check could
never be "bit-exact against run_cfg_poe".

---

## Why this mattered

Plan 00 specified a safety check: run the injection at dose 0, compare against
plain PoE, demand an exact match. Dose 0 should inject nothing, so the two
should agree. The check failed. This folder is the diagnosis.

## What was measured

Run on `a_cat__x__a_dog` seed 9, 50 steps, RTX 3090, fp16.

### A. One call, identical inputs, different batch size

`run_cfg_poe` feeds the UNet 3 prompts at once (A, B, unconditional).
`run_teacher_residual` feeds it 4 (A, B, joint, unconditional). Comparing only
the three branches both runs share, with byte-identical inputs:

| branch | max abs difference | identical? |
|---|---|---|
| eps_a | 1.953e-03 | no |
| eps_b | 1.953e-03 | no |
| eps_uncond | 1.953e-03 | no |

The unconditional branch is the clearest case: the exact same prompt, the exact
same latent, and the answer differs depending on what else was in the batch.

### B. Is it randomness, or is it the batch size?

| run | identical? |
|---|---|
| batch 3, run twice | **yes**, bit for bit |
| batch 4, run twice | **yes**, bit for bit |

So the model is deterministic. Fix the batch size and you get the same numbers
every time. Change it and you do not. The cause is kernel selection: the
library picks different matrix-multiply routines for different shapes, and in
fp16 those disagree in the last bits.

### C. Does a 0.002 difference matter?

Each denoising step feeds its output into the next, so the gap grows.

| step | max abs difference |
|---|---|
| 0 | 0.000 (same starting noise) |
| 1 | 8e-03 |
| 10 | 1.76e-02 |
| 25 | 1.38e-01 |
| final | **1.66** |

A difference in the fourth decimal place becomes a difference of 1.66 in the
final latent. That is not a rounding detail, it is a visibly different image.

## What this changed

The check as written could never pass on correct code, and a tolerance loose
enough to accommodate 1.66 would be far too loose to catch a real leak. Both
options were wrong.

The fix is to compare like with like. Every canary in
`tests/test_interaction_term_canaries.py` now holds the batch shape fixed at
four branches and varies only the dose, which is the thing under test. Within
one batch shape the sampler is deterministic (part B), so those assertions are
exact and meaningful.

## Reproduce

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PYTHONPATH=. /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
  artifacts/notes/batch-shape-nondeterminism/measure.py
```

Needs a GPU. About three minutes.

## Caveat

One pair, one seed, one GPU. The exact numbers will vary with hardware and
library versions. The three qualitative facts (batch size changes the answer,
same shape is deterministic, the difference compounds) are properties of fp16
kernel selection and will hold, but do not quote 1.953e-03 or 1.66 as universal
constants.
