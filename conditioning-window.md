# CFG conditioning-window ablation

`poe_repair/experiments/conditioning_window/` is the **no-LoRA baseline**
against which the LoRA experiment measures its marginal effect.

## Research question

For the prompt "a cat and a dog" at seed 42, which contiguous segments
of the 50-step DDIM trajectory genuinely require the conditional branch
of CFG? Cast as a causal-sensitivity-of-trajectory question on `t`: at
each step we either run the standard CFG combination
`ε̃ = ε_uncond + w·(ε_cond − ε_uncond)` or collapse to `ε_uncond` (zero
conditional contribution), keeping the σ-schedule and the bug-fixed
DDIM noise direction identical to the normal step.

No LoRA, no Mono. Clean SDXL base.

## Schedule grammar

Defined in [`poe_repair/experiments/conditioning_window/schedules.py`](poe_repair/experiments/conditioning_window/schedules.py).
A schedule is a length-50 boolean mask; `mask[i] = True` means the
conditional branch is applied at step `i`. Families:

| Family    | Form              | Examples                                  |
|-----------|-------------------|-------------------------------------------|
| `prefix`  | on for `[0, k)`   | `prefix_k05`, `prefix_k10`, …, `prefix_k50` |
| `suffix`  | on for `[n-k, n)` | `suffix_k05`, …, `suffix_k50`              |
| `window`  | on for `[a, b)`   | `window_00_05`, `window_10_30`, …          |
| `punctate`| 1–3-step pulse at one position | `pulse_t12_w2`, `pulse_t42_w3`, … |
| `sanity`  | all-on / all-off  | `sanity_all_on`, `sanity_all_off`          |

The `STANDARD_SUITE` enumerates 59 schedules — small enough to render in
one sitting, big enough to identify the minimum window by eye.

## Sanity protocol

Two equivalence checks (`--sanity-only`):

1. `run_cfg_masked(mask=all_on)` ≡ `run_cfg(...)` to ≤ 1e-5 max-abs
   latent delta — confirms the mask path is bitwise-identical to the
   standard CFG path when fully on.
2. `run_cfg_masked(mask=all_off)` ≡ `run_cfg(..., guidance_scale=0)` to
   the same tolerance — confirms a fully-off mask collapses cleanly to
   the unconditional sampler.

Results land at
`outputs/conditioning_window/cat_dog/seed_42/results/sanity/sanity.json`
plus four reference PNGs for visual spot-checks.

## Readout

**The interactive inspector is the primary readout** — visual
inspection only. Identify the minimum window by scrubbing the slider in
`scripts/lora_inspector.py` at `/conditioning_window`. The contact-sheet
PNG is the static fallback.

## Reproducing from a clean checkout

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
```

### Step 1 — sanity (fast, ~30s)

```bash
$PY -m poe_repair.experiments.conditioning_window --sanity-only
```

Non-zero exit if either check fails. Writes `sanity.json` either way.

### Step 2 — smoke (fast, ~90s for 2 schedules)

```bash
$PY -m poe_repair.experiments.conditioning_window --smoke
```

Renders `prefix_k10` and `window_10_20`, just to confirm sampling works.

### Step 3 — full sweep (~10–15 min for 59 schedules + sanity)

```bash
$PY -m poe_repair.experiments.conditioning_window
```

Renders the `STANDARD_SUITE`, runs the sanity checks, builds the
inspector manifest, renders the contact-sheet figure.

### Step 4 — launch the inspector

```bash
$PY scripts/lora_inspector.py --port 5050
# from your laptop:
#   ssh -L 5050:localhost:5050 mscluster106
#   open http://127.0.0.1:5050/conditioning_window
```

The slider scrubs through all 59 + 2 schedules. The 50-cell strip below
the slider shows the on/off mask (green = conditional ON, grey = OFF)
with tick-marks every 10 steps. Hover the strip to see the full
50-character binary mask string. The image panel updates as you scrub.

## Output layout

```
outputs/conditioning_window/cat_dog/seed_42/
  schedules/
    prefix_k05/image.png
    prefix_k05/summary.json
    …
    sanity_all_on/…
    sanity_all_off/…
  results/
    sanity/
      sanity.json
      masked_all_on.png         # run_cfg_masked(all_on)
      run_cfg.png               # run_cfg(...)
      masked_all_off.png        # run_cfg_masked(all_off)
      run_cfg_gs0.png           # run_cfg(guidance_scale=0)
    figures/
      contact_sheet.png         # static fallback
    inspector_manifest.json
```

## Why this is the right no-LoRA baseline

The conditioning_window experiment reuses **the same x_T as the LoRA
experiment** via `initial_latents_for_pair(cell=cell_for("a cat", "a dog", 42))`
— so a side-by-side at `(seed=42, "a cat and a dog")` between the LoRA
inspector's `epoch=max, λ=1` cell and the conditioning_window inspector's
`prefix_k50` (all-on) cell isolates exactly what the LoRA contributes
over the no-residual baseline.

When a partial schedule in conditioning_window already produces a
recognisable cat+dog, the LoRA's marginal contribution at that schedule
is "nothing" — the prompt alone sufficed. When the partial schedule
fails, the LoRA's contribution at the corresponding training step is
what closes the gap.
