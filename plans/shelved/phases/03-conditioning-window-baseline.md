# Phase 3 — Conditioning-window baseline (no LoRA)

## Question

For the prompt `"a cat and a dog"` at seed 42, which contiguous
segments of the 50-step DDIM trajectory genuinely need the conditional
branch of CFG? At each step we either run the standard CFG combination

```
ε̃ = ε_uncond + w · (ε_cond − ε_uncond)
```

or we collapse to `ε_uncond` (zero conditional contribution). A
schedule is a length-50 boolean mask of on/off steps. We sweep ~59
schedules (prefix-only, suffix-only, window, single-step pulse,
all-on, all-off) and look at the decoded image for each.

No LoRA. No Mono. Clean SDXL base.

## Why this phase exists

The LoRA contribution in Phase 4 only means something *relative to a
baseline that does nothing*. Without this phase we don't know whether
the LoRA is closing a 40% gap or a 4% gap. Concretely we want:

- The smallest *k* such that `prefix [0, k]` already gives a
  recognisable cat + dog without any LoRA. The LoRA's marginal
  contribution is large at schedules where the prompt alone *fails*.
- A sanity that the masked-CFG codepath is bit-equivalent to the
  standard sampler at the all-on and all-off limits.

## Code

- Sampler: `poe_repair/methods/_sampling.py::run_cfg_masked` (length-N
  mask; on-steps run the standard 2-branch CFG forward, off-steps run
  a single `ε_∅` forward, sharing scheduler state).
- Schedule grammar: `poe_repair/experiments/conditioning_window/schedules.py::STANDARD_SUITE`.
- Driver: `poe_repair/experiments/conditioning_window/__main__.py`.
- Inspector route: `scripts/lora_inspector.py:/conditioning_window`
  (dual-handle slider + 50-cell mask strip).

## Commands

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
```

### 1. Sanity (~30 s)

```bash
$PY -m poe_repair.experiments.conditioning_window --sanity-only
```

Two checks at `≤ 1e-5` max-abs latent delta:

- `run_cfg_masked(mask=all_on) ≡ run_cfg(...)`.
- `run_cfg_masked(mask=all_off) ≡ run_cfg(..., guidance_scale=0)`.

Non-zero exit if either fails. Writes
`outputs/conditioning_window/a_cat__x__a_dog/seed_42/results/sanity/sanity.json`.

### 2. Smoke (~90 s, two schedules)

```bash
$PY -m poe_repair.experiments.conditioning_window --smoke
```

Renders `prefix_k10` and `window_10_20` — confirms sampling and figure
writing both work.

### 3. Full sweep (~10–15 min, 59 schedules + sanity)

```bash
$PY -m poe_repair.experiments.conditioning_window
```

Renders `STANDARD_SUITE`, runs the sanity checks, builds the inspector
manifest, renders the contact-sheet figure.

### 4. Inspector

```bash
$PY scripts/lora_inspector.py --port 5050
# from laptop:
#   ssh -L 5050:localhost:5050 mscluster106
#   open http://127.0.0.1:5050/conditioning_window
```

The interactive inspector is the primary readout. The contact-sheet
PNG is the static fallback.

## How to read the result

| Bucket | What you see | Means |
|---|---|---|
| **Poor** | Sanity fails — `all_on` and `run_cfg` disagree above 1e-5; or all 59 schedule images are visually identical. | Mask codepath or scheduler state is broken. Nothing in this phase or Phase 4 is interpretable. |
| **Bad** | `all_on` and `prefix_k50` both produce recognisable cat + dog on their own (no LoRA needed). Other schedules fail. | The cell is not actually a hard PoE failure for *masked* CFG at this seed; the LoRA story doesn't have headroom here. Pick a harder cell or re-frame. |
| **Unknown** | Some short prefix (e.g. `prefix_k10`) produces something halfway — recognisable animal shapes but not clearly two animals. Smallest passing *k* is not obvious by eye. | This *is* the baseline. Note the ambiguous threshold and proceed to Phase 4; the LoRA's job becomes "make short prefixes pass cleanly." |
| **Good (the result on disk)** | Sanity passes bit-exactly. Long prefixes (`k ≥ 30`) produce recognisable but often chimeric outputs; short prefixes (`k ≤ 10`) fail entirely; the all-off output is structureless noise. There is a clear gap between "prompt is on long enough" and "prompt is on too briefly." | Phase 4 has a clean baseline to beat: the LoRA should rescue short-prefix schedules that no-LoRA fails on. |

## How this phase plugs into Phase 4

The conditioning-window experiment reuses **the same `x_T` as the LoRA
experiment** via `initial_latents_for_pair(cell=cell_for("a cat", "a dog", 42))`.
A side-by-side at `(seed=42, "a cat and a dog")` between the LoRA
inspector's `epoch=max, λ=1` cell and the no-LoRA `prefix_k50` cell
isolates exactly what the LoRA contributes over the no-residual
baseline.

A companion experiment, `conditioning_window_lora`, runs the same
schedule sweep *with* the trained LoRA in the loop — but that is part
of the Phase-4 inspection toolkit, not this phase. See
[04-lora-single-seed.md](04-lora-single-seed.md).

## What this phase does *not* do

- Train anything.
- Score automatically (CLIP or otherwise). Readout is visual.
- Other pairs or other seeds.

## Status — 2026-05-19

| Item | Done | To do |
|---|:---:|:---:|
| `run_cfg_masked` sampler landed | ✅ | |
| Sanity checks (`all_on` ≡ `run_cfg`, `all_off` ≡ `gs=0`) passing at ≤ 1e-5 | ✅ | |
| `STANDARD_SUITE` (59 schedules) rendered on cat × dog seed 42 | ✅ | |
| Inspector route `/conditioning_window` with dual-handle slider + mask strip | ✅ | |
| Contact-sheet static fallback PNG | ✅ | |
| Pinned x_T shared with the LoRA experiment (apples-to-apples comparison) | ✅ | |
| Visual identification of smallest passing prefix-*k* | | ⬜ (eyeball pass; only blocks Phase 4 writeup) |
| Automated minimum-*k* metric | | ⬜ (deferred — only if a paper claim needs a number) |
