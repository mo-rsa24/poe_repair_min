# Scene: the denoising step and log-SNR, as two names for one clock

The interactive companion to F3 in [../what-each-figure-argues.md](../what-each-figure-argues.md).
F3 is drawn on the denoising step. This scene shows what that axis buys, what the log-SNR axis
would have bought instead, and computes the measurement that decides between them, so the choice is
something you can check rather than take on trust.

## Run it

```bash
cd artifacts/scenes/logsnr-explainer
npm install
npm run dev --  --port 5175        # http://localhost:5175
```

Port 5173 is usually taken by the `dl-scene` app on this node, hence 5175. It runs on the session
node, so reach it from a laptop with a forward in the SSH command:

```bash
ssh -L 5175:localhost:5175 mmolefe@mscluster109.ms.wits.ac.za
```

Node project, own folder. It never touches the Python or CUDA environment.

## What is on the page

Six panels, one shared state. Nothing moves unless a named variable moved it. `story` plays the
argument in order; `explore` lights every panel and makes every control live.

1. **What the noise level actually is.** A noisy image is the mixture
   `x_t = √ᾱ·x₀ + √(1−ᾱ)·ε`, so `ᾱ` is the picture's share of the power. Drag `ᾱ` and watch one
   real signal drown while the odds `ᾱ/(1−ᾱ)` and their log print live. That log is log-SNR. The
   whole definition is first-year material: it is the log-odds that a unit of power in the noisy
   image came from the picture rather than the fuzz.
2. **The schedule.** `ᾱ` and `β` against the 1000 training timesteps, with dots on the timesteps
   the sampler in the controls visits.
3. **The same clock, relabelled.** `λ = log(ᾱ/(1−ᾱ))` against t, falling strictly from 7.07 to
   −5.36, so the two axes carry identical information.
4. **Even in t, very uneven in λ.** The real cached run steps 20 timesteps at a time, every time.
   In log-SNR those same steps move 0.16 apart in the middle and 2.46 at the last step, 19× the
   smallest jump. Under 20 evenly spaced log-SNR points, 7 carry one real step or none, and the six
   cleanest share 3 of the 50 steps.
5. **The same twenty measured values, on each axis.** F3's published median drawn against λ, and
   drawn again with x relabelled to the nearest real step. Four of the twenty points land on steps
   48 and 49, which is the previous panel's count showing up as shape.
6. **What the step axis gives up.** One curve read by two samplers disagrees by about 0.38 on the
   step-index axis and 0.01 on the log-SNR axis. That is log-SNR's one real advantage, and it buys
   a reader this paper does not have, since only DDIM at 50 steps is used. The trail-and-rest-stops
   version of the whole argument sits underneath for anyone who wants it without the diffusion.

Then **checks**, computed in the page rather than asserted: the round trip `σ(λ) = ᾱ`, the strict
monotonicity of λ across all 1000 timesteps, the starved-bin count, and the two sampler gaps side
by side.

## Where the numbers come from

| What | Source |
|---|---|
| `ᾱ` for 1000 timesteps | SDXL's own `DDIMScheduler.alphas_cumprod`, read through `poe_repair/experiments/interaction_term/cache.py` |
| the 50 timesteps actually visited | `meta.json` of `training_cache/train/a_cat__x__a_lion/seed_1`, not an assumed spacing rule |
| the correction curve | `snr_collapse.json` under `outputs/interaction_term/cache_analyses/refresh_20260810/prereg/`, produced by `scripts/snr_collapse.py` |
| the definition of `λ` | `Cell.log_snr()` in `cache.py`, verbatim |

All frozen into `src/data.json`. Regenerate after a new `snr_collapse.py` run:

```bash
python3 - <<'PY'
import json, pathlib
from poe_repair.experiments.interaction_term.cache import _alphas_cumprod
P = '/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/refresh_20260810/prereg/snr_collapse.json'
M = '/datasets/mmolefe/poe_repair_min/outputs/training_cache/train/a_cat__x__a_lion/seed_1/meta.json'
f3, meta = json.load(open(P)), json.load(open(M))
keep = ('collapse_spread_pct','normalize','n_pairs','n_curves','verdict',
        'peak_log_snr','peak_at_edge','log_snr_grid','median_curve','iqr')
pathlib.Path('artifacts/scenes/logsnr-explainer/src/data.json').write_text(json.dumps({
    'alphasCumprodSource': 'SDXL DDIMScheduler.alphas_cumprod, read from the model repo by poe_repair.experiments.interaction_term.cache',
    'alphasCumprod': [round(float(x), 9) for x in _alphas_cumprod().numpy()],
    'cachedTimesteps': meta['timesteps'], 'cachedTimestepsSource': M,
    'cachedSteps': meta['num_inference_steps'],
    'f3Path': P, 'f3': {k: f3[k] for k in keep},
}))
PY
```

## What this scene does not claim

- Nothing about whether the curves lie on top of each other. 19.7% is `loose` by
  `scripts/snr_collapse.py`'s own thresholds, and the scene prints that verdict as it stands.
- The step-axis panel is F3's published median with its x relabelled, not a recomputation from the
  per-cell values. The script saves only the median and the band, so a true step-axis redraw needs
  it to dump its `stack` array first, which is the open task in the figure's plan.
- The two samplers in the last panel are one measured curve re-sampled at two sets of timesteps,
  not two generation runs. Comparing real generations across samplers would be an experiment; this
  is the axis argument only.
- "Evenly spaced in log-SNR" is idealised, standing in for the EDM/Karras family. Only leading and
  trailing are real DDIM spacings, and the cached runs used leading at 50 steps.
