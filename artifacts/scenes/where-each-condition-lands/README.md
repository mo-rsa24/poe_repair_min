# Where each condition lands, step by step

An interactive page in the style of [Diffusion Explorer](https://github.com/helblazer811/Diffusion-Explorer):
one plane, points that move as the sampler denoises, a play button and a time slider. Each point is
one SDXL sampling run for cat × dog on one held-out seed, and its position is the run's current
estimate of the finished image (the Tweedie mean at that step, decoded), embedded with the compose
scorer's DINOv2 and placed on two axes fixed from the finished renders of the three single-prompt
conditions: left to right is cat-alone to dog-alone, up is toward where "a cat and a dog" ends up.

Six conditions per seed: cat alone, dog alone, the joint prompt, plain PoE, and PoE plus the
rank-32 step-30050 correction at λ 1.0 and λ 1.2 (held out on this pair). Hover a point to see its
thumbnail at that step; pick one seed to see all six side by side while you scrub.

## Open it

Build once (Node is on the cluster, nothing on the laptop):

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/scenes/where-each-condition-lands
npm install && npm run build      # writes dist/, about 26 MB with the thumbnails
```

Then serve and tunnel per the [scene-serving recipe](../../../runbook/looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md)
and open

```
http://localhost:8800/where-each-condition-lands/dist/index.html
```

For live editing instead: `npm run dev` binds `127.0.0.1:5174`, forward that port the same way.

## What is where

| Path | What it is |
|---|---|
| `src/App.svelte` | the page: the plane, hulls, trails, moving markers, thumbnails, controls |
| `vendor/ui/` | Diffusion Explorer's `packages/ui/src` copied verbatim (its `TimeSlider`, `Slider`, `FigureLegend` are what the page uses; `vendor/ui-package.json` is its manifest) |
| `public/data.json` | per run, per saved step, the (x, y) on the cloud axes, plus axis provenance and sampler settings |
| `public/frames/<condition>/seed_<n>/step_<kk>.png` | 160 px thumbnails of the decoded estimate at each saved step |
| `dist/` | the built static page |

## How the data was made

`scripts/showcase/where_each_condition_lands_trajectories.py` ran the 48 samplings on mscluster109
GPU 1 (`co3`) via `where_each_condition_lands_shared_device.sh` with `STAGE=frames`: 50 DDIM steps,
guidance 7.5, 1024 px, one pinned initial latent per seed from the training cache shared by all six
conditions. At steps 0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45 and 49 it decodes the Tweedie mean
from the sampler's own latent and guided noise prediction, and saves the final image as step 50.
The three reference conditions are rendered before the adapter is attached to the UNet, because
the windowed sampler leaves the adapter enabled when it returns. Frames live at
`/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames/`.

`scripts/showcase/where_each_condition_lands_frames_embed.py` embeds all 672 frames on CPU
(`XFORMERS_DISABLED=1`), fits the axes on this run's step-50 reference frames, and writes
`public/data.json`, the thumbnails, and `artifacts/results/where-does-each-condition-land/frames-dino-feats.npz`.

## What it cannot say

The joint cloud sits on the vertical axis by construction; the content is where PoE and the
corrected runs are relative to it and when they get there. Positions between saved steps are
straight-line interpolations. DINOv2 CLS is a global embedding, the validated compose scorer counts
instances, and the thumbnails are the ground truth. Eight seeds show where runs land, not the shape
of a distribution.
