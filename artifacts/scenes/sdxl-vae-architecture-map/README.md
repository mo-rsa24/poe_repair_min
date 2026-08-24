# dl-scene: the v4 pipeline's VAE, level by level

**ran under** Node/Vite, `npm run dev` on a cluster node, port 5173 (this README notes a
collision with the other scene on the same node, at `artifacts/scenes/logsnr-explainer/README.md:16`,
which uses 5175 instead); last run under this README's own instructions below (stated, not
independently re-verified this move).

**built from** `trace.json`, a CPU-only shape-trace of AutoencoderKL built by `tracer/trace_vae.py`
and `tracer/enrich_trace.py` from the SDXL config with `from_config` (random weights, no GPU); the
optional `cross_trace.json` / `tracer/trace_cross.py` / `tracer/spec_cross.py` trio for the
cross-attention side, present in this scene's source but not confirmed wired into `app/`'s current
UI (guessed, not verified this move — the README below documents only the VAE trace). Code panes
read the installed diffusers 0.39.0 source and this repo's own call sites (`runtime.py`,
`sdipc_utils.py`, `_sampling.py`, `export_vae_activations.py`).

**why** to make the SDXL VAE's architecture a clickable map instead of a diagram someone has to
redraw by hand: every shape and code snippet lands on the real traced value or the real source
line, tagged `real` / `your repo` where it applies.

**depends on** stated as of this move (2026-08-24), moved here from `dl-scene/` (top-level
`README.md`, `trace.json`, `cross_trace.json`, `tracer/`) and `dl-scene/app/` (the Vite app's
source) as part of the retrofit sweep's piles 1c and 5:
- `dl-scene/app/node_modules/` and `dl-scene/app/dist/` were **not** moved; they still sit at
  that old path, now orphaned from the source that built them (a loose end for a later pass, not
  fixed here). `dl-scene/` itself now holds nothing else.
- The "Run it" and "Regenerating the trace" commands below (`cd dl-scene/app`, `cd dl-scene`) name
  the old path and are not rewritten here; read them against the path above.
- `poe_repair/experiments/mechanism_study/export_vae_activations.py`'s `--out` example
  (`dl-scene/app/public/vae_act.json`) likewise names the old path.

A local web app where the SDXL VAE (the one `poe_repair/_sdxl/runtime.py:114` loads) is a
clickable map: the pipeline lane view descends into encoder/decoder funnels, into blocks,
into ResBlock chains, down to every conv, norm, and attention projection, each leaf landing
on the real source lines.

## Run it

```bash
cd dl-scene/app
npm install        # once
npm run dev        # http://localhost:5173
```

On the cluster, tunnel the port (VS Code forwards it automatically) or run
`npm run dev -- --host` and browse to the node's address.

## What is real

- `trace.json` is a shape-trace of AutoencoderKL built from the locally cached SDXL config
  (batch 1, input 3×1024×1024, CPU, no weights loaded, no GPU touched). Every shape in the
  app carries this provenance; nothing is hand-typed.
- Code panes show the installed diffusers 0.39.0 source and this repo's call sites
  (runtime.py, sdipc_utils.py, _sampling.py, export_vae_activations.py), tagged
  `real` / `your repo`.
- Toy arithmetic (attention softmax rows, conv size formula, param counts) is computed
  live by the page and checked against the traced numbers.
- Attention edge weights are illustrative and labelled as such.

## The real-activation slot

Feature-map thumbnails stay schematic until you run, on a GPU node:

```bash
python -m poe_repair.experiments.mechanism_study.export_vae_activations \
    --image <some 1024x1024 image> --out dl-scene/app/public/vae_act.json
```

The app picks up `app/public/vae_act.json` automatically at load.

## Regenerating the trace

```bash
cd dl-scene
python tracer/trace_vae.py vae_trace.json          # CPU, ~1 min, no GPU
python tracer/enrich_trace.py vae_trace.json trace.json
cp trace.json app/src/trace.json
```

`trace_vae.py` rebuilds AutoencoderKL from the cached `vae/config.json` with
`from_config` (random weights; shapes do not depend on weight values), hooks every
`named_modules()` entry, and runs one `encode` + `decode` at 1×3×1024×1024 under
`no_grad`. `enrich_trace.py` adds the diffusers source snippets and repo call sites.
