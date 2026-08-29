# Reading an adapter training run

What gets recorded while a pooled adapter trains, where each number lands, and how to read
it. The recorded things come in two kinds: **per-step curves** (logged to W&B on every
optimizer step) and the **tracking set**, the fixed list of images and curves saved at every
10k-step checkpoint. The tracking set is frozen before a run launches so that every
checkpoint of every run is comparable; its contents are decided in
[the showcase scope's decision ledger](../plans/01-showcase-the-trained-lora/decisions-taken-here.md).

W&B project: [prime_lab/poe-repair-animals-compose](https://wandb.ai/prime_lab/poe-repair-animals-compose).
Run folders: `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/<run_id>/`
(config, checkpoints, samples, history).

## 1. The per-step curves, and what each one means

Status: unverified (transcribed from `phase1_r8_100k`'s logged history; upgrade via
`--capture` on the next live run).

| Curve (W&B key) | What it is | How to read it |
|---|---|---|
| `train/loss` | How far the adapter's correction is from the cached true correction on training cells | Falling means still learning. Flat and noisy (as `phase1_r8_100k` was, around 1e-4) means more steps buy nothing at this capacity. |
| `train/loss_bucket/early`, `/commit`, `/late` | The same loss split by where in the 50 sampling steps the training example came from: before step 5, steps 5 to 25 (where the image picks its layout), after step 25 | If `commit` stays high while the others fall, the adapter is failing exactly where composition is decided. |
| `train/delta_target_norm` vs `train/delta_hat_norm` | The size of the true correction beside the size of the learned one | Learned much smaller than true means the adapter under-corrects. The known plateau: the learned one covers about 40% of the true one. |
| `train/grad_norm`, `train/lr` | Training health | Spikes in `grad_norm` with no loss change usually mean a bad cell, not a broken run. |

## 2. The tracking set, saved at every 10k-step checkpoint

Status: unverified until the instrument extension lands; items marked *new* are adopted but
not yet wired.

| Item | Where it lands | How to read it |
|---|---|---|
| Comparison strips: joint-prompt render, plain PoE, adapter, per cell | `samples/per_epoch/epoch_*/`, and W&B panels | The adapter column should drift from PoE-like toward the joint-prompt column over checkpoints. Left two columns never change; only the third evolves. |
| The repair cell (cat and dog, seed 1) | same folders | The one cell where the joint prompt itself fails. Watch for the step where the dog first appears. |
| Compose rate over the held-out pool | `compose_rate.json`, W&B | Fraction of held-out renders where the scorer counts two animals. The headline transfer number. |
| `direction_cosine` | W&B | Does the learned correction point the same way as the pool-average true correction: 1 same direction, 0 unrelated. |
| `frac_distance_reached` | W&B | How much of the true correction's length the learned one covers; about 0.4 is the known plateau. |
| *new* Learned-vs-true [cosine](/home-mscluster/mmolefe/goal-setting/learning/spectral-structure-of-the-correction/plans/01-dot-product-norm-cosine.md) at sampling steps 7, 15, 22 | to be wired | Same idea as `direction_cosine` but against this cell's own true correction at three fixed sampling steps. Shows which part of the trajectory is learned first. |
| *new* Embedding drift in the two scoring spaces (DINOv2, CLIP) | to be wired | Distance of the render to the joint-prompt image minus distance to the PoE image. Below zero means the render now sits closer to the target than to the failure. |
| *new* Spectral share of learned corrections | to be wired | Whether the learned corrections across cells concentrate in a few shared directions. Diagnostic only: the paper's low-rank claim is made on true corrections, never on these. |
| *new* Correction size across the 50 sampling steps | to be wired | Where in the sampling run the adapter acts. It should act before the layout is decided (steps 18 to 36 is where samples commit; the best injection window is steps 0 to 10). |

## 3. Filling the screenshot slots

Screenshot slot, unfilled: each table row above wants one captured W&B panel with a caption
saying what the healthy shape looks like.

The slots get filled while the showcase experiments run: the session capturing a run also
captures its W&B panels and files each one here, beside its table row, with a caption saying
what the healthy shape looks like. That needs the two connectors below, which only a human
can authorize (a non-interactive session cannot run the OAuth or install step).

**Both connectors are registered at user scope on this machine and connect** (check any
time with `claude mcp list`; they load at session start, so a session started before the
registration will not see them). The exact registered commands, should either ever need
re-creating:

```bash
# W&B MCP: serves runs, histories and panels as tools. Auth comes from ~/.netrc,
# no key in the config. The git+pin form is required: the package is not on PyPI,
# and its HEAD needs mcp pinned below 2.
claude mcp add --scope user wandb -- /home-mscluster/mmolefe/.local/bin/uvx \
    --with 'mcp<2' --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server

# Playwright MCP: drives a browser and takes screenshots. npx lives in the
# miniforge base, not on PATH; chromium 151 is installed under
# ~/.cache/ms-playwright and launches on this node.
claude mcp add --scope user playwright -- /home-mscluster/mmolefe/miniforge3/bin/npx \
    -y @playwright/mcp@latest
```

One caveat before relying on Playwright for W&B pages: wandb.ai panels need a logged-in
browser, and the Playwright MCP uses its own profile, so either log in once in headed mode or
let the W&B MCP pull the panel data and render the figure locally.

- **Generate the same figures locally today**, no browser and no new connectors, from the
  run history:

  ```bash
  PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
  $PY - << 'PYEOF'
  import wandb, matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  run = wandb.Api().run("prime_lab/poe-repair-animals-compose/<run_id>")
  df = run.history(keys=["train/loss", "train/delta_target_norm",
                         "train/delta_hat_norm"], samples=2000)
  ax = df.plot(x="_step", logy=True)
  ax.set_xlabel("optimizer step")
  plt.savefig("run_curves.png", dpi=150)
  PYEOF
  ```

  Save the output beside the run's other evidence and caption it from the tables above.
