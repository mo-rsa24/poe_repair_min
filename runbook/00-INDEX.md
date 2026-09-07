# Runbook: poe_repair_min

How to do the recurring things here by hand: checking the plan tree's state, and launching or
harvesting a run on the cluster. Built 2026-08-24 during stage 5 of the retrofit pass, from
`CLAUDE.md`'s own conventions section and `environment/hpc/`, not from a full excavation
dialogue (this sitting has no interactive user to walk one with). More themes get added the
next time something recurring happens by hand and isn't here yet.

## Lookup

| I want to... | Go to |
|---|---|
| See whether the plan tree has drifted (stale run claims, unharvested output, orphaned files, unjudged reviews) | [checking-the-plan-tree.md §1](checking-the-plan-tree.md#1-run-the-state-check) |
| Decide whether to submit a job or SSH onto a shared node | [launching-and-harvesting-a-run.md §1](running-things-on-the-cluster/launching-and-harvesting-a-run.md#1-decide-where-a-run-goes) |
| Launch something on a shared half-used `biggpu` node | [launching-and-harvesting-a-run.md §2](running-things-on-the-cluster/launching-and-harvesting-a-run.md#2-launch-on-a-shared-device) |
| Find every run in flight, including the ones Slurm can't see | [launching-and-harvesting-a-run.md §3](running-things-on-the-cluster/launching-and-harvesting-a-run.md#3-harvest-what-is-running) |
| Serve a static scene from `artifacts/scenes/` in a tmux session | [showing-a-scene-in-the-local-browser.md §1](looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md#1-serve-the-scenes-folder-from-a-tmux-session) |
| See a scene served on the node in my local browser (the SSH forward) | [showing-a-scene-in-the-local-browser.md §2](looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md#2-forward-the-port-and-open-it-on-the-laptop) |
| Know what a training run records and how to read each curve and image | [reading-a-training-run.md §1](looking-at-what-a-run-produced/reading-a-training-run.md#1-the-per-step-curves-and-what-each-one-means) |
| See what gets saved at every checkpoint (the tracking set) | [reading-a-training-run.md §2](looking-at-what-a-run-produced/reading-a-training-run.md#2-the-tracking-set-saved-at-every-10k-step-checkpoint) |
| Build the SuperDiff trajectory cache the adapters train on | [training-and-watching-a-superdiff-adapter.md §1](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#1-build-the-superdiff-trajectory-cache) |
| Train an adapter on SuperDiff's own residual (or resume one) | [training-and-watching-a-superdiff-adapter.md §2](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#2-launch-one-training) |
| See what a training adapter does inside SuperDiff, every 10k steps, without stopping it | [training-and-watching-a-superdiff-adapter.md §3](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#3-render-a-strip-from-every-checkpoint-while-it-trains) |
| Put every checkpoint strip on one sheet per rank and pair | [training-and-watching-a-superdiff-adapter.md §4](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#4-assemble-the-timelines) |
| Sweep λ on one checkpoint of a SuperDiff adapter | [training-and-watching-a-superdiff-adapter.md §5](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#5-sweep-λ-on-one-checkpoint) |
| Render SuperDiff alone across κ and λ (the seeds-by-λ sheets) | [training-and-watching-a-superdiff-adapter.md §6](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#6-render-superdiffs-own-kappa-by-lambda-sheets) |
| Inject a PoE-trained adapter into SuperDiff and measure which way its correction points | [training-and-watching-a-superdiff-adapter.md §7](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#7-inject-a-poe-trained-adapter-into-superdiff-and-measure-its-direction) |
| Stop a `nohup` run on a node without killing my own shell | [training-and-watching-a-superdiff-adapter.md §8](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#8-stop-a-run-cleanly) |
| Render the 8 held-out cat x dog seeds from one pooled-LoRA checkpoint, with the scorer's verdict on every tile | [probing-a-pooled-lora-checkpoint.md §1](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#1-render-the-8-seed-held-out-grid-for-one-checkpoint) |
| Measure how well a checkpoint fits the cached correction, per pair and step bucket, with no sampling | [probing-a-pooled-lora-checkpoint.md §2](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#2-measure-how-well-a-checkpoint-fits-the-cached-correction) |
| Resume a pooled (PoE) training from a checkpoint on whichever device is free | [probing-a-pooled-lora-checkpoint.md §3](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#3-resume-a-pooled-training-from-a-checkpoint-on-a-free-device) |
| Train two ranks at once on an idle node with one Slurm job | [probing-a-pooled-lora-checkpoint.md §4](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#4-train-two-ranks-on-one-idle-node-with-one-slurm-job) |
| Estimate how long a training run will take before launching it | [environment/hpc/throughput.md](../environment/hpc/throughput.md) |
| Prove the Langevin corrector is off when it is switched off | [running-the-langevin-corrector.md §1](running-things-on-the-cluster/running-the-langevin-corrector.md#1-prove-the-corrector-is-off-when-it-is-switched-off) |
| Fix the corrector's step size before reading any curve | [running-the-langevin-corrector.md §2](running-things-on-the-cluster/running-the-langevin-corrector.md#2-fix-the-step-size-before-reading-any-curve) |
| Run the corrector-count grid and print its three-way verdict | [running-the-langevin-corrector.md §3](running-things-on-the-cluster/running-the-langevin-corrector.md#3-run-the-corrector-count-grid-and-print-its-verdict) |
| Slide a corrector window across the run, or render the eight-seed sheets and the adapter tails | [running-the-langevin-corrector.md §4](running-things-on-the-cluster/running-the-langevin-corrector.md#4-slide-a-corrector-window-across-the-run), then [§5](running-things-on-the-cluster/running-the-langevin-corrector.md#5-render-the-eight-seed-sheets-the-adapter-tail-and-the-clean-tail) |
| Run a Langevin corrector on the product-of-experts score, fix its step size, and read what survives it | [running-the-langevin-corrector.md §1](running-things-on-the-cluster/running-the-langevin-corrector.md#1-prove-the-corrector-is-off-when-it-is-switched-off) to [§3](running-things-on-the-cluster/running-the-langevin-corrector.md#3-run-the-corrector-count-grid-and-print-its-verdict) |
| Slide a corrector window across the run, or render the eight-seed corrector sheets and the adapter tails | [running-the-langevin-corrector.md §4](running-things-on-the-cluster/running-the-langevin-corrector.md#4-slide-a-corrector-window-across-the-run) to [§6](running-things-on-the-cluster/running-the-langevin-corrector.md#6-draw-the-figures-and-log-the-set-to-wb) |
| Re-render the joint prompt, each concept alone, PoE and the corrected run on the held-out seeds, with per-step frames | [reproducing-where-each-condition-lands.md §1](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#1-render-the-references-and-the-per-step-frames-on-a-shared-device) |
| Redraw the endpoint figures and the sidecar behind the landing finding | [reproducing-where-each-condition-lands.md §2](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#2-make-the-endpoint-figures-and-their-sidecar) |
| Rebuild the animated "where each condition lands" page (Diffusion Explorer's UI over our frames) and open it | [reproducing-where-each-condition-lands.md §3](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#3-embed-the-frames-and-export-the-pages-data), then [§4](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#4-build-the-page-from-diffusion-explorers-ui-and-open-it) |
| Show that page on a laptop with no cluster tunnel | [reproducing-where-each-condition-lands.md §5](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#5-open-the-built-page-on-a-laptop-with-no-cluster-tunnel) |

## Themes

Grouped by the kind of work. The group is a listing convenience; every row above still points
straight at the recipe that answers it.

**Running things on the cluster**

| File | One place, one purpose | Recipes |
|---|---|---|
| [launching-and-harvesting-a-run.md](running-things-on-the-cluster/launching-and-harvesting-a-run.md) | A `biggpu` node over SSH: pick a device, launch with `nohup`, find what is running | 3 |
| [reproducing-where-each-condition-lands.md](running-things-on-the-cluster/reproducing-where-each-condition-lands.md) | The landing figures and their animated page, from renders to `dist/` | 5 (4 `ran`, 1 `unverified`) |
| [running-the-langevin-corrector.md](running-things-on-the-cluster/running-the-langevin-corrector.md) | A `biggpu` device or the session node: the corrector's leak checks, its step-size search, the corrector-count grid, the window sweep, the eight-seed sheets and the two adapter tails | 6 (all `ran`) |
| [probing-a-pooled-lora-checkpoint.md](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md) | A `biggpu` device or one Slurm job: render a pooled checkpoint's held-out grid, read its fit on the cache, resume or launch a pooled training | 4 (all `ran`) |
| [training-and-watching-a-superdiff-adapter.md](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md) | SuperDiff on the cluster: its cache, an adapter trained on its residual, strips from every checkpoint, λ sweeps, its own κ × λ sheets, a PoE adapter injected into it | 8 (all `ran`) |
| [running-the-langevin-corrector.md](running-things-on-the-cluster/running-the-langevin-corrector.md) | The Langevin corrector of scope 06: the two leak checks, the step-size search, the corrector-count grid and its verdict, the sliding window, the eight-seed sheets and the adapter tails, the figures and the W&B log | 6 (all `ran`) |

**Looking at what a run produced**

| File | One place, one purpose | Recipes |
|---|---|---|
| [reading-a-training-run.md](looking-at-what-a-run-produced/reading-a-training-run.md) | W&B: the curves and the tracking set a training run writes | 3 |
| [showing-a-scene-in-the-local-browser.md](looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md) | A static server on the node and the tunnel from the laptop | 2 |

**Flat, no group yet**

| File | One place, one purpose | Recipes |
|---|---|---|
| [checking-the-plan-tree.md](checking-the-plan-tree.md) | The repo on the session node: the plan-tree state check | 1 |

## Still open

No diagram prompts yet: neither theme crosses more than one system in a way that needs a
picture over a command. Only `reading-a-training-run.md` reserves places for screenshots, and
they are all still empty: they want captured W&B panels, which needs either a W&B or Playwright
MCP connected (a human step in claude.ai connector settings) or the local wandb-API figures in
that file's section 3. Both
recipes in `launching-and-harvesting-a-run.md` are `unverified`, transcribed from
`environment/hpc/execution-protocol.md` rather than run live this sitting; the next real
launch or harvest should upgrade them via `--capture`.
