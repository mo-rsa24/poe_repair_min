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
| Decide whether to submit a job or SSH onto a shared node | [launching-and-harvesting-a-run.md §1](launching-and-harvesting-a-run.md#1-decide-where-a-run-goes) |
| Launch something on a shared half-used `biggpu` node | [launching-and-harvesting-a-run.md §2](launching-and-harvesting-a-run.md#2-launch-on-a-shared-device) |
| Find every run in flight, including the ones Slurm can't see | [launching-and-harvesting-a-run.md §3](launching-and-harvesting-a-run.md#3-harvest-what-is-running) |
| Serve a static scene from `artifacts/scenes/` in a tmux session | [showing-a-scene-in-the-local-browser.md §1](showing-a-scene-in-the-local-browser.md#1-serve-the-scenes-folder-from-a-tmux-session) |
| See a scene served on the node in my local browser (the SSH forward) | [showing-a-scene-in-the-local-browser.md §2](showing-a-scene-in-the-local-browser.md#2-forward-the-port-and-open-it-on-the-laptop) |
| Know what a training run records and how to read each curve and image | [reading-a-training-run.md §1](reading-a-training-run.md#1-the-per-step-curves-and-what-each-one-means) |
| See what gets saved at every checkpoint (the tracking set) | [reading-a-training-run.md §2](reading-a-training-run.md#2-the-tracking-set-saved-at-every-10k-step-checkpoint) |

## Themes

- [checking-the-plan-tree.md](checking-the-plan-tree.md), 1 recipe
- [launching-and-harvesting-a-run.md](launching-and-harvesting-a-run.md), 3 recipes
- [showing-a-scene-in-the-local-browser.md](showing-a-scene-in-the-local-browser.md), 2 recipes
- [reading-a-training-run.md](reading-a-training-run.md), 3 recipes

## Still open

No diagram prompts yet: neither theme crosses more than one system in a way that needs a
picture over a command. Only `reading-a-training-run.md` reserves places for screenshots, and
they are all still empty: they want captured W&B panels, which needs either a W&B or Playwright
MCP connected (a human step in claude.ai connector settings) or the local wandb-API figures in
that file's section 3. Both
recipes in `launching-and-harvesting-a-run.md` are `unverified`, transcribed from
`environment/hpc/execution-protocol.md` rather than run live this sitting; the next real
launch or harvest should upgrade them via `--capture`.
