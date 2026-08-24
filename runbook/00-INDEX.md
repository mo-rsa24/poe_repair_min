# Runbook: poe_repair_min

How to do the recurring things here by hand: checking the plan tree's state, and launching or
harvesting a run on the cluster. Built 2026-08-24 during the retrofit sweep's stage 5, from
`CLAUDE.md`'s own conventions section and `environment/hpc/`, not from a full excavation
dialogue (this sitting has no interactive user to walk one with). Two themes only; more get
added by `--capture` the next time something recurring happens by hand and isn't here yet.

## Lookup

| I want to... | Go to |
|---|---|
| See whether the plan tree has drifted (stale run claims, unharvested output, orphaned files, unjudged reviews) | [checking-the-plan-tree.md §1](checking-the-plan-tree.md#1-run-the-state-check) |
| Decide whether to submit a job or SSH onto a shared node | [launching-and-harvesting-a-run.md §1](launching-and-harvesting-a-run.md#1-decide-where-a-run-goes) |
| Launch something on a shared half-used `biggpu` node | [launching-and-harvesting-a-run.md §2](launching-and-harvesting-a-run.md#2-launch-on-a-shared-device) |
| Find every run in flight, including the ones Slurm can't see | [launching-and-harvesting-a-run.md §3](launching-and-harvesting-a-run.md#3-harvest-what-is-running) |

## Themes

- [checking-the-plan-tree.md](checking-the-plan-tree.md), 1 recipe
- [launching-and-harvesting-a-run.md](launching-and-harvesting-a-run.md), 3 recipes

## Still open

No diagram prompts yet: neither theme crosses more than one system in a way that needs a
picture over a command. No screenshot slots: everything here is a terminal command. Both
recipes in `launching-and-harvesting-a-run.md` are `unverified`, transcribed from
`environment/hpc/execution-protocol.md` rather than run live this sitting; the next real
launch or harvest should upgrade them via `--capture`.
