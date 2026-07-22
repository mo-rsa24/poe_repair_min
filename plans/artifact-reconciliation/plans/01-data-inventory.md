# 📇 Data Inventory — experiments reconciled against W&B

## Description

List every saved file under `poe_repair/experiments/` plus any run outputs, LoRA checkpoints, caches, and eval results in the repo, then match each one to the project's W&B runs. Produce one table: experiment → runs → status → surviving file path.

## Purpose

This is the master list of what exists and which run made it. Every later task (integrity check, canonical layout, re-sweep) reads from it. The run status from W&B is the key input: `died-early` and `false-start` runs are the ones we treat as suspect later.

## Goal

`inventory/01-artifact-inventory.md` with an experiment → runs (id · step · state) → status → surviving-artifact-path table, plus a re-runnable `inventory/scripts/01_inventory.py`. W&B projects recovered from the local run dirs when not supplied.

## Tasks

- [x] ✅ Run the inventory + W&B reconciliation with the prompt below.
- [x] ✅ Write `inventory/01-artifact-inventory.md` and `inventory/scripts/01_inventory.py`.

Fully-qualified prompt (invoke via `/data-inventory`):

```
catalogue every artifact under poe_repair/experiments/ and any run outputs, LoRA
checkpoints, caches, and eval results in this repo. Then reconcile against these
W&B projects: poe-repair-lora, poe-repair-cross-seed, poe-repair-cross-pair,
poe-repair-group-a (poe-repair-m5-lora is legacy/out of scope). If project names
or per-run status are not supplied, recover them from the local W&B run dirs
under outputs/**/wandb/run-* (clean finish = "restore done"; early death =
stops at "Redirects installed"; "filestream: fatal error" = benign upload
failure, checkpoint still landed). Produce one table: experiment → runs
(id · step · state) → status (worked / crashed / failed) → surviving artifact path.
```

## Recommended skill

`/data-inventory` ✅

## Engagement Instructions

```
$ python inventory/scripts/01_inventory.py
# regenerates the outputs/ size + file-count roll-up used by the table.
# Expect: 22G outputs, four W&B projects, and an experiment→status→artifact table
# in inventory/01-artifact-inventory.md (group_a_failure worked, lora 3 pairs +
# 3 empty stubs, cross_seed/cross_pair per the table).
$ test -f inventory/01-artifact-inventory.md && echo OK
```
