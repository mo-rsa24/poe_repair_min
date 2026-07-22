# 🗂️ Two-Root Classified Sweep

## Description

Sweep both storage roots (`/datasets/mmolefe` and the repo) and classify every artifact owned by the three LoRA-pooling experiments into datasets / LoRA-checkpoints / caches / saved-results, grouped by owning experiment, with duplicates and orphans flagged.

## Purpose

The `01` inventory only covers the repo, but the heavy files (seed banks, the 22G training cache) live on `/datasets`. This task covers both roots at once: where each file physically lives, which experiment owns it, and where the same pair name shows up in both roots at different stages. That last point matters so we don't merge two different-stage files as if they were duplicates.

## Goal

`inventory/02-two-root-classified.md`: a path → type → group → owning-experiment → size → last-modified table for `lora`, `cross_seed_lora_pooling`, `cross_pair_lora_pooling` + their caches, with a duplicates section and an orphans section, and an explicit out-of-scope list for the other `/datasets` families.

## Tasks

- [x] ✅ Sweep both roots and classify per the prompt below.
- [x] ✅ Write `inventory/02-two-root-classified.md` with duplicates + orphans flagged.

Fully-qualified prompt (freeform, builds on `01`):

```
Sweep two roots: /datasets/mmolefe and /home-mscluster/mmolefe/Playground/PhD/poe_repair_min.
Classify every artifact into datasets / LoRA checkpoints / caches (training, eval,
group) / saved results, grouped by the experiment that produced it (lora,
cross_seed_lora_pooling, cross_pair_lora_pooling). One table: path → type → group
→ owning experiment → size → last-modified. Flag duplicates (same dir name across
roots — compare by size + mtime, not blind dedup: datasets copies are heavy seed
banks, repo copies are light eval samples) and orphans with no owning experiment.
List the other /datasets families (synthesizer, veracity*, idea*, mcmc*, medical
imaging checkpoints, etc.) as out of scope.
```

## Recommended skill

— custom; no skill fits (freeform two-root sweep, uses `du`/`ls`/`stat`).

## Engagement Instructions

```
$ test -f inventory/02-two-root-classified.md && echo OK
# Expect: a path→type→group→experiment→size→last-mod table; a "Duplicates flagged"
# section noting the cross_seed pair-name collision (heavy datasets banks vs light
# repo eval samples, different stages); an "Orphans" section (the 0-byte-looking
# lora alias dirs + manifold_cache); and an "Out of scope" list.
$ grep -c "Duplicates flagged\|Orphans\|Out of scope" inventory/02-two-root-classified.md   # expect 3
```
