# Storage layout

This repo's bulk results do **not** live in the working tree. They live under
`/datasets/mmolefe/poe_repair_min/outputs/` (NFS, large/cheap), and `outputs/`
in the working tree contains **symlinks** that point there.

See `/datasets/mmolefe/README.md` for the cross-project convention.

## Why

`/home-mscluster` is small and frequently near 100% full. `/datasets` has ~96 TB.
Results were dominated by `outputs/synthesizer/cache/pairs_*.pt` (63 GB encoded
pair cache, recreatable but slow). Moving everything off `/home-mscluster`
prevents disk-full incidents from blocking unrelated work.

## What is where

| Tree                                      | Location                                                       | Owner / notes                                 |
| ----------------------------------------- | -------------------------------------------------------------- | --------------------------------------------- |
| Code, configs, docs                       | `~/Playground/PhD/poe_repair_min/` (this repo)                 | git-tracked                                   |
| `outputs/<name>/` symlinks                | this repo                                                      | gitignored; recreated by `scripts/rehydrate_outputs.sh` |
| Actual experiment outputs                 | `/datasets/mmolefe/poe_repair_min/outputs/<name>/`             | not git-tracked                               |
| Live training runs (not yet migrated)     | `outputs/group_a/`, `outputs/m5_lora_sdxl/` (real dirs in repo) | migrate after runs finish                     |

In-code paths like `REPO_ROOT / "outputs/synthesizer/cache/..."` resolve through
the symlink transparently — no source changes were needed for the migration.

## Rehydrating on a fresh clone

After cloning the repo on a machine where `/datasets/mmolefe/poe_repair_min/`
already exists, run:

```bash
scripts/rehydrate_outputs.sh
```

This (re)creates the symlinks under `outputs/` pointing at the right targets.
It is idempotent and safe to re-run.

## Adding a new output directory

If a new experiment writes to `outputs/new_thing/`:

```bash
# 1. Create the real directory under /datasets
mkdir -p /datasets/mmolefe/poe_repair_min/outputs/new_thing

# 2. Symlink into the repo
ln -s /datasets/mmolefe/poe_repair_min/outputs/new_thing outputs/new_thing
```

Then run the experiment as usual. The symlink is transparent to all code.

## Migrating an existing real directory

If a directory got created as a real dir (e.g. a live training run that
finished), migrate it:

```bash
NAME=group_a   # example
# 1. Ensure no live writers
lsof +D outputs/$NAME 2>/dev/null

# 2. Copy with verification
rsync -aH outputs/$NAME/ /datasets/mmolefe/poe_repair_min/outputs/$NAME/

# 3. Spot-check
diff -rq outputs/$NAME/ /datasets/mmolefe/poe_repair_min/outputs/$NAME/ | head

# 4. Replace with symlink
rm -rf outputs/$NAME
ln -s /datasets/mmolefe/poe_repair_min/outputs/$NAME outputs/$NAME
```

## Things to remember

- `/datasets` is NFS — bandwidth is shared with other users. Avoid concurrent
  multi-GB rsyncs.
- `du -sh outputs/` follows symlinks if you pass `-L`; without `-L` you'll see
  just the size of the symlinks themselves (a few hundred bytes).
- The `pairs_*.pt` synthesizer cache (63 GB) was last touched 2026-05-05 and is
  recreatable from raw pair data via `scripts/train_synthesizer.sh`. Don't be
  afraid to delete it from `/datasets` if /datasets ever gets tight — keep the
  recipe, not the cache.
