# poe_repair_min: Environment and Architecture

HPC cluster, Slurm-scheduled, two separate filesystems with no automatic mirroring between
them, and a paper build that needs one specific LaTeX engine because nothing else is installed.
The thing a plan gets wrong most often here is treating "submit a job" as one action: it is a
several-step decision between an idle node, a shared half-used node reached over SSH, and a
disk guard that has to check the filesystem the script actually writes to, not an assumed one.

## Table of contents

- [Compute / runtime](#compute--runtime)
- [Access paths](#access-paths)
- [Auth](#auth)
- [Execution model](#execution-model)
- [Known gaps / non-obvious constraints](#known-gaps--non-obvious-constraints)
- [Provenance](#provenance)
- [Paper: LaTeX build](#paper-latex-build)

## Compute / runtime

Navigation: 📋 [TOC](#table-of-contents) | [Next](#access-paths) ➡️

The working session (VS Code + Claude) runs ON a compute node, currently `mscluster85`
(`bigbatch`: RTX 3090 24GB VRAM, 123GB RAM, 28 cores). The login node is never used for work.
Python is the `co3` mamba environment at
`/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`; every job script must use this
absolute path, never a bare `python`.

Full partition table, per-node GPU models, and the environment list are in
[hpc/nodes.md](hpc/nodes.md).

## Access paths

Navigation: ⬅️ [Compute / runtime](#compute--runtime) | 📋 [TOC](#table-of-contents) | [Next](#auth) ➡️

Repo and code on `/home-mscluster`; every checkpoint, cache, and sweep output on
`/datasets/mmolefe/poe_repair_min/`, a separate filesystem with its own quota and no automatic
mirroring to the repo side. W&B project: `prime_lab/poe-repair-animals-compose`.

Full paths, current `df` sizes, and the disk-guard rule that a script's guard must check the
filesystem it actually writes to are in [storage.md](storage.md).

## Auth

Navigation: ⬅️ [Access paths](#access-paths) | 📋 [TOC](#table-of-contents) | [Next](#execution-model) ➡️

- Cluster access: SSH, already established in-session; no per-job auth.
- W&B: API key present in `~/.netrc` (confirmed present, not read, live 2026-08-24) and
  `WANDB_API_KEY` (confirmed set in this session's shell, 2026-08-24); no human step needed.
- Hugging Face model downloads (SDXL, SD 1.5/2.1): anonymous pulls have sufficed; if a gated
  model is ever needed, that is a human browser step.

## Execution model

Navigation: ⬅️ [Auth](#auth) | 📋 [TOC](#table-of-contents) | [Next](#known-gaps--non-obvious-constraints) ➡️

"Running an experiment" is decided by a fixed order of checks: an already-allocated interactive
node first, then an idle `biggpu` node, then a shared half-used `biggpu` node reached over SSH
with a pinned free GPU device (invisible to `squeue`, harvested with `pgrep` instead), then
`bigbatch` as the fallback. Every job script carries a preflight block (disk guard, `co3` path
check, GPU-visibility guard) before it submits.

The full six-step protocol, the shared-device safety rules, and the SSH absolute-path rule
(the fix for a launch failure that recurred three times on 2026-08-19) are in
[hpc/execution-protocol.md](hpc/execution-protocol.md).

## Known gaps / non-obvious constraints

Navigation: ⬅️ [Execution model](#execution-model) | 📋 [TOC](#table-of-contents) | [Next](#provenance) ➡️

- `/home-mscluster` hit 100% once and silently killed checkpointing mid-run. Checkpoints go to
  `/datasets/...` only, and every job keeps the `df` guard (detail: [storage.md](storage.md)).
- The `hippo` single box takes a sequential or GNU-parallel loop over configs, NOT a Slurm
  array. Confirm partition and QOS before emitting any array.
- `sinfo` does not expose GPU GRES/VRAM on this cluster (reads `(null)`, reconfirmed live
  2026-08-24); node capability knowledge is tribal and recorded in
  [hpc/nodes.md](hpc/nodes.md), not queryable.
- Many nodes show down or drain at any given time; an idle-looking partition can still queue
  you. Check node STATE, not just partition availability (detail: [hpc/nodes.md](hpc/nodes.md)).
- fp16 end-to-end: cached eps tensors are float16; analyses that stack many of them (the SVD)
  should upcast to float32 before accumulating.
- No system LaTeX at all: `pdflatex`, `xelatex`, `latexmk`, and `bibtex` are absent from PATH
  (reconfirmed live 2026-08-24). Every LaTeX recipe except `tectonic` fails, including the
  extension's default. Detail, and an open question about whether a newly-added
  `.vscode/settings.json` changes this: [paper.md](paper.md).
- `tectonic`'s first build on a node downloads packages, so it needs network access. A node
  without it fails on the download, not on the document.

## Provenance

Navigation: ⬅️ [Known gaps](#known-gaps--non-obvious-constraints) | 📋 [TOC](#table-of-contents) | [Next](#paper-latex-build) ➡️

When and how these facts were established: live verification, a specific source read, or a
person who confirmed it. The table is in [provenance.md](provenance.md); it is not duplicated
here because it grows one row per verification and nobody reads it while planning.

## Paper: LaTeX build

Navigation: ⬅️ [Provenance](#provenance) | 📋 [TOC](#table-of-contents)

The manuscript is `paper/iclr/iclr2027_conference.tex`, built with `tectonic` (the only LaTeX
engine on this cluster) either from the terminal or through VS Code's LaTeX Workshop extension
with the recipe manually set to `tectonic`. A `.vscode/settings.json` was added to the repo on
2026-08-18 defining a custom `tectonic` recipe; whether that makes the plain "Build LaTeX
project" command work without manually picking the recipe is unverified (see "Still open" in
[00-INDEX.md](00-INDEX.md)).

Full build commands, the extension's recipe list, and the git-blame on `.vscode/settings.json`
are in [paper.md](paper.md).
