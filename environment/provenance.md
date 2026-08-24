# Provenance

Navigation: 📋 [Index](00-INDEX.md) | [Overview](overview.md#provenance)

One row per fact, wherever in this folder the fact lives. `verified` means probed live on the
date given and the result seen. `stated` means a person said it and it was not probed.
`inferred` means it was read out of the repository or a document and confirmed by neither a
person nor a run this sitting.

| Fact | Where it lives | Mark | How it was established |
|---|---|---|---|
| Session node is `mscluster85`, `bigbatch`, RTX 3090 24GB, 123GB RAM, 28 cores | `hpc/nodes.md` | verified | `nvidia-smi` on 2026-08-04 |
| Session node's current partition and state (`bigbatch`, `alloc`) | `hpc/nodes.md` | verified | `sinfo -N -o "%N %P %t"`, live, 2026-08-24 |
| Partition table (nodes, time limits) for `biggpu`, `bigbatch`, `batch`, `stampede`, `gpuexpress` | `hpc/nodes.md` | verified | `sinfo -o "%P %a %l %D"`, live, 2026-08-24 (re-confirms a 2026-08-04 check) |
| `sinfo` GRES column reads `(null)` for every partition | `hpc/nodes.md`, `overview.md` | verified | `sinfo -o "%P %N %G"`, live, 2026-08-24 |
| `mscluster106`: 2x Quadro RTX 8000, 49GB each | `hpc/nodes.md` | verified | `nvidia-smi` over SSH, 2026-08-19 |
| `mscluster110`: Blackwell-class | `hpc/nodes.md` | stated | not machine-verified |
| `co3` python resolves to `python3.10` at the stated absolute path | `hpc/nodes.md`, `overview.md` | verified | `ls -la`, live, 2026-08-24 |
| Env list under `~/miniforge3/envs/` (13 environments including `co3`, `co3_bw`, `superdiff`) | `hpc/nodes.md` | verified | `ls`, live, 2026-08-24 |
| `co3_bw`'s exact purpose versus `co3` | `hpc/nodes.md` | inferred | seen only as a launcher reference in one plan file; not probed this sitting |
| `superdiff` exists for SuperDiff work if `co3` dependencies conflict | `hpc/nodes.md` | stated | not re-verified this sitting |
| Shared-device path and its safety rules | `hpc/execution-protocol.md` | verified | live over SSH on `mscluster106`, 2026-08-19: GPU 1 at 1MiB/0% while GPU 0 carried another user's 8GB process; a torch matmul from `co3_bw` on GPU 1 succeeded without touching GPU 0 |
| SSH launch line must use absolute paths (`poe-launch-001`) | `hpc/execution-protocol.md`, `known-failures.md` | verified | reproduced three times, 2026-08-19, on a smoke-run launch on `mscluster106` |
| `/datasets` and `/home-mscluster` sizes, used, free | `storage.md` | verified | `df -h /datasets /home-mscluster`, live, 2026-08-24 (supersedes a 2026-08-09 `df -h` reading kept alongside it for the delta) |
| No per-user `quota` command on these nodes | `storage.md` | verified | `which quota`, live, 2026-08-24 |
| `/home-mscluster` hit 100% once and killed checkpointing | `storage.md`, `overview.md` | stated | hard-won operational history, carried from the project's earlier cluster notes; no run reproduces this on demand |
| Disk guard on `run_dose_sweep.sh` checked the wrong mount, 3.4GB landed on `/home-mscluster` | `storage.md`, `known-failures.md` | stated | recorded in the project's own `CLAUDE.md`; not re-run this sitting |
| Cached residual trajectory contents (18 train + 58 heldout pairs, ~12 seeds, 50 steps) | `storage.md` | verified | direct filesystem inspection and a loaded residual file, 2026-08-04 |
| W&B API key present in `~/.netrc`; `WANDB_API_KEY` set in-session | `overview.md` | verified | file existence checked (not read) and env var presence checked, live, 2026-08-24 |
| `pdflatex`, `xelatex`, `latexmk`, `bibtex` absent from PATH | `paper.md`, `overview.md` | verified | `which` for each, live, 2026-08-24 (re-confirms a 2026-08-05 check) |
| `tectonic` present and executable at the stated absolute path | `paper.md` | verified | `ls -la` and `which`, live, 2026-08-24 |
| `tectonic` builds the stock template successfully | `paper.md` | stated | carried from the 2026-08-05 note; not rebuilt this sitting (a build would write a PDF outside this folder's scope) |
| LaTeX Workshop extension present, version `10.15.2`, server-side | `paper.md` | verified | `ls ~/.vscode-server/extensions/`, live, 2026-08-24 (re-confirms a 2026-08-05 check) |
| Extension's 9 default recipes, `latexmk` first, `tectonic` last | `paper.md` | inferred | read from `package.json` on 2026-08-05; not re-read this sitting |
| `.vscode/settings.json` exists, tracked, defines a custom `tectonic` recipe, no `recipe.default` key | `paper.md`, `overview.md` | verified | file read directly, live, 2026-08-24; `git log` shows it added in commit `2c4d0b6` |
| Whether the new `.vscode/settings.json` makes plain "Build LaTeX project" work without picking a recipe | `paper.md` | **open** | deliberately not tested this sitting; testing means running a build, which writes outside this folder's scope |
| VS Code keystroke build route (`Ctrl+Shift+P` -> Build with recipe -> tectonic) | `paper.md` | stated | user-stated 2026-08-05; the command palette cannot be driven from a shell, so this has never been machine-verified |
| `*.pdf` gitignored at `.gitignore:31` | `paper.md` | inferred | read from the file on 2026-08-05; not re-read this sitting |
| Fraction-of-distance-reached, direction-cosine, and scorer failure patterns (`poe-score-001/002`, `poe-lora-001/002`) | `known-failures.md` | stated | carried verbatim from `docs/EXPERIMENT_ERROR_CATALOG.md`, first discovered at step-09; not re-run this sitting |
