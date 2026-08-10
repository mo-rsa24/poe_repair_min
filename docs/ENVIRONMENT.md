# poe_repair_min: Environment and Architecture

## Compute / runtime
HPC cluster, Slurm-scheduled. The working session (VSCode + Claude) runs ON a
compute node, currently mscluster85 (bigbatch: RTX 3090 24GB VRAM, 123GB RAM,
28 cores, verified via nvidia-smi 2026-08-04). The login node is never used
for work: Claude does not run there. So the real execution split is
"in-session on the current node" vs "submitted/interactive job on a bigger
GPU", not "login vs compute". Partitions (verified via sinfo, 2026-08-04):

| Partition | Nodes | Time limit | Notes |
|---|---|---|---|
| biggpu | mscluster[106-112] | 3 days | FIRST CHOICE for GPU jobs: ~100GB VRAM (user-stated; sinfo GRES reads null, so not machine-verified) |
| bigbatch | mscluster[42-89] | 3 days | fallback when no biggpu node is idle |
| batch | mscluster[120-219] | 1 day | short jobs only |
| stampede | mscluster[22-41] | 3 days | rarely used for this project |
| gpuexpress | mscluster116 | 1 hour | smoke tests only |

Python: mamba/miniforge env `co3` at
`/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`. This is the project
env; jobs must use this absolute path, never a bare `python`. A separate
`superdiff` env exists for SuperDiff-related work if its dependencies conflict.

## Access paths
- Repo and code: `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min`.
- Large artifacts (checkpoints, caches, outputs): `/datasets/mmolefe/poe_repair_min/`.
  These are separate filesystems with different quotas; there is no automatic
  mirroring between them.
- Cached residual trajectories (the r_t training targets, all four eps branches
  per step): `/datasets/mmolefe/poe_repair_min/outputs/training_cache/`
  (18 train + 58 heldout pairs, ~12 seeds, 50 steps each).
- W&B project for the animals program: `prime_lab/poe-repair-animals-compose`.

## Auth
- Cluster access: SSH (already established in-session; no per-job auth).
- W&B: API key present in `~/.netrc` and `WANDB_API_KEY`; no human step needed.
- Hugging Face model downloads (SDXL, SD 1.5/2.1): anonymous pulls have
  sufficed; if a gated model is ever needed, that is a human browser step.

## Execution model
"Running an experiment" means one of two paths, decided by this protocol, in
order:

1. `squeue -u mmolefe`: if an interactive node is already allocated on biggpu,
   run directly there (no sbatch).
2. Else `sinfo -p biggpu`: if a biggpu node is idle, target biggpu; otherwise
   target bigbatch.
3. Every job script carries a preflight block: df disk guard on the checkpoint
   target (abort at 90% full), `co3` python path check, `nvidia-smi` guard
   (abort in seconds if no GPU is visible).
4. Submit, then poll `squeue`/`sacct`. On failure, read the log and classify:
   OOM (reduce batch/resolution or move to biggpu), wrong env (fix the path),
   missing GPU or bad #SBATCH directives (fix and resubmit), node failure
   (resubmit elsewhere). Bounded retries; never silently loop.
5. In-session tier: cache-only analyses (SVD, SNR curves, language probes,
   scoring cached pngs) and light GPU inference run directly on the current
   session node (mscluster85, RTX 3090 24GB), no queue, while bigger jobs
   wait. Reference point: phase1_r8_100k TRAINING peaked at 22.95GB VRAM, so
   training-scale work goes to biggpu; SDXL inference-only sweeps generally
   fit the 3090, but check nvidia-smi for co-tenants before launching.

Skill wiring: `/run-experiment` drives GPU tasks; every experiment logs to W&B
including the qualitative Mono vs PoE vs LoRA triptych panels so `/analyze-run`
can sweep runs later; `/execute-plan-tree` may run tasks unattended, using each
plan's pre-registered falsification rules and /demonstrate checkpoints as its
stop conditions.

## Paper: where the LaTeX lives and how it is built
The manuscript is `paper/iclr/`, owned by the `plans/paper-iclr/` scope. Editing
and building it is a different loop from running experiments: no GPU, no queue,
no job.

**Where to make changes.** All prose goes in
`paper/iclr/iclr2027_conference.tex`, the single manuscript file. The other
files in that folder are the ICLR distribution and are not edited:
`iclr2027_conference.sty` (layout), `.bst` (bibliography style), `natbib.sty`,
`fancyhdr.sty`, and `math_commands.tex` (macro definitions, worth reading before
defining a new macro since it likely already exists). References go in
`iclr2027_conference.bib`. Figures are referenced, never copied in: they are
produced by `plans/interaction-term/` and `plans/animals-compose-transfer/`, and
`paper/iclr/README.md` holds the path rule.

**Building it, the VS Code route (the normal one).** LaTeX Workshop
(`james-yu.latex-workshop`, 10.15.2, installed server-side) is the extension.
`Ctrl+Shift+P` → `LaTeX Workshop: Build with recipe` → `tectonic`, then
`LaTeX Workshop: View LaTeX PDF` to see it.

The recipe has to be picked by hand because **the extension's default recipe is
`latexmk`, which does not exist on this cluster.** Plain "Build LaTeX project"
fails for that reason, not because anything is wrong with the document. Nine
recipes ship by default; `latexmk` is first, `tectonic` is last. There is no
`.vscode/settings.json` in this project, so the extension runs on stock defaults.
Setting `latex-workshop.latex.recipe.default` to `tectonic` would make plain
Build and build-on-save work; that is a one-line change nobody has made yet.

**Building it, the terminal route (what Claude and any script uses).**
```bash
cd paper/iclr
/home-mscluster/mmolefe/.local/bin/tectonic iclr2027_conference.tex
```
`tectonic` is a self-contained engine: it needs no TeX installation and fetches
packages and fonts on demand into `~/.cache/Tectonic`. The FIRST build on a node
therefore needs network access. Later builds are offline and fast.

**Reading the output.** The build writes `iclr2027_conference.pdf` next to the
`.tex`. `*.pdf` is gitignored (`.gitignore:31`), so the PDF is never committed:
you look at the paper by building it. Open it through the VS Code PDF viewer, or
from a laptop over the SSH tunnel the same way the LoRA Inspector is reached.

**Build noise that is not failure.** A successful build still prints underfull
`\vbox` warnings and `TeX rerun seems needed, but stopping at 6 passes`. Both are
normal here. Judge a build by whether the PDF was written, not by whether the
output was silent.

## Storage: size, retention, and what a `✓ verified` tag depends on
Both filesystems are NFS mounts, sized and used as follows (`df -h`, 2026-08-09):

| Mount | Size | Used | Free | What belongs there |
|---|---|---|---|---|
| `/datasets` | 201T | 24T (12%) | 178T | every checkpoint, cache, and sweep output |
| `/home-mscluster` | 73T | 26T (35%) | 48T | the repo, the conda envs, small results and logs |

There is no per-user `quota` command on these nodes, so a user's share is not
queryable: the only guard is the df check every job script runs before writing.

Neither filesystem is under version control and neither is snapshotted. Every
`✓ verified (...)` evidence tag in the plan tree points at a path on one of
them, so an artifact deleted or moved silently breaks the claim it backs. That is
what `plans/standing/artifact-reconciliation/plans/05-resweep-on-new-runs.md` exists to
catch, and why it is a standing recurring plan rather than a finished one.

A script writing large output must check the filesystem it actually writes to.
`scripts/mechanism_study/run_dose_sweep.sh` sets its output root under the repo
on `/home-mscluster` while its disk guard reads `df /datasets/mmolefe`, so 3.4GB
of sweep cells landed on the wrong mount with the guard reporting healthy.

## Known gaps / non-obvious constraints
- `/home-mscluster` hit 100% once and silently killed checkpointing mid-run.
  Checkpoints go to `/datasets/...` only, and every job keeps the df guard.
- The `hippo` single box takes a sequential or GNU-parallel loop over configs,
  NOT a Slurm array. Confirm partition/QOS before emitting any array.
- sinfo does not expose GPU GRES/VRAM on this cluster (reads null); node
  capability knowledge is tribal and recorded here, not queryable.
- Many nodes show down/drain at any given time; an idle-looking partition can
  still queue you. Check node STATE, not just partition availability.
- fp16 end-to-end: cached eps tensors are float16; analyses that stack many of
  them (the SVD) should upcast to float32 before accumulating.
- No system LaTeX at all: `pdflatex`, `xelatex`, `latexmk`, and `bibtex` are
  absent from PATH. Every LaTeX recipe except `tectonic` fails, including the
  extension's default. Any instruction, script, or CI step that assumes a normal
  TeX install is wrong here.
- `tectonic`'s first build on a node downloads packages, so it needs network
  access. A node without it fails on the download, not on the document.

## Provenance
Partitions, node lists, time limits, idle counts, and empty allocation state
verified live via `sinfo`/`squeue` on 2026-08-04. The biggpu 100GB VRAM figure
is user-stated (not machine-verifiable via sinfo here). Disk-guard and
hippo/array rules carried from EXPERIMENTS.md cluster notes (hard-won during
earlier runs). Cache contents verified by direct filesystem inspection and a
loaded residual file on 2026-08-04. Env list verified by ls of
`~/miniforge3/envs/` on 2026-08-04.

The LaTeX section was added 2026-08-05. Absence of `pdflatex`/`xelatex`/
`latexmk`/`bibtex` verified by `which`. The `tectonic` build verified by running
it on the stock template (73 KB PDF written, package and font downloads
observed). Extension version and its nine default recipes read from
`~/.vscode-server/extensions/james-yu.latex-workshop-10.15.2/package.json`;
`latexmk` first and `tectonic` last confirmed there. Absence of project or
machine settings confirmed by ls. The VS Code keystroke route is user-stated
(2026-08-05) and consistent with those defaults, but not machine-verified, since
the extension's command palette cannot be driven from a shell.
