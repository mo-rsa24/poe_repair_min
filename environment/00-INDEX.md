# 🗺️ Environment: poe_repair_min

What is true about the system this work runs on. Migrated 2026-08-24 from the single
`docs/ENVIRONMENT.md` file (the old shape) into this folder, as part of stage 2 of the
`/retrofit-repo` sweep ledgered in `RETROFIT.md`.

Read [overview.md](overview.md) first, always. Then the row below matching what you are about
to touch.

## Before I...

| The moment | Read this | Because |
|---|---|---|
| submit a job, or write an `sbatch` header | [hpc/nodes.md](hpc/nodes.md) | partitions differ in GPU model and time limit, and `sinfo` cannot tell you which node has a GPU: asking for the wrong one queues silently |
| decide whether to submit or SSH into a shared device | [hpc/execution-protocol.md](hpc/execution-protocol.md) | "run an experiment" is a six-step decision here, not one action, and the shared-device path has mandatory safety rules |
| launch anything over SSH onto a node this session is not on | [hpc/execution-protocol.md](hpc/execution-protocol.md) | a relative path in the launch line resolves against `$HOME`, not the repo, and the failure is silent (`poe-launch-001`) |
| read or write project data, or add a disk guard to a job script | [storage.md](storage.md) | two filesystems, different quotas, no automatic mirroring, and a guard that checks the wrong mount has already caused a real loss (`poe-disk-001`) |
| build or edit the paper | [paper.md](paper.md) | no system LaTeX exists here; the extension's default recipe fails outright, and a recent config change to fix that is unverified |
| hit an error that looks like it has happened before | [known-failures.md](known-failures.md) | seven catalogued patterns with symptom, root cause, and solution, so a failure is not re-diagnosed from scratch |
| want to know how sure a fact in this folder actually is | [provenance.md](provenance.md) | every fact is marked verified, stated, or inferred, and several were re-checked live on 2026-08-24 |

## The systems

| Folder | What you operate there | Leaves |
|---|---|---|
| [hpc/](hpc/) | the cluster: partitions, nodes, GPUs, the launch protocol | 2 |
| [storage.md](storage.md) | the two filesystems, sizes, and the disk-guard rule | 1 (flat, no second leaf yet) |
| [paper.md](paper.md) | the LaTeX build for `paper/iclr/` | 1 (flat, no second leaf yet) |

How every fact here was established is in [provenance.md](provenance.md). The project failure
catalog is [known-failures.md](known-failures.md).

## Still open

- **Whether `.vscode/settings.json` (added 2026-08-18, commit `2c4d0b6`) makes VS Code's plain
  "Build LaTeX project" work without manually picking the `tectonic` recipe.** Deliberately
  untested this sitting, because testing means running a build, which writes a PDF outside this
  folder's scope. See [paper.md](paper.md).
- **`co3_bw`'s exact difference from `co3`.** Confirmed to exist (`ls`, 2026-08-24) and referenced
  by at least one launcher, but not probed for what specifically it changes. See
  [hpc/nodes.md](hpc/nodes.md).
- **Whether every job script's disk guard now reads the same resolved root as its output path,**
  after the `paths.resolve()` commits. Not re-audited this sitting. See `poe-disk-001` in
  [known-failures.md](known-failures.md).
- **The wider discovery-route wiring.** The project `CLAUDE.md` and the root `MASTER_PLAN.md`
  now point at `environment/overview.md` and `environment/00-INDEX.md`. Roughly two dozen other
  files still name `docs/ENVIRONMENT.md`: about ten sub-scope `MASTER_PLAN.md` files under
  `plans/` (each with its own `## Environment Context` pointer) and about a dozen plan and review
  files that link into specific anchors of the old file. Repointing all of them is out of scope
  for this sitting; see the audit output for the full list. This is flagged for whichever stage
  of the retrofit sweep handles repo-wide path rewrites (the plan's stage table names stage 4,
  `tidy-repo`, as "the bulk of it, both filesystems," which is the pass already doing repo-wide
  renames).
- **No `reference`-type memory entry names the `environment/` folder.** The closest existing
  entry, `hpc-and-execution-conventions` (`type: project`), duplicates several facts inline and
  points at the old `docs/ENVIRONMENT.md` path. Per `ENVIRONMENT_CONTEXT_FORMAT.md`'s discovery
  routes, a `reference`-type entry naming this folder's path and purpose is the route that
  survives working outside the repo; writing memory entries is the orchestrating session's job,
  not this skill's, so this is reported rather than created.
- **Five of the six `overview.md` sections' internal cross-references were narrowed to "two or
  three facts plus a link"** per the format's own rule, rather than keeping every sentence of
  the original `docs/ENVIRONMENT.md` inline. This is a real content move, not a pure copy: check
  a leaf's wording against `docs/ENVIRONMENT.md` (still present, unmodified, for this comparison)
  if a fact seems to be missing rather than moved.
