# poe_repair_min

## This project runs scientific experiments

Read `~/.claude/EXPERIMENT_CONVENTIONS.md` before drafting, checking, or updating any plan. It
holds how runs are classified, why the design and the verdict live in separate files, what a
failed bar does, and the four-step pass for coming back after time away.

This pointer is the opt-in. `populate-plans` emits review files for this repo's scopes and
`verify-plan` checks for them because this line is here.

## How the research gets done

Read `~/.claude/RESEARCH_PRACTICE.md` before designing, running, or writing up anything. It is
the practice, shared across projects: write the results section before running the experiments,
name a run's cost and what it buys before it starts, look at the data before designing on top of
it, which visual skill to pick by what you are pointing at, the seven kinds of result and what
each may do to the paper, the six-step diagnosis procedure when a result contradicts the premise,
attack the work before a reviewer does, and kill work that is bleeding rather than only work that
failed. This pointer is the opt-in, same as the conventions pointer above.

[context/research-guidelines.md](context/research-guidelines.md) is the local half: where each of
those rules lands in this repo (paths, commands, the canary), and the specific mistakes this
project has already made. It does not repeat the practice.

The mechanics are elsewhere and not repeated in either: `~/.claude/EXPERIMENT_CONVENTIONS.md` for
what a run may change, and `~/.claude/skills/WORKFLOWS.md` for the eighteen end-to-end skill
chains with their handoff files marked.

## The environment

This project's real environment and architecture facts live in `environment/`. Read
`environment/overview.md` before drafting or checking any plan: partitions and walltime limits,
the `co3` absolute python path, `/datasets` versus `/home-mscluster` and their sizes, the disk
guard, the fp16 upcast rule, and the fact that no system LaTeX exists here. Then read the row in
`environment/00-INDEX.md` matching what you are about to touch. Every plan names the facts it
actually depends on in its `## Environment Facts This Plan Depends On` field.

If a task depends on an environment fact that folder does not cover, stop and ask. Do not infer
it, and do not proceed on a plausible assumption. An unanswered question costs one message; a
wrong assumption costs a run, plus the time to work out why it failed. Saying "I don't know how
this runs here" is a correct and useful answer.

## Two checkouts, one launch path

The repo is checked out twice: on the laptop at `/home/molef/PhD/poe_repair_min`, where code is
written, and on mscluster at `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min`, where it
runs. GitHub carries commits between them. From the laptop checkout, every cluster action goes
through `scripts/cluster.sh`:

- `scripts/cluster.sh run "sbatch scripts/<job>.sbatch"` commits nothing on its own. It pushes
  the current branch, fast-forwards the cluster checkout to it, then runs the command from the
  cluster repo root. Commit first; it refuses to run with uncommitted tracked changes.
- `scripts/cluster.sh jobs` shows `squeue` for this user; `scripts/cluster.sh sh "<cmd>"` runs a
  command there without syncing (tailing a log, reading an output).
- `scripts/cluster.sh pull <path>` rsyncs a git-ignored output folder back through mscluster84,
  skipping model weights.

If the cluster checkout cannot fast-forward (someone committed on the cluster side), the script
stops. Reconcile the two with `/reconcile-machines` rather than merging on the login node.

## Context

What this project is about in the real world (why PoE composition on SDXL fails, what a chimera,
an animal pair, the interaction term, and a compose rate are, where the pipeline's data comes
from, and what every field in a `summary.json` or a scorer output means) lives in `context/`.
Start at `context/00-INDEX.md`, which lists it as questions.

Before answering a question about what something means (a column, a code, an ID, a symbol like
`r_t` or `d_T`, a number in an output), read the file that owns that term. If the term is not
there, say so and ask rather than inferring a meaning from its name. A plausible wrong meaning is
worse than an admitted gap, because it gets built on.

## What the conventions look like in this repo

**The state check** is `python3 scripts/plan_pulse.py`. Report-only, four checks, about 7 seconds
over the whole tree. The session-start hook runs checks 1 and 2; the session-end hook runs 3 and 4.

**Bars in code, not prose.** `MIN_MEDIAN_RATIO` and `MIN_FRACTION_ABOVE_ONE` in the mechanism
re-probe scorer are the pattern to copy: the threshold sits in the source, so it cannot be adjusted
after seeing the answer without showing up in a diff.

**Harvest reads three execution modes, not one.** `squeue -u mmolefe` for the queue, and
`pgrep -af 'sweep|train'` on the session node, because biggpu allows one job per user, so the long
sweeps are started with `nohup` outside Slurm and Slurm is blind to them. Then the output count
against what the plan expected, and the tail of the log.

**Check where output landed, not just that it exists.** Large artifacts go to `/datasets` only,
because `/home-mscluster` hit 100% once and silently killed checkpointing. A script's disk guard
must check the filesystem it actually writes to.

**The tracker** is W&B, project `prime_lab/poe-repair-animals-compose`. W&B owns the numbers, the
plan tree owns the verdict. Never copy a curve into markdown; copy the verdict, the run id, and the
bar it was judged against.

## Runbook

How to do the recurring things here by hand (checking the plan tree's state, launching or
harvesting a run on the cluster) is in `runbook/`. Start at `runbook/00-INDEX.md`, which lists
them as questions.

Before writing out a sequence of commands or click paths for something done here regularly,
check whether a recipe already exists. If one does and it is now wrong, fix the recipe rather
than answering around it.

## Report

What this project found (each question, its verdict, the figure and the statistic that back it)
lives in `report/`. Start at `report/00-INDEX.md`, which lists every question with its verdict
and headline number. The older files beside it are pre-registrations, a log and an evidence
register, not findings.

Before asserting what this project showed, read the finding that owns the question. If no
finding exists, say the question is unanswered rather than reconstructing a verdict from chat
memory or from a results file alone. A misremembered verdict gets built on, and it is worse than
an admitted gap.

## Where to look first

The root [MASTER_PLAN.md](MASTER_PLAN.md) opens with `## Where things stand`, a snapshot of what
is running and what to do next, then carries the one order across every plan in every scope in
`## The paper: what has to land` and its three background lists. No scope keeps an order of its
own, and `## One plan, one table` states that rule.

`plans/` holds only work still to do: seven live scopes numbered `01` to `07` in the order a
reader meets them, plus `standing/`. The numbers are a reading order, not the step order, which
interleaves across scopes and lives in the root running order. Anything finished, parked or cold
has left for `artifacts/plans/`.

## Folders this repo adds to the top level

Beyond the global four (`plans/`, `artifacts/`, `runbook/`, `environment/`) and `data/`, this
repo keeps two more at the root: `context/` for what the project means in the real world (found
through `context/00-INDEX.md`), and `report/` for pre-registrations, instrument provenance and
results summaries per `~/.claude/EXPERIMENT_CONVENTIONS.md`. Both are staples here; neither is a
filing mistake.

## Six folders under `artifacts/` that are not one of the eight kinds

All six are declared here so no census reports them, and none may be moved.

`artifacts/_shared/cross_pair_pool_configs/` holds the pair, prompt and seed pool YAMLs that
define which cells an experiment runs over. It is config the code reads, not an artifact: the
path is hard-coded at `poe_repair/paths.py:150` as `GROUP_POOL_CONFIGS`, so moving it breaks
every runner. Config stays where the tooling expects it.

`artifacts/_quarantine/` is the holding pen the retrofit sweep created for files whose keep,
re-run or discard call had not been made. It is not junk: `results-archive/` inside it holds the
early findings the repository `README.md` cites nine times, so the front door depends on it. The
dispositions are owned by the parked `artifact-reconciliation` scope, which is where an emptying
pass would start, and until that scope resumes the pen stays as it is.

`artifacts/caches/` holds run bytes a script reads back rather than a person looks at: the
training cache and the manifold cache, hard-coded at `poe_repair/paths.py:153-154` as
`TRAINING_CACHE` and `MANIFOLD_CACHE`. It stays gitignored and outside the eight kinds because
nothing in it is evidence; it is memoization other runs depend on.

`artifacts/rung2-survive-noise/`, `artifacts/rung3-group-wise/` and `artifacts/rung4-scale/`
hold the per-pair, per-seed evidence cells for the causal-experiment ladder's three later rungs,
hard-coded at `poe_repair/paths.py:137,144-145` as `HELD_OUT_SEEDS`, `WITHIN_GROUP` and
`ALL_GROUPS`. Each is too large to fold into a single `artifacts/results/` grouping without
drowning it, so the rung stays its own top-level tree, gitignored like `results/`.

## Root files this repo adds to the list no check reports

`~/.claude/CLAUDE.md`'s "Files at the root that no check reports" table is the base list. This repo
extends it with:

| File | What it is for |
|---|---|
| `RENAMES.md` | the retrofit sweep's old-path-to-new-path table, owned by `retrofit-repo`/`tidy-repo` |
| `learning-map.md` | the learning weave's illustrated map; renders and process snapshots in `learning-diagrams/`, owned by `learning-pulse` |

## Learning journeys about this repo

One learning journey tours this codebase from outside it, under `~/goal-setting/learning/`. It
reads this repo and never writes to it.

**`sampler-correctors-for-composition/`** builds the sampling theory behind why product-of-experts
composition fails, and ends by measuring how much of `r_t` a Markov-chain corrector removes per
step on the cached trajectories. Before arguing in the paper about whether the compose-rate cliff
at steps 0 to 10 is a sampler artefact or a model artefact, read its master plan: the question is
posed there with the falsification criterion already written.
