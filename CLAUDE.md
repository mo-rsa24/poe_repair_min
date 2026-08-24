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

[docs/RESEARCH_GUIDELINES.md](docs/RESEARCH_GUIDELINES.md) is the local half: where each of those
rules lands in this repo (paths, commands, the canary), and the specific mistakes this project has
already made. It does not repeat the practice.

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

## Where to look first

The root [MASTER_PLAN.md](MASTER_PLAN.md) carries `## Do this next` above a flat `## Running order`
table covering every plan in every scope. No scope keeps an order of its own.
