# poe_repair_min

## This project runs scientific experiments

Read `~/.claude/EXPERIMENT_CONVENTIONS.md` before drafting, checking, or updating any plan. It
holds how runs are classified, why the design and the verdict live in separate files, what a
failed bar does, and the four-step pass for coming back after time away.

This pointer is the opt-in. `populate-plans` emits review files for this repo's scopes and
`verify-plan` checks for them because this line is here.

## How the research gets done

[docs/RESEARCH_GUIDELINES.md](docs/RESEARCH_GUIDELINES.md) is the practice: write the results
section before running the experiments, name a run's cost and what it buys before it starts, look
at the data before designing on top of it, which visual skill to pick by what you are pointing at,
attack the work before a reviewer does, and kill work that is bleeding rather than only work that
failed. It names the skill to reach for at each moment.

The mechanics are elsewhere and not repeated there: `~/.claude/EXPERIMENT_CONVENTIONS.md` for what
a run may change, and `~/.claude/skills/WORKFLOWS.md` for the eighteen end-to-end skill chains
with their handoff files marked.

## The environment

Read [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) before drafting or checking any plan. It holds
the live-verified cluster facts: partitions and walltime limits, the `co3` absolute python path,
`/datasets` versus `/home-mscluster` and their sizes, the disk guard, the fp16 upcast rule, and
the fact that no system LaTeX exists here. Every plan names the facts it actually depends on in
its `## Environment Facts This Plan Depends On` field.

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
