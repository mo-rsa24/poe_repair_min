# poe_repair_min: how experiments are planned, run, and recorded

Read [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) before drafting or checking any plan. It holds
the live-verified cluster facts: partitions and walltime limits, the `co3` absolute python path,
`/datasets` versus `/home-mscluster`, the disk guard, the fp16 upcast rule. Every plan names the
facts it actually depends on in its `## Environment Facts This Plan Depends On` field.

## Every run declares what it is allowed to change

Decide this before the run starts, not after the number arrives. The kind decides what the
result may touch.

| Kind | May change the claim? | Where the result lands | What it answers to |
|---|---|---|---|
| Tests the claim | Yes, and only by failing its bar | The claim, the paper | The experiment design and the bar written before it ran |
| Tries an idea | No | `PARKING_LOT.md`, or a proposal for a new claim-testing run | The literature: is this method sound |
| Makes a figure | No | The paper's figures | The already-settled result it draws |
| Checks an old number | No, it can only confirm or break one | The evidence tag on the existing claim | The original run's config and seed |
| Produces a competitor | No, and it freezes once it lands | The comparison column | What the field actually reports |

There is no separate kind for tuning to push a number up. That is trying an idea. Naming it
separately makes it feel like progress toward the claim when it is not.

## Every run belongs to exactly one plan, and the plan lists it

Each plan file carries a `## Runs` table: the run's identity (W&B id, job id, or log path),
its kind from the table above, the commit it was launched at, where its output landed, and its
state. A run that belongs to no plan is the run that quietly moves the goalposts.

This block is what makes the other three things possible. Harvest has a list to check against
instead of a guess. A number is reproducible in six months because the plan says which run made
it and at what commit. And an unowned run becomes visible instead of invisible.

## Only a claim-testing run may move the plan

And only by failing a bar that was written before it ran.

Bars live in code wherever a scorer exists, not in prose. `MIN_MEDIAN_RATIO` and
`MIN_FRACTION_ABOVE_ONE` in the mechanism re-probe scorer are the pattern to copy: the threshold
is in the source, so it cannot be adjusted after seeing the answer without showing up in a diff.

A striking number from a run that was trying an idea does not rewrite the hypothesis. It goes in
`PARKING_LOT.md`. What it earns is the right to propose an experiment.

## A failed bar finishes the plan and writes the next one

Failing is a completed result, not an interruption. Tick the task, record the numbers against the
bar, and link one follow-on plan file:

```
- [x] ❌ full sweep: bar not met. Median 1.09 against the 1.2 bar, 61% of rows above 1
      against the 75% bar. Follow-on: plans/interaction-term/plans/12-why-the-bar-missed.md
```

Do not leave the plan open. Do not loosen the bar to rescue the result. `/frame-hypothesis` runs
inside the follow-on plan, never inside the plan that just finished.

A result that is neither pass nor fail by the pre-registered rule is ticked `🟡` with the cause
named and the next action named. It is not left as `⚠️`.

## Harvest before sync

A plan file is not evidence about a run. Before updating any status, read the real state:

- `squeue -u mmolefe` for queued and running Slurm jobs.
- `pgrep -af <script>` on the session node, because biggpu allows one job per user, so long work
  often runs outside Slurm and Slurm cannot see it.
- The output directory cell count against the count the plan expects.
- The tail of the run log, which is where a finished sweep prints its verdict.

Classify every run as not started, running, finished but unharvested, dead, or done. A run in the
fourth state is the expensive one: the result exists and nothing has read it.

## Coming back after time away

Four steps, and only the last one writes. There is no skill that wraps them, on purpose: the
decision in step 3 is the one that must not be automated, because a result that folds itself into
the tree is a result that moved the goalposts without being asked.

1. `python3 scripts/plan_pulse.py` says what the machine knows and the plan files do not: a task
   line claiming a run is in flight when nothing is, output newer than the plan that owns it,
   markdown no task names, narration debris. It reads only. The session-start hook already runs it.
2. `/orient --progress` retells the story and reads what landed against it, then says wait, pick
   something up, or start something new.
3. You decide what each result is allowed to change, by the run-kind table above.
4. `/integrate-plans` writes new or changed tasks. `/sync-plan-tree --clean` writes statuses and
   tidies prose.

End every such pass with one line: either the next task, or an honest wait naming what is in
flight and what it will produce. Never both, and never neither.

## Evidence tags carry provenance for numbers that reach the paper

Any task line producing a number that will appear in the manuscript carries the run id, the step
or cell count, the metric against its comparison, and the commit the run was launched at:

```
✓ verified (run wandb-8p1spi5b, 480 cells, oracle AUC 0.422 against control 0.059, commit a21ac8b)
```

The launch commit is stamped by the run harness, not typed in by hand, because the repo moves
after a run starts. Task lines that do not feed the paper keep the shorter artifact-path form
already in use across the tree.

## Where plans live

One scope per claim, with the paper reading and the experiment design as its first plan files.
Do not split reading, designing and running into separate scopes: the three are not independent
and the split triples the tree.

Paper reading that belongs to no single claim goes in its own standing scope, the same recurring
shape as `plans/artifact-reconciliation/`.

## Keeping the tree readable

Every scope root carries `## Do this next` naming the single file to open, above a
`## Running order` table covering every plan across all levels. Plan numbering is per folder and
is not the order.

Run `sync-plan-tree --clean` when leaving a scope. Every markdown in a scope must be named by at
least one task, or it should not exist.
