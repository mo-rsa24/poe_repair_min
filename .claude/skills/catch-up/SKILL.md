---
name: catch-up
description: The standing pass for coming back to the experiments after time away, or after any run finishes. Harvests every run's real state, analyses anything unread, folds the findings into the plan files, re-syncs the tree, recomputes the running order, and ends with either the next task or an honest wait. Use whenever the user says "catch up", "where was I", "what happened while I was away", "any results to read", "what should I do next", or comes back to the project after a break. Idempotent: running it twice with nothing new does nothing.
---

# Catch up

The failure this fixes: a run finishes at 3am, produces a real result, and nothing
reads it. The plan tree keeps saying a job is in flight for days while the answer
sits in a log file. Nothing is broken, and nothing is moving.

Six steps, in order. Do not reorder them. Step 1 is what makes the rest true.

## 1. Harvest: read the real state before believing any plan file

```bash
python3 scripts/plan_pulse.py
```

That reports stale task lines, unharvested output, orphan markdown, and narration
debris. It reads only. Then fill in what it cannot see:

```bash
squeue -u mmolefe                       # Slurm jobs
pgrep -af 'sweep|train'                 # bare processes; biggpu allows one job
                                        # per user, so long work often runs
                                        # outside Slurm and squeue is blind to it
hostname                                # which node this session is on
```

Classify every run in every live scope's `## Runs` table into one of five states:
not started, running, finished but unharvested, dead, done.

**The expensive state is the fourth one.** A run whose result exists and which
nothing has read. That is the state this whole pass exists to find.

For a run that looks finished, check three things before believing it: the output
count against what the plan expected, the tail of the log for a verdict line, and
whether the output landed on the filesystem the plan said it would.

## 2. Analyse what is unread

One pass per unharvested run.

- W&B training runs: the `training-analyst` agent already does this, idempotently.
- Everything else: `/analyze-run`, or read the scorer's own output file.

Do not re-derive a number the run already computed. Read what it wrote.

## 3. Fold the findings into the task lines

This is the step that was missing, and it is the point of the pass.

For each analysed run, edit its plan file:

- Tick the task with the marker the result earns: `✅` passed, `❌` failed the
  pre-registered bar, `🟡` inconclusive by that bar, `◑` partly done.
- Write the finding on the line: the numbers, against the bar they were judged by.
- Add or update the run's row in the plan's `## Runs` table.
- On `❌`, follow the rule in `CLAUDE.md`: the plan **finishes**, and one
  follow-on plan file is written and linked. Do not leave it open. Do not loosen
  the bar.
- On `🟡`, name the cause and the next action on the line.

Full provenance (`run id, cells, metric against comparison, commit`) is required
only where the number will appear in the manuscript.

## 4. Sync the tree

```
/sync-plan-tree --clean
```

Roll child statuses up into each `## Sub-Scopes` entry, absorb prose debris, and
report orphan markdown. It now has real evidence to work from, which it did not
have at step 1.

## 5. Recompute the running order

Update the `## Running order` table in the root `MASTER_PLAN.md`: statuses from
step 3, and the `Waits on` column against what is actually queued. A row whose
dependency just landed becomes available; a row waiting on a dead run does not.

## 6. End with one line

Either the next task, or an honest wait. Never both, never neither.

- **Do this next:** the single file to open, and the first thing to do in it.
  Write it into the `## Do this next` block at the top of the root master plan.
- **Wait:** what is in flight, on which node or job, what it will produce, and
  when it should land. Then name the best thing to do while waiting, which on
  this cluster is usually a cache-only analysis needing no GPU and no queue.

## What not to do

- Do not skip step 1 and start from the plan files. They are the thing being
  checked, not the evidence.
- Do not tick a task from a plan file's own claim. Tick it from the output.
- Do not let a good number from an idea-trying run move a hypothesis. It goes to
  `PARKING_LOT.md`. Only a claim-testing run that fails its bar moves a plan.
- Do not run the analysis again on a run already folded in. This pass is
  idempotent by design; a second run with nothing new should change nothing.
- Do not end without either a next task or a wait.
