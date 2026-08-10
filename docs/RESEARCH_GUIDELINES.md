# How this research gets done

The practice, not the mechanics. The mechanics are in `~/.claude/EXPERIMENT_CONVENTIONS.md`
(what a run may change, design against verdict, the bar) and `~/.claude/skills/WORKFLOWS.md`
(eighteen end-to-end skill chains with their handoff files marked). This document is what to do
when, and which skill to reach for at that moment.

Read with the root [MASTER_PLAN.md](../MASTER_PLAN.md), whose four lists say what is happening
now, what is happening in the background, and what to do next.

## Write the results section before running the experiments

The blank results section is the most useful planning document available, and it is cheaper than
any experiment. Empty tables. Captions saying what each figure will claim. Numbers replaced by
`XX`.

Once it exists, "what should I run next" becomes "what is still blank", and every run has an
obvious answer to "what does this buy". Written afterwards instead, the paper is shaped by
whatever the experiments happened to produce, which is how a program of work turns into a
collection of results.

The skeleton is `plans/closing-the-compositional-gap/plans/paper-iclr/plans/05-results-skeleton.md`. Move it early, keep it ugly,
and fill it in as answers arrive.

## Every run names its cost and what it buys, before it starts

Two lines in the design plan:

- **Cost:** the cell count times the seconds per cell, in hours. `480 cells at 50s, about 6h.`
- **Buys:** the review question it answers and the figure it feeds. `Answers question 1, feeds
  figure 3.`

A run that cannot name what it buys does not launch. This is the cheapest check in this document
and it is the one that protects GPU-days, which are the actual constraint on a thesis.

`/experiment-planner` produces the matrix and the pre-registered protocol. `/frame-hypothesis`
comes first when the question is not yet falsifiable, and it will route you to background study
instead if the mechanism is not predictable even roughly.

## Look at the data before designing anything on top of it

`/visualize-data-samples` before a scorer, a metric, or a training run. A threshold chosen from a
number and never checked against a picture is a threshold that fails silently, which is what
happened to the composition scorer: it read three instances on an image of two cats.

Pair every measure with something to look at. `/pair-figure` for one claim,
`/evidence-ladder` when three or more claims build an argument.

## When you cannot picture something, pick by what you are pointing at

- **Code you have lost hold of** → `/deep-learning-scene`. Traces the real module tree, drills to
  source lines.
- **A maths object you cannot feel** → `/polish` to file it properly, then `/math-scene` to drag
  the point and watch everything move.
- **An experiment's shape** → `/experiment-atlas`. Reads the scope and computes the real cell
  counts.
- **A process, where the drawing must be exactly right** → `/visualize-procedure`. Emits Mermaid
  or TikZ, reproducible.
- **A document you already wrote** → `/picture-speak` to click through its claims, or
  `/illuminate` to read it as one piece.

Do not run two of these on the same thing hoping one lands. Pick on the input.

## Read the literature in the background, continuously

Reading is a pool, not a step. `/paper-scout` to find, `/unpack-paper` to read one properly,
`/drip --paper` when you want it paced across a week, `/master-paper` only when you intend to
reimplement. Every read lands as a row in the register at
`plans/standing/literature/plans/01-reading-register.md`.

The rule that makes this pay off: **every run that tries an idea names the paper it came from.**
An idea with no source is a hunch wearing a method's clothes.

## Let a result be unclear, and say which kind of unclear

Three outcomes, not two. A result that passes its bar, a result that fails it, and a result that
the pre-registered rule cannot decide. The third is the most common and the one people fail to
record.

Failing finishes the plan and opens one follow-on. Unclear names the cause and the next action,
usually a fix to the instrument rather than to the hypothesis. Neither is a reason to move the
bar.

## Seven kinds of result, and what each one is allowed to do to the paper

Decide which kind a result is before writing a word around it. The kind decides where it goes,
what its caption may claim, and what happens next.

| Kind | Enters the paper? | Where | The caption may claim | Next |
|---|---|---|---|---|
| Passes its bar, central | Yes | Results, a main figure | Exactly the bar's sentence, with the numbers | Fill the figure slot, add the register row |
| Passes, supporting | Yes | Appendix, or folded into a table | The narrow thing it showed, nothing wider | Register row only if the paper cites it |
| Suspiciously good | Not yet | Nowhere until cleared | Nothing | Treat as contamination until proven otherwise: the λ=0-style canary, what each flag actually selected (print the count), leakage between train and held-out. Only a clean check makes it kind 1 or 2 |
| Fails in a bounded regime | Yes, and this strengthens the paper | The limitation, stated plainly where the claim is made | Where it holds AND where it does not, as one sentence | Review answer ❌ with the boundary named; a follow-on plan only if the boundary threatens the main claim |
| Fails centrally, contradicts the premise | Nothing gets written around it | Nowhere, yet | Nothing | Stop. The diagnosis procedure below, before any prose |
| Unclear by the pre-registered rule | Not until typed | Nowhere, yet | Nothing | 🟡 with the cause named. Instrument first: most unclear results are the scorer, not the science |
| Good but orthogonal to the claim | No | `PARKING_LOT.md` | Nothing in this paper | It earns the right to propose an experiment |

The dangerous mistake is writing the fourth kind as if it were the fifth (padding a real
limitation into vagueness) or the fifth as if it were the fourth (shrinking a contradiction into
a "limitation" paragraph). The bar's pre-registered sentence is the arbiter: a bounded miss fails
in a regime the claim never needed; a central miss fails where the claim lives.

## When a result contradicts the premise or cannot be typed: the diagnosis procedure

Six steps, in this order, because the cheap explanations come first. The habit this formalises is
the one that already works: look at the figures, ask questions, work through it in `/drip` and
`/plain-speak --drip` sessions until the data's statement is plain.

1. **Look at actual cells, not curves.** Open the images the number came from. `/analyze-figure`
   on the key figure.
2. **Check the instrument against your eyes.** Score five cells by eye, compare with the scorer.
   The composition scorer once counted a 162px limb as a third animal; this step is why.
3. **Check the harness.** Do the canaries still pass? Did every flag select a non-empty set
   (print the count)? A knob with nothing to select is a silent no-op that reports a plausible
   number for a configuration that never ran.
4. **Check what actually ran.** The `## Runs` row's launch commit against the config the result
   claims. The repo moves after launch; the row is the truth.
5. **Understand it.** `/drip` on the mechanism the result touches; `/plain-speak --drip` on any
   writeup of it that resists a cold read.
6. **Attack your own reading.** `/grill-me` or `/defend-results` on the interpretation you are
   about to commit to.

**Exit condition: one sentence stating what the data says, typed as one of three.** If the six
steps do not produce that sentence, the answer is the third exit by default, and the follow-up
experiment is designed to distinguish the remaining candidates.

- **A limitation.** Route: review answer `❌` with the boundary named, the caption and the figure
  slot's claim rewritten to the bounded sentence, the limitation written where the claim is made.
- **A contradiction.** Route: the plan finishes (a failed bar is a completed result), the figure
  slot is marked broken and not built, one follow-on plan owns the question, and
  `/frame-hypothesis` runs inside that follow-on, never in the plan that just closed.
- **An artefact of the setup.** Route: a procedure file for the fix (the instrument, the harness,
  or the config), the review answer stays `🟡` with the procedure as its named next action, and
  the run re-scores or re-runs. The bar does not move.

## Attack the work before a reviewer does

`/defend-results` and `/grill-me` on any result heading for the paper, and `/supervisor-prep`
before a meeting. A hostile read from someone who knows the field is the highest-signal feedback
available and it costs nothing but discomfort.

`/pressure-test` answers a different question: is this new, and is the method sound. Both are
needed, and passing one says nothing about the other.

## Kill work that is bleeding, not just work that failed

A clean negative is easy: the bar fails, the plan closes, a follow-on opens. The expensive case
is the plan that has not moved in a month and has not failed either, so no rule fires.

Set the rule explicitly. A plan whose last commit is more than a month old gets a continue-or-kill
decision, recorded with its reason. Killed work moves whole into `plans/shelved/` with one line
saying what would bring it back.

## Organise on the way out, not as a separate project

`sync-plan-tree --clean` when leaving a scope. `plan_pulse.py` reports the mechanical problems and
never fixes them: a task claiming a run is in flight when nothing is, output newer than the plan
that owns it, markdown no task names, run state in a design plan, a finished run with unanswered
questions, prose that will not read cold, a plan in none of the root lists.

Tidying is cheap when it rides along with work you are already doing in that folder, and expensive
as a dedicated week.

## Prefer the short chain when the work feeds the paper

Counter-intuitive and load-bearing. Care belongs in the experiment design and the pre-registered
bar, both fixed before the run. By figure time the result is settled and the only remaining risk
is not shipping.

So: paper work takes the short version of its chain, and every plan's `## Next` block names that
short version explicitly. Background work takes the long version, because nothing waits on it.

## Keep the handoff on disk

Every step of every chain writes a file the next step reads. A result passed through the
conversation is lost at session end, which is exactly when you next need it. This is why
`docs/ENVIRONMENT.md` works and why nothing has had to re-explain the cluster.

## Know the deadline, and check the remainder against it

The tree gives an order and cannot tell you whether the remainder fits. Put the submission date
at the top of the paper list and count backwards from it once a week. An order without a deadline
is how the last three steps become the ones you did not reach.
