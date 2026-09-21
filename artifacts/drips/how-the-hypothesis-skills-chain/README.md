# In what order do frame-hypothesis, hypothesis-to-scope, and init-master-plan fit together?

Before scoping the next experiment, this checks which of three planning skills to reach for and
in what order: `frame-hypothesis`, `hypothesis-to-scope`, and `init-master-plan`. The walk is a
single chat answer split into five short pieces so it could be checked one claim at a time
instead of absorbed as one block. All five pieces were read and confirmed; nothing in it is
still open.

## What is in here

One thread record, `dispatch.md`, with five confirmed pieces and no side documents or images.

## The pieces

**The three skills meet in a fixed order.** `frame-hypothesis`, then `hypothesis-to-scope`, then
`init-master-plan`, matching workflow 20 of `~/.claude/skills/WORKFLOWS.md`. Confirmed without
further questions.

**`frame-hypothesis` compiles a five-part artifact, then stops.** It writes no plan files itself.
Confirmed; no example was requested, though one was offered (the samplers-or-models gap).

**`hypothesis-to-scope` is the bridge.** It translates the compiled artifact, checks it against
the environment, and annotates it, but never authors plan files itself. Confirmed with no
questions.

**`init-master-plan` is the one that writes `MASTER_PLAN.md` and the `plans/` folder.** The full
path from there is workflow 20's seven-hop spine, from `env-pulse` through `populate-plans`.
Confirmed; the distinction that this skill is the one that writes the actual file landed without
a pause.

**Where it came from and what judged it.** No plan or context entry in this repository points
back at this thread by path. It is kept as a short, settled reference for which planning skill to
invoke and in what order, drawn from `~/.claude/skills/WORKFLOWS.md` workflow 20 and the three
skills' own `SKILL.md` headers, rather than as an open question needing further work.
