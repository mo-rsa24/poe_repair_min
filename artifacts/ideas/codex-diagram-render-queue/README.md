# Should every planning skill's diagram prompts feed one background render queue instead of sitting unrendered?

Planning skills across this project's tooling write diagram prompts into `diagram-prompts.md`
files but nothing drained them automatically. The idea is a single worker that drains the queue in
file order: dispatch each prompt to Codex with its reference images, check the render against its
own faithfulness note, revise in the same session up to three times, save to its `Save as:` path,
and flip its mark to `[built]`. All six layers (what happens on render, the state machine, who
talks to whom, the worker's contract, where state lives, and what to build first) are decided and
the walk is compiled.

## What is in here

One file, `IDEA_MAP.md`, the full `/drip-idea` walk: the six decided layers, the constraints
pinned from the repo's diagram-prompts contract, an appended extension for pointing the same
worker at documentation folders (`context/`, `environment/`, `report/`), and a "Route results
folded back" section recording that slices 1 and 2 were already built.

## Where it came from and what judged it

**Held since 2026-09-05.** No plan or document in this repo cites the idea map itself; the one
mention elsewhere in the tree (`RENAMES.md`) is incidental, naming a file the walk's own render
test produced, not a reference to the idea. Kept because the walk's own record shows it already
shipped most of its build: `~/.claude/skills/render-diagrams/` and the Rendering section of
`~/.claude/DIAGRAM_PROMPTS_FORMAT.md` exist per the map's "Route results folded back" entry, and
what remains (draining the sync sweep and the documentation producer) is still named in its "Next
step" line.
