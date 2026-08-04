# Build spec — Mechanism Study → Todoist

**Account:** Molefe Molefe (momolefe24@gmail.com), Todoist Free
**Project:** 🧑‍🎓 PhD — `6XHxp94hx7x3PMMR`
**Section:** 🎱 LoRA Experiments — `6h7CH6wCmr9C85F2`
**Mapping:** B-variant (one parent task + real subtasks), house format (emoji + bold title,
no dash separators), matching the section's existing style but deviating from the existing
rungs' single-task-with-Do-list pattern by using real subtasks per user request.
**Label:** `mechanism-study` (new, violet) — `2184434274`

## Written (2026-07-27)

| Item | Todoist id | Parent |
|---|---|---|
| **🔬 Mechanism Study** (parent, sibling to the publishable-bar umbrella task) | `6h8fmJ325vgwqXj2` | — (section-level) |
| **🔧 Instrument: wire attention capture into the LoRA path** | `6h8fphjV86pr7gH2` | `6h8fmJ325vgwqXj2` |
| **🔬 Baseline-Compare: build the Attend-and-Excite-equivalent intervention** | `6h8fphvvGcXh9GqR` | `6h8fmJ325vgwqXj2` |
| **📊 Read the Mechanism: LoRA vs Attend-and-Excite comparison** | `6h8fpj6g6fhmm99R` | `6h8fmJ325vgwqXj2` |

Source: `plans/mechanism-study/todoist/{00-INDEX,01,02,03}.md`, translated from
`plans/mechanism-study/plans/*.md` (themselves generated from EXPERIMENTS.md EXP-06).

## Re-run semantics
Re-running `/todoist-publish plans/mechanism-study/todoist` re-reads this project/section,
matches by task `content` under the parent, and only creates genuinely new/changed items —
this run is a full create, a re-run should be a no-op skip on all 4 items.
