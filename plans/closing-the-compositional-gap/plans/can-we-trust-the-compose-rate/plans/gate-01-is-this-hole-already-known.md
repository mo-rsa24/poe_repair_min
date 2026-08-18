# 🔍 Has someone already said this?

**No step number: nothing in the paper order waits on this.** This scope runs in its own internal order, and earns numbered steps only on the big-promotion condition its own `MASTER_PLAN.md` sets. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Within this scope | Plan | Status |
|---|---|---|
| **1 of 4** | **this plan** | **⚠️ not started** |
| 2 of 4 | [instrument-01-the-three-state-labelled-set](instrument-01-the-three-state-labelled-set.md) | ⚠️ not started |

Design only. The verdict lives in [../review/gate-01-is-this-hole-already-known.md](../review/gate-01-is-this-hole-already-known.md).

## What this asks, in one line
Two sentences this scope wants to build on may already be in the literature, in which case
most of the scope should not be built. Asking "is there a cat?" cannot catch a cat-dog fusion,
because the fusion really does contain cat features and answers yes. Counting animals cannot
catch a repeat, because two dogs is two animals. One `/pressure-test` pass decides whether
either of those is news.

## Why this plan exists
This scope's value depends entirely on the answer. If both halves are already named and
measured in a published benchmark, the honest move is a methods paragraph citing it, not a new
metric. Finding that out costs one session and no GPU. Finding it out after building the metric
costs the metric.

## Description
Write the claim down first, in one paragraph, exactly as it will be tested. Then run
`/pressure-test` against the compositional-evaluation literature: T2I-CompBench (arXiv
2307.06350), TIFA, VQAScore, Davidsonian Scene Graph, VISOR, and the attribute-binding work.
Record the verdict three ways. Act on it in the same session.

## Purpose
Serves Objective 1 and Definition-of-Done item 1. It is also the switch for the rest of the
scope: `idea-01` and `gate-02` do not start until it returns. `instrument-01` runs regardless.

## Goal
A verdict in `review/gate-01-is-this-hole-already-known.md` naming one of three outcomes, with
citations, plus one line saying what happens to `idea-01` and `gate-02`.

## Environment Facts This Plan Depends On
- Runs in session. No GPU, no Slurm queue, no disk.
- Needs working web access for the literature search. If it is unavailable, the plan halts
  rather than guessing at what a paper says.

## Success/Failure Outcomes
- **Already known and named.** The scope shrinks to a methods paragraph handed to
  `writing-06-mechanism-and-limitations`. `idea-01` and `gate-02` are cancelled and marked so.
  `instrument-01` still runs, because the paper still needs the band on 94%.
- **Said informally, never measured.** The scope continues, and the contribution becomes the
  measurement rather than the observation. This is the outcome the verdict most needs to
  distinguish, because it looks like "already known" and is not.
- **Not addressed.** The scope continues at full size.
- **The failure mode to avoid:** a verdict with no citations, or one that names a paper without
  saying which of its metrics is the presence family. That is an opinion, not a check, and it
  cannot gate anything.

## Tasks
- [ ] Write the claim as one paragraph BEFORE searching, so the search cannot soften it.
  Both halves stated separately, each with the failing case named (a cat-dog fusion for the
  first, two dogs for the second).
- [ ] Run `/pressure-test` on that paragraph against T2I-CompBench (arXiv 2307.06350),
  TIFA, VQAScore, Davidsonian Scene Graph, VISOR, and attribute-binding work. For each,
  record which question the metric actually asks of an image.
- [ ] Record the verdict in `review/gate-01-is-this-hole-already-known.md`, three ways, with
  citations.
- [ ] Write the consequence line: whether `idea-01` and `gate-02` proceed or are cancelled.
  If cancelled, mark both plan files cancelled and hand the methods paragraph to
  `writing-06-mechanism-and-limitations`.

## Engagement Instructions
The verdict file exists, names exactly one of the three outcomes, and carries at least one
citation with an arXiv id or venue per benchmark examined. Every benchmark listed above appears
in the file, including any that turned out irrelevant, with one line saying why. The consequence
line is present and names `idea-01` and `gate-02` by filename.

STOP: no web access → halt, do not write a verdict from memory. A verdict with no citations is
not a verdict and does not release the plans it gates.

## Recommended skill
▶ `/pressure-test` ✅: this plan is one invocation of it, on the paragraph written in task 1.
