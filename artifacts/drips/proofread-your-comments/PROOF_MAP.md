# 🔎 Proofread map: the manuscript, walked comment by comment

The walk's persistent state. Read it first on every invocation, write it before every round ends.

## Position in the walk

| Field | Value |
|---|---|
| Selection | `paper/overleaf-iclr/iclr2027_conference.tex`, whichever chunk each comment points at |
| Kind | manuscript |
| Grain | paragraph, sentence where a comment is about one sentence |
| Cadence | close, one comment per round |
| Current chunk | waiting on the next comment |
| Marks | 2 settled in Overleaf, 0 queued |
| Compiled | nothing queued; the clone was pulled from Overleaf to `ee209cb` on 2026-09-25, and line numbers for chunk 2 are from before that pull |
| Owner | this session |

## Table of contents

- [Position in the walk](#position-in-the-walk)
- [Quick context: what this is](#quick-context-what-this-is)
- [The chunk map](#the-chunk-map)
- [Accepted, not yet written](#accepted-not-yet-written)
- [Originals held](#originals-held)
- [Figures](#figures)
- [Open questions on the text](#open-questions-on-the-text)
- [Routes and threads](#routes-and-threads)
- [Compile log](#compile-log)
- [Next step](#next-step)

## Quick context: what this is

Navigation: ⬅️ [Position](#position-in-the-walk) | 📋 [TOC](#table-of-contents) | [Next](#the-chunk-map) ➡️

**What is being proofread**

The ICLR submission, at the places the writer comments on.

**Why this pass is happening**

The writer has their own comments on the draft and wants help acting on each one.

**What the text is trying to do**

Show that product-of-experts composition drops a measurable term, that adding it back early
restores two objects, and that an adapter can learn it and apply it to pairs it never trained on.

**Other walks over the same file**

The [full-pass proof map](../proofread-full-pass/PROOF_MAP.md) holds uncompiled work on chunks
1 to 6, 15 and 16 of its own list. A comment landing on one of those chunks reads that entry first.

## The chunk map

Navigation: ⬅️ [Quick context](#quick-context-what-this-is) | 📋 [TOC](#table-of-contents) | [Next](#accepted-not-yet-written) ➡️

One row per comment, in the order they arrive.

| # | Line | Kind | First words | The comment | Mark | Fault | The read |
|---|---|---|---|---|---|---|---|
| 1 | 154 | sentence | "The property lost in this collapse is what we call" | remove the comma, read more naturally; one sentence, no colon | settled in Overleaf | apposition chain; repetition ("property ... the property") | the definition hangs off the name as a comma apposition and repeats "property" |
| 2 | 248 | caption | "A larger model does not rescue" | cite arXiv 2403.03206 for SD 3.5 | settled in Overleaf | missing citation (added to the fault list this session) | the figure renders SD 3.5 Medium, SD 1.4, SD 2.1 and SDXL, and the caption names none of them; the bib has no SD3 entry |

## Accepted, not yet written

Navigation: ⬅️ [The chunk map](#the-chunk-map) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

| # | Control | The accepted text |
|---|---|---|

Chunks 1 and 2 were dropped from this queue: the writer made both changes in Overleaf in their own words (line 154, "We call the ability to render each named concept as a distinct object plurality."; `esser2024scaling` in the bib, cited in a new paragraph at line 271). The laptop's uncommitted line 154 edit is in the clone's `stash@{0}`.

## Originals held

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#figures) ➡️

| # | The original text |
|---|---|
| 2 | ...columns are four text-to-image models, oldest to newest, at seed 42. *(bib: no esser2024scaling entry)* |
| 1 | The property lost in this collapse is what we call \emph{plurality}, the property that every concept a prompt names appears in the scene as its own distinct object. |

## Figures

Navigation: ⬅️ [Originals held](#originals-held) | 📋 [TOC](#table-of-contents) | [Next](#open-questions-on-the-text) ➡️

None opened yet.

## Open questions on the text

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

**Overleaf line 154** defines plurality without `\emph`. **Overleaf line 271**: "a cat-dog pair fuse" should be "fuses". Both are the writer's text, unwalked.

## Routes and threads

Navigation: ⬅️ [Open questions](#open-questions-on-the-text) | 📋 [TOC](#table-of-contents) | [Next](#compile-log) ➡️

None yet.

## Compile log

Navigation: ⬅️ [Routes and threads](#routes-and-threads) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing written yet.

## Next step

Navigation: ⬅️ [Compile log](#compile-log) | 📋 [TOC](#table-of-contents)

Take the writer's next comment.
