# Template: one finding, its figures, and the explainer beside them

Copy this when a result lands. Three files, three places, one question. The finding and the
explainer shapes are pinned in `~/.claude/REPORT_FORMAT.md` and
`~/.claude/FIGURE_EXPLAINER_FORMAT.md`; this page shows how they sit together in this repo, with
the navigation layer this repo adds, and with
[where each condition lands](when-does-the-outcome-lock-in/where-does-each-condition-land.md) and
[its explainer](../artifacts/results/where-does-each-condition-land/figure-explainer.md) as the
worked instance.

## Table of contents

- [The three places](#the-three-places)
- [How it reads](#how-it-reads)
- [The finding file](#the-finding-file)
- [The explainer file](#the-explainer-file)
- [The card](#the-card)
- [The rules that keep the three honest](#the-rules-that-keep-the-three-honest)

## The three places

Navigation: 📋 [TOC](#table-of-contents) | [Next](#how-it-reads) ➡️

| File | Where | What it holds |
|---|---|---|
| `report/<question>.md` | flat in `report/`, or in a question group once it has a sibling | the claim, where the idea came from, the aim and hypothesis, the pre-registered bar, what was tried in order, numbered rungs (figure + number + source), what it cannot say, provenance, links, still open |
| `artifacts/results/<question>/README.md` | beside the images | one entry per file: what it shows, how it was made, what it cannot say |
| `artifacts/results/<question>/figure-explainer.md` | beside the images | every image read in plain words, one section each, with the documents that own each term |

The index row in `report/00-INDEX.md` carries the verdict and the headline number. The finding
carries the evidence. The explainer carries the reading. Nothing is restated across the three:
the finding embeds the images, the explainer links the finding for the numbers, the card names
the scripts.

## How it reads

Navigation: ⬅️ [Previous](#the-three-places) | 📋 [TOC](#table-of-contents) | [Next](#the-finding-file) ➡️

**The verdict is on the title line.** Nobody reads three paragraphs to learn the answer.

**A table of contents, then a navigation line under every section heading.** The same
`⬅️ Previous | 📋 TOC | Next ➡️` line the plan files carry, so a reader can move through a long
finding without scrolling back.

**Bold mini-titles, not bullets, for prose.** A section made of several related points writes
each as a short bold title on its own line, the point beneath it, and a blank line before the
next. Lists are for true sets: steps, options, files.

**Short sentences, one idea each, in the words you would say out loud.** Paths, numbers, dates,
field names and verdict words stay exactly as they are. The framing around them is what gets cut.

**Every picture has a caption that says what to notice**, never what it is, and a backlink line
under it pointing at the explainer section that reads it.

**Every number has its unit, its meaning and its file.** "0.42, DINOv2 cosine units, mean of 8
seeds, from `<sidecar>` field `<field>`" is checkable. "The correction clearly helps" is not.

## The finding file

Navigation: ⬅️ [Previous](#how-it-reads) | 📋 [TOC](#table-of-contents) | [Next](#the-explainer-file) ➡️

```markdown
# <The question?>   <✅ support | ⚪ null | ❌ null on the bar> · verified <date>

**The claim**
<Two to six plain sentences. Lead with what was found. Say which part is pre-registered and
which is post-hoc, if both exist.>

## Table of contents
- [Where the idea came from](#where-the-idea-came-from)
- [What we set out to see](#what-we-set-out-to-see)
- [What would have counted](#what-would-have-counted)
- [What was tried, in order](#what-was-tried-in-order)
- [1. <first rung>](#1-...)
- ...
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## Where the idea came from
Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-we-set-out-to-see) ➡️
**The tool / the paper / the post.** <Half a line each on what it says, linked to its entry in
context/sources.md, never re-described here.>
**The connection to this project.** <Why it bears on this question.>

## What we set out to see
Navigation: ⬅️ [Previous](#...) | 📋 [TOC](#table-of-contents) | [Next](#...) ➡️
<The design in plain words: conditions, seeds, what is held fixed.>
**The hypothesis.** <What was expected, in one paragraph.>

## What would have counted
Navigation: ...
<The criterion, quoted, and a link to the plan section where it was pinned before the run.
If it failed, say so here, in one sentence. Name any number reported without a bar as post-hoc.>

## What was tried, in order
Navigation: ...
| When | What | Where it landed | Outcome |
|---|---|---|---|
| <date time> | <the run or the attempt> | <path> | <what happened, including the ones that died> |

## 1. <The first rung: what the effect looks like>
Navigation: ...
![<axes and data in one line>](../artifacts/results/<question>/<figure>.png)
*<What to notice, never what it is.>*
📊 Drawn in [Figure N of the figure explainer](../artifacts/results/<question>/figure-explainer.md#figure-n-...).

**The number.** <value, unit, meaning, range> From `<results file>`, field `<field>`, written by
`<script>` under [<the plan>](plans/<scope>/plans/<file>.md).

## 2. <The second rung: the pictures behind the points, or the control>
...

## What this cannot tell you
Navigation: ...
<The view's blind spot. Any comparison where more than one axis differs, marked mixed. Any
threshold chosen after the data existed.>

## Where this came from
Navigation: ...
| What | Source | Mark |
|---|---|---|
| <the numbers> | `<sidecar path>`, read <date> | verified |
| <the renders> | `/datasets/.../<folder>/`, `<manifest>`, rendered <date> on <node> | verified |
| The run | task **<bolded task name>** in [<the plan>](plans/...); verdict in [its review file](plans/.../review/...) | verified |
| Regenerate with | [<runbook recipe>](runbook/<file>.md#<anchor>), then the scripts named | |

## Depends on
Navigation: ...
- <a term>: [<context entry>](context/world/<file>.md)
- <a system fact>: [<environment entry>](environment/<file>.md)
- <the design's sources>: [<sources.md entry>](context/sources.md#<anchor>)
- Each picture, read one at a time: [the figure explainer](artifacts/results/<question>/figure-explainer.md)

## Still open
Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)
- [ ] <a named gap, with the plan task that would close it, quoted by its bolded name>
```

## The explainer file

Navigation: ⬅️ [Previous](#the-finding-file) | 📋 [TOC](#table-of-contents) | [Next](#the-card) ➡️

```markdown
# <Question>: what these pictures mean

<One line: which finding, which outputs, whether a diagram map exists.>
N figures explained · N waiting on a render.

## Table of contents
<every section, figures listed by number and title>

## Where this sits
Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-cast) ➡️
**The system drawn.**  **What the pictures were read off.**  **Which plan opened it.**
**What the set was meant to achieve.** <with the outside sources linked from context/sources.md>

## The cast
Navigation: ...
| Drawn as | It is | Owned by |

## Figure N: <its own title>
Navigation: ⬅️ [Previous](#...) | 📋 [TOC](#table-of-contents) | [Next](#...) ➡️
[observed] · Explains: `<png>`, rendered <date> by `<script>` · Read off <sidecar or folder>
![...](<png>)
**What you are looking at**   **The components** (table)   **The flows**
**What it teaches**   **What it does not show**   **Read next**

## How the figures connect
## Where this touches the repo
| Figure | Points at | Why | Backlink |
## Still open
```

## The card

Navigation: ⬅️ [Previous](#the-explainer-file) | 📋 [TOC](#table-of-contents) | [Next](#the-rules-that-keep-the-three-honest) ➡️

`README.md` beside the images, per `~/.claude/ARTIFACT_TREE_FORMAT.md`: a one-paragraph
statement of the question, a `What is here` table with one row per file, `How it was made` with
the scripts and nodes, `What it says` in two or three plain paragraphs, and `What it cannot say`.
Provenance lives here as prose; the finding links it rather than repeating it.

## The rules that keep the three honest

Navigation: ⬅️ [Previous](#the-card) | 📋 [TOC](#table-of-contents)

- Every number in the finding names its file and field, and is marked `verified` only after that
  field was read in the session that wrote it.
- Every image is embedded from `artifacts/results/`, never copied into `report/`, and every
  image was looked at before its section was written.
- The bar is quoted from where it was pinned before the run. A read taken after the result is
  filed under "Asked after the result" in the review file, named as post-hoc in the finding, and
  never promoted to the bar.
- Attempts that died are rows in "What was tried", with what killed them, so the next reader
  does not repeat them.
- Links to plan material are root-relative and name the thing; tasks are quoted by their bolded
  name. Links to context, environment, runbook and sources are pointers, never restatements.
- A figure the finding wants and does not have is a 🖼️ slot in the finding and a row on the
  index, and it stays until the render exists.
- Outside sources (a paper, a tool, a post) get an entry in `context/sources.md` and are linked
  from there, so the finding says what the design borrowed without re-describing the source.
