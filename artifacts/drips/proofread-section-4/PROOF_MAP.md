# 🔎 Proofread map: section 4 of the ICLR manuscript

The walk's persistent state. Read it first on every invocation, write it before every round ends.

## Position in the walk

| Field | Value |
|---|---|
| Selection | `paper/overleaf-iclr/iclr2027_conference.tex` lines 257-299, `\section{Restoring the Plurality Term Restores Composition}` (`sec:analysis`) |
| Kind | manuscript |
| Grain | paragraph |
| Cadence | sweep |
| Current chunk | 7 of 7, walked |
| Marks | 2 ok, 5 changed, 0 work, 0 cut, 0 unwalked |
| Compiled | chunks 2, 3, 5, 6 and 7 written on 2026-09-23, build passed |
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

The timing section of the paper: three paragraphs and two figures arguing that a ten-step window of
the measured correction, placed at the start of the run, is all that composition needs.

**Why this pass is happening**

Sections 3 and 1 have already been walked and compiled; section 4 is the next one in reading order
and is the last section written from measured corrections before the adapter takes over.

**What the text is trying to do**

Establish two things a learned correction depends on: the correction is only needed in the first
steps, and that boundary holds across every pair and seed rather than in one example.

## The chunk map

Navigation: ⬅️ [Quick context](#quick-context-what-this-is) | 📋 [TOC](#table-of-contents) | [Next](#accepted-not-yet-written) ➡️

| # | Kind | First words or file | Mark | Fault | The read |
|---|---|---|---|---|---|
| 1 | heading | "Restoring the Plurality Term Restores Composition" | ok | | names what is under it |
| 2 | paragraph | "If $r_t$ is the signal composition is missing" | changed | bloat, stacked ideas | seven sentences to set up one question, and the answer is given in sentence 2 before the question is finished being asked |
| 3 | paragraph | "One pair shows what the sweep that follows measures" | changed | claim past the evidence | "later than step ten" does not match what either figure or the next paragraph says the boundary is |
| 4 | figure | `figures/when-the-correction-arrives/` | ok | | four renders compared in the built PDF (three-column rebuild, trimmed five-column, sideways, split). Sideways costs a page and a turn, split costs two pages and drifts to page 8, and the trim gains 11 percent. Kept exactly as it is, so the caption still carries the column meaning |
| 5 | caption | "Ten corrected steps already reach the outcome" | changed | stacked ideas | reading instructions and the claim in one block; the last sentence asks the reader to do the comparison the caption already stated |
| 6 | paragraph | "The sweep across all eight pairs and four seeds" | changed | stacked ideas, no join | eight sentences carrying the detector, the method, the result, the interpretation, a size control, an open question and the bridge to section 5 |
| 7 | caption | "The final image for a cat and a dog at seeds 9 to 12" | changed | caption past the picture (inverted) | the text claims eight pairs and four seeds, the caption shows one pair at four seeds, and it is the only caption in the section with no bold claim sentence |

## Accepted, not yet written

Navigation: ⬅️ [The chunk map](#the-chunk-map) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

| # | Control | The accepted text |
|---|---|---|
| | | all written on 2026-09-23; see the compile log |

## Originals held

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#figures) ➡️

| # | `was:` |
|---|---|
| 7 | "The final image for a cat and a dog at seeds 9 to 12, as the ten-step correction window slides across the run. A column is the window's first and last step, and everything outside it runs plain composition, so a row varies only in where the correction sat. The green frame is the detector verdict, a cat box and a dog box with little overlap. Seed 11 composes at 5--15 but not at 0--10, the one break in the left-to-right decay." |
| 6 | "The sweep across all eight pairs and four seeds draws the same boundary. A generated image counts as composed when an object detector finds two distinct animals in it, and all counts were manually verified. We restrict the correction to a ten-step window, leave the remaining forty steps uncorrected, and slide the window across the run in strides of five. The window is kept narrow so the sweep can separate timing from coverage, since a window spanning most of the run would contain the early steps at every position. Windows starting within the first five steps restore composition for every pair we tried, and windows starting at step 20 or later restore it for none (Figure~\ref{fig:where-the-correction-lands}). The correction is therefore not needed for most of the run: once the early steps have settled which scene the trajectory is in, plain product-of-experts composition carries the image to the end without it. In every pair we measured, the correction is at its smallest, relative to that pair's typical per-step size, in the steps where it matters, so size does not explain the timing. Why the earliest steps are decisive is open, and what we can show is that the correction they need is learnable. Every result above used the joint prompt, which composition is not allowed to have. The next section removes that need." |
| 5 | "\textbf{Ten corrected steps already reach the outcome, and no late start recovers it.} A cat and a dog at seed 12. Two grids of the same run, differing only in which stretch of it receives the correction: from the beginning outward in \textbf{(a)}, from a later start through to the end in \textbf{(b)}. Along any row the five pictures are that run decoded after 10, 20, 30, 40 and 50 steps, a green frame marks a decode taken while the correction was active, and the row marked \emph{off} never corrects. Read down the rightmost column of each panel: in (a) it never changes, and in (b) it changes exactly once, between the row that starts at step ten and the row that starts at step twenty." |
| 3 | "One pair shows what the sweep that follows measures across every pair. Extending the correction later into the run buys nothing, and starting it later than step ten loses composition altogether (Figure~\ref{fig:when-the-correction-arrives}). Both grids are one pair at one seed, so nothing here says how often that boundary holds." |
| 2 | "If $r_t$ is the signal composition is missing, then adding back a fraction $\lambda$ of it should compose the pair, and both $\lambda$ and which steps receive it should matter. We show that a small fraction of the measured plurality term, applied in the first steps of the denoising run, is enough to restore composition. The strength axis says nothing about when the correction has to be present. A composed run visits fifty states and the correction is defined at every one of them, so restoring composition could need the whole trajectory or only part of it. The difference decides what a learned correction has to do: a term needed everywhere must be produced at every step of every run, while a term needed only at the start has to be right in a narrow window and can leave the rest to plain composition. We settle it by choosing which steps receive the correction and leaving the others uncorrected." |

## Figures

Navigation: ⬅️ [Originals held](#originals-held) | 📋 [TOC](#table-of-contents) | [Next](#open-questions-on-the-text) ➡️

| Figure | Image file | Opened | What to change | Slot decided |
|---|---|---|---|---|
| 4 | `figures/when-the-correction-arrives/when-the-correction-arrives.pdf` | no | | main text, after chunk 3 |
| 7 | `figures/where-the-correction-lands.pdf` | no | | main text, after chunk 6 |

## Open questions on the text

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

- [x] ✅ **Boundary confirmed off the image:** in panel (b) the row starting at step ten composes and every row from step twenty down does not, so the prose boundary is step twenty, not "later than step ten".
- [x] ✅ **Settled by opening it: one pair, cat x dog, at seeds 9 to 12 across nine window positions.** The eight-pair evidence is the heat map `how-often-the-correction-lands`, commented out at lines 278 to 286. Chunk 6 was narrowed to the shown figure rather than restoring it. **Still owed:** the eight-pair result is measured and now appears nowhere in the paper.
- [ ] ⚠️ **Old question, kept for the record: does figure `where-the-correction-lands` carry all eight pairs?** (blocks chunks 6 and 7) The caption says one pair; the paragraph claims the sweep across eight. Settled by opening the image and the review file behind it.

## Routes and threads

Navigation: ⬅️ [Open questions](#open-questions-on-the-text) | 📋 [TOC](#table-of-contents) | [Next](#compile-log) ➡️

| Chunk | Where it went | What it owes back | Returned |
|---|---|---|---|
| | | nothing routed yet | |

## Compile log

Navigation: ⬅️ [Routes and threads](#routes-and-threads) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| When | Chunks written | Build | Notes |
|---|---|---|---|
| 2026-09-23 | 2, 3, 5, 6, 7 | passed | pdflatex plus bibtex, no errors, no undefined citations, 11 pages as before, figure 4 still on page 6 |

## Next step

Navigation: ⬅️ [Compile log](#compile-log) | 📋 [TOC](#table-of-contents)

**Where a cold session resumes**

Every chunk is walked. Five changes are queued and the file is untouched: run `compile` to write them and rebuild. The one thing the walk did not resolve is that the eight-pair sweep result now appears nowhere in the manuscript, since chunk 6 was narrowed rather than the heat map restored.
