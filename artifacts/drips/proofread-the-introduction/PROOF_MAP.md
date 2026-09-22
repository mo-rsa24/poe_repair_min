# 🔎 Proofread map: the introduction of the ICLR manuscript

The walk's persistent state. Read it first on every invocation, write it before every round ends.

## Position in the walk

| Field | Value |
|---|---|
| Selection | `paper/overleaf-iclr/iclr2027_conference.tex` lines 127 to 163, `\section{Introduction}` up to `\section{Background and Related Work}` |
| Kind | manuscript section |
| Grain | paragraph |
| Cadence | sweep |
| Current chunk | walk complete and compiled |
| Marks | 3 ok, 7 changed, 3 cut, 0 still to settle |
| Compiled | 2026-09-22, the whole introduction, into `paper/overleaf-iclr/` only |
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

The introduction of the ICLR 2027 submission, in the copy that came down from Overleaf on
2026-09-21. The same text now sits in `paper/iclr/iclr2027_conference.tex`, so a compile has two
possible targets and the walk has to name one.

**Why this pass is happening**

The Overleaf sitting rewrote the title, the abstract and several joins in this section, and the
section has not been read end to end since.

**What the text is trying to do**

Get the reader from "images contain several objects" to three contributions, by naming the
setting (joint prompting), the alternative (product-of-experts composition), what that
alternative loses (plurality), and the quantity this paper measures and learns (the plurality
term).

## The chunk map

Navigation: ⬅️ [Quick context](#quick-context-what-this-is) | 📋 [TOC](#table-of-contents) | [Next](#accepted-not-yet-written) ➡️

Every chunk in reading order, with its mark. `ok` means leave it alone, `work` means there is
something to do, `cut?` means it may not belong.

| # | Kind | Line | First words or file | Mark | Fault | The read |
|---|---|---|---|---|---|---|
| 1 | heading | 127 | Introduction | ok | | names what is under it |
| 2 | paragraph | 137 | "Natural images usually depict several objects" | changed | rule-of-three list, bloat | replaced by the author's two-paragraph opening: the combinatorial problem, joint prompting named, then composition at inference |
| 3 | paragraph | 139 | "Recently, text-to-image diffusion models" | cut | decoration, uncited claim, repeat | the whole paragraph goes. The introduction stays on composition's own failure and drops the single-model benchmark failures, so `huang2025t2icompbench`, `conwell2024relations`, `saharia2022photorealistic`, `chen2023beyond` and `du2024generative` leave the paper |
| 4 | paragraph | 141 | "In text-to-image diffusion models, the conditioning signal" | cut | repeat | every piece is carried elsewhere: $c$ by chunk 6, the example and the name by ¶1, Mono by line 239. Its last clause, "the reference the rest of this paper measures against", moved onto ¶1's naming sentence |
| 5 | paragraph | 143 | "We contrast this with inference-time composition" | changed | claim past the evidence, half a repeat | fused with chunk 3's remnant: keeps the product-of-experts rule, the collapse and the plurality definition; loses its setup to chunk 2 and its promise to explain why composition fails |
| 6 | paragraph | 147 | "In diffusion models, the noise prediction is a scaled negative score" | changed | dangling reference | the assumption is named in the sentence and cited to Liu, $c$ is defined in the where-clause, and equation 1 stays in the introduction |
| 7 | equation | 148 | eq. poe-composition | ok | | the rule as used later; the symbols arrive close enough to their first prose use |
| 8 | paragraph | 153 | "This linear composition is exact only when" | changed | claim past the evidence, rule-of-three lists | rewritten around the two settings side by side, with $r_t$ defined as section 3 defines it: the gap between the joint-prompt and composed predictions at the same point in the same run |
| 9 | paragraph | 156 | "To make this concrete, consider two visually similar concepts" | cut | contrast frame, duplicate | every part of it is now in the fused paragraph: the cat and dog, what the joint prompt returns, the fused animal, the Figure 1 reference. Its last sentence was a contrast frame saying nothing chunk 8 does not. The `\looseness=-1` on line 155 goes with it |
| 10 | figure | 129 | `figures/where-the-product-lands-on-the-manifold-seed12-surface.png` | ok | kept as drawn | opened and read; the top label belongs to the blue thumbnail and reads as a title, and the surface height is unlabelled. A redraw prompt was offered and declined |
| 11 | caption | 132 | "Product-of-experts composition lands between the two concepts" | changed | caption past the picture | cut to its bold title and the four-renders sentence |
| 12 | paragraph | 158 | "Understanding and fixing this gap matters" | changed | stance, repeats, colon reveal, drumbeat | four of five sentences repeated ¶1 to chunk 8; replaced by the one tension no earlier paragraph states (the residual needs the joint prompt, composition has none) posed as the question the contributions answer |
| 13 | list | 159 | the three contributions | changed | repeat, undefined abbreviation, lead mismatch | bullet 1 no longer redefines $r_t$, bullet 2 names its three controls and adds the timing result, bullet 3 answers chunk 12's question; "PoE" is now defined in the fused paragraph |

## Accepted, not yet written

Navigation: ⬅️ [The chunk map](#the-chunk-map) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

Nothing waiting. Everything accepted was written on 2026-09-22; see the compile log.

## Originals held

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#figures) ➡️

Released after the compile. The pre-walk text is commit `e793c5e` in `paper/overleaf-iclr/`.

## Figures

Navigation: ⬅️ [Originals held](#originals-held) | 📋 [TOC](#table-of-contents) | [Next](#open-questions-on-the-text) ➡️

| Figure | Image file | Opened | What to change | Slot decided |
|---|---|---|---|---|
| 1 | `paper/overleaf-iclr/figures/where-the-product-lands-on-the-manifold-seed12-surface.png` | yes | nothing; kept as drawn. The caption no longer says the surface is drawn rather than measured, which is the author's decision | top of page 1, unchanged |

## Open questions on the text

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

- [ ] ⚠️ **¶3 and chunk 8 both end on a "we show that" claim.** (blocks nothing) ¶3 closes "and we
      show that what is missing can be measured and put back"; chunk 8 closes "We show that adding
      the residual back during denoising restores plurality". The author kept both; the trim that
      would remove the repeat is cutting ¶3's second clause.
- [ ] ⚠️ **The Bradley and CO3 keys are now split.** (blocks nothing) `bradley2025mechanisms` sits on
      the condition sentence and `dutta2026steer` on the hybrid outcome, which revises the draft
      map's record of both keys together on the outcome. Reversible either way.
- [ ] ⚠️ **One name for the failed object, across the whole file.** (blocks nothing, owed at compile)
      The manuscript uses four: `hybrid` (abstract line 101, line 143), `blended animal` (figure
      captions 132, 235, 377, and lines 156 and 392), `entangled` (143) and `chimera` (252). The
      introduction now names it a hybrid, so the other three want a decision: keep "blended" as
      plain description in captions, or make every naming use read hybrid.
- [ ] ⚠️ **Does the motivation sentence live in the opening or in chunk 12?** (blocks chunks 2 and 12)
      Both would cite `du2024compositional` for the same point.
- [ ] ⚠️ **Which file does `compile` write to?** (blocks the compile, not the walk) The clone at
      `paper/overleaf-iclr/` is what pushes to Overleaf; `paper/iclr/` is what this repo tracks.
      They hold identical text right now.
- [ ] ⚠️ **Is the gap named in chunk 8 meant to be the gap to the true joint model, or the gap
      between the two predictions one network makes?** (blocks chunk 8) Line 271 of the same file
      states the second, so either the introduction or that line has to move.

## Routes and threads

Navigation: ⬅️ [Open questions](#open-questions-on-the-text) | 📋 [TOC](#table-of-contents) | [Next](#compile-log) ➡️

| Chunk | Where it went | What it owes back | Returned |
|---|---|---|---|

## Compile log

Navigation: ⬅️ [Routes and threads](#routes-and-threads) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| When | Chunks written | Build | Notes |
|---|---|---|---|
| 2026-09-22 | 2 to 13 and the Figure 1 caption, one hunk at lines 129 to 165 of `paper/overleaf-iclr/iclr2027_conference.tex` | passed, pdflatex and bibtex on the laptop, no errors, no undefined references or citations | 12 pages before, 11 after, both built the same way; the conclusion moves from page 11 to 10. `paper/iclr/` not written. Not committed in the clone, not pushed to Overleaf |

## Next step

Navigation: ⬅️ [Compile log](#compile-log) | 📋 [TOC](#table-of-contents)

The introduction is compiled into the Overleaf clone and builds. Owed: commit and push it to
Overleaf from the cluster (the laptop has no Overleaf token), decide whether `paper/iclr/` gets the
same text, and settle the open questions above (the failed object's four names, the double \"we
show that\", the split Bradley and CO3 keys).
