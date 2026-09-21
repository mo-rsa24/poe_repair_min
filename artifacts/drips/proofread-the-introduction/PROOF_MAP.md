# 🔎 Proofread map: the introduction of the ICLR manuscript

The walk's persistent state. Read it first on every invocation, write it before every round ends.

## Position in the walk

| Field | Value |
|---|---|
| Selection | `paper/overleaf-iclr/iclr2027_conference.tex` lines 127 to 163, `\section{Introduction}` up to `\section{Background and Related Work}` |
| Kind | manuscript section |
| Grain | paragraph |
| Cadence | sweep |
| Current chunk | 8 of 13, the interaction-term paragraph |
| Marks | 2 ok, 4 changed, 2 cut, 5 unwalked |
| Compiled | nothing written |
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
| 4 | paragraph | 141 | "In text-to-image diffusion models, the conditioning signal" | work | now a repeat | nothing unique left: chunk 2 names joint prompting, chunk 6 now defines $c$, line 239 introduces the Mono subscript. Cut once its last clause, "the reference the rest of this paper measures against", moves onto chunk 2's naming sentence |
| 5 | paragraph | 143 | "We contrast this with inference-time composition" | changed | claim past the evidence, half a repeat | fused with chunk 3's remnant: keeps the product-of-experts rule, the collapse and the plurality definition; loses its setup to chunk 2 and its promise to explain why composition fails |
| 6 | paragraph | 147 | "In diffusion models, the noise prediction is a scaled negative score" | changed | dangling reference | the assumption is named in the sentence and cited to Liu, $c$ is defined in the where-clause, and equation 1 stays in the introduction |
| 7 | equation | 148 | eq. poe-composition | ok | | the rule as used later; the symbols arrive close enough to their first prose use |
| 8 | paragraph | 153 | "This linear composition is exact only when" | work | claim past the evidence | calls the gap to the true joint model the plurality term, which line 271 says it is not |
| 9 | paragraph | 156 | "To make this concrete, consider two visually similar concepts" | cut | contrast frame, duplicate | every part of it is now in the fused paragraph: the cat and dog, what the joint prompt returns, the fused animal, the Figure 1 reference. Its last sentence was a contrast frame saying nothing chunk 8 does not. The `\looseness=-1` on line 155 goes with it |
| 10 | figure | 129 | `figures/where-the-product-lands-on-the-manifold-seed12-surface.png` | work | not yet opened | read pending; the image has to be looked at before anything is said |
| 11 | caption | 132 | "Product-of-experts composition lands between the two concepts" | work | caption past the picture | derives the geometry in the caption, ahead of the body text that derives it |
| 12 | paragraph | 158 | "Understanding and fixing this gap matters" | work | anaphoric drumbeat, colon reveal, now a repeat | two adjacent sentences open "This paper", and its combinatorial motivation is now chunk 2's opening sentence |
| 13 | list | 159 | the three contributions | ok | | a real set, each item says what was done and how far it goes |

## Accepted, not yet written

Navigation: ⬅️ [The chunk map](#the-chunk-map) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

One chunk accepted. The only change to the author's wording is the four-key citation added after "Recent methods" in paragraph two: `liu2022compositional`, `du2023reduce`, `skreta2024superdiff`, `zhang2025product`. Three of those four are cited nowhere else but line 170.

| # | Control | The accepted text |
|---|---|---|
| 2 | `improve`, author's own draft | Two paragraphs replacing the current opening. Paragraph one: "The number of scenes a text-to-image model can generate grows combinatorially as concepts are added, requiring increasingly large training datasets to cover their possible combinations \citep{du2024compositional}. A single prompt is sufficient for combinations that are common in the training data. For example, prompting Stable Diffusion with ``a cat and a dog'' can produce a cat sitting beside a dog \citep{rombach2022high,podell2023sdxl}. We refer to this setting as \emph{joint prompting}. Extending this approach to new, unseen compositions requires additional data and training." Paragraph two: "Recent methods instead compose concepts at inference time \citep{liu2022compositional}. Each concept is assigned its own prompt, such as ``a cat'' and ``a dog'', and the model produces a separate prediction for each. These predictions are then combined to guide the generation toward both concepts. The pretrained model remains unchanged, so a new composition requires additional inference rather than additional data and training." |
| 3 + 5 | `fuse` | One paragraph replacing both: "However, composition fails for pretrained diffusion models when the two concepts compete for the same region of the image \citep{bradley2025mechanisms}. Combining their predictions under a product-of-experts rule \citep{liu2022compositional, zhang2025product} favours a sample that scores highly under both experts. Composing a cat with a dog returns one animal with the features of both fused into a single body \citep{dutta2026steer}, a hybrid rather than a cat beside a dog. Figure~\ref{fig:poe-failure} places that hybrid between the two single-concept peaks, away from the peak the joint prompt reaches. The property lost in this collapse is what we call \emph{plurality}, the property that every concept a prompt names appears in the scene as its own distinct object. We study this gap between joint prompting and inference-time composition, and we show that what is missing can be measured and put back." |
| 6 + 7 | `improve` | Lead-in rewritten, equation 1 kept: "In diffusion models, the noise prediction is a scaled negative score: $\epsilon_\theta(x_t, t \mid c) = -\sigma_t \nabla_{x_t} \log p(x_t \mid c)$ \citep{dhariwal2021diffusion}, where $c$ is the text prompt and $\sigma_t$ is the noise standard deviation at step $t$. Treating the two concepts as conditionally independent given the image then gives the product-of-experts composition rule \citep{liu2022compositional}:" Equation 1 itself is unchanged except its trailing comma becomes a full stop, since a new paragraph follows. |

## Originals held

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#figures) ➡️

Chunk 2's original is held; the rewrite is offered and not yet accepted.

| # | `was:` |
|---|---|
| 2 | "Natural images usually depict several objects that appear together and interact as part of a coherent scene. A cat may sit beside a dog, or a cup may rest on a table, so generating such a scene takes more than producing each object correctly. The model must also place them, so a cup described as resting on a table sits on the tabletop rather than floating above it, appearing underneath it, or merging into its surface." |

## Figures

Navigation: ⬅️ [Originals held](#originals-held) | 📋 [TOC](#table-of-contents) | [Next](#open-questions-on-the-text) ➡️

| Figure | Image file | Opened | What to change | Slot decided |
|---|---|---|---|---|
| 1 | `paper/overleaf-iclr/figures/where-the-product-lands-on-the-manifold-seed12-surface.png` | no | pending chunk 10 | top of page 1, unchanged so far |

## Open questions on the text

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

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

## Next step

Navigation: ⬅️ [Compile log](#compile-log) | 📋 [TOC](#table-of-contents)

Chunks 2, 3 and 5 are accepted and held in the map. The walk moves to chunk 4, which now keeps only
Mono, since chunk 2 names joint prompting.

Still owed: chunk 9 loses its blended-animal reveal and its Figure 1 reference to the fused
paragraph, and chunk 12 loses its combinatorial motivation to chunk 2. `saharia2022photorealistic`,
`chen2023beyond` and `du2024generative` are now cited nowhere; `compile` should confirm no undefined
or unused-key warnings beyond those three.
