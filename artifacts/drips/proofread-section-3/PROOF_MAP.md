# 🔎 Proofread map: section 3 of the ICLR manuscript, the plurality term

The walk's persistent state. Read it first on every invocation, write it before every round ends.

## Position in the walk

| Field | Value |
|---|---|
| Selection | `paper/overleaf-iclr/iclr2027_conference.tex` lines 194 to 266, `\section{Missing Implications: The Plurality Term}` up to `\section{Restoring the Plurality Term Restores Composition}` |
| Kind | manuscript section |
| Grain | paragraph |
| Cadence | sweep |
| Current chunk | walk complete and compiled |
| Marks | 7 ok, 12 changed, 2 cut |
| Compiled | 2026-09-22, the whole of section 3, into `paper/overleaf-iclr/` only |
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

Section 3 of the ICLR 2027 submission, in the Overleaf clone at `paper/overleaf-iclr/`. The repo
copy at `paper/iclr/` still carries the old introduction and is not touched.

**Why this pass is happening**

The introduction was walked and compiled on 2026-09-22, and settled the names (joint prompting,
PoE, hybrid), equation 1, and the definition of $r_t$. Section 3 has to agree with all of them.

**What the text is trying to do**

Show what product-of-experts composition drops (the ratio $R_t$), define the measurable stand-in
for it (the plurality term $r_t$), write the corrected sampler, and show on one pair that adding
$r_t$ back composes where matched controls do not.

## The chunk map

Navigation: ⬅️ [Quick context](#quick-context-what-this-is) | 📋 [TOC](#table-of-contents) | [Next](#accepted-not-yet-written) ➡️

| # | Kind | Line | First words or file | Mark | Fault | The read |
|---|---|---|---|---|---|---|
| 1 | heading | 194 | Missing Implications: The Plurality Term | changed | point not landed | "Missing Implications" names nothing a reader can picture; the section is about what composition drops and the term that measures it |
| 2 | paragraph | 197 | "At each denoising step, product-of-experts composition evaluates" | changed | colon reveal, rule-of-three list, term before thing | now opens on the model (SDXL, its three parts, the latent shape) and what a seed fixes; the list of three is cut to the one property that returns in ¶19 |
| 3 | paragraph | 199 | "When the two concepts are independent given the image" | changed | repeat (one line) | kept: the only place the density factorisation and $p_{\mathrm{PoE}}$ are defined; the prose now names equation 1 |
| 4 | equation | 200 | align block, $\log p_{\mathrm{PoE}}$ down to $\hat\epsilon_{\mathrm{PoE}}$ | changed | repeat (one line) | third line, equation 1 written out again, removed; the sum form that chunk 5 needs stays |
| 5 | paragraph | 211 | "Attaching a weight $w_i$ to each conditional term" | ok | | the only place the weights enter, since equation 1 has none |
| 6 | paragraph | 213 | "The gap between the joint conditional and the product form is the ratio" | ok | | defines $R_t$ before anything uses it |
| 7 | paragraph | 215 | "At every step the sampler combines the two experts as though" | changed | citation placement | each key now sits on its own claim; the introduction splits them, Bradley on the condition, CO3 on the hybrid |
| 8 | equation | 216 | align block, joint score to $\epsilon_{\text{true}}$ | ok | | each line follows from the one above by the step the prose names |
| 9 | paragraph | 225 | "Nothing in the run evaluates $R_t$" | changed | symbol before its definition, no join, stacked ideas | figure sentence first and matched to the four-models grid, no $r_t$ before its definition, timing promise cut; the size point is left to chunk 21 |
| 10 | figure | 227 | `figures/same-rule-composes-one-pair-and-blends-another.pdf` | ok | | opened: a butterfly meadow with both concepts, and one white cat-dog animal; shows what the text says |
| 11 | caption | 230 | "The same rule composes one pair and blends another." | changed | caption past the picture, point not landed | new four-models grid; the caption now states the takeaway: a larger model does not rescue product-of-experts composition |
| 12 | paragraph | 234 | "We define the plurality term $r_t$ as the difference" | ok | | agrees with intro ¶8: two predictions of one model at the same latent |
| 13 | equation | 235 | eq. interaction-term | ok | | |
| 14 | paragraph | 240 | "The identity ... holds by construction and is not a result" | changed | contrast frame, term before thing | the frame is gone and the three measured things are stated directly; "held out of training" became "a model trained to predict it reaches concept pairs it never saw" |
| 15 | equation | 241 | eq. corrected-sampler | ok | | |
| 16 | paragraph | 247 | "The strength axis separates the correction from any disturbance" | changed | numbers in prose, naming, claim past the picture | inline λ 0.75 and λ 2; "chimera"; the grid's λ 0.75 column already shows two animals, and the wrong-seed row past λ 1.25 needs a second look before "no control reaches two animals" stands |
| 17 | figure | 249 | `figures/how-much-is-added/strength-grid-0to1.pdf` | changed | two jobs in one panel | cropped to λ 0 to 1: the columns past the joint-prompt prediction are the only place a control reaches a second, damaged animal, and no other part of the paper uses λ above 1 |
| 18 | caption | 252 | "Only the measured correction composes, and over-driving it does no visible harm." | changed | caption past the picture | "over-driving it does no visible harm" and the columns-past-1 clause describe cells the crop removed; the seed leaves the caption and stays in the paragraph |
| 19 | paragraph | 262 | "Product-of-experts sampling never sees the joint prompt." | cut | claim past the evidence, no join, repeat | "why composition fails" contradicts line 299; its measurability point is chunk 21's, and its direction clause moved to chunk 16 |
| 20 | paragraph | 264 | "Independence can fail in more than one way" | changed | claim past the evidence | the rule-versus-model attribution is narrowed to what the renders show: the model draws both animals when the joint prompt asks for them |
| 21 | paragraph | 266 | "The composed prediction and the joint prediction are built from the same four" | cut | repeat, contrast frames | four of its six sentences are said in chunks 2, 7, 9 and 12; the ceiling ("the correction can at best reach what joint prompting already produces") moved into chunk 14 and the trajectory clause into chunk 12 |

## Accepted, not yet written

Navigation: ⬅️ [The chunk map](#the-chunk-map) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

| # | Control | The accepted text |
|---|---|---|
| 1 | `improve`, option 2 | `\section{What Product-of-Experts Composition Drops: The Plurality Term}` |
| 2 | `improve`, author's opening, denoising-step sentence dropped | In this paper we use Stable Diffusion XL \citep{podell2023sdxl} as our latent diffusion model \citep{rombach2022high}. It consists of an autoencoder that compresses a $1024 \times 1024$ image into a $4 \times 128 \times 128$ latent and decodes it back, two text encoders that turn a prompt into the conditioning the network reads, and a U-Net that predicts the noise in a latent. A run starts from a Gaussian draw $x_T \sim \mathcal{N}(0, I)$ in this latent space, fixed by its random seed. At each denoising step, product-of-experts composition evaluates the U-Net on the same $x_t$ under three conditions, the first prompt, the second prompt, and no prompt at all. With no prompt, the network's prediction moves the latent toward images in general rather than toward either concept. These three predictions are then combined to form the PoE noise estimate given in \eqref{eq:poe-composition}, by adding the two conditional predictions and subtracting the unconditional one. Each conditional prediction contains the unconditional prediction plus a term carrying its own prompt. The network is never shown both prompts at once. |
| 3 | `improve`, change 2 | "Taking logarithms and passing to noise predictions gives \eqref{eq:poe-composition}, regrouped around the unconditional prediction, writing $p_{\mathrm{PoE}}$ for the product above." (first two sentences unchanged) |
| 4 | `improve`, change 1 | the align's third line, `\hat\epsilon_{\mathrm{PoE}}(x_t, t) &= \epsilon_\theta(x_t, t \mid c_1) + \epsilon_\theta(x_t, t \mid c_2) - \epsilon_\theta(x_t, t) \\`, deleted, and the next line's left side becomes `\hat\epsilon_{\mathrm{PoE}}(x_t, t)` |
| 7 | `improve` | "...and for two subjects contending for one region \citep{bradley2025mechanisms} it peaks between them rather than on an image holding both \citep{dutta2026steer}." (other sentences unchanged) |
| 9 | `improve`, option B after Figure 2 became the four-models grid | The same rule composes a butterfly with a meadow on three of the four models and returns a hybrid for a cat and a dog on all four (Figure~\ref{fig:two-regimes}), so the outcome follows the pair, and a larger model does not change it. The factor the rule drops, $R_t$, depends on the pair, but nothing in the run evaluates it, so the paper measures the gap between two predictions the network does produce. |
| 11 | `improve`, reading 1 | \includegraphics[width=\textwidth]{same-rule-on-four-models.pdf} / \caption{\textbf{A larger model does not rescue product-of-experts composition.} Rows are prompt pairs and columns are models of increasing size, each cell showing the joint prompt on the left and product-of-experts composition on the right. The same composition rule returns one hybrid animal for a cat and a dog on every model, so the failure is not solved by moving to a larger pretrained model.} |
| 14 | `improve` | The identity $\epsilon_{\mathrm{Mono}} = \hat\epsilon_{\mathrm{PoE}} + r_t$ holds by construction. What the paper measures is the size of $r_t$, how it changes across the denoising steps, and whether a model trained to predict it reaches concept pairs it never saw. (the second sentence unchanged) |
| 16 | `improve`, plus chunk 19's surviving clause | The strength axis separates the correction from any disturbance of the same size and schedule. The correction is a full noise-prediction vector at every step, so it has a direction as well as a magnitude, and at a given $\lambda$ every row of the grid carries the same magnitude. In Figure~\ref{fig:strength-grid} the hybrid holds while small amounts are added and separates into a cat and a dog before $\lambda$ reaches 1, and no control row reaches two animals anywhere on the axis. So composition is restored by the direction the correction points in; the three controls push the prediction just as hard and not one of them composes. One pair at one seed shows what the axis does, not how often it does it. |
| 17 | crop to λ 0 to 1 | \includegraphics[width=\textwidth]{how-much-is-added/strength-grid-0to1.pdf} |
| 18 | `improve` | \caption{\textbf{Only the measured correction composes.} A cat and a dog. A column is the strength $\lambda$ added at every step, up to the full correction at $\lambda = 1$; a row is what was added. The top row is the pair's own plurality term, and beneath it sit three controls, each breaking one thing about it while matching its size and schedule: another pair's term, the same term from a different seed, and the same term with its step order deranged. Nothing is added at $\lambda = 0$, so that column is one sample shown four times.} |
| 19 | `cut` | (deleted; its direction-and-magnitude clause moved into chunk 16) |
| 20 | `improve` | Independence can fail in more than one way, and the one this paper studies is spatial, since a patch of pixels that reads strongly as one animal cannot read equally strongly as the other. In every pair we study both concepts are foreground subjects, rather than a subject and a setting that could occupy different parts of the image. The same network draws two distinct animals when the joint prompt asks for them, so composition is not failing for want of a model that can render the pair. |
| 21 | `cut` | (deleted; the ceiling sentence moved into chunk 14 and the trajectory clause into chunk 12) |

## Originals held

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#figures) ➡️

| # | `was:` |
|---|---|
| 1 | `\section{Missing Implications: The Plurality Term}` |
| 2 | At each denoising step, product-of-experts composition evaluates the pretrained network on the same latent $x_t$ under three conditions, the first prompt, the second prompt, and no prompt at all. With no prompt, the network's prediction moves the latent toward images in general rather than toward either concept. These three predictions are then combined to form the PoE noise estimate given in \eqref{eq:poe-composition}, by adding the two conditional predictions and subtracting the unconditional one. Each conditional prediction contains the unconditional prediction plus a term carrying its own prompt. Three properties of this construction carry into the rest of the section: nothing is trained, the network is never shown both prompts at once, and the composed prediction is a fixed linear combination of three of its own outputs. |
| 3 | "Taking logarithms and passing to noise predictions gives the estimate the sampler uses, writing $p_{\mathrm{PoE}}$ for the product above." |
| 4 | lines 205 to 207: `\hat\epsilon_{\mathrm{PoE}}(x_t, t)` / `&= \epsilon_\theta(x_t, t \mid c_1) + \epsilon_\theta(x_t, t \mid c_2) - \epsilon_\theta(x_t, t) \\` / `&= \epsilon_\theta(x_t, t) + \sum_{i=1}^{2}` |
| 7 | "...and for two subjects contending for one region it peaks between them rather than on an image holding both \citep{bradley2025mechanisms, dutta2026steer}." |
| 9 | Nothing in the run evaluates $R_t$, so the paper measures the gap between the two predictions the network does produce instead. The same rule composes one pair and blends another (Figure~\ref{fig:two-regimes}), so whatever separates the two cases is not the rule itself. $R_t$ is a property of the two concepts and $r_t$ is a property of one model's two predictions, so no reading of size carries from one to the other, and two pairs cannot say whether the size of the correction predicts which pairs fail. When the correction has to arrive is what we measure next. |
| 11 | \includegraphics[width=0.62\textwidth]{same-rule-composes-one-pair-and-blends-another.pdf} / \caption{\textbf{The same rule composes one pair and blends another.} One real sample per prompt pair, seed 42, both sampled with uncorrected product-of-experts composition at the same settings. The butterfly pair puts both concepts in the picture; the animal pair returns one blended animal.} |
| 14 | The identity $\epsilon_{\mathrm{Mono}} = \hat\epsilon_{\mathrm{PoE}} + r_t$ holds by construction and is not a result, whereas the size of $r_t$, its structure across denoising steps, and its transfer to concept pairs held out of training are. |
| 16 | The strength axis separates the correction from any disturbance of the same size and schedule. In Figure~\ref{fig:strength-grid} the fusion holds until $\lambda$ reaches 0.75 and then resolves into a cat and a dog, while no control reaches two animals at any strength, so what restores composition is where the correction points rather than how hard the prediction is disturbed. The rows part company hardest past $\lambda = 1$, where the measured correction still gives two clean animals at $\lambda = 2$ and every control has collapsed into colour artefacts or a single chimera. One pair at one seed illustrates the strength axis rather than measuring it. |
| 17 | \includegraphics[width=\textwidth]{how-much-is-added/strength-grid.pdf} |
| 18 | \caption{\textbf{Only the measured correction composes, and over-driving it does no visible harm.} A cat and a dog at seed 9. A column is the strength $\lambda$ added at every step; a row is what was added. The top row is the pair's own plurality term, and beneath it sit three controls, each breaking one thing about it while matching its size and schedule: another pair's term, the same term from a different seed, and the same term with its step order deranged. Nothing is added at $\lambda = 0$, so that column is one sample shown four times, and the columns past $\lambda = 1$ carry the correction beyond the joint-prompt prediction.} |
| 19 | Product-of-experts sampling never sees the joint prompt. The difference $r_t$ therefore serves as a signal we can measure, one that establishes what the missing correction is, when in the denoising trajectory it matters, and why composition fails without it. $r_t$ is a full noise-prediction vector at every step, so it has a direction as well as a magnitude. |
| 20 | Independence can fail in more than one way, and the one this paper studies is spatial, since a patch of pixels that reads strongly as one animal cannot read equally strongly as the other. In every pair we study both concepts are foreground subjects, rather than a subject and a setting that could occupy different parts of the image. The same network produces two distinct animals when it is given the joint prompt, so the failure belongs to the composition rule and not to the model. |

## Figures

Navigation: ⬅️ [Originals held](#originals-held) | 📋 [TOC](#table-of-contents) | [Next](#open-questions-on-the-text) ➡️

| Figure | Image file | Opened | What to change | Slot decided |
|---|---|---|---|---|
| 2 | `paper/overleaf-iclr/figures/same-rule-on-four-models.pdf` (was `same-rule-composes-one-pair-and-blends-another.pdf`) | yes, the grid at 90 dpi and each surprising cell at full size | replaced by a new grid: rows butterfly × flower meadow, camel × forest, cat × dog; columns SD 1.4, SD 2.1, SDXL, SD 3.5; each cell joint prompt left, PoE right, seed 42 | not yet walked |
| 3 | `paper/overleaf-iclr/figures/how-much-is-added/strength-grid-0to1.pdf` | yes, at 400 dpi and column by column | cropped to λ 0 to 1: the columns past 1 are cut, since no other part of the paper uses λ above 1 and the wrong-seed row's second damaged face at λ 1.5 to 2 undercut "no control reaches two animals" | main text, unchanged position |

## Open questions on the text

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

- [x] **Does section 3 keep its own derivation of equation 1?** Yes, minus the line that restates equation 1. (blocks chunks 3 and 4) The
      introduction carries the rule and its assumption; cutting here means naming
      $p_{\mathrm{PoE}}$ where chunk 6 first needs it.
- [ ] ⚠️ **What may the new caption say about the camel row?** (blocks chunk 11) The pair is the GLIDE
      example in the README of Liu et al.'s released code; their paper's text never mentions a camel.
- [ ] ⚠️ **Does the wrong-seed row reach two animals past λ 1.25?** (blocks chunk 16) At 200 dpi a
      second small cat face appears behind the dog at λ 1.5 to 2.

## Routes and threads

Navigation: ⬅️ [Open questions](#open-questions-on-the-text) | 📋 [TOC](#table-of-contents) | [Next](#compile-log) ➡️

| Chunk | Where it went | What it owes back | Returned |
|---|---|---|---|
| 10, 11 | [the four-models figure plan](../../../plans/03-does-the-correction-cause-composition/plans/figures/12-the-same-rule-on-four-models.md) | `paper/overleaf-iclr/figures/same-rule-on-four-models.pdf` (three pairs by four models, joint and PoE per cell, seed 42) and the review file's cell-by-cell read. Drawn 2026-09-22 from jobs 58176 and 58190, 24 of 24 cells; read cell by cell in the review file | yes |

## Compile log

Navigation: ⬅️ [Routes and threads](#routes-and-threads) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| When | Chunks written | Build | Notes |
|---|---|---|---|
| 2026-09-22 | 1 to 4, 7, 9, 11, 12, 14, 16 to 18, 20, and the cuts of 19 and 21, across lines 194 to 266 of `paper/overleaf-iclr/iclr2027_conference.tex` | passed, pdflatex and bibtex on the laptop, no errors, no undefined references or citations, no overfull boxes | 11 pages, conclusion on page 10, matching the baseline. Figure 2 sits at `width=0.85\textwidth`: at full width the manuscript ran to 12 pages. `paper/iclr/` not written, nothing committed in the clone, nothing pushed to Overleaf |

## Next step

Navigation: ⬅️ [Compile log](#compile-log) | 📋 [TOC](#table-of-contents)

**Where a cold session resumes**

Section 3 is walked end to end and compiled into the Overleaf clone, and it builds. Owed: push the
clone to Overleaf from the cluster, decide whether `paper/iclr/` takes the same text, and settle the
open questions below.
