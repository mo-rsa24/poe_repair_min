# 🔎 Proofread map: section 3 of the ICLR manuscript, the plurality term

The walk's persistent state. Read it first on every invocation, write it before every round ends.

## Position in the walk

| Field | Value |
|---|---|
| Selection | `paper/overleaf-iclr/iclr2027_conference.tex` lines 194 to 266, `\section{Missing Implications: The Plurality Term}` up to `\section{Restoring the Plurality Term Restores Composition}` |
| Kind | manuscript section |
| Grain | paragraph |
| Cadence | sweep |
| Current chunk | 11 of 21, Figure 2's caption, waiting on the redrawn figure |
| Marks | 8 ok, 6 work, 1 cut?, 6 changed |
| Compiled | nothing yet |
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
| 9 | paragraph | 225 | "Nothing in the run evaluates $R_t$" | changed | symbol before its definition, no join, stacked ideas | alt A: figure sentence first, no $r_t$ before its definition, timing promise cut; the two-pairs caveat is dropped and the size point is left to chunk 21 |
| 10 | figure | 227 | `figures/same-rule-composes-one-pair-and-blends-another.pdf` | ok | | opened: a butterfly meadow with both concepts, and one white cat-dog animal; shows what the text says |
| 11 | caption | 230 | "The same rule composes one pair and blends another." | work | naming | "one blended animal" where the introduction says hybrid; the same words are drawn inside the image |
| 12 | paragraph | 234 | "We define the plurality term $r_t$ as the difference" | ok | | agrees with intro ¶8: two predictions of one model at the same latent |
| 13 | equation | 235 | eq. interaction-term | ok | | |
| 14 | paragraph | 240 | "The identity ... holds by construction and is not a result" | work | contrast frame, rule-of-three list | "is not a result, whereas ... are" |
| 15 | equation | 241 | eq. corrected-sampler | ok | | |
| 16 | paragraph | 247 | "The strength axis separates the correction from any disturbance" | work | numbers in prose, naming, claim past the picture | inline λ 0.75 and λ 2; "chimera"; the grid's λ 0.75 column already shows two animals, and the wrong-seed row past λ 1.25 needs a second look before "no control reaches two animals" stands |
| 17 | figure | 249 | `figures/how-much-is-added/strength-grid.pdf` | ok | | opened: four labelled rows by nine λ columns, labels on the thing |
| 18 | caption | 252 | "Only the measured correction composes, and over-driving it does no visible harm." | work | caption past the picture | one seed; at λ 1.75 and 2 both animals stay but the dog changes breed and pose |
| 19 | paragraph | 262 | "Product-of-experts sampling never sees the joint prompt." | work | claim past the evidence, no join | claims $r_t$ establishes "why composition fails"; "therefore" does not follow from the sentence before it |
| 20 | paragraph | 264 | "Independence can fail in more than one way" | work | claim past the evidence | "the failure belongs to the composition rule and not to the model": sampler error and model error are not separated, and the sampler-correctors journey poses exactly this as open |
| 21 | paragraph | 266 | "The composed prediction and the joint prediction are built from the same four" | cut? | repeat, contrast frames | restates chunks 9 and 12 after the results; its new facts are the four evaluations per step and that $r_t$ is read on the trajectory it corrects |

## Accepted, not yet written

Navigation: ⬅️ [The chunk map](#the-chunk-map) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

| # | Control | The accepted text |
|---|---|---|
| 1 | `improve`, option 2 | `\section{What Product-of-Experts Composition Drops: The Plurality Term}` |
| 2 | `improve`, author's opening, denoising-step sentence dropped | In this paper we use Stable Diffusion XL \citep{podell2023sdxl} as our latent diffusion model \citep{rombach2022high}. It consists of an autoencoder that compresses a $1024 \times 1024$ image into a $4 \times 128 \times 128$ latent and decodes it back, two text encoders that turn a prompt into the conditioning the network reads, and a U-Net that predicts the noise in a latent. A run starts from a Gaussian draw $x_T \sim \mathcal{N}(0, I)$ in this latent space, fixed by its random seed. At each denoising step, product-of-experts composition evaluates the U-Net on the same $x_t$ under three conditions, the first prompt, the second prompt, and no prompt at all. With no prompt, the network's prediction moves the latent toward images in general rather than toward either concept. These three predictions are then combined to form the PoE noise estimate given in \eqref{eq:poe-composition}, by adding the two conditional predictions and subtracting the unconditional one. Each conditional prediction contains the unconditional prediction plus a term carrying its own prompt. The network is never shown both prompts at once. |
| 3 | `improve`, change 2 | "Taking logarithms and passing to noise predictions gives \eqref{eq:poe-composition}, regrouped around the unconditional prediction, writing $p_{\mathrm{PoE}}$ for the product above." (first two sentences unchanged) |
| 4 | `improve`, change 1 | the align's third line, `\hat\epsilon_{\mathrm{PoE}}(x_t, t) &= \epsilon_\theta(x_t, t \mid c_1) + \epsilon_\theta(x_t, t \mid c_2) - \epsilon_\theta(x_t, t) \\`, deleted, and the next line's left side becomes `\hat\epsilon_{\mathrm{PoE}}(x_t, t)` |
| 7 | `improve` | "...and for two subjects contending for one region \citep{bradley2025mechanisms} it peaks between them rather than on an image holding both \citep{dutta2026steer}." (other sentences unchanged) |
| 9 | `improve`, alt A | The same rule composes one pair and blends another (Figure~\ref{fig:two-regimes}), so the rule alone does not decide the outcome. The factor the rule drops, $R_t$, depends on the pair, but nothing in the run evaluates it, so the paper measures the gap between two predictions the network does produce. |

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

## Figures

Navigation: ⬅️ [Originals held](#originals-held) | 📋 [TOC](#table-of-contents) | [Next](#open-questions-on-the-text) ➡️

| Figure | Image file | Opened | What to change | Slot decided |
|---|---|---|---|---|
| 2 | `paper/overleaf-iclr/figures/same-rule-composes-one-pair-and-blends-another.pdf` | yes, rendered at 70 dpi | replaced by a new grid: rows butterfly × flower meadow, camel × forest, cat × dog; columns SD 1.4, SD 2.1, SDXL, SD 3.5; each cell joint prompt left, PoE right, seed 42 | not yet walked |
| 3 | `paper/overleaf-iclr/figures/how-much-is-added/strength-grid.pdf` | yes, rendered at 200 dpi | not yet walked | not yet walked |

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
| 10, 11 | [the four-models figure plan](../../../plans/03-does-the-correction-cause-composition/plans/figures/12-the-same-rule-on-four-models.md) | `paper/overleaf-iclr/figures/same-rule-on-four-models.pdf` (three pairs by four models, joint and PoE per cell, seed 42) and the review file's cell-by-cell read. Tasks 1.1 to 1.3 done 2026-09-22; the load check (1.4) waits on a commit | no |

## Compile log

Navigation: ⬅️ [Routes and threads](#routes-and-threads) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| When | Chunks written | Build | Notes |
|---|---|---|---|

## Next step

Navigation: ⬅️ [Compile log](#compile-log) | 📋 [TOC](#table-of-contents)

**Where a cold session resumes**

Chunk 11, Figure 2's caption at line 230, once the four-models grid exists. The caption is rewritten
against the grid as rendered, and the `\includegraphics` path and width change with it. Chunk 9's
queued sentence ("The same rule composes one pair and blends another") is re-read against the grid
then too.
