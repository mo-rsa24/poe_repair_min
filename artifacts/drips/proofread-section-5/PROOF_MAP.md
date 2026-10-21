# 🔎 Proofread map: section 5, "Learning the Plurality Term"

The walk's persistent state. Read it first on every invocation, write it before every round ends.

## Position in the walk

| Field | Value |
|---|---|
| Selection | `paper/overleaf-iclr/iclr2027_conference.tex` lines 315 to 501 (section 5, Results, Limitations) |
| Kind | manuscript section |
| Grain | paragraph |
| Cadence | sweep |
| Current chunk | ¶16, "All runs use 50 DDIM steps" (appendix; walk otherwise paused at ¶21) |
| Marks | 13 ok, 7 work, 4 move, 1 fuse, 3 cut or cut?, 0 changed |
| Compiled | 2026-09-25: section 5, Results, Limitations, `app:loss-weight`, the macro block and four bib entries are in `paper/overleaf-iclr/iclr2027_conference.tex`. Builds with `pdflatex`, 16 pages, no errors, no undefined references or citations. |
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

The method section of the ICLR submission: the section that trains an adapter to produce the
plurality correction from the two concept prompts alone, so a corrected run no longer needs the
joint prompt.

**Why this pass is happening**

It has to read well and be clear, and the two architecture diagrams it carries (Figures 6 and 7)
have problems that a caption cannot fix.

**What the text is trying to do**

Say what was trained, on what data, with what loss, and what it costs at inference, so a reader
can tell that the correction now comes out of the weights rather than out of the joint prompt.

## The chunk map

Navigation: ⬅️ [Quick context](#quick-context-what-this-is) | 📋 [TOC](#table-of-contents) | [Next](#accepted-not-yet-written) ➡️

Second pass, 2026-09-25, over lines 315 to 501: section 5, Results and Limitations together,
because the ask is a restructure. Derivations go to a new appendix that starts over and explains
each step; the main text keeps one clean equation per idea with a pointer; Results becomes a
subsection of section 5; "Where it breaks" and "Limitations and outlook" become one or two
paragraphs of failures and limitations beside the six-panel failure gallery.

| # | Kind | First words or file | Mark | The read |
|---|---|---|---|---|
| 1 | heading | `\section{Learning the Plurality Term}` | ok | names what is under it |
| 2 | ¶ | "In the previous section we measured" | ok | the bridge from section 4, does its job |
| 3 | ¶ | "We construct the training pool" | changed | the "Thus" sentence folded in; "can compose under a joint prompt" narrowed to what the filter checks; noise sentences kept |
| 4 | ¶ | "The training data consists of states" | cut | the user's call; its first sentence (uncorrected states) survives only in Appendix C |
| 5 | ¶ | "We train a LoRA adapter" | ok | kept on the user's word; the "every cross-attention layer" wording against the cross+self run is noted under open questions |
| 6 | ¶ + Eq 14 | "The two concepts are never passed" | ok | defines the three adapted passes; every symbol introduced |
| 7 | ¶ + Eq 15 | "Each concept prediction is first classifier-free guided" | changed | one displayed line stays; the expansion opens the new derivations appendix, in the user's wording |
| 8 | ¶ + Eq 16 | "During training, these same three adapted passes" | ok | the target, stated once |
| 9 | ¶ + Eq 17 | "Only the LoRA parameters" | ok | the loss, with the weight already pointed at the appendix |
| 10 | ¶ + Eqs 18, 19 | "During training, the joint prompt is used only" | changed | the user's text: both equations stay, a closing sentence and an appendix pointer added; the loss expansion opens the appendix's second paragraph |
| 11 | ¶ + Eqs 20, 21 | "This relation explains why the joint prompt" | changed | Eq 20 and the early-steps claim kept; Eq 21, the wrong appendix pointer and the repeated close cut |
| 12 | ¶ + Eq 22 | "At inference, only $c_1$, $c_2$, and $\emptyset$" | fuse | says what ¶11 and ¶13 say; one inference paragraph with one equation |
| 13 | ¶ + Eq 15 | "This form also gives the learned analogue" | changed | reopened 2026-09-25: introduces the equation a second time, after ¶12's last sentence already says "gives the corrected sampler below"; "This form" has no clear antecedent; its left side is $\hat\epsilon_{\mathrm{corr}}$ where Eq 13 writes $\hat\epsilon_\lambda$ |
| 14 | figure | `four-pairs-four-methods.png` | ok | rebuilt today, sixteen cells, read against the caption |
| 15 | caption | "The adapter repairs the composition" | ok | claims no more than the picture carries |
| 16 | ¶ | "All runs use 50 DDIM steps" | changed | first paragraph of `app:implementation`; one topic per sentence, same facts; the table was offered and declined. Batch size and what an epoch is remain unstated |
| 17 | ¶ | "What remains to test is whether" | cut | the user's call |
| 18 | heading | `\section{Results}` | work | becomes `\subsection{Results}` under section 5 |
| 19 | ¶ | "The adapter composes all four pairs" | changed | reopened 2026-09-25 at sentence grain: "The animal pairs come from the concept-disjoint evaluation split; the tent and lighthouse pairs lie outside the training pool" restates the sentence before it, and is not quite true for the lighthouse, which is trained on as a_lighthouse__x__a_sailboat in pair_pool_v58.yaml |
| 20 | ¶ | "The second column shows" | ok | kept on the user's word; "in each case ... not preserved as distinct entities" reads past row 4, where the lighthouse only shrinks |
| 21 | ¶ | "CO3 also recovers the compositions" | ok | the CO3 caveat, needed once |
| 22 | ¶ | "The bottom two rows are especially informative" | cut? | repeats ¶19's last sentence and the caption; survives deletion |
| 23 | heading | `\subsection{Where it breaks}` | work | becomes the failures-and-limitations paragraph, with the gallery |
| 24 | ¶ | "Training the adapter is not stable" | work | keeps its three causes; each one gets a panel of the gallery to point at |
| 25 | figure | `how-a-render-fails.png` (new, the paper layout of the gallery) | work | read 2026-09-25: six panels, each a real render with the telling patch boxed and enlarged; title, cause lines and provenance dropped for the paper, gaps closed. caption accepted; slot: the failures subsection, a `[!t]` float declared before chunk 24, `\textwidth`, about four-tenths of a page. Budget from the 2026-09-25 build: main text ends on page 11; Limitations plus the moved derivations free about one page, so the text should close on page 10 |
| 26 | heading | `\section{Limitations and outlook}` | cut | the section goes |
| 27 | ¶ | "How much correction is needed" | cut? | one sentence of it ("measure but cannot predict") may join ¶24; the rest goes |
| 28 | ¶ | "What the paper leaves is a correction" | cut | the outlook; the conclusion already carries it |

## Accepted, not yet written

Navigation: ⬅️ [The chunk map](#the-chunk-map) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

| # | Control | The accepted text |
|---|---|---|
| 16b | compiled 2026-09-25, `improve`, 4 changes accepted, prose kept | Every run samples with 50 DDIM steps at guidance weight 7.5, at $1024\times1024$ resolution, in fp16. The adapter has rank \adrank{} with $\alpha = r$, so its update is added at full scale. It adds \adparams{}\,M trainable parameters to the \adbase{}\,B-parameter frozen base. We train it with AdamW for \adsteps{} optimizer steps, grouped as \adepochs{} epochs of 50 steps. The learning rate starts at $10^{-4}$ and decays to zero on a cosine schedule, and gradients are clipped at norm 1.0. |
| 13b | compiled 2026-09-25, `improve`, 4 changes accepted | ...Writing the learned correction as $\hat r_\psi = \hat\epsilon^{\psi}_{\mathrm{PoE}} - \hat\epsilon_{\mathrm{PoE}}$ gives the learned analogue of \eqref{eq:corrected-sampler}, with $\hat r_\psi$ in place of $r_t$: [equation, left side $\hat\epsilon_\lambda$ in place of $\hat\epsilon_{\mathrm{corr}}$] Appendix~\ref{app:training-inference} derives the training and inference relation. |
| 19b | `improve`, user-directed, compiled 2026-09-25 | All four are held-out pairs, and for each one the joint-prompt, plain-product, and adapter samples share the starting noise. |
| 3 | `cut`, confirmed | *(deleted)* Its one load-bearing fact, one adapter across every pair, is carried by paragraph 5's "one set of weights $\psi$ is trained on all 48 pairs together". No seam edit needed on paragraph 3. |
| 2 | `improve`, 8 rounds | In the previous section we measured the plurality term along composed runs and showed that adding a small amount of it back, early in the run, restores composition. In this section we learn it. We train an adapter inside the model so that its own composed prediction from the current latent and the two concept prompts carries the correction at each step. During training the joint prompt supplies the target the adapter learns from; at inference it is withheld, and the corrected run does not need it. |

**What changed in chunk 2, and why.** "A model that produces the correction" became "an adapter
inside the model so that its own composed prediction carries the correction", because the old
phrasing builds a second network that emits a correction vector, and nothing adds anything at
inference. The closing three sentences became one, ordered training first and inference second at
the user's direction. 89 words before, 89 after.

| 3 | `improve`, accepted | We train a low-rank adapter (LoRA) \citep{hu2022lora} on the query, key and value projections of every cross-attention layer, with the base model frozen. We do not adapt the output projections, the self-attention layers or the feed-forward layers. The adapter stays attached on every pass, including the one under the null token, so a corrected run's unconditional prediction comes from the adapted model rather than the frozen one. Appendix~\ref{app:adapter-ablations} reports what changes when the rank and the adapted layer set are varied. |
| 5b | settled, five paragraphs plus four displays | The adapter and training block, final: LoRA on cross-attention q/k/v with the base frozen giving $\epsilon_{\theta,\psi}$; queries from the latent, keys and values from the prompt tokens; one pass per prompt. Three passes per step because the concepts are never passed together, with $c_1$, $c_2$, $c_{12}$ named as cat, dog and the joint prompt, and $a_\psi$, $b_\psi$, $u_\psi$ defined in `eq:adapted-predictions`. The guided composition expanded over four lines to $u_\psi + w[a_\psi + b_\psi - 2u_\psi]$ in `eq:adapted-poe`, with the $w=1$ check back to `eq:poe-composition` and $w = 7.5$ stated. The frozen guided joint-prompt target in `eq:mono-target`. The weighted squared error in `eq:adapter-loss`, only $\psi$ updated. Conditioning uses section 2's mid bar throughout. |

**Decisions that bind the whole section.** The adapter spec is folded into the head of the training
paragraph, so the order is: what the correction is, the pool, the training states, the adapter and
its training, the loss weight, the run. Caching is never mentioned: it is an implementation detail,
and the load-bearing fact is that the joint prompt exists at training and not at inference. What
must not go with it is that the training states come from *uncorrected* composition while inference
visits corrected ones.

**Debts section 5 now owes section 3.** State once whether the hat means guided, then make
`eq:poe-composition` (written raw, named $\hat\epsilon_{\mathrm{PoE}}$) and `eq:interaction-term`
(where $\epsilon_{\mathrm{Mono}}$ has no hat but is guided at 7.5) agree with it.

**Moved to the appendix, not dropped.** The residual splits into $w$ times the composition's own
error minus $(w-1)$ times the drift of the adapted null branch. Algebra verified. It belongs beside
the `v58-04-freezenull` comparison (`mwrgppyw`), which pins the null branch and reads compose 0.0
at 8,461 steps.

| 5c | accepted, user-authored verbatim | The implicit-residual block as the user wrote it: the decomposition as its own display, substitution into `eq:adapter-loss` giving `eq:implicit-residual`, $\hat r_\psi$ defined in `eq:learned-residual`, "Importantly, $\hat r_\psi$ is not produced by a separate residual head", the correction emerging from the difference between adapted and frozen compositions, and $\hat\epsilon_{\mathrm{corr}} = \hat\epsilon_{\mathrm{PoE}} + \lambda\hat r_\psi$. Wording kept as authored: "interaction residual", "Importantly", and $\hat\epsilon_{\mathrm{corr}}$ rather than $\hat\epsilon^{\psi}_{\lambda}$. |

**One outstanding correction in this block.** "This form also gives the correction used by
`\eqref{eq:corrected-sampler}` directly" misstates that equation, which uses the measured $r_t$ and
not $\hat r_\psi$. Proposed fix, not yet accepted: "the learned analogue of
`\eqref{eq:corrected-sampler}`".

**Why this block exists.** A reviewer arriving from the intro ("a model can learn to produce the
residual") and from `eq:corrected-sampler` will expect a residual head whose output is added at
inference. Without the substitution, "we fit the prediction rather than $r_t$" reads as a dodge. With
it, the implicit design is a derivation.

**Spelling debt.** The paper is British ("minimising" at line 192, "memorised" in the last
paragraph). The accepted training block writes "We minimize the weighted squared error".

| 5d | accepted | Main text keeps only: "The weight $\gamma_t$ measures the error on the clean latent rather than on the noise, which concentrates the loss on the earliest steps; Appendix~\ref{app:loss-weight} gives its form." Placed straight after `eq:adapter-loss`. No values in the main text. |
| A1 | drafted | `app:loss-weight`: the Tweedie inversion and the $(1-\bar\alpha_t)/\bar\alpha_t$ factor, cited to \citet{kingma2023understanding} for the weighting view and \citet{li2025basics} for preferring the clean-latent distance; the factor's range 172 to 0.002 on SDXL's 50-step DDIM grid; $\kappa = 22$, $Z = 7.43$; the equivalence to Min-SNR at $1/\kappa \approx 0.045$ against their default 5; the clamp binding through step 10 with $\gamma_t = 2.96$ flat and $2.3\times10^{-4}$ at step 49; the two consequences, that the objective is noise-space where the clamp binds and that the flat window is Section 4's decisive window by construction; and the schedule dependence. |

**Where the concession lives.** That the objective concentrates on the early window by construction
rather than by discovery is stated in the appendix, not the main text. In the appendix it reads as
rigour; in the main text it reads as a weakness volunteered.

| 5e | accepted, user-authored with five corrections | The training-to-inference bridge: joint prompt used only to define the target; frozen model evaluated under $c_{12}$ **and** $\emptyset$; `eq:interaction-term` giving $\hat\epsilon_{\mathrm{Mono}} = \hat\epsilon_{\mathrm{PoE}} + r_t$; `eq:training-implicit` as what the objective enforces; `eq:learned-residual` and `eq:inference-implicit` as the exact decomposition; the plurality term learned through the adapter's changes to the three per-prompt predictions; inference evaluating only $c_1$, $c_2$, $\emptyset$ from $x_T \sim \mathcal{N}(0,I)$, with the final equation showing $\hat\epsilon^{\psi}_{\mathrm{PoE}} = \hat\epsilon_{\mathrm{PoE}} + \hat r_\psi \approx \hat\epsilon_{\mathrm{PoE}} + r_t$; and inference retaining the same three prompt evaluations per step as ordinary product-of-experts composition. Corrections applied: the frozen null evaluation added, "selected" for "sampled", "plurality term" for "interaction", "each step evaluates the network to give" for "each DDIM step produces", both prose uses of PoE spelled out. |
| 5f | accepted | The qualification on $\hat r_\psi \approx r_t$: "Together with \eqref{eq:training-implicit}, training makes $\hat r_\psi \approx r_t$ over the early steps, which Section~\ref{sec:analysis} found is where the outcome is decided. Agreement falls away over the rest of the run, and Appendix~\ref{app:loss-weight} reports it step by step." No $\gamma_t$ here: its gloss a page earlier already says the loss concentrates early. |

**Why the qualification exists.** $\hat r_\psi \approx r_t$ was the block's load-bearing claim and was
unqualified, while the runs measure it: `learned_actual_cosine` is about 0.45 over the early bucket
and near zero over the commit and late buckets, on both `tiiz0c0c` and `v58-01-plain`. The
qualification must not present the early agreement as confirming Section 4's timing result, since
the loss weights those steps heaviest by design.

**`app:loss-weight` now has two jobs:** the weight's form and values, and the step-by-step agreement
between $\hat r_\psi$ and $r_t$.

| R1 | accepted, user-authored with three corrections | Results paragraphs 1 to 4, result-first: the adapter composes all four pairs using only the two concept prompts and the null token, separating cat and dog, elephant and penguin, placing the tent in front of the mountain and restoring the lighthouse, with the joint prompt never evaluated and three predictions per step; the second column as the uncorrected product-of-experts samples, animals fusing, tent and mountain collapsing to one cone, the lighthouse reduced; CO3 recovering the compositions but retaining the joint prompt, and its different starting noise meaning composition-level rather than pixel-wise comparison; the bottom two rows never trained on, with the tent and mountain case resolving a geometric collapse. Corrections: "null token" for "null prompt", product-of-experts spelled out, and "the tent and mountain case" for an en-dashed compound. |

**Results section structure agreed.** `\section{Discussion}` is replaced by `\section{Results}`
carrying the new four-pairs-by-four-methods figure. The old Discussion's six paragraphs split:
the attention-maps paragraph is held pending a finding, `fig:adapter-samples` and its caption go to
the appendix, the method-recap paragraph is cut except its transfer sentence, the training-instability
paragraph becomes the Results subsection "Where it breaks", and the last two become
`\section{Limitations and outlook}`.

**The restructure (accepted 2026-09-25, with the improve; previewed, not yet compiled)**

The queue is one script, `apply_edits.py` in the session scratchpad, which rebuilds the preview
from the untouched manuscript. Every sentence it places is either the manuscript's original or
text the user wrote in this walk; the only non-vetted words are four appendix run-in headings and
the one pointer sentence under Figure 6. It supersedes the per-chunk entries for chunks 6 to 13
below, including the fused inference paragraph, which is dropped.

1. Chunk 3 as accepted.
2. Main text, in original order: the loss paragraph (Eq 14 and the γ pointer); the user's
   "During training, the joint prompt is used only…" paragraph; the user's Eq 18 and Eq 19 text
   and its closing "Thus, … Appendix B expands this relation through the training loss."; the
   $\hat r_\psi$ definition as accepted in chunk 11; the original two closing inference
   sentences, the first without its leading "Thus,".
3. Appendix B, "Derivations for the adapter", in original order: the user's intro; the three
   adapted passes (original, Eq 18 there); the user's adapted-product expansion carrying the
   `eq:adapted-poe` label; the training target (original, `eq:mono-target`); the user's "What the
   objective enforces" paragraph with its own lead-in and the loss expansion; the original
   inference paragraph through Eq 22 and the original Eq 23 paragraph.
4. `\section{Results}` becomes `\subsection{Results}` and the `\FloatBarrier` before it goes.
5. "Where it breaks" and "Limitations and outlook" move verbatim to Appendix C.
6. The failure gallery with the accepted caption, fixed in place with `[H]` directly above its one
   pointer sentence, both on page 10; this leaves about eight empty lines at the foot of page 9.

7. Chunk 4 (the training data) cut from the main text, on the user's call.
8. Chunk 17 ("What remains to test…") cut from the main text, on the user's call.
9. Chunk 16 (the run details) moved verbatim to a new appendix section, "Training and sampling
   details" (`app:training-details`), after Appendix B.
10. Floats: Figure 5 declared before the LoRA paragraph (`[!t]`), Figure 6 declared before
    `\subsection{Results}` with `[!b]`; this replaces the `[H]` fix in item 6, which overflowed
    page 9 once chunk 4 was cut.

11. Figure 6's caption text becomes the Results paragraph that discusses it, the user's accepted
    wording verbatim, opened by "Figure~\ref{fig:how-a-render-fails} shows five ways a corrected
    cat--dog render goes wrong." and closed by the Appendix C pointer; the caption becomes a
    description of the panels: "Six cat--dog renders: the uncorrected product-of-experts sample
    (top left) and five corrected runs, each labelled with the failure it shows. The red box marks
    the patch enlarged in the corner of the same panel.", under the same bold title.
12. Float placement chosen by an exhaustive search (`search_floats.py` in the scratchpad) over six
    anchors for Figure 5, five for Figure 6 and five placement options each, scored on the page
    distance from each figure to its citing sentence, blank stretches and overflow.

Preview build 2026-09-25, after items 7 to 10: 17 pages, no errors, no undefined labels, no
overfull pages. Figure 5 page 8 with the text reading it on page 9 (facing pages in print);
Figure 6 at the foot of page 9 with its sentence at the top of page 10; Conclusion page 10;
Appendix B page 12, training details page 13, Appendix C page 14.

**Chunks 12 and 13, inference (fuse, drafted 2026-09-25, awaiting acceptance)**

was: chunk 12 ("At inference, only $c_1$, $c_2$, and $\emptyset$ are evaluated. ... Eq 22 ... Thus, the joint-prompt information ... Inference therefore retains the same three prompt evaluations per step as ordinary product-of-experts composition.") and chunk 13 ("This form also gives the learned analogue of \eqref{eq:corrected-sampler} directly:" Eq `eq:corrected-sampler-learned`).

now, main text (one paragraph, no display):

```latex
At inference only $c_1$, $c_2$ and $\emptyset$ are evaluated. Each step computes $a_\psi$, $b_\psi$
and $u_\psi$ and combines them by \eqref{eq:adapted-poe}, so the sampler follows
$\hat\epsilon^{\psi}_{\mathrm{PoE}} = \hat\epsilon_{\mathrm{PoE}} + \hat r_\psi \approx
\hat\epsilon_{\mathrm{PoE}} + r_t$ without ever evaluating $c_{12}$. This is the corrected sampler
of \eqref{eq:corrected-sampler} at $\lambda = 1$, with the learned $\hat r_\psi$ in place of the
measured $r_t$, and it keeps the three prompt evaluations per step of ordinary product-of-experts
composition. Every result below uses $\lambda = 1$.
```

now, appendix, third paragraph of `app:adapter-derivations`:

```latex
\paragraph{Inference without the joint prompt.}
At sampling time the adapted network is evaluated on $c_1$, $c_2$ and $\emptyset$ only, and the
three predictions are combined by \eqref{eq:adapted-poe}. By \eqref{eq:learned-residual} the
result is $\hat\epsilon^{\psi}_{\mathrm{PoE}} = \hat\epsilon_{\mathrm{PoE}} + \hat r_\psi$, and by
\eqref{eq:training-implicit} this approximates $\hat\epsilon_{\mathrm{PoE}} + r_t$, the corrected
prediction of Section~\ref{sec:analysis}. For a correction strength other than one,
\begin{equation*}
    \hat\epsilon_{\mathrm{corr}}
    = \hat\epsilon_{\mathrm{PoE}} + \lambda \hat r_\psi
    = \hat\epsilon_{\mathrm{PoE}}
      + \lambda\left(\hat\epsilon^{\psi}_{\mathrm{PoE}} - \hat\epsilon_{\mathrm{PoE}}\right),
\end{equation*}
which needs both the adapted and the frozen composition at every step, so six evaluations instead
of three. The joint prompt enters neither.
```

Check before compile: that no figure in Results or the appendices is rendered at a $\lambda$ other than 1 (the strength-sweep appendix is, by design; the sentence says "every result below", so confirm the sweep is not "below" in the reader's sense, or narrow it to "every render in Section~\ref{sec:results}").

**Chunk 11, the decomposition (accepted 2026-09-25)**

was: "This relation explains why the joint prompt is no longer needed at inference. Defining the change induced by the adapter as" Eq `eq:learned-residual`, "we have the exact decomposition" Eq `eq:inference-implicit`, "Together with ... over the early steps, which Section~\ref{sec:analysis} found is where the outcome is decided. Agreement falls away over the rest of the run, and Appendix~\ref{app:loss-weight} reports it step by step. The adapter therefore learns, ..."

now, main text (opening sentence restored on the user's word, 2026-09-25):

```latex
This relation explains why the joint prompt is no longer needed at inference. Defining the change induced by the adapter as
\begin{equation}
    \hat r_\psi = \hat\epsilon^{\psi}_{\mathrm{PoE}} - \hat\epsilon_{\mathrm{PoE}},
    \label{eq:learned-residual}
\end{equation}
training makes $\hat r_\psi \approx r_t$ over the early steps, which Section~\ref{sec:analysis} found is where the outcome is decided.
```

now, appendix: the closing sentence of chunk 10's appendix paragraph gains the clause "writing $\hat r_\psi$ for this change, the last line of \eqref{eq:adapter-loss-expanded} is a weighted regression of $\hat r_\psi$ onto $r_t$." placed after "approximate the missing interaction $r_t$".

Why: Eq 21 was Eq 20 rearranged and neither label is referenced; "Appendix~\ref{app:loss-weight} reports it step by step" pointed at an appendix that reports $\gamma_t$, not agreement.

**Chunk 10, what training enforces (accepted 2026-09-25; both texts are the user's)**

was: three sentences restating chunk 8, the unnumbered equation $\hat\epsilon_{\mathrm{Mono}} = \hat\epsilon_{\mathrm{PoE}} + r_t$, "The training objective therefore enforces", and `eq:training-implicit`.

now, main text:

```latex
During training, the joint prompt is used only to define what the adapted
composition should reproduce. At each selected state $(x_t,t)$, the adapted
network is evaluated separately on $c_1$, $c_2$, and $\emptyset$ to form
$\hat\epsilon^{\psi}_{\mathrm{PoE}}$. The frozen model is evaluated on the
joint prompt $c_{12}$ and the null prompt $\emptyset$ to form the mono target
$\hat\epsilon_{\mathrm{Mono}}$.

By \eqref{eq:interaction-term}, this target can be written as
\begin{equation}
    \hat\epsilon_{\mathrm{Mono}}
    =
    \hat\epsilon_{\mathrm{PoE}} + r_t,
\end{equation}
where $\hat\epsilon_{\mathrm{PoE}}$ denotes the frozen model's PoE composition
at the same $(x_t,t)$ and $r_t$ is the interaction missing from that
composition. The training objective therefore enforces
\begin{equation}
    \hat\epsilon^{\psi}_{\mathrm{PoE}}
    \approx
    \hat\epsilon_{\mathrm{PoE}} + r_t.
    \label{eq:training-implicit}
\end{equation}
Thus, the joint prompt supplies the missing interaction during training,
while the adapted composition itself is formed only from $c_1$, $c_2$, and
$\emptyset$. Appendix~\ref{app:adapter-derivations} expands this relation
through the training loss.
```

now, appendix, second paragraph of `app:adapter-derivations`:

```latex
\paragraph{What the objective enforces.}
The target in \eqref{eq:mono-target} is the frozen model's guided prediction
under the joint prompt $c_{12}$. By \eqref{eq:interaction-term}, the same
prediction can be written at each $(x_t,t)$ as
\begin{equation*}
    \hat\epsilon_{\mathrm{Mono}}
    =
    \hat\epsilon_{\mathrm{PoE}} + r_t,
\end{equation*}
where $\hat\epsilon_{\mathrm{PoE}}$ is the frozen model's PoE composition
formed from $c_1$, $c_2$, and $\emptyset$, and $r_t$ is the interaction
missing from that composition.

Substituting this decomposition into \eqref{eq:adapter-loss} gives
\begin{align}
\mathcal{L}(\psi)
&=
\mathbb{E}_{t,x_t}
\left[
\gamma_t
\left\|
\hat\epsilon^{\psi}_{\mathrm{PoE}}
-
\hat\epsilon_{\mathrm{Mono}}
\right\|_2^2
\right]
\nonumber\\
&=
\mathbb{E}_{t,x_t}
\left[
\gamma_t
\left\|
\hat\epsilon^{\psi}_{\mathrm{PoE}}
-
\left(
\hat\epsilon_{\mathrm{PoE}} + r_t
\right)
\right\|_2^2
\right]
\nonumber\\
&=
\mathbb{E}_{t,x_t}
\left[
\gamma_t
\left\|
\left(
\hat\epsilon^{\psi}_{\mathrm{PoE}}
-
\hat\epsilon_{\mathrm{PoE}}
\right)
-
r_t
\right\|_2^2
\right].
\label{eq:adapter-loss-expanded}
\end{align}

The objective therefore makes the change induced by the adapter,
$\hat\epsilon^{\psi}_{\mathrm{PoE}}
-
\hat\epsilon_{\mathrm{PoE}}$,
approximate the missing interaction $r_t$. Equivalently,
\begin{equation*}
    \hat\epsilon^{\psi}_{\mathrm{PoE}}
    \approx
    \hat\epsilon_{\mathrm{PoE}} + r_t,
\end{equation*}
which is \eqref{eq:training-implicit}. The joint prompt is needed only to
construct the training target; the adapted PoE prediction remains a
composition of the separate concept prompts and the null prompt.
```

**Chunk 7, the adapted product (move, accepted 2026-09-25; appendix text is the user's)**

was: the four-line align `eq:adapted-poe` with its two lead sentences and the $w=1$ sentence.

now, main text:

```latex
Each concept prediction is first classifier-free guided against the same adapted null prediction with guidance weight $w$. The two guided predictions are then added, and the shared null prediction is subtracted once, which gives
\begin{equation}
\hat\epsilon^{\psi}_{\mathrm{PoE}} = u_\psi + w\left[ a_\psi + b_\psi - 2u_\psi \right],
\label{eq:adapted-poe}
\end{equation}
with the expansion in Appendix~\ref{app:adapter-derivations}. At $w=1$ this reduces to $a_\psi + b_\psi - u_\psi$, the composition rule of \eqref{eq:poe-composition}. In our experiments $w=7.5$.
```

now, appendix (new section before `\section{The loss weight}`; chunks 10 to 12 add their paragraphs to it):

```latex
\section{Derivations for the adapter}
\label{app:adapter-derivations}

This appendix expands the three relations used in
Section~\ref{sec:method}: how the three adapted predictions form the PoE
composition, how the training objective relates this composition to the
missing interaction, and why the joint prompt is not needed at inference.

\paragraph{The adapted product.}
At a state $(x_t,t)$, the adapted network is evaluated separately on
$c_1$, $c_2$, and $\emptyset$, producing $a_\psi$, $b_\psi$, and $u_\psi$
as defined in \eqref{eq:adapted-predictions}. Each concept prediction is
first classifier-free guided against the same adapted null prediction.
The two guided predictions are then added, and the shared null prediction
is subtracted once:
\begin{align*}
\hat\epsilon^{\psi}_{\mathrm{PoE}}
&=
\underbrace{\left[
u_\psi + w(a_\psi-u_\psi)
\right]}_{\text{guided }c_1}
+
\underbrace{\left[
u_\psi + w(b_\psi-u_\psi)
\right]}_{\text{guided }c_2}
-u_\psi
\\
&=
u_\psi
+w(a_\psi-u_\psi)
+w(b_\psi-u_\psi)
\\
&=
u_\psi
+w\left[
(a_\psi-u_\psi)
+
(b_\psi-u_\psi)
\right]
\\
&=
u_\psi
+w\left[
a_\psi+b_\psi-2u_\psi
\right].
\end{align*}
This recovers \eqref{eq:adapted-poe}. At $w=1$, the expression reduces to
$a_\psi+b_\psi-u_\psi$, which is the PoE composition rule in
\eqref{eq:poe-composition}. We use $w=7.5$ throughout our experiments.
```

**Chunk 3, the training pool (accepted 2026-09-25)**

was: "We construct the training pool from concept pairs that the model can compose under a joint prompt, ... Thus, holding out ``a cat'' and ``a dog'' also removes every training pair containing either concept, so the adapter encounters both concepts for the first time at evaluation. We similarly hold out ..."

now: "We construct the training pool from concept pairs for which a joint-prompt render shows both concepts, since a pair that fails even under joint prompting provides no useful target for the adapter to imitate. For each candidate pair, we generate one image using plain joint prompting and one using CO3 \citep{dutta2026steer}, and retain the pair if either image contains both concepts. This filtering leaves \trainpairs{} training pairs, with the remaining pairs reserved for evaluation. The split is concept-disjoint: no concept appearing in a held-out pair occurs in any training pair, so holding out ``a cat'' and ``a dog'' removes every training pair containing either concept. We similarly hold out the starting noise used for evaluation. Because the same starting noise is shared across pairs, including it during training would expose the adapter to the exact noise from which an evaluation trajectory begins."

Changes: the filter described as what was checked (a joint-prompt render, plain or CO3, shows both); the "Thus" sentence folded into the concept-disjoint sentence. The noise sentences kept on the user's word. 148 words to 133.

**Chunk 25, caption (accepted 2026-09-25)**

```latex
\caption{\textbf{Five ways a corrected cat--dog render goes wrong.}
The uncorrected product collapses the two concepts into a single hybrid animal (top left), while
unsuccessful corrected runs depart from this failure in different ways. Some introduce unrelated
content, such as human figures, and others lose semantic structure almost entirely. Restricting
adaptation to the null-token prediction changes the image style without recovering a second
animal. Broader training can recover plurality without preserving identity, producing an extra
animal, while an image-target variant with a cached null-token prediction separates the
composition but replaces the requested dog with a horse.}
\label{fig:how-a-render-fails}
```

**Figure 5, `four-pairs-four-methods.png`: `improve`, written 2026-09-25 (see the compile log).** Three edits, each
refusable on its own.

**Edit A, the layout.** The two side-by-side 2×4 halves become one 4×4 grid at 0.57 of the text
width, with the Results text wrapped on its left. Each image goes from about 0.63 in to about
0.82 in wide. The column headers are cropped out of the PNG and typeset at `\scriptsize` to match
the two-line side labels. Tested in a scratch copy: 15 pages as before, Figure 5 on page 8,
Figure 6 and the Conclusion still on page 9, no overlap and no gap with the wrap at 27 lines.

was: the `\begin{figure}[!htbp]` block at `iclr2027_conference.tex:383`, two `minipage`s of
`0.495\textwidth`, images at `0.925\linewidth` (left) and `0.975\linewidth` (right), headers inside
the PNG. It moves to sit directly before the paragraph "The adapter composes all four pairs".

```latex
\begin{wrapfigure}[27]{r}{0.57\textwidth}
\vspace{-12pt}
\centering\setlength{\tabcolsep}{0pt}\begin{tabular}{@{}c@{\hspace{2pt}}c@{}}
& {\scriptsize\makebox[0.1361\textwidth][l]{Joint prompt}\makebox[0.1361\textwidth][l]{Plain product}\makebox[0.1361\textwidth][l]{CO3}\makebox[0.1361\textwidth][l]{Ours}} \\[-1pt]
\rotatebox{90}{\scriptsize\makebox[0.1361\textwidth]{\shortstack{cat and\\dog}}} & \includegraphics[width=0.544\textwidth,trim=34 932 0 40,clip]{four-pairs-four-methods.png} \\[-1pt]
\rotatebox{90}{\scriptsize\makebox[0.1361\textwidth]{\shortstack{elephant and\\penguin}}} & \includegraphics[width=0.544\textwidth,trim=34 624 0 348,clip]{four-pairs-four-methods.png} \\[-1pt]
\rotatebox{90}{\scriptsize\makebox[0.1361\textwidth]{\shortstack{tent and\\snowy mountain}}} & \includegraphics[width=0.544\textwidth,trim=34 316 0 656,clip]{four-pairs-four-methods.png} \\[-1pt]
\rotatebox{90}{\scriptsize\makebox[0.1361\textwidth]{\shortstack{lighthouse and\\stormy sea}}} & \includegraphics[width=0.544\textwidth,trim=34 8 0 964,clip]{four-pairs-four-methods.png} \\[-1pt]
\end{tabular}
\vspace{-4pt}
\caption{<edit B>}
\label{fig:four-pairs-four-methods}
\vspace{-10pt}
\end{wrapfigure}
```

**Edit B, the caption.** Fault: caption past the picture. "Each group" names groups the grid no
longer has.

was: \textbf{The adapter repairs the composition without the joint prompt.} Each group shows two unseen concept pairs, with columns for the joint prompt, plain product, CO3, and our adapter. CO3 uses different starting noise; the other three columns share it.

now: \textbf{The adapter repairs the composition without the joint prompt.} Rows are four unseen concept pairs, and columns are the joint prompt, plain product, CO3, and our adapter. CO3 uses different starting noise; the other three columns share it.

**Edit C, the body sentence at `iclr2027_conference.tex:405`.** Same fault in prose.

was: The second column in each group shows the corresponding uncorrected product-of-experts samples.

now: The second column shows the corresponding uncorrected product-of-experts samples.

**Check at compile.** Edit B changes the caption's length, so the wrap's 27-line count is
re-checked on the built page: too few and the caption overlaps the next paragraph, too many and a
gap opens under it.

**Figure 6, `how-a-render-fails-paper.png`: `improve`, written 2026-09-25 (see the compile log).** Fault:
the figure sat at 0.60 of the text width with empty margins on both sides, and its panel labels,
drawn into the PNG, printed at about 4pt. Full width fills the margins and brings the labels to
about 5.5pt. Tested in a scratch copy: 14 pages as before, Figure 6 and the Conclusion still on
page 9, the references start about 8 lines lower.

was: `\includegraphics[width=0.60\textwidth]{how-a-render-fails-paper.png}` at `iclr2027_conference.tex:418`

now: `\includegraphics[width=\textwidth]{how-a-render-fails-paper.png}`

Not taken: typesetting the six panel labels in LaTeX at `\scriptsize`, which would need the PNG
cropped into its panels the way Figure 5 is.

**The paragraph reading Figure 6 (`iclr2027_conference.tex:410` to 416): `improve`, version A
written 2026-09-25 (see the compile log).** Faults: term before thing ("adapting only the null-token prediction",
"broader training", "an image-target variant with a cached null-token prediction" appear nowhere
else in the manuscript), no join (five failures as parallel sentences), point not landed (the
conclusion comes last and only two of five cases support it). Owed before or at compile: `claim`
on "adapters trained with other settings", that all five corrected panels come from trained
variants.

was: `Figure~\ref{fig:how-a-render-fails} shows that correcting the product does not always recover the requested composition. Some variants introduce human figures or lose semantic structure almost entirely. Adapting only the null-token prediction changes the style without recovering a second animal. Broader training produces an extra animal, while an image-target variant with a cached null-token prediction separates the animals but replaces the requested dog with a horse. Recovering plurality is therefore insufficient to establish that the requested cat and dog have both been preserved.`

now (version A, chosen over the first draft and version B): `Figure~\ref{fig:how-a-render-fails} shows five ways a corrected cat-and-dog render goes wrong, taken from adapters trained with other settings. Three are easy to see: invented people, a scene dissolved into texture, and a style change that still leaves one animal. The other two hold more than one animal and still miss the prompt, one with a third animal and one with a horse in place of the dog. A count of subjects would pass both, so recovering plurality does not by itself show that the requested cat and dog are both there.`

## What is in the built PDF

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

`\section{Discussion}` is gone. The paper runs Introduction, Background, `sec:interaction`,
`sec:analysis`, `sec:method`, **`sec:results`** (with the four-pairs figure and the subsection
"Where it breaks"), **`sec:limitations`**, Conclusion, then `app:strength-sweep`,
**`app:loss-weight`**, `app:adapter-ablations`, `app:adapter-samples` and
`app:where-the-correction-acts`.

**Swap-in surface.** A macro block after `\newcommand{\methodname}` holds `\adrank`, `\adparams`,
`\adbase`, `\adepochs`, `\adsteps`, `\advram`, `\adgpu`, `\adhours`, `\trainpairs`,
`\statesperrun`, `\lossclamp`, `\lossnorm`. The figure is `figures/four-pairs-four-methods.png`,
written by `build_grid.py` in this session's scratchpad, so a new checkpoint is a re-run plus a
rebuild with no tex edit.

**Hat convention settled.** All eight occurrences of the mono target are hatted, matching `eq:cfg`.
Verified against the code: $\hat\epsilon_{\mathrm{Mono}} - \hat\epsilon_{\mathrm{PoE}}$ expands to
$w(\epsilon_J - \epsilon_a - \epsilon_b + \epsilon_\emptyset)$, which is `delta_t_from_raw`.

**Still open.** `eq:poe-composition` defines a hatted symbol from raw predictions, the one place the
convention does not hold. The figure carries a mockup banner and per-tile checkpoint tags until one
run covers all four rows, and three CO3 cells and an object-and-object row are missing.
`app:where-the-correction-acts` has no finding in `report/` behind it. `paper/overleaf-iclr/` is
untracked, so there is no undo. `environment/paper.md` says `pdflatex` is absent on this machine
and it is present at `/usr/bin/pdflatex`.

## Originals held

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#figures) ➡️

| # | `was:` |
|---|---|
| 16b | All runs use 50 DDIM steps at guidance 7.5, $1024\times1024$, in fp16. The adapter is rank \adrank{} with $\alpha = r$, so the update enters unscaled, and adds \adparams{}\,M trainable parameters to a \adbase{}\,B frozen base. Training runs for \adepochs{} epochs of 50 optimizer steps, \adsteps{} in all, with AdamW at learning rate $10^{-4}$ decayed to zero on a cosine schedule and gradient clipping at 1.0. |
| 13b | ...gives the corrected sampler below; Appendix~\ref{app:training-inference} derives the training and inference relation. / This form also gives the learned analogue of \eqref{eq:corrected-sampler} directly: [equation with left side $\hat\epsilon_{\mathrm{corr}}$] |
| 19b | All four pairs and their starting noise are unseen during adapter training. The animal pairs come from the concept-disjoint evaluation split; the tent and lighthouse pairs lie outside the training pool. For each pair, the joint-prompt, plain-product, and adapter samples share the starting noise. |
| 3 | The model learns its correction from the residuals themselves. For each training pair we therefore store the residual at every denoising step, over several runs, so the training data covers whole trajectories rather than isolated steps. We train a single adapter on every pair together to learn the correction rule that restores the plurality lost during composition. |
| 2 | In the previous section we measured the plurality term along composed runs and showed that adding a small amount of it back, early in the run, restores composition. In this section we learn it. We train a model that produces the correction at each step, from the current state and the two concept prompts, so a corrected run no longer needs the joint prompt. The joint prompt is barred only at inference, so the term can still be measured in advance, and those measurements are what the model trains on. |

## Figures

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#open-questions-on-the-text) ➡️

**Figure 6, `training-the-compositional-adapter.png`**

Read off the image, not the caption.

1. Panel 1 says "50 DDM steps". The sampler is DDIM, named as DDIM everywhere else in the paper.
2. Panel 1 caches $\varepsilon_{\emptyset}$ and nothing in the figure ever consumes it. Panel 2
   produces its own $\varepsilon_{\emptyset}$ from the adapted model, which is what
   `trainer.py:507` does by default.
3. Panel 2 draws a $+$ node and a $-$ node into a box that already states
   $\tilde\varepsilon_{\mathrm{PoE}} = \tilde\varepsilon_a + \tilde\varepsilon_b - \varepsilon_{\emptyset}$.
   The arithmetic is written twice, once as glyphs and once as algebra.
4. The prompt symbol is $C_{ab}$, $C_a$, $C_b$ here and $c_a$, $c_b$ in Figure 7.
5. The LoRA panel says $r = 8$ (single pair), $r = 32$ (pooled). This section trains the pooled
   adapter and the prose says rank 8.

**Figure 7, `sampling-with-the-adapter.png`**

1. The $+$ nodes sit before the guidance boxes, so guidance reads as an addition of
   $\varepsilon_a$ and $\varepsilon_{\emptyset}$ rather than as the formula the box states.
   Same double-counting as Figure 6 item 3, and here the $-$ node does it a third time.
2. The caption line under $x_t$ is clipped at the bottom edge: "three passes, one latent," and
   then a cut-off line.
3. Prompt symbols are lowercase here, uppercase in Figure 6.

## Open questions on the text

**The lighthouse is not held out by line 324's definition.** The v58 pool trains on `a_lighthouse__x__a_sailboat`, so "all four are held-out pairs" (¶19) and "concept-disjoint held-out pairs" (line 365) overstate the fourth row. Written as held-out on the user's call, 2026-09-25; the fix if a reviewer raises it is a scene pair with no concept in the pool.

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

Chunk 5 says the adapter sits on every cross-attention layer; the reported run (v58-05, `cross+self`) also adapts self-attention q, k, v. Kept as written on the user's word on 2026-09-25; worth one line if a reviewer checks the ablation appendix.

## Settled by checking

**The section reports rank 8, and the prose is right.** `config.json` of `phase1_r8_100k`, read on
the cluster at `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/`,
gives `lora.rank 8`, `lora.alpha 8`, `target_modules ["attn2.to_q", "attn2.to_k", "attn2.to_v"]`,
`optim AdamW lr 1e-4 grad_clip 1.0`, `sampler ddim 50 steps guidance 7.5 at 1024x1024 fp16`.
`paper/iclr/figures.md:40` names the same run as the source of the held-out sample figure, so the
section and its figure are the same adapter.

Every number in chunk 4 and chunk 5 therefore checks out, and `alpha = rank` means the figure's
"$\alpha = r$, so the scale is 1" is right too. The one thing that does not is Figure 6's LoRA
panel, which says the pooled adapter is $r = 32$.

## The run the section must describe

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

**`tiiz0c0c`, display name `v57-11b-x0-self-decay`, still running at epoch 996 of 1000.**
Config read from W&B on 2026-09-23. It is a different adapter from `phase1_r8_100k`, which is what
the prose and both figures currently describe.

| Thing | Section 5 and Figures 6 and 7 say | `tiiz0c0c` |
|---|---|---|
| rank, alpha | 8, 8 | 32, 32 |
| adapted layers | `attn2` Q, K, V only: 70 blocks x 3 = 210 | `attn2` and `attn1` Q, K, V: 420 |
| loss space | epsilon | x0 (`loss_eps` logged beside `loss_fit`) |
| extra loss terms | none | `orth_weight 1`, bucket weights (`loss_weight_mean 1.42`), `loss_null_anchor`, `loss_undialled` |
| learning rate | 1e-4 constant | 1e-4 cosine decay, at 4.87e-9 by step 49,849 |
| optimizer steps | 2,000 epochs x 50 = 100,000 | 1,000 x 50 = 50,000 |
| peak memory | 23 GB | 28.0 GB |
| the pool | 19 animal pairs, 11 train and 8 eval | `all_groups`; the tracking set holds `a_typewriter__x__a_cactus` alongside the animals |
| sampler | DDIM 50 steps, guidance 7.5, 1024x1024, fp16 | identical |

**What this costs each chunk.** Chunks 4, 5, 7 and the whole of Figure 6's bottom row are about an
adapter the paper is no longer training. The self-attention addition is the one that changes the
story rather than a number: Figure 6 says the prompt enters only through cross-attention and that
this is why the adapter sits there, and with `attn1` adapted the adapter also sits where the image
attends to itself.

**What it does not cost.** The sampler is unchanged, so Figure 7's green block is wrong only in the
ways round 1 found, not because of this run.

**Not settled.** This run is in flight and is variant 11b of a cohort. Whether it is the one the
paper reports is the user's call, and nothing in the section is rewritten until it is made.

## Which render is in the paper

Navigation: ⬅️ [The run](#the-run-the-section-must-describe) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

**The paper still holds the 2026-09-17 render.** `paper/iclr/figures/` and
`paper/overleaf-iclr/figures/` both hold md5 `ce095d9db716dcdc2dd2c52e2da0bf43` for
`training-the-compositional-adapter.png`. The two folders are identical; the 2026-09-21 mtime on
`paper/iclr/` is a copy, not a new render.

**A newer candidate exists outside the repo** and has been seen only as a screenshot. Against the
render in the paper it fixes two things and breaks one.

| | In the paper | The newer candidate |
|---|---|---|
| panel 1 step count | "50 DDM steps" | "50 DDIM steps", correct |
| the cached $\varepsilon_{\emptyset}$ | nothing says it is consumed | "read back from the cache" added, correct |
| the guidance boxes | $\tilde\varepsilon_a = \varepsilon_{\emptyset} + w(\varepsilon_a - \varepsilon_{\emptyset})$, correct | $\tilde\varepsilon_a = \varepsilon_a + w(\varepsilon_a - \varepsilon_{\emptyset})$, **wrong on both boxes** |
| the nodes before those boxes | circled $+$ | empty circles, no operator |

The candidate must not replace the file until the leading term is back to
$\varepsilon_{\emptyset}$. Classifier-free guidance is
$\varepsilon_{\emptyset} + w(\varepsilon_c - \varepsilon_{\emptyset})$, and the caption in the
manuscript still states it that way.

## Which panels are the training architecture

The bottom row of Figure 6, three orange panels. The green block above is the training procedure
and carries no architecture fact.

| Panel | What is wrong under `tiiz0c0c` |
|---|---|
| SDXL denoising UNet (overview) | "70 cross-attention blocks x 3 projections = 210 adapted layers" is 420; "the text prompt enters here and nowhere else" no longer explains where the adapter is |
| One cross-attention block | "the output projection, self-attention and feed-forward are not [adapted]" is false: 210 of the 420 modules are `attn1` |
| One adapted projection (LoRA) | "$r = 8$ (single pair), $r = 32$ (pooled)" is $r = 32$; the $d_{\mathrm{in}}$ line holds for cross-attention only |

Counted from `lora_step_050000.pt`: 840 tensors, 420 modules, 70 each for `attn1.to_q/k/v` and
`attn2.to_q/k/v`, `lora_A` shaped `(32, 640)`.

## Routes and threads

| Chunk | Where it went | What it owes back | Returned |
|---|---|---|---|
| the paragraph reading Figure 6 (`iclr2027_conference.tex:410`), a "why these fail" addition | `/paper-scout` on Dahary 2024 (Bounded Attention), Chefer 2023 (Attend-and-Excite), Rassin 2023 (SynGen), routed 2026-09-25 | reading-register rows and bib keys; which paper shows subject duplication or species substitution in SDXL-class models | yes, 2026-09-25: Dahary et al. (arXiv 2403.16990, `dahary2024bounded`) selected, register row added; Attend-and-Excite and SynGen set aside with reason in the register |

**Integrated, queued behind the missing appendix homes.** The addition after version A, with the
citation on its own sentence, because Dahary et al. document subjects leaking into each other on
SDXL and say nothing about why an adapter would do it:

`Our other experiments point to causes for three of them. The grey render comes from an adapter trained far past its best checkpoint, by which point its weights have kept growing and its renders have drifted away from the joint-prompt target. The third animal and the horse match a separate test in which both prompts name the same animal: the plain product draws one animal and the adapter draws two. The adapter has therefore learned to produce two subjects, and neither its training loss nor a count of subjects checks which two. Subjects taking on each other's features is a documented failure of multi-subject generation on SDXL \citep{dahary2024bounded}.`

Bib entry to add to `paper/overleaf-iclr/iclr2027_conference.bib`:

```bibtex
@article{dahary2024bounded,
  title   = {Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation},
  author  = {Dahary, Omer and Patashnik, Or and Aberman, Kfir and Cohen-Or, Daniel},
  journal = {arXiv preprint arXiv:2403.16990},
  year    = {2024}
}
```

Still owed before this addition compiles: an appendix home for each finding it states, from
`report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md`
and `report/is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md`.
Dahary's Figures 2 to 4 were described from an automated read of the HTML and are not cited by
number. Invented people left out: the target drawing a person is filed at one seed only.

## Compile log

**2026-09-25, ¶13 (Equation 15 lead-in, left side $\hat\epsilon_\lambda$) and ¶16 (Appendix A settings).** Written in the same pass as the full-pass walk's chunks 7, 8, 9 and 13; see the [full-pass compile log](../proofread-full-pass/PROOF_MAP.md#compile-log). 14 pages, no errors.

**2026-09-25, ¶19 held-out sentence.** Three sentences on where the four pairs come from replaced by one, at `iclr2027_conference.tex:387`; pre-edit file in the session scratchpad as `iclr2027_conference.tex.bak-p19`. `pdflatex` twice: no errors, no undefined references, 13 pages.

**2026-09-25, the paragraph reading Figure 6, version A.** Written; pre-edit file in the session scratchpad as `iclr2027_conference.tex.bak-fig6para`. `pdflatex` twice: no errors, no undefined references, 13 pages. The drop from 14 is not this edit: the pre-edit file already builds to 13, because a parallel session had reworked section 5 since the Figure 6 width compile (`eq:adapter-loss` is gone). The paragraph now sits on page 8, Figure 6 at the top of page 9, Conclusion on page 9.

**2026-09-25, Figure 6 at full width.** Written to `iclr2027_conference.tex:418`, previous file kept as `iclr2027_conference.tex.bak-fig6`. `pdflatex` twice: no errors, no undefined references, 14 pages. Figure 5 on page 8, Figure 6 and the Conclusion on page 9, references begin at the foot of page 9.

**2026-09-25, Figure 5 edits A to C.** Written to `paper/overleaf-iclr/iclr2027_conference.tex`, with the previous file kept as `iclr2027_conference.tex.bak-fig5` beside it because the folder is untracked. Applied to the file as it stood after a parallel session's appendix edits, each target matched exactly once. `pdflatex` twice: no errors, no undefined references, 14 pages (the appendix edits, not this change, took it from 15). Figure 5 on page 8, Conclusion on page 9, the 27-line wrap closes on the caption with no overlap and no gap.

## Next step

The pool paragraph, which is still my draft and has never had a sentence pass.

**Missing since the restructure.** The two sentences tying the loss back to $r_t$ are gone:
`\eqref{eq:interaction-term}` gives $\hat\epsilon_{\mathrm{Mono}} = \hat\epsilon_{\mathrm{PoE}} + r_t$,
so fitting the target is supplying $r_t$, and fitting the prediction rather than $r_t$ keeps a second
estimate out of the sampler. That was the only link from section 5 back to the paper's own definition
of the term it is learning.

**Carried forward:** paragraph 6 says "The correction rule therefore transfers" and nothing now
establishes "correction rule". Fix it in paragraph 6, not by restoring paragraph 2.
