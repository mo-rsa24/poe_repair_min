# 🔎 Proofread map: the whole manuscript, consistency and technical accuracy

The walk's persistent state. Read it first on every invocation, write it before every round ends.

## Position in the walk

| Field | Value |
|---|---|
| Selection | `paper/overleaf-iclr/iclr2027_conference.tex`, the whole file, only the chunks the full read flagged |
| Kind | manuscript |
| Grain | paragraph |
| Cadence | close |
| Current chunk | 6 of 16, "The same composition rule succeeds for a butterfly" |
| Marks | 2 ok, 3 changed, 1 cut, 2 skipped, 8 unwalked |
| Compiled | chunks 7, 8, 9, 13 on 2026-09-25 (see compile log); previously nothing; the section 5 walk holds two queued edits of its own (¶13 and the Appendix A settings paragraph) in the [section 5 proof map](../proofread-section-5/PROOF_MAP.md) |
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

The ICLR submission end to end, restricted to the sixteen places a full read on 2026-09-25 found
an inconsistency between the text and its figures, a claim with nothing behind it on the page, or
a technical inaccuracy.

**Why this pass is happening**

To decide whether the paper is ready to submit. The full read said not yet: five findings a
reviewer would catch, and a handful of smaller technical points.

**What the text is trying to do**

Show that product-of-experts composition drops a measurable term, that adding it back early
restores two objects, and that an adapter can learn it and apply it to pairs it never trained on.

## The chunk map

Navigation: ⬅️ [Quick context](#quick-context-what-this-is) | 📋 [TOC](#table-of-contents) | [Next](#accepted-not-yet-written) ➡️

In reading order. Line numbers are the file as it stood on 2026-09-25.

| # | Line | Kind | Opens | Mark | Read |
|---|---|---|---|---|---|
| 1 | 114 | ¶ abstract | "Text-to-image diffusion models generate" | work | "a general correction for plurality rather than the appearance of any specific concept" has no evidence in the paper; "a small fraction" has no number |
| 2 | 173 | list item | "We show that adding it back restores composition" | work | the controls claim has no figure; `strength-grid-0to1.png` would carry it |
| 3 | 182 | ¶ | "Existing methods keep the product as the target" | work | SuperDiff is cited as a sequential Monte Carlo corrector; it is the Itô density estimator |
| 4 | 189 | ¶ + Eq 2 | "Diffusion models." | work | $\alpha_s$ called "per-step noise levels"; they are the signal kept per step |
| 5 | 237 | Eqs, derivation | "Writing $\epsilon_{\text{true}}$" | work | guided $\hat\epsilon_{\mathrm{PoE}}$ inside an unguided identity; $r_t$ never tied to $R_t$ (under guidance $r_t = w \cdot (-\sigma_t\nabla\log R_t)$ for an exact model) |
| 6 | 241 | ¶ | "The same composition rule succeeds for a butterfly" | work | "unchanged across four models" one sentence after butterfly succeeds on three of four |
| 7 | 265 | ¶ | "The strength axis separates the correction" | changed | the λ sweep and its three controls, described with no figure; `strength-grid-0to1.png` shows exactly this |
| 8 | 269 | ¶ | "Independence can fail in more than one way" | cut | "in every pair we study both concepts are foreground subjects" is false for tent and mountain, lighthouse and sea, butterfly and meadow |
| 9 | 274 | ¶ | "If $r_t$ is the signal composition is missing" | changed | "a small fraction of the measured plurality term" is wrong: the window runs inject the full term (λ 1, "full dose" per `poe_repair/experiments/interaction_term/window_grid.py:10`) over a fraction of the steps; the abstract repeats it (chunk 1) |
| 10 | 313 | caption | Figure 4, "Composition survives only while" | ok | kept on the user's word; the caption's "a cat box and a dog box" is stricter than the instance-count scorer that drew the frames |
| 11 | 326 | ¶ | "We train a LoRA adapter" | skipped (user: leave Figure 5 alone) | cross-attention only in the text; every Figure 5 "Ours" checkpoint adapts self-attention too |
| 12 | 332 | ¶ | "During training, we select ten states" | ok | kept on the user's word; "uncorrected" checked against the v58 recipe and is right; that the ten are steps 0 to 9 goes unstated |
| 13 | 365 | ¶ | "We evaluate on concept-disjoint held-out pairs" | changed | promises a detector evaluation the paper never reports; "concept-disjoint" false for the lighthouse |
| 14 | 370 | figure | Figure 5, `four-pairs-four-methods.png` | skipped (user: leave Figure 5 alone) | three different checkpoints across the four rows, one with a different loss, undisclosed |
| 15 | 388 | sentence | "The joint-prompt sample provides the target" | work | reads as if held-out pairs get a joint-prompt target |
| 16 | 441 | ¶ | "This matters because the value of composition" | work | "Our results show" on four qualitative images |

## Accepted, not yet written

Navigation: ⬅️ [The chunk map](#the-chunk-map) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

| # | Control | The accepted text |
|---|---|---|
| 13 | compiled 2026-09-25, `improve`, accepted | Training and sampling details are given in Appendix~\ref{app:implementation}. |
| 9 | compiled 2026-09-25, `improve`, accepted; full dose assumed from `window_grid.py:10`, render outputs not opened | We settle it by choosing which steps receive the correction and leaving the others uncorrected, and find that the full measured correction, applied only in the first ten of the fifty steps, is enough to restore composition. |
| 8 | compiled 2026-09-25, `cut`, confirmed | *(deleted, the paragraph at line 269)* Its spatial point is in the introduction and the $R_t$ paragraph; the cat and dog are named in Figures 3 and 4; the joint prompt drawing the pair is shown in Figure 2 and required by the Section 5 training filter. |
| 7a | compiled 2026-09-25, `improve`, accepted | ...no control reaches two animals anywhere on the axis (Figure~\ref{fig:strength-grid}, Appendix~\ref{app:strength-sweep}). This is one pair at one seed. |
| 7b | compiled 2026-09-25, `improve`, accepted, new appendix section after `app:training-inference` | \section{The strength sweep and its controls} \label{app:strength-sweep}, one `[!htbp]` figure: `how-much-is-added/strength-grid-0to1.pdf` at `\textwidth`, label `fig:strength-grid`, caption "\textbf{Only the measured correction composes.} A cat and a dog at seed 9. A column is the strength $\lambda$ added at every step, up to the full correction at $\lambda = 1$; a row is what was added. The top row is the pair's own plurality term, and beneath it sit three controls, each breaking one thing about it while matching its size and schedule: another pair's term, the same term from a different seed, and the same term with its step order deranged. Nothing is added at $\lambda = 0$, so that column is one sample shown four times." |

## Originals held

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#figures) ➡️

| # | The original text |
|---|---|
| 13 | We evaluate on concept-disjoint held-out pairs using the composition criterion of Section~\ref{sec:analysis}: an object detector finding two distinct subjects in the image. Training and sampling details are given in Appendix~\ref{app:implementation}. |
| 9 | We settle it by choosing which steps receive the correction and leaving the others uncorrected, and find that a small fraction of the measured plurality term, applied in the first steps of the run, is enough to restore composition. |
| 8 | Independence can fail in more than one way, and the one this paper studies is spatial, since a patch of pixels that reads strongly as one animal cannot read equally strongly as the other. In every pair we study both concepts are foreground subjects, rather than a subject and a setting that could occupy different parts of the image. The same network draws two distinct animals when the joint prompt asks for them, so composition is not failing for want of a model that can render the pair. |
| 7a | Sweeping $\lambda$ from 0 to 1 on a cat and a dog, the hybrid separates into two animals before the full correction is reached, and no control reaches two animals anywhere on the axis. |
| 7b | *(no appendix section; nothing to restore)* |

## Figures

Navigation: ⬅️ [Originals held](#originals-held) | 📋 [TOC](#table-of-contents) | [Next](#open-questions-on-the-text) ➡️

**`figures/how-much-is-added/strength-grid-0to1.png`, candidate for chunks 2 and 7.** Opened
2026-09-25. Five columns, λ = 0, 0.25, 0.5, 0.75, 1; four rows, labelled only in a line under
the grid: real correction, wrong pair, wrong seed, wrong order. The λ = 0 column is the same
hybrid in every row. The real-correction row stays one animal through 0.5 and is a cat beside a
dog at 0.75 and 1. The wrong-pair row degrades into a cartoon single animal; the wrong-seed and
wrong-order rows stay one dog-like animal at every λ. It shows both claims as written. Cat and
dog at seed 9, from `scripts/compose_rate_vs_correction.py --layout grid5`, per
[how big the strength grid should be](../../results/how-big-the-strength-grid-should-be/README.md).
The section 3 walk decided it belongs in `Appendix~\ref{app:strength-sweep}`, "The strength sweep
and its controls", with the main text keeping the argument; that appendix section is not in the
file, and neither the label nor the figure appears anywhere in it.

## Open questions on the text

**The v58 recipe answers two gaps in the Appendix A settings paragraph** (queued in the section 5 walk): `--train-batch-size 1`, so an epoch is 50 single-state steps; and it also sets `--ema-decay 0.9999`, `--loss-space x0` and `--contrast-weight 0.01`, none of which the paper states. Source: `scripts/launch_recipe_run.sh`, the BASE argument list.

**A held-out number exists for chunk 13.** [Where does each condition land](../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md), section 8: instance-count compose fraction over 8 seeds on 8 held-out pairs, plain product 0.00 to 0.25, corrected 0.75 to 1.00 on every pair, joint prompt 0.62 to 1.00; `artifacts/results/where-does-each-condition-land/pairs/verdict.json`, field `per_pair[].instance_count_compose_fraction_post_hoc`. Rank-32 adapter at step 30,050, λ 1.0; read post-hoc; not the checkpoints in Figure 5.

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

**Page budget.** The main text currently closes on page 9. A new Section 3 figure pushes it
toward page 10; something has to give if the limit is nine.

## Routes and threads

Navigation: ⬅️ [Open questions](#open-questions-on-the-text) | 📋 [TOC](#table-of-contents) | [Next](#compile-log) ➡️

None yet.

## Compile log

Navigation: ⬅️ [Routes and threads](#routes-and-threads) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**2026-09-25, chunks 7, 8, 9 and 13.** Written to `paper/overleaf-iclr/iclr2027_conference.tex` together with the section 5 walk's ¶13 and Appendix A edits; pre-edit file in the session scratchpad as `iclr2027_conference.tex.bak-fullpass`. Every target matched exactly once. `pdflatex` twice: no errors, no undefined references, 14 pages (13 before; the new Appendix C with the strength grid is the added page). Conclusion on page 9, Appendix A on page 12, Appendix C on page 13.

## Next step

Navigation: ⬅️ [Compile log](#compile-log) | 📋 [TOC](#table-of-contents)

Walk chunk 7, the strength-axis paragraph, with `strength-grid-0to1.png` as its figure; chunk 2
then only needs a pointer to it.
