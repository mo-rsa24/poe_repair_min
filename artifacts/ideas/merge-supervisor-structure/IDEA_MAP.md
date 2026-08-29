# 💡 Merge the supervisors' paper structure into the spine

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. Intro and Background map one-to-one | holds | their items 1 and 2 are the spine's sections 1 and 2 under the same names |
| 2. Their "missing implications" item is section 3 renamed, carrying a new worked comparison | holds as a rename; the comparison is held | the heading maps to section 3; the three-product comparison is parked and routed to its own /drip-idea walk, verdict returns via integrate |
| 3. "Plurality as an instantiation of the missing context" is the draft's existing framing | holds, enacted; the name itself is conditional | the rename is in the tex: candidate 3 is the abstract, section 3 is "Missing Implications: The Plurality Term", r_t is the plurality term with the interaction-term anchor kept at first intro use; build passed; whether the name survives Richard's earn-the-name test is walked separately and held below |
| 4. Their one "LoRA fixes it" slot can hold sections 4 and 5 without collapsing the oracle-vs-learned split | holds with the two-section repair, enacted | their item maps to the pair: section 4 is now "Restoring the Plurality Term Restores Composition", section 5 "Learning the Plurality Term"; the boundary sentence is parked in DRAFT_MAP's Loose lines; one meeting question outstanding |
| 5. "Showing generalisations from the LoRA" is section 6 renamed | holds | the transfer result exists: 96.9% compose on unseen pairs where plain PoE composes 0% |
| 6. The omitted closers (7 Discussion, 8 Conclusion) merge back unchanged | holds | nothing in their five items occupies the closers' ground; a meeting list omits closers by habit, not proposal |

Load-bearing: claim 4, settled. The split survived under the two-section repair, so the merge stays a rename, not a restructure.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

The supervisors' five-item paper structure, proposed live in a meeting, folds into the existing eight-section spine by renaming rather than restructuring, and the walk finds where the two genuinely disagree.

**Where the walk is**

All six claims settled. The renames are enacted in the tex, SPINE.md and DRAFT_MAP, and the build passes. Awaiting compile.

**What compile would produce today**

Destination A. The reconciled section list, one claim per section, ready to seed /drip-write. Five items stay open past compile: the routed three-product walk (returns via integrate), the parked earn-the-plurality-name walk whose verdict decides whether the plurality name stands (returns via integrate), the one meeting question on the two-section banner, the per-pair size-curve diagnostic on section 3's shared-shape claim (returns via integrate), and section 4's teaching figure, layout decided in [the dial-figure design note](../../paper/iclr/section-4-dial-figure-design.md), build outstanding.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

The draft's spine (SPINE.md) carries eight sections, one claim each, split by which correction is in play: sections 3 and 4 use the oracle correction as a measurement instrument, sections 5 and 6 use the learned one. The supervisors proposed five items: Intro, Background, a theory item ("missing implications from individual isolated experts cannot be added by PoE", with the worked comparison p(cat)·p(dog) vs p(cat and thing)·p(dog) vs p(cat and thing)·p(dog and thing)), a plurality item ("plurality as an instantiation of the missing context: LoRA fixes it"), and a generalisation item. The idea is that these are mostly the same sections under different names, that renames are cheap and rewrites are not, and that the two genuinely new wordings (the three-product comparison and the instantiation framing) must be checked against what the results actually support before either enters the spine.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. ✅ Intro and Background map one-to-one

Mark: holds. Their items 1 and 2 are the spine's sections 1 and 2. No renaming needed, no content moves.

### 2. ✅ Their "missing implications" item is section 3 renamed, carrying a new worked comparison

Mark: holds as a rename, with the comparison held. The claim "missing implications from isolated experts cannot be added by PoE" is section 3's claim, so the heading merges by renaming. The three-product comparison is parked in this walk and routed to its own /drip-idea walk (veracity, framing against the experimental framework, and a dig into context-augmented experts). One candidate remains for section 3 if the routed walk confirms it: the algebraic point that conditioning each expert on extra context changes each factor but the product still factorises, so any term coupling the two concepts is still absent. That is a paper-and-pencil claim, not a run, and it waits for the routed walk's verdict before entering the spine.

### 3. ✅ "Plurality as an instantiation of the missing context" is the draft's existing framing

Mark: holds, enacted in the tex. The framing was already the draft's own (the residual is the broad missing signal, plurality the aspect this setting isolates), and the naming is now carried through. Abstract candidate 3 is the manuscript abstract. The intro names plurality where the collapse is first described and keeps one interaction-term anchor clause at the first use of r_t, so the Background stays searchable under the field's name. Section 3 is titled "Missing Implications: The Plurality Term". SPINE.md and DRAFT_MAP's quoted spine renamed in the same pass. Build passed with tectonic, no overfull lines.

The name carries a condition from the 2026-08-20 meeting: Richard's earn-the-name test, that "dog times plurality" applied to one concept must give two dogs, or plurality narrows back to a claim about conflicting pairs and the general object stays an interaction term. That test has its own /drip-idea walk, [the earn-the-plurality-name walk](../../artifacts/ideas/earn-the-plurality-name/IDEA_MAP.md), parked at its claim 1 (pinning the operational form of "dog times plurality"). Its verdict returns here via `integrate`: name earned means this claim and claim 4 stand as written; name demoted means the narrowed wording plus a bounded tex rename sweep, with the fallback clause already in place at `paper/iclr/iclr2027_conference.tex:103`.

### 4. ✅ Their one "LoRA fixes it" slot can hold sections 4 and 5 without collapsing the oracle-vs-learned split

Mark: holds with the two-section repair, enacted. Load-bearing, and it survived: their item maps to the pair of sections, not one. Section 4 is retitled "Restoring the Plurality Term Restores Composition" (oracle diagnosis, F2 to F5), section 5 "Learning the Plurality Term" (the adapter, F6, F7), in the tex, SPINE.md and DRAFT_MAP in one pass, with contribution 2 aligned to the same vocabulary and the 4-to-5 boundary sentence parked in DRAFT_MAP's Loose lines for the section-4 walk. The literal one-section reading was rejected because the F2-to-F5 evidence block argues causality, not the LoRA, and the spine's lead needs the diagnosis before the fix. Build passed. Outstanding: one meeting question, whether the supervisors accept two sections under their one banner.

### 5. ✅ "Showing generalisations from the LoRA" is section 6 renamed

Mark: holds. Section 6's claim is exactly this: the learned correction composes pairs it never trained on and matches the oracle without seeing the joint prompt. The result exists (96.9% compose on unseen pairs against 0% for plain PoE).

### 6. ✅ The omitted closers merge back unchanged

Mark: holds. Discussion and Conclusion appear nowhere in the five items and nothing in the five items occupies their ground.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| the missing context | the interaction term / the residual `r_t` | what the product of independent experts drops; measured as the joint-prompt prediction minus the PoE prediction | confident, the project's own term |
| plurality | plurality (minted in this draft's abstract) | rendering each named concept as its own object in the scene; the aspect of the residual this setting isolates | confident, minted at abstract c1.6 |
| p(cat and thing)·p(dog and thing) | context-augmented experts | each expert conditioned on its concept plus a placeholder for the other's presence | no run in the repo does this; naming is provisional |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|
| 2, the three-product comparison | whether the symbolic argument holds algebraically, whether the third product still fails empirically, and what prompt wording "and thing" becomes | the routed /drip-idea walk on the third product; its verdict returns to this walk via `integrate` |
| 3, whether the plurality name survives Richard's earn-the-name test | whether "dog times plurality" applied to one concept gives two dogs; the operational form, the pre-registered bar, and the scorer's one-class check are all still open | resume [the earn-the-plurality-name walk](../../artifacts/ideas/earn-the-plurality-name/IDEA_MAP.md), parked at its claim 1; run the test it designs; its verdict returns here via `integrate` |

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 4 | ask the supervisors whether two sections under their "Prompt Plurality" banner is acceptable | yes: the merge stands as enacted; no: the disagreement is genuine and the one-section version must be argued against with the spine's lead (no credible fix without the diagnosis first) |
| 4 | ask whether their item 4 absorbs the causal diagnosis or compresses it | absorbs: the merge keeps sections 4 and 5 as two sections under their plurality banner; compresses: the diagnosis (F2, F4) loses its home and the merge becomes a restructure |
| 3, 4 | the four probes designed in [what the size measure means, and the four probes that would go past it](../../artifacts/drips/how-big-the-correction-is/what-the-size-measure-means-and-what-it-misses.md), none built: the step-by-step alignment matrix (does the correction hold one direction through an early window, and where does that window end), the images-over-curves grid for cat×dog beside eagle×hawk, the rank read on a run's 50 correction vectors, and correction size against blendedness over the 17 pool pairs. Each carries the pattern it must show and what a null looks like; the file also holds the two pasteable prompts and the interactive walk of the measure itself. Four facts the build needs, already located: the measure is `curve_for` in `scripts/snr_collapse.py` (‖r_t‖/‖eps_PoE‖ per step, then scaled to the curve's own median), imported rather than re-derived; cat×dog's cell is `training_cache/heldout/a_cat__x__a_dog/`, a real 50-step trajectory across 17 seeds, so its curve is computed from that cell and never reused from the 17-pair population sidecar it is absent from; each seed folder already holds rendered `poe.png` and `mono.png`, but the per-step x̂₀ thumbnails at steps 2, 12, 25 and 45 are not cached and need a VAE decode, which is the one part of this pair that is not cache-only and must be confirmed as runnable here before it is started; blendedness for the scatter's y axis comes from whichever read `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json` marks as trusted | a clean early window in direction plus a rising size-against-blendedness trend: sections 3 and 4 gain the qualitative face of the size-curve story and the plurality term reads as something that acts in a locatable place; a fading matrix or a flat scatter: the size curve stays a size curve, and neither section may claim the correction is orderly where it matters. Under this walk's own renames `r_t` is the plurality term, so every outcome lands as figure text in sections 3 and 4, never as a change to the section list |
| 4 | build section 4's teaching figure per [the dial-figure design note](../../paper/iclr/section-4-dial-figure-design.md): the four-vector dose grid rebuilt with the injection formula in the column headers, the λ=1 by-construction footnote, and the clipping fixed, then register it in DRAFT_MAP's figure slots for section 4 | built and registered: section 4 carries the figure that keeps the dial (λ, fraction of r_t injected) and the measure (the 5 to 14% per-step size ratio) apart, which is the confusion the supervisors' "LoRA fixes it" wording invites; not built: section 4's λ-sweep evidence stays in a figure that clips its own legend and lets the two numbers blur |
| 3 | per-pair size-curve diagnostic, not yet built: 8 small multiples (6 pool pairs chosen to span the spread, plus cat×dog and elephant×penguin) from the unsmoothed curves in `cache_analyses/step_collapse.json` (17 pool pairs, two seeds each, both seeds drawn per panel), each pair's curve and its median-scaled version over the 50 steps, a thumbnail strip of one existing render per pair beneath, saved to `artifacts/results/does-every-pair-share-one-size-curve/`; cat×dog and elephant×penguin are absent from that cache and need their curves from another (`fork_curve.json` or `quality_control_cache.json`, verify same quantity first) | all 8 rise then flatten: the pooled shape is genuinely shared and section 3's size-curve story stands unqualified; any pair deviates: the deviation goes to the owning review file as an open question and the section 3 sentence stating the shared shape picks up the caveat via DRAFT_MAP's loose lines; verdict returns via integrate |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|

## Sources

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| paper/iclr/SPINE.md | the eight sections and the one claim each carries; the oracle-vs-learned split stated as load-bearing | read this session |
| paper/iclr/DRAFT_MAP.md | section states (1 and 3 compiled, rest bare headings); plurality minted at the abstract | read this session |
| paper/iclr/what-each-figure-argues.md | which results exist behind each section: F1 to F8 with their numbers | read this session |
| paper/iclr/abstract_candidate_01.tex, abstract_candidate_02.tex | the existing plurality-as-narrowed-residual framing, verbatim | read this session |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Compile: produce the reconciled section list and the /drip-write seed. After that, the open items are the routed three-product walk, the meeting question on the two-section banner, the per-pair size-curve diagnostic, the four probes past the size measure (none built; both carry their full spec in Checks outstanding), and section 4's teaching figure, whose layout is already decided in [the dial-figure design note](../../paper/iclr/section-4-dial-figure-design.md) and only needs building and registering in DRAFT_MAP.

The two figure items share one subject: what the correction-size curve can and cannot argue in sections 3 and 4. Read [the parked size-measure walk](../../artifacts/drips/how-big-the-correction-is/what-the-size-measure-means-and-what-it-misses.md) before either is built, and resume that walk at its piece 3 if the section-4 text needs the direction result rather than only the size one.
