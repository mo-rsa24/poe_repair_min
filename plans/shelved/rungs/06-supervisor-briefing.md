# 🎤 Supervisor Briefing — remind him where the program stands, and name the next moves

## Description
Turn the current program state into a short briefing a supervisor gets in one sitting: one slide per
pyramid rung showing where that rung actually is (status pulled from `MASTER_PLAN.md` goals and
`DECISION_TIMELINE.md`, backed by a real generated picture, not a claim), then a short
new-directions section that promotes the two ripe deferred ideas from `PARKING_LOT.md` into candidate
next steps. The whole thing ends with a `/plain-speak` pass so it reads for a tired supervisor, not a
reviewer fluent in our jargon. This is a communication deliverable, not a research rung: it consumes
the tree, it does not advance it.

## Purpose
The reorganisation on 2026-07-21 answered "where are we" for us, but nothing in the tree turns that
into something the supervisor can absorb. This plan closes that gap and makes the "passed through
plain-speak" claim true by making it an explicit task rather than an aspiration. It also forces the
honest framing: five rungs, only the first is fully landed, and the binding constraint (delivery, not
transfer) is the thing worth his attention.

## Goal
A single self-contained briefing (markdown + rendered figures, optionally a slide deck) with: five
rung cards each pairing a one-line status with one real artifact image; a "biggest constraint" card
stating the ~40% delivery ceiling in plain words; and a two-item new-directions card (SLERP-merge,
degradation-curve). Every quantitative status line is paired with something to look at. Nothing in it
uses a term the glossary would have to translate.

## Tasks

### Group 1 — State of things (one card per rung)

- [ ] ⚠️ Pull the current status for all five rungs from the two sources of truth into a single table.
  Prompt: `/progress-brief read MASTER_PLAN.md Goals (lines 28-44) and DECISION_TIMELINE.md, and produce a 5-row table: rung → one-line status → the single most convincing artifact path for it → what is still owed. Overfit=G04 landed cat×dog + G4/G6 trained, G1-G3 owed; Survive-Noise=G6 pool verdict ok, held-out-seed enactment pending; Cross-Pair=code ready (Plan 12), not run; Group-Wise=g6 smoke only; Scale=trained to step 30k, crossbar never evaluated.`

- [ ] ⚠️ Make sure every rung has ONE real picture to show. Overfit already has `recap/figs/beachhead_strip.png`; find or render the one image that best enacts each other rung.
  Prompt (`/run-experiment`): `for each rung lacking a figure, render the single most legible artifact from its owning experiment — Survive-Noise: the G6 held-out-seed contact sheet from artifacts/rung2-survive-noise/; Scale: the crossbar quadrant grid if any cells exist. Write them to recap/figs/<rung>.png. If a rung has no runnable artifact yet, say so explicitly on its card rather than faking one.`

- [ ] ⚠️ State the real limiter in one plain card: the trained cell reaches only ~40% of the PoE→Mono distance and plateaus, so delivery is the binding constraint, not transfer (from PARKING_LOT.md:7-9). This is the one thing the supervisor should leave remembering.

### Group 2 — New directions (promote the ripe deferred items)

- [ ] ⚠️ Write the SLERP-merge candidate as a supervisor-facing next step: spherically interpolate the Plan-09 per-pair LoRAs into a group corrector as a cheap second transfer route, reusing existing artifacts (PARKING_LOT.md:11-13, ref LoRA Soups arxiv 2410.13025). One paragraph, the cost, the risk (dilution to the 40% ceiling).

- [ ] ⚠️ Write the degradation-curve candidate: report transfer as recognisable-composition rate vs fraction of pairs held out, instead of per-group pass/fail (PARKING_LOT.md:15-17, ref arxiv 2508.20783). One paragraph, what data it needs (more held-out cells than the current 3-per-group plan produces).

### Group 3 — Assemble and plain-speak

- [ ] ⚠️ Assemble the cards into one briefing document.
  Prompt: `/slidecraft build a short briefing from the Group-1 table + figures and the Group-2 candidates: title, five rung cards (status line + image), one "biggest constraint" card, one "two candidate directions" card. Qualitative-first, one idea per card.`
  alt: `/recap-plan-tree` if you want the interactive HTML gallery form instead of slides.

- [ ] ⚠️ Final pass — this is what makes "passed through plain-speak" true.
  Prompt: `/plain-speak rewrite the assembled briefing so a tired supervisor gets every card in one read: no term the MASTER_PLAN glossary would have to translate, PoE/Mono/residual said in plain words, one idea per sentence. Keep the load-bearing numbers (40%, step 30k, ~ep600) and artifact paths exact.`

## Recommended skill
▶ `/progress-brief` ✅ — pulls the current rung statuses into the table Group 1 needs.
   then `/slidecraft` ✅ to assemble, `/plain-speak` ✅ for the final register pass.
   alt: `/recap-plan-tree` for the interactive-gallery form of the same briefing.

## Engagement Instructions
```
# Done when a single briefing file exists with five rung cards, each carrying a status line AND a real
# image path that resolves on disk:
$ for f in recap/figs/beachhead_strip.png recap/figs/survive-noise.png recap/figs/scale.png; do test -f "$f" && echo "have $f" || echo "MISSING $f (card must say 'not yet rendered')"; done
# And when the final file has been through /plain-speak — verify no glossary term survives unexplained:
$ grep -niE "PoE|Mono-free|residual|crossbar|out_out|Task D" <briefing-file>   # each hit must sit next to a plain-words gloss
```

**▶ View the briefing:** open the assembled deck/gallery; the fastest live look at any single rung's
underlying evidence is still the LoRA Inspector (`bash scripts/run_lora_inspector.sh`).
