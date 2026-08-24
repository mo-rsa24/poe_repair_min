# Can We Trust The Compose Rate

## Where this scope sits in the order

**This scope owns none of the 22 numbered steps, by its own design.** Nothing in the paper order
waits on it, so it carries no step number and blocks nothing. Its four plans run in the internal
order below, and it earns numbered steps in the one `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md) only on the big-promotion condition in the
Definition of Done: the false-compose rate comes back contaminated at the ten-point bar, or a
candidate detector clears the 95-versus-85 bar.

**Next in this scope:** [gate-01-is-this-hole-already-known](plans/gate-01-is-this-hole-already-known.md),
one `/pressure-test` verdict. Nothing but instrument-01 may start before it comes back.

| Within this scope | Plan | What it does | Status |
|---|---|---|---|
| 1 of 4 | [gate-01-is-this-hole-already-known](plans/gate-01-is-this-hole-already-known.md) | is this hole already published | ⚠️ not started |
| 2 of 4 | [instrument-01-the-three-state-labelled-set](plans/instrument-01-the-three-state-labelled-set.md) | the labelled set and the band | ⚠️ not started, runs whatever gate-01 says |
| 3 of 4 | [idea-01-what-the-current-benchmarks-score](plans/idea-01-what-the-current-benchmarks-score.md) | score the candidate detectors | ⚠️ blocked by gate-01 |
| 4 of 4 | [gate-02-promote-or-close](plans/gate-02-promote-or-close.md) | promote or close, in writing | ⚠️ blocked by the other three |

## Mission
Every compose rate in the paper comes from a detector we ask "how many animals", never "which
animals". Two dogs scores the same as a cat and a dog. So 94% is an upper bound, and nobody has
measured how far above the truth it sits. This scope measures it.

The number that decides whether the paper changes is not how big the error is. It is whether
the error gets bigger as λ gets bigger. If it stays the same size, F2's shape is real and we
add one caveat. If it grows with λ, then part of F2's slope is just the detector becoming
easier to please.

## Run kind
Group 2, tries a new idea. It may propose experiments and may not change any claim in the
paper. Anything it finds goes to `PARKING_LOT.md`, or becomes a proposed group-1 run (the kind
that is allowed to change a claim) in a sibling scope, until the promotion criterion below is
met.

## Depends on
- `outputs/compose_scorer/scorer_validated.json`: the file this scope is checking. Read only.
  `does-the-fix-reach-unseen-pairs` will not start unless it exists, so nothing here edits it.
- `artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/`: all 32 cells at λ=1, sorted by hand, with `calls.json` as the
  table. Kept back so the labelling pass can be scored against it.
- `outputs/interaction_term/dose/pairs/`: 19 pairs × 5 seeds × λ ∈ {0, 0.25, 0.50, 0.75, 1.00},
  the real-correction row plus the `_random` and `_wrong_pair` control rows.

## What the scorer cannot do
Its rule: COMPOSE iff distinct-instance-count("animal", NMS iou<0.5, conf>=0.30, min box side
>= 0.25 of image) >= 2.

**It cannot spot a repeat.** Two dogs is two animals. The hand audit found one such image in 30
scored successes.

**On most of this pool it cannot be checked at all.** Six of the eight held-out pairs were
picked because they blend (leopard/jaguar, cow/buffalo, frog/toad). That is the same thing that
makes "are both animals there?" unanswerable from the picture. A better detector does not help,
so how much of the pool is uncheckable is one of the numbers this scope reports.

## Objectives
1. **Check nobody has said this already.** One `/pressure-test` verdict on the claim that
   asking "is there a cat?" cannot catch a fused animal, and counting animals cannot catch a
   repeat. Nothing after this runs until it comes back, except objective 2.
2. **Build the labelled set** on pairs a person can actually judge, with the rule for which
   pairs those are written down before any image is opened. Needed whatever objective 1 says:
   it bounds the 94% and it certifies whatever metric the paper ends up using.
3. **Try the best available tools against it.** Published compositional benchmarks and better
   detectors than grounding-dino-tiny, each named to the paper it came from, scored on the same
   labels.
4. **Decide in writing** whether this gets promoted or closed, against the bars below.

## Goals
1. **Literature verdict recorded**, three ways: already known and named, said informally but
   never measured, or not addressed. Already-known shrinks this scope to a methods paragraph in
   `writing-06` and cancels objectives 3 and 4.
   [checkpoint: one `/pressure-test` verdict with citations, in `review/gate-01-*.md`]
2. **The judgeable pairs picked before any labelling.** A pair is judgeable if each animal has a
   visible feature you can name that the other lacks. Split all 19 pairs on that rule.
   [checkpoint: a 19-row table, written before any label exists]
3. **Labelled set built.** Four labels per image: both requested animals, separate; one animal
   or a fusion; two or more animals but not the two asked for; cannot tell. About 150 images
   (judgeable pairs × 5 seeds × 5 λ, real-correction row), plus the λ=1 cells of both control
   rows.
   [checkpoint: label counts per λ, and the pass scored against the 32 audit cells]
4. **The error measured against λ.** The false-compose rate is how often the scorer says
   compose and a person says it did not. Bar written in code before the labels exist.
   - *contaminated* if the false-compose rate at λ=1 is 10 points or more above the rate at
     λ=0.50. F2's caption gets capped, and a group-1 run is proposed in
     `does-the-correction-cause-composition`.
   - *clean* if the two rates are within 5 points of each other. The shape stands, 94% prints as
     an upper bound with its band, one paragraph in `writing-06`, scope closes.
   - *inconclusive* between 5 and 10 points. Report both numbers, add the newly generated
     separable pairs, do not pick a side.
   [checkpoint: both rates with their denominators, and the coverage number beside them]
5. **Candidates scored** against the same labels. Keep a candidate only if it agrees with the
   people at 95% or better where the current scorer agrees at 85% or less, on judgeable pairs.
   [checkpoint: agreement table, one row per candidate, current scorer first]

## Expected Outcome
A defensible band on every compose rate the paper prints, and an answer on whether F2's slope is
partly its own measurement.

Two predictions, written down now so they cannot be adjusted later. The error stays about the
same size across λ, the shape survives, and the paper gains a paragraph saying 94% is an upper
bound between 75% and 94%. The published benchmarks hit the same wall, because they are built on
the same "is there a cat?" question.

## Definition of Done
1. ⚠️ `/pressure-test` verdict recorded, with citations, three ways.
2. ⚠️ The judgeable-pair rule and the 19-pair split written down before any image is labelled,
   so it cannot be adjusted to suit the labels. [inferred prerequisite]
3. ⚠️ The labelling tool strips λ out of the file path and shuffles before showing an image, so
   the labeller cannot see which setting produced it. [inferred prerequisite]
4. ⚠️ The labelling pass scored against the 32 cells in `artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/`.
   Disagreement above 10% throws the pass away and the user labels by hand. That bar lives in
   the labelling script, not in prose. [inferred prerequisite]
5. ⚠️ Labelled set on `/datasets`, with label counts per λ and the coverage number.
6. ⚠️ False-compose rate at λ=1 and at λ=0.50, both with denominators, judged against the
   three-way bar.
7. ⚠️ Every candidate named to its source paper, run with a working `--device cpu` route, judged
   against the 95%-versus-85% bar.
8. ⚠️ `artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` gives the band as 75% to 94% and says it cannot be
   narrowed from those images. It currently says 87% to 94%, which counts the 17 uncallable
   cells as successes.
9. ⚠️ Promotion decision written in `review/gate-02-*.md` against the two levels below.

**Promotion, two levels.**

*The small one, which we expect to happen.* The band and the coverage number go to
`writing-the-paper` as proposed wording for `writing-06`, plus a cap on F2's caption reading 94%
as an upper bound. Wording only. No row in the root `MASTER_PLAN.md` paper table.

*The big one, the actual contribution.* Either goal 4 comes back *contaminated* at the 10-point
bar, or goal 5 produces a candidate that clears the 95%-versus-85% bar. Either one earns a
numbered step in the root `## Running order` table and a group-1 plan in
`does-the-correction-cause-composition`. `gate-02` must then say whether the winner re-certifies
or replaces `scorer_validated.json`, and what that means for the runs
`does-the-fix-reach-unseen-pairs` has already finished against the old one.

*Neither happens.* The scope closes with the limitations paragraph and the labelled set as a
reusable instrument. That is written down as the finding.

## Sub-Scopes
(none)

## Plans

Grouped by the run group each answers to. Statuses live in the review/ files.

**Not a run: a literature check before print**

| Plan | What it does | Status |
|---|---|---|
| gate-01-is-this-hole-already-known | `/pressure-test` on the claim (DoD 1) | ⚠️ |

**Instrument: changes no claim, needed whatever gate-01 says**

| Plan | What it does | Status |
|---|---|---|
| instrument-01-the-three-state-labelled-set | judgeable-pair rule, labels, coverage, the band (DoD 2-6, 8) | ⚠️ |

**Idea runs: may change nothing, land in PARKING_LOT.md**

| Plan | What it does | Status |
|---|---|---|
| idea-01-what-the-current-benchmarks-score | `/paper-scout`, then score them against the labels (DoD 7) | ⚠️ blocked by gate-01 |

**Not a run: the promotion decision**

| Plan | What it does | Status |
|---|---|---|
| gate-02-promote-or-close | reads the above, writes the verdict (DoD 9) | ⚠️ blocked by all three |


## Environment Context
See `environment/00-INDEX.md`. Read before drafting or checking any plan in this scope. What this
scope leans on: the GPU is shared and often full, so every detector pass needs a working
`--device cpu` route; large artifacts go to `/datasets` only; thresholds live as named constants
in source, following `MIN_BOX_FRACTION` in
`poe_repair/experiments/compose_scorer_validation/detection_scorer.py`.

## Glossary
Terms only this scope uses. Shared vocabulary is in the root `MASTER_PLAN.md`.

- **Judgeable pair:** each animal has a visible feature you can name that the other lacks
  (stripes, tusks, a trunk), so a person can say from the picture which two animals are there.
  The other pairs cannot be judged at all, because the pool picked them for blending.
- **Coverage:** how much of the pair pool any "is it there?" metric can be scored on at all.
  Reported as a result, not worked around.
- **False-compose rate:** among images the scorer calls compose, how often a person says the two
  requested animals are not both there as separate animals. How this changes across λ is what
  decides whether F2's shape is safe.
- **`wrong_pair` and `random`:** the two control rows in the dose grid. `wrong_pair` injects a
  correction computed for a different animal pair, `random` injects a size-matched random
  vector. Both are the same size as the real correction and neither should compose, so they say
  whether the effect comes from this correction or from pushing the model at all.
