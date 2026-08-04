# 📊 Figures: the F2–F5 evidence cascade

## Why this plan exists
The run outputs from plans 03 and 04 only become an argument once someone can see
them. This plan turns them into four reviewer-facing figures, one claim each, in
reading order: delivery (F2), transfer (F3), delivery-vs-transfer split (F4), and the
pool contrast (F5). Four simple figures instead of one dense one.

## Description
The four figures that carry the scope's argument (F1 belongs to compose-scorer). Each
is designed through /design-figure before it is built, so form and honest-limits are
decided first.

## Purpose
Serves Definition-of-Done item 5. Turns the run outputs into the reviewer-facing
evidence set.

## Goal
F2–F5 produced, each designed via /design-figure then built.

## Tasks
- [ ] ⚠️ **[needs /design-figure]** F2 delivery-live: fraction-of-distance-reached over
  training, one faint line per held-out pair, the ~40% plateau drawn as a reference line.
- [ ] ⚠️ **[needs /design-figure]** F3 (A) transfer: compose-rate vs fraction held out,
  do-no-harm baseline as a band, real held-out thumbnails pinned at a couple of points.
- [ ] ⚠️ **[needs /design-figure]** F4 (A) real: compose-rate and direction-cosine as
  twin panels sharing an x-axis, so divergence (delivery vs transfer) is visible.
- [ ] ⚠️ **[needs /design-figure]** F5 (B) pool: paired bars, animals vs mixed, per
  held-out animal pair, same-pair pairing explicit.

## Engagement Instructions
GATE (pass/fail): each of F2–F5 exists as a non-empty figure file, and each was
designed via /design-figure before building (a design note per figure exists). A script
asserts four figure files present and non-empty.
STOP: if a figure's underlying data is missing (e.g. the leaderboard from plan 03 or the
contrast from plan 04 hasn't landed), halt that figure: do not fabricate a plot from
absent data. Figures depend on plans 03/04 completing.

## Recommended skill
▶ `/design-figure` ✅: designs each figure (form, computation, honest limits) before
   it's built. alt: `/plan-figures` to sequence the F2–F5 set as one coherent cascade.
