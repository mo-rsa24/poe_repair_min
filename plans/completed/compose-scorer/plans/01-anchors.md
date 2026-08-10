# 🎯 Anchors: generate the three reference images per validation pair

## Description
The scorer measures every output against three reference anchors: animal-A alone,
animal-B alone, and the joint prompt ("A and B") through the good composer. This
plan produces those anchors for the two validation pairs, cat×dog and wolf×husky.
Partly on disk already: cat×dog joint(Mono) + PoE references live at
`artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/trajectory_diagram/seed_42/`
(`mono.pt`, `poe.pt`, `projection_meta.json`), and wolf×husky corrected samples at
`artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/heldout_pair/a_wolf__x__a_husky/sample_seed_09..12.png`.
Missing and to-generate: the four single-animal anchors (cat-alone, dog-alone,
wolf-alone, husky-alone) and the wolf×husky joint reference.

## Purpose
Prerequisite for the whole scope. Serves Objective 1 (the scorer needs anchors to
measure against). No anchors, no scorer.

## Goal
For each validation pair, three non-empty anchor images on disk (A-alone, B-alone,
joint), plus a contact sheet showing all anchors per pair.

## Tasks
- [ ] ⚠️ Inventory what exists: confirm the cat×dog joint/PoE refs and the
  wolf×husky corrected samples load, and list exactly which single-animal anchors
  are missing. Output: a short manifest of have/need.
- [ ] ⚠️ Generate the missing single-animal anchors (cat-alone, dog-alone,
  wolf-alone, husky-alone) at the validation seed(s), same generation settings as
  the existing references. ~4 images.
- [ ] ⚠️ Generate the wolf×husky joint reference (the good-composer "a wolf and a
  husky") so wolf×husky has its positive anchor. Output under an anchors dir per pair.
- [ ] ⚠️ Build the anchor contact sheet per pair (A-alone, B-alone, joint side by side).

## Engagement Instructions
GATE (unattended pass/fail): for each pair in {cat×dog, wolf×husky}, assert exactly
three anchor image files exist and are non-empty (size > 0). A script check:
`for pair; do test -s A_alone && test -s B_alone && test -s joint; done`, exit 0 =
pass. The contact-sheet PNG per pair exists and is non-empty.
STOP: if any anchor generation fails (non-zero exit, empty file), halt that pair and
log which anchor failed. Do not proceed to plan 02 with an incomplete anchor set for
either validation pair.

## Recommended skill
▶ `/run-experiment` ✅: drives the anchor generation on the GPU node and confirms the
   images land. alt: `/visualize-data-samples` for the contact-sheet assembly.
