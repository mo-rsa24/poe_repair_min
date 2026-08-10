# 🔧 Build: the 3-anchor scorer, in both embedding spaces

## Description
The one genuinely new code in this scope. A scorer that, given an output and a pair's
three anchors, decides compose vs blend. Compose = the output is far from both
single-animal anchors AND near the joint anchor. Blend = the output is nearest one
single-animal anchor. It runs in two embedding spaces: DINOv2 (external,
structure/instance-aware) and the project MDS/latent space (the same space the
trajectory diagrams already use, `mono.pt`/`poe.pt`/`projection_meta.json`). The
distance form is RELATIVE (is the output closer to the joint than to either
single-animal anchor), with the threshold determined empirically from the validation
pairs, not a hand-picked absolute cutoff. That threshold decision is the one open
design choice, made here at build time.

## Purpose
Serves Objective 1 and Definition-of-Done item 1 (scorer module, both spaces).

## Goal
A scorer module that emits a compose/blend label per output in both embedding spaces,
plus the DINOv2-vs-MDS agreement table over the validation outputs.

## Tasks
- [ ] ⚠️ **[needs /pressure-test]** Before building the MDS-space read, pressure-test
  the claim "the project MDS/latent space separates blend from compose." DINOv2 has no
  such risk; the own-latent-space read could be scoring "did the LoRA move" rather than
  "did it compose." The guard is the joint positive anchor (a real compose is far from
  both single-animal anchors AND near joint, not merely far from the monos). Record the
  verdict before relying on the MDS read.
- [ ] ⚠️ Implement the DINOv2 embedding + relative-distance read: embed output and the
  three anchors, compute distance-to-each-anchor, emit a compose/blend label.
- [ ] ⚠️ Implement the MDS/latent read reusing the existing trajectory space
  (`mono.pt`, `poe.pt`, `projection_meta.json`), same relative-distance logic, with the
  joint anchor as the positive-anchor guard.
- [ ] ⚠️ Determine the relative threshold empirically from the validation pairs (the
  separation between the known cat×dog compose and the wolf×husky blend), and record it.
- [ ] ⚠️ Produce the DINOv2-vs-MDS agreement table: per output, {DINOv2 label, MDS
  label, distances to the three anchors}.

## Engagement Instructions
GATE (unattended pass/fail): the scorer runs on the validation outputs and emits a
compose/blend label per output in BOTH spaces without error (exit 0), and the
agreement table is written non-empty with one row per output and both label columns
populated.
STOP: if the scorer errors (import failure, missing anchor, embedding crash, non-zero
exit), halt. Do not proceed to validation with a scorer that can't produce labels.
Note: passing this gate means "the scorer runs and labels," NOT "the labels are
correct": correctness is plan 03's gate.

## Recommended skill
▶ `/pressure-test "the project MDS/latent space separates a two-animal composition
   from a chimera blend, given the joint anchor as positive"` ✅: the load-bearing
   feasibility check before the MDS read is trusted. alt: `/scaffold` for the module skeleton.
