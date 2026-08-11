# 🧪 What the published metrics score on the same images

Design only. The verdict lives in [../review/idea-01-what-the-current-benchmarks-score.md](../review/idea-01-what-the-current-benchmarks-score.md).

## What this asks, in one line
Run the current compositional benchmarks and the better detectors on exactly the images
`instrument-01` labelled, and see whether any of them agrees with people where ours does not.

## Why this plan exists
Swapping a certified instrument for a new one needs evidence, not preference. This plan
produces the one table that could justify it: every candidate scored on the same images against
the same human labels, with our current scorer as the first row.

## Description
`/paper-scout` first, because a method with no source paper is a hunch wearing a method's
clothes. Two families to scout. The published compositional benchmarks (T2I-CompBench, TIFA,
VQAScore, Davidsonian Scene Graph and their successors), and detectors and segmenters stronger
than `grounding-dino-tiny` (larger Grounding DINO checkpoints, OWLv2, YOLO-World, SAM 2,
vision-language models used directly as judges). Then score each on the labelled set.

## Purpose
Serves Objective 3 and Definition-of-Done item 7. Group 2, tries a new idea: it may change
nothing. A winner earns a row in `PARKING_LOT.md` and the right to propose a group-1 run, never
a direct change to a paper number.

## Goal
An agreement table, one row per candidate, current scorer first, scored on the judgeable pairs
of `instrument-01`'s labelled set. Every row names its source paper and records whether it ran
on CPU.

## Environment Facts This Plan Depends On
- The GPU is shared and often full. Every candidate needs a working `--device cpu` route and is
  recorded as CPU-runnable or not. A candidate that only runs on a free GPU is a candidate this
  project cannot rely on, and that is part of its score.
- Model weights and any cached scores go to `/datasets`, never `/home-mscluster`.
- No external API key is assumed. A candidate that needs a hosted model is recorded as such and
  not run.
- The 95% and 85% agreement thresholds are constants in the bake-off script, following
  `MIN_BOX_FRACTION`.

## Success/Failure Outcomes
- **A candidate clears the bar.** It goes to `PARKING_LOT.md` with its agreement number, and
  `gate-02` decides whether it re-certifies or replaces `scorer_validated.json`.
- **Every candidate reproduces the same hole.** The expected outcome, written down before
  running: the benchmarks are built on presence questions, so a fusion answers yes to both. This
  is the strongest evidence for the limitations paragraph and is recorded as a finding.
- **A candidate scores well on all images including the unjudgeable ones.** That is a red flag,
  not a win. No metric can be right about a picture no person can call, so it is guessing and
  the score is measuring the guess. Score on judgeable pairs only.
- **The failure to avoid:** tuning a candidate's thresholds until it beats ours. Every candidate
  runs at its published defaults, recorded in the table.

## Tasks
- [ ] `/paper-scout` the current compositional-evaluation benchmarks and open-vocabulary
  detectors. Produce a candidate table: method, source paper with arXiv id, checkpoint, whether
  it runs offline on this cluster, whether it runs on CPU.
- [ ] Write the prediction before running anything: which candidates are presence-family and
  therefore expected to reproduce the hole. It goes in the review file, not in a comment.
- [ ] Score each candidate at its published defaults on the judgeable pairs of the labelled
  set. No threshold tuning.
- [ ] Build the agreement table, current scorer as row 1, judged against the 95%-versus-85%
  bar held as constants in the script.
- [ ] Any candidate clearing the bar goes to `PARKING_LOT.md` with its number and its source
  paper, as a proposed group-1 run. Nothing here changes a paper number.

## Engagement Instructions
The table exists with one row per candidate and the current scorer first. Every row carries a
source paper with an arXiv id or venue, a checkpoint name, and a CPU-runnable yes or no. Every
candidate that ran, ran at published defaults, and the script asserts no threshold was passed
that differs from the paper's. Scores are computed on judgeable pairs only, and the row count
matches `instrument-01`'s judgeable denominator exactly.

STOP: `gate-01` returned already-known → this plan is cancelled, not run. `instrument-01`'s
labels do not exist yet → halt, there is nothing to score against.

## Recommended skill
▶ `/paper-scout` ✅ for tasks 1 and 2, then `/run-experiment` ✅ for tasks 3 to 5.
   alt: `/pressure-test` on any single candidate that looks like it solves the problem, before
   believing it.

## Recommended Prompts
- **On task 3** (scoring the candidates): `/debug-config` on the first candidate's checkpoint
  load. Getting one open-vocabulary detector running offline on this cluster is the step most
  likely to eat a day, and it is worth doing once carefully before the other candidates queue
  behind it.
- **On task 4** (the agreement table): `/pair-figure` if the table is going anywhere near the
  paper. A table of agreement rates wants one example image per disagreement type beside it.
