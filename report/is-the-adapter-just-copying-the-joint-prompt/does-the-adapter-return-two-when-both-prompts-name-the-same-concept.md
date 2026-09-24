# Does the adapter return two objects when both prompts name the same concept?   ✅ support: two animals on four of five cells, where the plain product returns one on all five · verified 2026-09-24

**The claim**

Asked for a dog and a dog, the two experts agree, so nothing contends for the same region on
semantic grounds, and plain product-of-experts composition still returns a single dog. The
adapter returns two. The same holds for a cat and a cat. What the adapter supplies is therefore
not a way of telling two different concepts apart, since there are no different concepts here.

**What would have counted**

No bar was pre-registered. The read is descriptive and by eye over five cells: dog and dog at
seeds 1, 2 and 9, cat and cat at seed 9, tiger and tiger at seed 9. A cell counts as composed
when two whole animals of the named kind are in the frame. The adapter is `v58-03-self` at step
50,000, correction on all fifty steps at strength 1.

## 1. One concept asked for twice, three methods

![Joint prompt, plain product and the adapter on five cells naming one concept twice](../../artifacts/results/what-the-main-study-grids-show/the-same-concept-twice.png)
*Rows are cells, columns are the joint prompt, the plain product and the adapter, each row from
one starting noise. What to notice: the middle column holds exactly one animal in every row, and
the right column holds two in four of the five.*

**The number.** Five cells. Plain product: one animal in five of five. The adapter: two animals
in four of five, the exception being dog and dog at seed 1, where the second animal is dark and
horse-like rather than clearly a dog. The joint prompt: two animals in three of five, a person
beside one dog at seed 1, and one dog with a malformed second at seed 2. Read by eye from the
tiles listed in
[the figure's sidecar](../../artifacts/results/what-the-main-study-grids-show/the-same-concept-twice.json).

## What this cannot tell you

- **Whether the second animal is the right kind.** At seed 1 it is not, and the read counts
  animals rather than species.
- **Whether this generalises past three pairs.** Dog, cat and tiger, at one or three seeds each.
- **Whether a different checkpoint behaves the same way.** One adapter, `v58-03-self` at step
  50,000, was used for every cell.
- **What the adapter does to the picture's quality here.** Nothing in this read measures
  sharpness or colour, and the tiger row's adapter render is softer than its joint prompt.

## Still open

- **The same five cells through two more checkpoints**, so the claim is about the method rather
  than one adapter.
- **A species check on the second animal**, which needs a reader or a detector that can tell a
  dog from a horse, and the project's instance counter cannot.

## Where this came from

| What | How it was established | When |
|---|---|---|
| The five cells and their counts | read by eye from the renders named in the figure's sidecar | 2026-09-24 |
| The figure | `scripts/main_study_grids.py` | 2026-09-24 |
| The renders | `scripts/figure_candidates.py`, columns `mono`, `poe` and `ours`, adapter `v58-03-self` step 50,000 | 2026-09-24 |
