# Run names

A run name says what that run varies, and nothing else. `P`, `C1`, `C2` said which slot a run held
in a design that has since changed twice, so a name no longer matched what the run did.

## Words this uses

**cell** — one pair at one seed: a 50-step cached trajectory plus its joint-prompt render.
**all-50 / early-25** — whether training reads all 50 cached steps of a cell or only steps 0 to 25.
**orth3** — the error outside the plane the two experts can already reach is charged three times.

## The runs already finished

| Name it had | What it actually did | Result on cat×dog seed 9 |
|---|---|---|
| `phase1_r32_100k` | the reference: 88 cells, 11 look-alike pairs, all 50 steps, weight decay 0, 100k steps | 0.46 to 0.51 |
| `v54d_C2_allsteps` | 54 cells, **all 50 steps** | 0.458 |
| `v54d_P` | the same 54 cells, **steps 0 to 25 only** | 0.389 |
| `v54d_C1_allseeds` | 182 cells, every cached seed of the same 24 pairs, steps 0 to 25 | 0.359 |
| `v54d_P_orth3` | the same 54 cells, steps 0 to 25, **orth3** | 0.343 |
| `v56_wd0` | 72 cells with objects, all 50 steps, **weight decay 0** | pending |
| `v56_wd1e2` | the same 72 cells, **weight decay 1e-2** | pending |

`v54d_P` and `v54d_C2_allsteps` have no differing config field at all: the step range is the whole
difference between them, and it is worth 0.07 of cosine.

## The names to use from here

`<pool size>-<step range>[-<what else changed>]`. The pool size is the cell count, so a name stops
being right the moment the pool changes, which is the point.

| Name | What it does |
|---|---|
| `pool43-all50` | the parent: 43 curated cells, all 50 steps, plain loss, weight decay 1e-2 |
| `pool43-early25` | the same cells, steps 0 to 25 only |
| `pool43-all50-orth3` | the same cells and steps, orthogonal error charged three times |
| `pool43-animals` | the 22 animal pairs only, the 7 object pairs dropped |
| `pool43-unfiltered` | the same 29 pairs, every cached seed instead of the judged ones |

Each child differs from `pool43-all50` in exactly one thing, so a gap between a child and the
parent is attributable. `pool43-animals` is the exception worth stating: dropping the object pairs
changes the cell count as well as the kind of pair, so read its gap as both.
