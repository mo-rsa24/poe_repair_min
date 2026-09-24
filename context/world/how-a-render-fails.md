# 🌍 How a render fails

A render that goes wrong here goes wrong in a small number of recognisable ways, and each look
points at a different cause. This file names the looks, says what causes each and how sure that
is, and points at a real tile showing it, so a failed render can be read for its cause instead of
only being called bad. The single failure the project is built around, one fused animal, has its
own entry in [chimera.md](chimera.md).

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [The seed sets the look first](#the-seed-sets-the-look-first)
- [The looks](#the-looks)
- [The gallery](#the-gallery)
- [Where this came from](#where-this-came-from)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-seed-sets-the-look-first) ➡️

- **Register**: whether a render reads as a photograph or as a drawing (pencil sketch, engraving,
  flat colour illustration, clipart). Two renders can compose equally well and sit in different
  registers.
- **The comparison sheets**: three images, one per held-out seed, each rendering every trained
  adapter on the same cat and dog cell. Filed in
  [which adapter composes cat and dog](../../artifacts/results/which-adapter-composes-cat-and-dog/README.md).
  A tile is named below by its seed and the label printed above it on the sheet.

## The seed sets the look first

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-looks) ➡️

**Whether a render is a photograph or a drawing is decided mostly by the seed's starting noise,
and nearly every adapter keeps its seed's register.** ✅

With no adapter at all, seed 12 draws a photograph and seed 11 draws a pencil sketch. Counted tile
by tile on the comparison sheets:

```text
seed 12 (photograph with no adapter)   21 of 23 adapters draw a photograph
                                         1 draws an illustration ("empty branch from cache")
                                         1 draws mush ("54-cell + plural prompt")
seed 11 (sketch with no adapter)       19 of 23 draw a sketch or flat illustration
                                         2 draw a photograph, both trained against filtered renders
                                         2 draw mush
```

So a drawing is usually the seed's doing rather than the adapter's, and an adapter's fidelity read
on a single seed mostly measures that seed. The two adapters that override a drawing seed were both
trained against filtered renders ("target is a picture" and "picture + cached empty"). The first of
them draws the same scene on every seed, which is its own failure below.

## The looks

Navigation: ⬅️ [The seed sets the look first](#the-seed-sets-the-look-first) | 📋 [TOC](#table-of-contents) | [Next](#the-gallery) ➡️

✅ seen on all three seeds or settled by a comparison that moved one thing · 🟡 seen, cause
inferred · ❓ seen once

| What it looks like | What causes it | How sure | Where to see it |
|---|---|---|---|
| one fused animal | no correction, or one too weak to carry to this pair | ✅ | every sheet, "no adapter" |
| flat clipart or line art of one animal, often in a dark circle | only the empty branch was free to move: it can restyle the whole picture and cannot place a second named animal | ✅ | seeds 9 and 11, "only the empty branch"; `v57w10-04-nullonly` at step 3,000 |
| brown or grey mush with no animal in it | trained far past its peak | ✅ | every sheet, "original, rank 8, 450k" |
| the same mush, from a changed branch prompt | branch prompts rewritten to "two animals" | ✅ | every sheet, "54-cell + plural prompt" |
| grey pencil sketch or engraving, often in a round frame | the seed's own register; on a photographic seed, the larger pools (54, 72, 182 cells) also fail to hold the photograph | 🟡 | seed 11, "no adapter"; seed 9, "54 cells" and "72 cells" |
| flat colour illustration of two animals | the seed's register again, rendered in colour by the v57 and v54 pools | 🟡 | seed 11, "43 cells, all 50 steps" |
| invented people | the plain loss trained past its peak on a broad pool | 🟡 | `v57w10-01-base` at step 10,000, seeds 9 and 10; seed 11, "43 cells, first 10, on the picture" |
| oil painting or impressionist scene | unknown; seen early in one run that froze the empty branch | ❓ | `v57w10-03-nulloff` at steps 1,000 and 3,000 |
| three or more animals on the right backdrop | plurality learned without identity | 🟡 | seed 9, "43 cells, all 50 steps" and "43 cells, first 10, on the picture" |
| two animals of the wrong species | identity lost while plurality holds | 🟡 | seed 11, "picture + cached empty" (a horse); seed 9, "judged on the picture" (two cats) |
| the same scene on every seed | trained against a filtered set of renders and learned one scene from it | 🟡 | every sheet, "target is a picture": a dog and a cat in a green mountain field each time |

The `v57w10-*` renders sit under
`/datasets/mmolefe/poe_repair_min/outputs/showcase/v57w10/<run>/samples/per_epoch/` on the cluster,
one folder per checkpoint.

## The gallery

Navigation: ⬅️ [The looks](#the-looks) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

<a href="../../artifacts/results/which-adapter-composes-cat-and-dog/how-a-render-fails-gallery.png"><img src="../../artifacts/results/which-adapter-composes-cat-and-dog/how-a-render-fails-gallery.png" width="420" alt="Six panels, each a failed render with the telling patch enlarged in the corner"></a>

Six of the looks above, one panel each, every panel a render already on disk. A red box marks the
patch, and the same patch is enlarged in the corner of its own panel, so the fused face, the
invented person, the third animal and the horse standing in for a dog can be seen rather than taken
on trust. Built by `scripts/failure_gallery.py`; the panels, their source files and their crop boxes
are listed in the sidecar beside the image.

Five looks still have no panel: the mush from a rewritten branch prompt, the pencil sketch, the flat
colour illustration, the oil painting, and the same scene on every seed. The last one needs several
seeds side by side rather than one crop.

## Where this came from

Navigation: ⬅️ [The gallery](#the-gallery) | 📋 [TOC](#table-of-contents)

Read by eye from the three comparison sheets (seeds 9, 11 and 12, twenty-three adapters and the
no-adapter control, rendered 2026-09-21 by `scripts/across_adapters.py`) and from the per-checkpoint
renders of the `v57w10` cohort. The clipart row is also backed by a comparison that moved one thing:
the adapter with only the empty branch free against the plain adapter, same step, same seed, where
the plain one draws several animals and this one draws one. No automated scorer was used; the
[instance counter](compose-rate.md#what-people-get-wrong) cannot tell a cat from a second dog, and the [edge measure](../../report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md#what-would-have-counted) prefers a sketch to a
photograph.
