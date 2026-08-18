# F1 left panel: the image prompt

Paste the fenced block below into ChatGPT, save what it returns as
`assets/F1-schematic.png`, and `scripts/make_f1.py` does the rest.

**The drawn panel contains no animals.** It is contour geometry and labels, nothing
else. Every animal a reader sees in figure 1 is a real model output, placed by the
build script:

| position in the figure | what goes there |
|---|---|
| the marked product peak, left panel | a leader line crossing to the right panel |
| the right panel | `outputs/interaction_term/dose/pairs/a_cat__x__a_dog/seed_9/teacher_residual_const_lam000/`, large |
| the dashed "cat beside dog" region | nothing, deliberately |
| the two contour-cluster centres | labels only |

The dashed region stays empty because plain PoE never produces an image there. The
absence is the claim, and it says it better than a picture would. The real cat beside
a real labrador exists for this pair and seed at `teacher_residual_const_lam100`, but
that image required the correction, so it belongs to F2 and not to the figure about
the failure.

Art direction is paper minimal from `~/.claude/ILLUSTRATION_STYLE_GUIDE.md`, picked
because this is a single abstract subject where any card, badge, or dashboard chrome
would make a drawn argument look like a measured result. Light background throughout.

Two accent colors carry meaning and nothing else does. Warm terracotta means what one
prompt on its own asks for, used identically for both single-concept regions. Muted
slate blue means what the product asks for.

```
Style: warm off-white paper background. One subject per image, drawn as clean
geometry in fine ink-like lines. A restrained palette of exactly two accent
colors carrying meaning, never decoration. All labels in a clean serif type,
sparse, lowercase. No pills, no cards, no flowchart boxes, no UI chrome. Arrows
only where the arrows are the subject.

Scene: a single wide rectangular field standing for the space of all images,
drawn with a very faint thin border and nothing else, no axes, no ticks, no
grid, no numbers. Two large overlapping contour-ring clusters sit side by side
across the middle of the field, like two hills on a topographic map, each drawn
as five or six nested closed rings getting tighter toward its centre. The left
cluster's centre sits at roughly one third across, the right cluster's centre at
roughly two thirds. They overlap in a lens-shaped region in the middle. Inside
that lens, and only inside it, a small tight region is filled with a soft even
wash and carries a single filled dot at its densest point. Lower left of the
field, well away from both cluster centres and outside every ring, a rounded
region is outlined in a fine dashed line with no fill at all. Generous empty
space around everything.

Cast: contour rings and labels only. No animals, no creatures, no figures, no
objects of any kind sit anywhere in the image. Both contour-ring clusters are
drawn in the same warm terracotta ink at the same weight, because they mean the
same kind of thing, one concept on its own. The filled lens region, its wash,
and its peak dot are drawn in muted slate blue, meaning what the product of the
two asks for. The peak dot is small, solid, and clearly the single densest point
of the blue region, because a line will later be drawn from it to a photograph
placed outside this image. The dashed region is completely empty inside except
for its label.

Flows: no arrows and no leader lines anywhere. The overlap is communicated by the
rings crossing, and nothing in the image points at anything else.

Text in the image: none. The image contains no words, no letters, no numerals and
no captions of any kind. Every label is added afterwards by other software, so
leave clear empty space beneath each contour cluster, beneath the blue region,
inside the dashed region, and along the bottom edge of the field.

Exclusions: no text, no letters, no words, no numbers, no captions, no titles, no
legend, no signature. No animals anywhere in the image, not drawn, not sketched,
not photographic, not as silhouettes, not as icons, not inside the dashed region
and not at the contour centres. No paw prints, no ears, no tails, no cartoon
faces. No axis lines, no tick marks, no gridlines, no colorbar, no third concept,
no additional contour clusters, no shading outside the lens region, no arrows or
leader lines of any kind, no logos, no watermark.
```

## Why the drawing carries no words

Text baked into a generated image arrives at whatever size the image model chose. On
the first version of this schematic the labels measured 2.4pt once the panel was placed
in the ICLR column, and 4.4pt even if the drawing took the full column width, against
10pt body text. Nothing about the layout fixes that, because the label size is fixed
relative to the drawing.

So `make_f1.py` sets every label itself: it finds the two contour clusters, the product
peak, and the dashed region in the image, then writes "a cat", "a dog", "the product",
"cat beside dog", "no product mass" and "schematic, not measured" at a size that reads
in print. Labels can be moved without regenerating anything, and the image model never
gets a chance to garble a word.

## What the returned image must get right

Reject and regenerate if any of these fail.

The blue filled region sits strictly inside the lens where the two ring clusters
overlap and never spills past either cluster. The whole argument is that the product
concentrates where both concepts are satisfied at once.

The dashed "cat beside dog" region lies fully outside every contour ring. A reading the
sentence intends but the product gives no mass to is the point of the panel; if that
region touches the rings, the image says the opposite of what it is for.

There is no animal anywhere in the returned image. A drawn chimera in this panel would
be an invented picture of the paper's own result sitting next to a real one, which is
the failure this whole panel is arranged to avoid.

There is no text anywhere in the returned image, not even a stray letter or a signature.
The script assumes the only near-black ink in the picture is the dashed region's
outline, and any stray lettering moves the labels to the wrong place.

The peak dot is a single small solid mark, not a cluster and not a starburst. The build
script anchors both a label and a leader line to it.

## Then

Save the accepted image as `assets/F1-schematic.png`. `scripts/make_f1.py` reads it,
draws it on the left axes, puts the real λ=0 cell on the right axes, and connects the
peak dot to it. The script is the only thing that decides sizes and placement, so the
figure can be rebuilt from the same PNG without regenerating anything.
