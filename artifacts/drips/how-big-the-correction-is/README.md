# What does the correction-size figure's population band actually divide by, and what can it claim?

Figure F3 of the paper pools 17 pairs' correction-size curves into population bands, and a reader
cannot trust the shape of those bands without knowing what the curve divides by, twice, before
pooling. This is a `/drip --math` walk through that two-stage normalization, plus a parked
`/plain-speak --drip` walk on what the resulting magnitude curve can and cannot claim about
composition. The math walk delivered its first piece and a worked toy example; the meaning walk
delivered two of five pieces and paused at the third, direction, with four unbuilt figure probes
recorded for whoever resumes it.

## What is in here

One rendered figure (the toy normalization-collapse plot) with its rendering script, one worked
numeric example, one parked explainer document carrying four unbuilt figure probes and two
pasteable build prompts, and the ladder's own index, `00-INDEX.md`.

## Four toy pairs collapsing onto one shape after each divides by its own median

![](figures/normalization-collapse.png)

**What it shows**

Top panel: two invented three-point curves (near-noise, mid, near-image toy denoising steps) at
completely different scales, 0.6 to 1.0 for one toy pair and 1.2 to 2.0 for the other. Bottom
panel: the same two curves after each divides by its own median across its three points,
collapsing onto the identical sequence (0.75, 1.25, 1.0).

**How we got this image**

`render_normalization_collapse.py` in this folder plots four invented numbers (no real run data)
worked by hand in `worked-example-toy-numbers.md`, to make F3's real two-stage normalization,
`‖r_t‖/‖ε_PoE‖` per step then divided by that pair's own median before pooling, concrete as
arithmetic rather than only as a rule in words.

**What it lets you say**

Two pairs can have completely different raw correction scales yet the same shape across
denoising steps (low near noise, a peak mid-run, settling near the image), and dividing each
pair by its own median is exactly the operation that reveals that shared shape and lets 17
differently-scaled pairs stack into one set of population bands.

**What it cannot say**

Every number in the figure is invented. It shows only that the arithmetic behaves as claimed on
toy numbers with the denominator `‖ε_PoE‖` held artificially fixed across steps and pairs; it is
not a measurement of the real 17-pair population, and the toy population median (n=4) is
explicitly not a claim about where the real percentile bands would sit.

## The pieces

**The worked toy example** (`worked-example-toy-numbers.md`) carries the arithmetic behind the
figure above through five steps: per-vector norms, the raw ratio, each pair's own median, the
pair-normalized value, and a toy population median across four pairs, tying every number back to
the two equations F3's real code applies.

**The parked meaning walk** (`what-the-size-measure-means-and-what-it-misses.md`) settles two
things: the denominator is `‖ε_PoE‖`, not `‖ε_mono‖`, because the correction should be measured
against what it corrects rather than partly measured against itself, a choice pre-registered in
`report/normalization_preregistration.md`; and the curve is a budget and a schedule, not a cause,
with the real per-curve medians (0.094 for eagle-hawk seed 9, 0.215 for lion-tiger seed 1) small
enough that learning and re-adding the term is plausible, but only the dose and window
experiments, not this curve, can show the term causes composition. It records four cache-only
probes that were designed but never built (a step-by-step direction-alignment matrix, an
images-over-curves grid for two pairs, a singular-value rank read on one run's 50 correction
vectors, and correction size plotted against blendedness across 17 pairs), each with the pattern
a real result and a null result would show, plus two pasteable prompts to build the first two.

**Where it came from and what judged it**

`artifacts/ideas/merge-supervisor-structure/IDEA_MAP.md` names this folder's parked meaning walk
twice, as the source of the four unbuilt figure probes for the paper's sections 3 and 4 and as
required reading before either probe is built or the walk is resumed at its piece 3.
