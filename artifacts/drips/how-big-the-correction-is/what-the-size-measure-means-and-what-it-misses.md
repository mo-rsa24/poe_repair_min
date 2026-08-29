# What the size measure means, and the four probes that would go past it

```
walk:       /plain-speak --drip on the significance of ||r_t|| / ||eps_PoE||
state:      parked at piece 3 of 5; pieces 1 and 2 delivered, one tangent answered in full
measure:    curve_for in scripts/snr_collapse.py
artifact:   https://claude.ai/code/artifact/31277c54-e582-49dd-80e8-52788af234ba
            (six-frame interactive walk: one r_t tensor -> norm -> ratio -> curve -> two pairs -> median scaling)
```

## What the walk covers

Five pieces, in order: why the denominator is the PoE prediction and not the joint one; what the
magnitude number buys and what it cannot claim; the λ = 0.5 on steps 0 to 10 idea against the dose
and window experiments; direction, and what the cross-seed cosine of 0.002 forces; trajectories on
the manifold and the distance metrics that behave there.

Pieces 1 and 2 are delivered. Piece 3 is where the walk resumes.

## What is settled

**The denominator is the thing being corrected.**

`r_t = eps_mono - eps_PoE`, so `eps_mono = eps_PoE + r_t`. Dividing by `||eps_mono||` puts the
correction inside its own yardstick, and a large correction would partly shrink its own reading.
`||eps_PoE||` is independent of the numerator in a way `||eps_mono||` is not. The choice is
pre-registered in `report/normalization_preregistration.md` and lives once, in `curve_for`.
Numerically the two denominators differ by a few percent at ratios near 0.1; the difference is
conceptual, not arithmetic.

**The curve is a budget and a schedule, not a cause.**

Per-curve medians of 0.094 (eagle x hawk, seed 9) and 0.215 (lion x tiger, seed 1) say the missing
piece is a tenth to a fifth of the prediction it lands on. Small enough that learning it and adding
it back is a plausible repair. Had the medians come out near 1, the repair story would be dead,
because that is replacing the prediction rather than correcting it. The mid-run rise locates where
PoE's error concentrates, which is what makes a windowed intervention worth testing rather than a
guess.

**Three things the magnitude cannot say.**

It cannot say the term causes composition; adding it back has to be tested, and is (the dose and
window experiments). It cannot say which way the correction points. And a norm is one scalar per
step, so two very different tensors can share one: the curve never shows that two pairs need the
same correction, only same-sized ones.

## The four probes, with the pattern each would have to show

Each is cache-only. None needs new sampling.

### The step-by-step alignment matrix

Cell `(i, j)` is the cosine between the correction at step `i` and at step `j`, over the 50 steps
of one cell. Agreeing steps read as a bright block, and the block's edge is a window boundary read
off structure instead of eyeballed.

**Expect:** one solid bright square over the early steps, then a sudden break into dark speckle.

**Good:** the square's edge sits near step 12, so the window has support from direction as well as
from size. **Bad:** brightness fading with no edge, meaning there is no clean window in direction
and the inject-early story rests on size and the ablation alone. **Also worth catching:** a bright
patch far off the diagonal, meaning a late step has swung back to where an early step pointed,
which no step-to-step read can show.

### The 2x2 grid: images over two curves, two pairs

`a_cat__x__a_dog` on the left, `an_eagle__x__a_hawk` on the right. Decoded x-hat-0 thumbnails
across the top at steps 0, 5, 10, 15, 30, 40. Beneath each, two flat curves sharing the step axis:
consecutive-step direction stability above, `||r_t|| / ||eps_PoE||` below. No third axis; a 3D
curve makes both variables harder to read.

**Expect:** thumbnails staying an undecided animal-blob through the early steps while the direction
curve is flat and the size curve sits near a tenth, then a committed image, and only after that the
direction curve going ragged.

**Good:** the ragged region starts after the image has committed, so the correction is orderly
exactly while the picture is still fixable. **Bad:** chaos starting while the blob is still
undecided, which breaks the story; or the two pairs' stable bands sitting at different steps, which
means the window is per-pair and a single global window is the wrong design.

**The cross-pair check with teeth:** across the 17 population pairs, length of the smooth region
against compose gain, one scatter, 17 points. A rising trend validates the sign. A flat cloud means
smoothness belongs to the noise schedule rather than to the pairs, and the paper should not lean
on it.

### The rank read

Stack a run's 50 correction vectors into a 50 x 65,536 matrix. The singular values count how many
directions the correction ever uses across the whole run, which is a different question from either
of the two above (they are local and pairwise; this is global).

**Expect:** a singular-value plot with a cliff.

**Good:** a cliff after about 3 values, so even the chaotic late phase is fast turning inside a tiny
room and the adapter only has to learn that room. This is what would license the low-rank framing.
**Bad:** a slow slide across about 30 values, meaning the correction is genuinely high-dimensional
and the adapter's job is harder.

**Companion picture:** project all 50 steps onto the top two directions. A traced path means the
correction evolves along something parameterizable; a shapeless cloud means it does not.

**Pinned rule:** run the decomposition on the true corrections `r_t`, never on the adapter's
outputs.

### Correction size against blendedness

Per pair, mid-run mass of the ratio curve on x, blendedness of that pair's PoE render on y, 17
points. Blendedness comes from the validated scorer artifacts; check `scorer_validated.json` for
which read is trusted.

**Good:** a rising trend, which is the quantitative half of "the ratio rises where the render is
still deciding what animal this is". **Bad:** a flat cloud, meaning correction size and visible
blending are unrelated and the qualitative figure is decoration.

## Why direction matters despite 65,536 dimensions

Size cannot be injected; only a vector can, so direction is the payload and the norm curve only
says how much of it and when. The same pair re-run under a different seed gives correction vectors
with cosine about 0.002, essentially orthogonal. There is no fixed per-pair correction direction
sitting in latent space: the direction is a function of the current state `x_t`. That is what
forces the fix to be a learned function rather than a stored vector.

Faithful projections exist and are limited. A cosine matrix and a singular-value spectrum stay
inside the true space. A 2-D scatter of 65,536-dimensional vectors does not, and a 2-D replica
scenario is a teaching device that has to be labelled illustrative.

One caution that applies to every cosine above: random vectors in 65,536 dimensions are almost
always near-orthogonal, so a cosine near zero is only meaningful against that null.

## Two prompts this walk produced, not yet run

Both are pasteable as they stand.

**The curve with images, plus the blendedness scatter.**

```
In poe_repair_min. Build the "why the correction-size curve matters" evidence pair, cache-only,
no new sampling.

The measure: curve_for in scripts/snr_collapse.py (||r_t|| / ||eps_PoE|| per step; median
scaling optional here). Cache root: /datasets/mmolefe/poe_repair_min/outputs/training_cache.
Population sidecar: paper/iclr/figures/correction-size-over-the-denoising-run-across-17-pairs.json.

Figure A (qual-quant pair): a_cat__x__a_dog's own ratio curve, x = denoising step 0-49,
with decoded x-hat-0 thumbnails from the cached PoE trajectory at steps 2, 12, 25, 45 placed
under the curve at their step positions. The read the figure must support: the ratio rises
exactly where the render is still deciding what animal this is. Caveat to honor: cat x dog is
not one of the 17 population pairs, so its curve gets computed from its cached cell, not
reused. If decoding x-hat-0 needs the VAE on GPU, say so and stop rather than silently queuing.

Figure B (the correlation): one scatter, 17 points, one per population pair. x = mid-run mass
of the ratio curve (define it in the sidecar: e.g. mean ratio over the measured window, and
name the window source). y = blendedness of that pair's PoE render, scored by the validated
scorer artifacts (chimera scorer / caption readback; check scorer_validated.json for which
read is trusted). Write the sidecar JSON next to the PNG per repo convention.

Context that motivates it: the window experiments already showed injecting the correction only
in the early window composes, so the window is measured, not locked. This figure pair is the
qualitative face of that result.
```

**The twin-panel scene for direction.**

```
/picture-speak borrowing math-scene's interactive style. Target claim, verbatim: "Size cannot
be injected; only a vector can. The correction's direction is a function of the current state
x_t, not a fixed pair vector: the same pair re-run with a different seed gives cosine ~0.002.
In 65,536 dimensions this is abstract, so replicate the setting on a 2D plane."

Build two linked panels, one scrub control (the denoising step), one clock:
- Left panel, the real lens: x = denoising step 0-49, y = ||r_t|| / ||eps_PoE||, the real
  eagle x hawk seed 9 curve (cache: /datasets/mmolefe/poe_repair_min/outputs/training_cache,
  measure: curve_for in scripts/snr_collapse.py). A marker rides the curve at the scrubbed step.
- Right panel, the toy lens, LABELED ILLUSTRATIVE: a 2D cartesian plane where a point x_t
  walks a denoising path, and at the scrubbed step two arrows leave it: the PoE prediction
  and the correction r_t drawn as a small arrow attached to its tip. As the scrub moves, the
  correction arrow's length tracks the left panel's marker (real) while its angle turns
  smoothly early and erratically late (illustrative, mimicking the measured consecutive-cosine
  behavior: smooth early, chaotic late).
- The teaching beats, one frame each: (1) the norm is the arrow's length only; (2) two arrows,
  same length, opposite angles, opposite effects on the path; (3) same pair, second seed:
  the arrow at the same step points somewhere else entirely (cosine 0.002), so no stored arrow
  works; (4) a function that reads x_t and emits the arrow does work, and that is what the
  adapter is.
Label real against illustrative on every frame. The 2D plane never claims to be a projection
of the true 65,536-dim space; it is a replica scenario, and the page says so.
```

## Where this lands, and where it does not

These are measurement and figure designs, so they belong to the figure ladder and to a plan under
`does-the-correction-cause-composition`, not to
[the supervisors' structure merge](../../ideas/merge-supervisor-structure/IDEA_MAP.md). That walk
is about section naming and its six claims are settled. The seam between the two is only that
under its framing `r_t` is the plurality term, which makes these probes candidate figures for the
paper's sections 3 and 4. No task lines are written for any of the four probes yet.

## Resuming

Piece 3: the λ = 0.5 on steps 0 to 10 idea, read against the dose sweep and the window ablation.
Then piece 4 (direction) and piece 5 (trajectories and distance metrics), both of which the tangent
above has already previewed.
