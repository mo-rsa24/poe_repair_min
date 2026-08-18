# What each of the seven figures argues, and why we bothered

Read this before designing or building any paper figure. The plan file
[figure-01-the-seven-paper-figures.md](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/plans/figure-01-the-seven-paper-figures.md)
holds the checkboxes. This holds the reasons. Neither repeats the other.

Every figure below answers the same eight questions in the same order, so you can
read one section or compare two:

**The question** it answers · **Why bother**, meaning what would change if the
answer came out the other way · **What is plotted**, axes named in plain words ·
**What to look at**, the thing your eye should find · **What it may not claim** ·
**The numbers**, and where they live · **Built** or what is still missing ·
**Go deeper**, a prompt to paste when the maths behind the figure has stopped
meaning anything and you want it derivable again rather than memorised.

## The words

- **PoE**, the broken way of composing: ask the model about "a cat" and about "a
  dog" separately, then add the two answers. It usually paints one fused animal.
- **Mono**, the target: hand the model the joined prompt "a cat and a dog". It
  composes fine. That is why it is only ever the thing we aim at, never the method.
  Needing the joined prompt is the problem we are trying to solve.
- **The correction**, written `r_t`: at each denoising step, Mono's prediction minus
  PoE's prediction. It is exactly the piece PoE throws away. "The correction" and
  `r_t` mean the same thing throughout.
- **λ (lambda)**: how much of the correction we add back into PoE. 0 is none, 1 is
  all of it.
- **A cell**: one generated picture. One animal pair, one λ, one starting noise (one
  seed). Everything counted here is counted in cells.
- **Compose rate**: over a set of cells, the fraction showing two separate animals
  instead of one fused one. Decided by the validated scorer, an object detector that
  counts animal boxes and calls a cell composed at 2 or more. Never decided by eye.
- **The two fake corrections**, our controls: a random vector stretched to the same
  length as the real correction, and the real correction belonging to a *different*
  animal pair. Same size, wrong direction. They exist because "adding something
  helped" is not the claim. "Adding *this*, pointing *this way*, helped" is.
- **The fork step**: walk the PoE path and the Mono path from the same starting
  noise, and measure the distance between them at each of the 50 denoising steps.
  Early on they track each other. Then at some step the distance takes off. That
  takeoff is the fork, measured at step 16 from cached trajectories.

## The argument the seven figures make together

The paper makes one chain and each figure is one link. Losing a link does not soften
the story, it breaks it at that point.

1. Composing with PoE fails, and it fails in one specific way: one fused animal
   rather than two (**F1**).
2. The thing PoE is missing is `r_t`, and it is missing *causally*. Put it back and
   composition comes back. Put back something the same size pointing elsewhere and
   it does not (**F2**).
3. `r_t` is not a per-pair quirk. Its size is set by how far through the denoising
   run you are, not by which animals were asked for. That is the first reason to
   believe one learned adapter could serve pairs it never saw (**F3**).
4. `r_t` is not needed uniformly. There is a point in the run where the outcome is
   decided, and the correction has to be there for it (**F4**).
5. The λ dial is not an artefact of the scorer. Three measurements that share none
   of the scorer's machinery move together as λ rises (**F5**).
6. `r_t` is low-rank, which is the second and stronger reason a small adapter can
   learn it, and specifically why rank 8 was enough (**F6**).
7. When we train that adapter, it changes what a word paints rather than where the
   word looks, which is the mechanism the paper claims (**F7**).

F8 lives in the sibling scope and closes the chain: the trained adapter transfers to
held-out pairs.

## F1: the two readings of "a cat and a dog"

**The question.** What does the failure actually look like?

**Why bother.** Before any causal claim, the reader has to know what is broken. A
reader who has not seen the failure reads the rest of the paper as arithmetic. The
phrase "a cat and a dog" has two readings: two animals in one scene, or one animal
that is catlike and doglike. PoE reliably produces the second. This figure is the
only place the paper shows that, and it does it with a picture rather than a rate.

**What is plotted.** No chart. Two panels. Left: two concept densities and the
result of multiplying them, drawn as a schematic and labelled inside the panel. The
peak of the product sits between the two concepts rather than at either one. Right:
the real cat × dog seed 9 picture at λ=0, one head split down the middle, cat ear,
eye and whiskers on the left, dog ear, muzzle and tongue on the right. One arrow
joins the product's peak to the picture.

**What to look at.** That arrow. It is the whole argument: the fused animal *is* the
most likely point under the product, so this failure is what PoE asks for, not a bug
in how it was run.

**What it may not claim.** Anything about how often this happens. It shows one cell.
The λ=0 rate is owed to the results section.

**The numbers.** None on the figure, by design.

**Built.** `python scripts/make_f1.py` → `paper/iclr/figures/F1-two-meanings.pdf`.
The left panel is drawn by the script rather than generated by an image model, for
two reasons found the hard way. The image model's own lettering measured 2.4pt once
the panel sat in the ICLR column. And a generated fused animal sitting beside a real
one is exactly the confusion the panel exists to prevent.

**Go deeper:**
> `/math-scene the product of two concept densities in a diffusion model: show two
> 2-D densities, their product, and where the argmax of the product sits relative
> to each mode, with a slider on how far apart the two modes are, so I can see the
> fused-animal regime appear`

## F2: more correction, more composition

**The question.** Is the correction the *cause* of composition, or just something
that helps?

**Why bother.** This is the paper's central claim and the first figure a reviewer
will attack. The obvious objection: maybe disturbing the sampler at all shakes it
out of the fused mode, in which case `r_t` is not special and the paper has no
mechanism. The two fakes kill that objection inside the picture rather than in the
caption.

**What is plotted.** Bottom half: compose rate on y against λ on x, at 0, 0.25, 0.5,
0.75 and 1. Three curves, one for the real correction and one for each fake. Top
half: a 3 by 5 grid of real generated pictures on that same λ axis, one row per
injected vector, one column per λ, same pair and same seed throughout.

**What to look at.** Read *down* the λ=1 column: the real correction gives two
animals while both fakes still give one fused animal. Read *across* the top row: the
fusion separates as λ rises. The grid is the half that persuades. The curves are the
half that generalises.

**What it may not claim.** The 3% at λ=0 is not a real success. It is one cell (frog
× toad, seed 10) where the detector drew three boxes on one fused body with a
duplicated head and extra limbs. The scorer's rule does not get changed to remove
it. The note travels beside the number wherever the number goes.

**The numbers.** The real correction takes the compose rate from 3% at λ=0 to 94% at
λ=1. Both fakes stay at or below 6% everywhere. AUC is the area under each
compose-rate-against-λ curve on a 0-to-1 scale, which summarises the whole curve
rather than just its endpoint: 0.387 for the real correction against 0.023 and 0.039
for the fakes, so roughly a tenfold gap. 32 cells behind every point.

**A layout rule that is not cosmetic.** The control rows are drawn exactly like the
real row and never greyed out. Greying them tells the reader which row is meant to
lose before they have looked, which is the opposite of what a control is for.

**Built.** `python scripts/make_f2.py` → `paper/iclr/figures/F2-dose-response.pdf`.
Cell is cat × dog seed 9. `F2b-dissimilar-pair.pdf` is elephant × penguin, held in
reserve for the "they only fuse because they look alike" objection.

**Go deeper:**
> `/math-scene the correction r_t = the joint-prompt prediction minus the
> Product-of-Experts prediction, with a slider on lambda so I can watch the
> corrected prediction move along the line from the PoE point to the Mono target,
> and see what partial lambda actually means geometrically`

## F3: the correction's size follows how far through the run you are

**The question.** Is `r_t` a different object for every pair of animals, or one
object whose size is set by how noisy the image currently is?

**Why bother.** If `r_t` were specific to cat × dog, then fixing cat × dog would
teach you nothing about eagle × hawk. The only honest method would be one correction
per pair, which is useless: you would need the joined prompt for every pair you
wanted to fix, and needing the joined prompt is the problem. The paper's whole claim
to generality rests on `r_t` being roughly the same function of the noise level for
every pair. This figure is the first evidence for that, and it is why the transfer
results later read as believable rather than lucky. A null here would not be a weak
figure. It would send the method back to per-pair corrections.

**What is plotted.** x is the denoising step, 0 to 50 left to right, with "noise"
under the left end and "image" under the right. DDIM runs from noise to image, and
the step number is an axis the reader already has.

y is the correction's size, with two scalings on it. First ‖r_t‖ is divided by the
size of the prediction it corrects. Then each curve is divided by its own median. So
y=1 is imposed rather than measured: it marks "this pair at its own typical amount
of correction", and 1.4 means 40% above that pair's own typical. It says nothing
about whether this pair needs more correcting than another. The axis label says so.
The measure was fixed by
`instrument-02-fix-the-size-measure-before-any-result` before any of this ran, so it
cannot be chosen after seeing the answer.

The population is the seventeen pool pairs, shown as two grey bands and a median
line rather than as seventeen curves: the middle half, and everything but the top
and bottom tenth. Bands rather than named extremes, because the two most extreme
pool pairs are outside the middle half by construction, and drawing them as coloured
lines reads as disagreement while telling the reader less than the outer band does
about how far the population reaches.

Two pairs are named on top, in colour. Cat × dog, the running example, drawn with its
two individual seed curves faint beneath its mean, so seed agreement is shown once on
a worked case rather than asserted. And elephant × penguin, which answers "they only
fuse because they look alike". Neither is in the pool, and the pool is seventeen
look-alike pairs, so laying these two over the pool's band is the harder test rather
than the easier one. Every band label says "pool" for that reason: without it a reader
takes the coloured lines for two of the seventeen and reads their departure as the
population disagreeing with itself.

Every drawn line, bands included, is a rolling median over five steps, and the figure
says so in the corner. Adjacent steps differ by more than the shape does, so the raw
curves read as jitter and the shared shape is invisible. Nothing is smoothed relative
to anything else, and the unsmoothed curves stay in
`cache_analyses/step_collapse.json`.

Above the curve, sharing its x-axis, four decoded frames at steps 0, 16, 33 and 49
from cat × dog seed 9 with the correction on throughout.

**What to look at.** The shape of the median: the correction is small at the noisy
start, rises over roughly the first twenty steps, then sits flat. Then whether the
bands are narrow around it, which is what "the same function of the noise level for
every pair" would look like.

**What it may not claim.** Four things. The frames along the top anchor the axis,
telling the reader what step 33 looks like; they are not evidence for the y-value and
the caption says so. The uncorrected run is not shown beside the corrected one,
because those two diverging is F2's whole argument and repeating it here makes this
figure argue two things and land as neither. The pool pairs agree *better late than
early*, so no sentence may say the collapse holds uniformly. And the two named pairs
both leave the band in the second half, so the shared-shape claim is made for the
pool and demonstrated, not claimed, for pairs outside it.

**The numbers, and where they live.** How far apart the curves sit is measured as the
width of the band holding the middle half, divided by the height of the median at
that point. Over the seventeen pool pairs on the step axis with each pair's two seeds
averaged first: 21.4% as a median across the fifty steps, which the code's own
thresholds call `loose` rather than `tight`
([snr_collapse.py:171-172](../../scripts/snr_collapse.py#L171-L172)). Averaging the
seeds barely moved it, 22.9% to 21.4%, so the verdict word did not change and no
better number was quietly adopted. A band a fifth as tall as the curve puts a typical
pair a *tenth* from the middle, not a fifth.

The spread is very uneven and the median hides it: 30.0% over steps 0 to 24 against
15.6% over steps 25 to 49, and 65.1% at step 0 against 13.1% at step 49. The pairs
agree more as the image forms. All of these live in the appendix table rather than on
the plot, because a reader cannot act on a percentage while looking at a curve.

**Why not log-SNR.** log-SNR is the log signal-to-noise ratio of the current noisy
image. Its one advantage is staying comparable if someone reruns this with a
different sampler, which buys a reader the paper does not have and costs every reader
it does. Only DDIM is used here. Plotting against the step also removes an artefact:
the log-SNR version interpolated each cell onto twenty evenly spaced log-SNR points,
and steps are not evenly spaced in log-SNR, so the noisy end was a few real steps
stretched across several points.

**The fan at the noisy end is real.** It was worth checking whether the wide
disagreement at the start was an artefact of the log-SNR interpolation, and it is not:
74.5% at the noisy end survives the switch to the step axis unchanged. So the figure
shows the pairs disagreeing there, with the bands at their widest on the left. A
tidier drawing was available and would have been a false claim made with a picture
instead of a sentence.

**Built.** `python scripts/make_f3.py` → `paper/iclr/figures/F3-size-follows-noise.pdf`.
It writes a sidecar `.json` recording which pairs were drawn, which seeds, the
selection rule and the frame cell, so a caption can be checked later without rerunning
anything. The measure comes from `curve_for` in `scripts/snr_collapse.py` rather than
being repeated here, because two copies of a measure drift and the drift is invisible
in the figure. The frames cost no sampling: `cross/pairs/a_cat__x__a_dog/seed_9/
call__rall` already saved `latent_trajectory.pt` with the model's estimate of the
finished picture at all 50 steps, and `scripts/decode_trajectory_frames.py` turns
those into pictures with the VAE alone at 0.5s a frame.

**One thing left open.** Cat × dog swings hard from step 26 onward, and both its seeds
swing together, so it is not seed noise: its mean step-to-step change is 0.249 in the
second half against 0.045 in the first. Elephant × penguin does not do this (0.053
against 0.070). It is one cell out of nineteen behaving unlike the rest in the second
half of the run, and it is the running example, so a reader will look straight at it.
Worth understanding before the caption is frozen.

**Go deeper.** That axis decision is built as a scene you can operate:
[scene-logsnr/](scene-logsnr/README.md), run with `cd docs/figures/scene-logsnr && npm
install && npm run dev`. It draws SDXL's own schedule, derives log-SNR from it, and
computes both sides of the choice live. For the step axis: the cached run's steps are 20
timesteps apart every time, but move 0.16 apart in log-SNR mid-run against 2.46 at the
final step, so 7 of 20 evenly spaced log-SNR points carry one real step or none. For
log-SNR: one curve read by two samplers disagrees by 0.38 on the step axis against 0.01
on the log-SNR axis, in units of the plotted y. It opens on the first-year reading, that
log-SNR is the log-odds a unit of power in the noisy image came from the picture rather
than the noise.

## F4: when in the run the correction is needed

**The question.** Does the correction have to be present at a particular moment, or
does it just need to be present overall?

**Why bother.** Two reasons. Scientifically, if there is a moment where the outcome
is decided, that is a mechanism, and it can be checked against a completely separate
measurement (the fork step) that used no new generations at all. Two independent
estimates of the same moment agreeing is much stronger than one. Practically, a
narrow window means the fix can be applied for a fifth of the run rather than all of
it.

**What is plotted.** The correction is switched on inside a 10-step window and off
everywhere else, and that window slides across the 50 steps in strides of 5, giving
nine positions. The prompt stays on the whole time, so the only thing changing
between positions is *when* the correction acts. y is compose rate, x is the centre
step of the window. The fork step, 16, is drawn on the same axis as a shaded vertical
band.

**What to look at.** Whether there is a peak at all, and whether the shaded band sits
under it. A flat curve is a real answer: it says the correction is needed throughout.

**What it may not claim.** Anything about sliding the window over the *prompt* instead
of over the correction. That experiment was designed and deliberately not run, and
the caption names it as future work rather than leaving the reader to assume it was
covered.

**The numbers.** Nine window positions, and the fork step at 16 measured over 19
cells with 15 of them landing between steps 13 and 20.

**Why width 10, and not a width derived from the correction's size.** The obvious rule
was to make the window as wide as the narrowest band of steps carrying half the total
‖r_t‖. That rule gives 25 steps, half the run, because the raw ‖r_t‖ barely varies
when you add it up over blocks: every fifth of the run carries 15 to 22% of the
total. At width 25 every placement contains the fork step, so the curve comes out
flat whatever the truth is, and the experiment tests nothing. So the width is set by
what the sweep needs to be able to resolve. Width 10 at stride 5 puts the fork step
inside only two of the nine placements, and if timing matters those two win. Both
numbers live in `poe_repair/experiments/interaction_term/window_grid.py`, which the
runner, the scorer, the strip and the inspector all read, so none of them can
disagree about which grid was run.

This does not contradict F3, though a reader meeting the two claims ten pages apart
might think it does. F4 adds up the raw correction over blocks of steps. F3 plots, at
each single step, the correction divided by the size of the prediction it corrects.
Whichever figure is built second names the other's quantity in one clause.

**Not built.** Waits on hypothesis-03's 288-cell grid.

**Go deeper:**
> `/unpack the sliding-window design in hypothesis-03: why switching only the
> injected correction on and off, while the prompt stays on at every step, is the
> only version that isolates timing, and what the other version (sliding the
> window over the prompt instead) would have confounded it with`

## F5: one dial, three independent measurements

**The question.** Is the λ effect real, or is it something about the scorer?

**Why bother.** F2's whole result is read through one object detector. A reviewer can
reasonably ask whether λ is improving pictures or improving *detectability*. This
figure answers with three measurements that share none of the detector's machinery,
so agreement between them is agreement between independent instruments rather than
the same instrument three times.

**What is plotted.** Three panels sharing one λ colour bar, all reading the same
dose-sweep pictures:

- **The manifold walk.** Each picture is placed on the existing CLIP image-similarity
  axes, and the points slide from the fused region towards the two-animal region as λ
  rises. The control is a same-sized push in a random direction, which should not
  slide.
- **The caption crossover.** Each picture is matched against a bank of captions
  including a "blended animal" caption, and the best-matching caption switches from
  the blend to the two-animal one as λ rises. Same claim as the walk, made in
  language space rather than image space.
- **The density climb.** Whether the correction pushes along the direction sampling
  is already moving, rather than at right angles to it. Two controls: a random
  direction, and the right correction taken from the wrong step.

**What to look at.** Whether all three panels move in the same direction as λ rises,
and whether each panel's control stays put.

**What it may not claim, and this one has to be in the caption.** On the manifold
walk, the wrong pair's correction still travels 44% of the distance the right one
travels, against a 50% bar written beforehand. It passes, barely. The caption says
plainly that a mis-aimed correction gets you nearly half the slide, because the
figure will be read by someone who checks.

**The numbers.** 44% against a 50% bar on the walk. Normalised climb median +0.397
with 0 of 38 cells negative, against +0.000 for the random control and +0.11 for the
wrong-step control.

**One thing to know when reading these.** At λ=1 the corrected prediction reproduces
the Mono prediction exactly, measured at 1.9 grey levels out of 255. So the λ=1 point
is the joint render by construction and carries no information about whether the
method works. Every dose comparison is read at λ=0.75, the largest interior dose.

**Not built.**

**Go deeper:**
> `/unpack why lambda=1 reproduces the joint prediction exactly, straight from the
> definition of r_t, and what that means for which points on a dose curve carry
> evidence`

## F5b: the correction steers meaning from the start, and the fork is the midpoint

Held beside F5 the way F2b is held beside F2: same instrument as F5's manifold
walk (an embedding axis that shares none of the scorer's machinery), extended
over time instead of over λ.

**The question.** Does the image's *meaning* commit to fused-or-composed at a
particular moment, or does the correction steer it continuously?

**Why bother.** The fork step is measured on raw latents, and raw-latent
distance is noise-dominated early, so "the paths part at step 16" could be an
artefact of where the noise floor sits rather than a fact about the image. This
figure re-asks the question in a space that knows what a fused animal is. The
answer also carries a practical stake: a sharp commitment window would justify
applying the fix briefly; continuous steering says the correction is doing
semantic work from the first step. And the same read yields a per-pair
difficulty signal measured before most of the run has happened.

**What is plotted.** For each cell, both arms of the run, correction on
throughout (which reproduces the Mono prediction) and correction off (pure
PoE), are decoded step by step into the model's current estimate of the
finished picture, embedded with CLIP, and placed on the axis from that cell's
own PoE endpoint to its Mono endpoint. y is the two arms' separation along
that axis: 0 means indistinguishable, 1 means the full final separation. x is
the schedule timestep, noise left, image right, because the population runs a
20-step schedule and the named example a 50-step one, and the timestep is the
one axis both live on with no interpolation. The seventeen pool pairs are two
grey bands and a median line, same convention as F3. Cat × dog is drawn on top
from the 50-step grid with its seed curves faint beneath the mean, elephant ×
penguin from the 20-step grid. The fork band, steps 13 to 20 of 50 in timestep
terms, is shaded with the fork step dashed. Above the curve, two rows of
decoded frames from cat × dog seed 9, correction-on arm above correction-off,
at steps 0, 16, 33 and 49: they anchor the axis and are not evidence for the
y-value.

**What to look at.** Where the curves sit as they cross the shaded band:
already at roughly half their final separation, and still climbing on the far
side. No plateau before it, no jump inside it. The fork step is the midpoint
of a drift that starts almost immediately, not the moment the paths part.

**What it may not claim.** Four things. It does not contradict F4: F4 asks
when the correction must be *present* to flip the outcome, a causal question
only the window sweep answers; this figure is observational. It says nothing
about raw latents, where the fork remains a real feature of when drift clears
the noise floor. The naive version of this read, raw pairwise embedding
distance between the arms, is not evidence either way: two early mush frames
are far apart in any embedding space for texture reasons, and that read shows
step 0 as far apart as step 49. And the 20-step curves rest on estimates of
the finished picture recovered in closed form from consecutive noisy latents;
the recovery is validated at 2.6% median error against saved ground truth, but
its step-0 point is 14% off and is the least trustworthy on the plot.

**The difficulty correlation, and its bars.** Written before the run: mean
separation over the early window (timesteps 950 to 750) against each pair's
area under its real-correction compose-rate-against-λ curve, support at
Spearman |ρ| ≥ 0.5, dead below 0.3, sign reported rather than pre-committed.
Result: ρ = −0.500 in CLIP, exactly at the bar, and −0.690 in DINOv2, over the
8 pairs the dose sweep scored. The sign says pairs whose arms separate more
early respond worse to the correction. Eight points is too few to call this
more than support; it earns a scatter in the appendix, not a panel here, per
the one-panel-one-job rule.

**The numbers.** The pre-registered sharp-window checks failed and the
failure is the finding: steepest rise inside steps 13 to 20 for 3 of 9 cells
in CLIP and 1 of 9 in DINOv2 on the 50-step grid, with separation near zero at
step 0 and roughly half the axis covered by the fork band. All curves, checks
and bars live in `cache_analyses/trajectory_divergence/` with copies in
`evidence/f5b-trajectory-divergence/`.

**Built.** `python scripts/make_f5b.py` →
`paper/iclr/figures/F5b-gradual-commitment.pdf`, sidecar `.json` beside it.
Curves from `scripts/trajectory_divergence.py` (50-step cells) and
`scripts/dose_trajectory_divergence.py` (20-step cells, recovery validation,
correlation), both with their bars as constants in the source.

**Go deeper:**
> `/unpack why the model's per-step estimate of the finished picture can be
> recovered in closed form from two consecutive DDIM latents and the schedule,
> and why the recovery degenerates at the very first step`

## F6: the correction is low-rank enough to learn

**The question.** Can a small adapter represent `r_t`, and why was rank 8 enough?

**The answer this slot cannot give.** The intended argument was that almost all the
energy of the stacked corrections lives in a handful of directions, so rank 8 is the
right size read off the data rather than a lucky hyperparameter. That argument does
not survive its own control and the slot is open. The floor it rested on, a Gaussian
matrix of identical shape, gives every row the same expected norm, while real ‖r_t‖
spans a factor of 4.5 because it tracks the noise level, and rows of unequal size
concentrate energy without sharing any direction. Against random directions carrying
the real norms the pooled stack is 1.4x at k=8, not 11x. Scaling rows to unit norm
leaves genuine direction structure (7.8x at k=8) but it sits inside single runs
(4.8x) and not across them (1.2x), which is D1 and D3 restated rather than a new
claim. Full argument and tables in
`docs/evidence/F6-what-the-spectrum-measures/QUERY.md`.

**What still answers the question.** The adapter's behaviour, not the geometry of the
cached vectors: rank 8, 11 training pairs, 96.9% compose on six unseen pairs where
plain PoE composes 0%. Two cache-only geometric proxies for learnability have now
been tried and both failed, the subspace-overlap test and this spectrum.

**What is plotted, if the slot survives.** Energy captured against the number of
directions k, with two floors: the equal-norm Gaussian, and random directions
carrying the real ‖r_t‖. Only the gap to the second one is evidence about shared
directions, and reporting the first alone is what produced the retracted reading.

**What it may not claim.** That the held-out inset says anything about whether the
method transfers. It does not. The inset is 13.3% at k=64 on unseen pairs against
63.0% on training pairs, while the same adapter composes 96.9% on those pairs where
plain PoE composes 0%. The `r_t` vectors are near-orthogonal to each other, about
0.00 cosine even between training pairs, so no fitted subspace can contain unseen
pairs whether or not the correction transfers. Any sentence built on "shared
subspace" wording gets rewritten to that bounded form. Full argument in
`docs/evidence/F6-subspace-vs-transfer/QUERY.md`.

**The numbers.** 440 rows from `scripts/spectrum.py --pool --stride 10 --max-seeds 8`.
Against the equal-norm Gaussian floor: k=1 3.8% against 0.3% (14 times), k=8 22.6%
against 2.1% (11 times), k=64 63.0% against 16.4% (4 times). Against random
directions carrying the real ‖r_t‖, which is the floor that controls for size: 1.5,
1.4 and 1.1 times at the same three k. Quoting the first row of numbers without the
second is the error this slot was caught making.

**What one row is.** The correction at one denoising step of one cell, a cell being
one pair at one seed, flattened over the 4x128x128 latent into 65536 numbers. The
caption says so, because "low-rank" means nothing until the row is named. Averaging
a cell over its timesteps was rejected: the adapter is called once per step and
never produces such an average, and the steps of one cell share no direction to
average over (cosine +0.81 between steps 1 apart, +0.012 between steps 20 to 49
apart). The energy agrees: over 11 pairs at 8 seeds keeping every 10th step, the
per-step stack is 22.6% at k=8 against a 2.1% floor (10.7 times), where averaging
each cell gives 39.1% against a 9.8% floor (4.0 times), a higher percentage carrying
a weaker claim. Reasoning in hypothesis-04's review.

**How much of the cache the numbers come from.** All 11 training pairs at 8 seeds,
keeping every 10th step, and the invocation travels with the numbers because the
ratio depends on it. Adjacent steps are near-copies (cosine +0.81 one step apart)
and near-copy rows concentrate energy in a way a Gaussian floor of independent rows
does not correct for, so denser sampling flatters the claim: the ratio at k=8 is
21.8 keeping every step at 1 seed, 14.6 keeping every 3rd at 3 seeds, and 10.7 at
the setting quoted here. The claim holds at all three.

**Note on the arithmetic.** The cached tensors are fp16. Upcast to fp32 before any
accumulation, especially the SVD, or the singular values come back with NaNs.

**Not built.**

**Go deeper:**
> `/math-scene the SVD energy spectrum: show a matrix, its singular values, the
> cumulative energy curve, and the same curve for a random matrix of identical
> shape, with a slider on the true rank so I can see what "low-rank" looks like
> against its floor`

## F7: what the trained adapter changes inside the model

**The question.** When the trained adapter is switched on, does it change *what* a
word paints or *where* that word looks?

**Why bother.** Everything up to here is about a correction computed from the joined
prompt. F7 is about the adapter trained to produce that correction *without* the
joined prompt, and it is the paper's mechanism claim. Every prompt word does two
things inside the model: the attention weights decide where in the image it acts, and
the painted values decide what gets written there. The paper says the adapter changes
the second. If it changed the first, the story would be about spatial layout and the
paper would need a different explanation.

**What is plotted.** For each of the pair's two subject words, at three points in the
run, both maps are captured with the adapter off and again with it on, from the
identical starting state, over 64 cells of pairs the adapter never trained on. The
panel compares how much each map's spatial pattern moved. Beside it sit the transfer
table and the replication strip.

**What to look at.** Whether the painted content's bar is taller than the attention
bar, and by how much against the 1.2 line.

**What it may not claim.** The same effect shows up on the control pair. So the
sentence is "this is what the adapter does to any pair it touches", not "this is why
the fix works". The caption is capped at the narrower claim and the review file says
so.

**The numbers.** Median pattern ratio 1.52 across 64 cells against a bar of 1.2
written before the run: the painted content's pattern moves half again as much as the
attention pattern.

**Why the comparison is not the obvious one.** The obvious measure is each map's total
change, `||on − off|| / ||off||`. Under that measure the attention weights move 1.70
times more than the painted content, which contradicts the hypothesis. That reading
is wrong for two reasons about the instrument rather than the science. The two maps
are not on the same footing: attention weights are rows summing to one, painted
content carries raw magnitudes, so their norms are not comparable quantities. And the
adapter dims the attention weights by roughly 25% overall, a uniform brightness
change that swamps the spatial change the hypothesis is about. So each change is
split into **gain**, the single best uniform rescaling of the off-map onto the on-map,
and **pattern**, whatever a rescaling cannot explain. Only pattern is compared. Full
argument with its guards in
[../evidence/F7-mechanism-reprobe/measure-fairness.md](../evidence/F7-mechanism-reprobe/measure-fairness.md).

**Blocked on one decision.** One point per seed or one point per pair.
`/pair-figure` decides it.

**Not built.**

**Go deeper:**
> `/deep-learning-scene on poe_repair, focused on the cross-attention path the
> adapter touches, so "where the word looks" and "what the word paints" become two
> tensors I can point at in the real forward pass rather than two phrases`
