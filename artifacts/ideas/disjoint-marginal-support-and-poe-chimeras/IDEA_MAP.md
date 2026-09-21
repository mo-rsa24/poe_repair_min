# 💡 Does the chimera failure come from the composition math or the training data

## Where this came from

Raised while reviewing [the basins-by-hand plan](../../../plans/05-when-does-the-outcome-lock-in/plans/reading/01-basins-by-hand.md)
in this project, but it is not part of that scope or this paper. It needs new training runs on
a purpose-built dataset, which is a project of its own. Parked here to revisit later, once the
current paper ships.

## The question

Score addition, the operation this project's PoE composition actually performs, is exact for
combining log-densities: if two experts are independent, the gradient of the log of their
product equals the sum of their individual score gradients, with no approximation. So when
composing "a cat" and "a dog" produces a fused hybrid animal instead of a cat and a dog in one
scene, is that because the addition is the wrong operation, or because the two experts' learned
marginals happen to overlap in a way that makes the true product's mode a fused hybrid? The
first would mean the failure is inherent to product-of-experts composition. The second would
mean it is a property of how these two specific experts were trained, and the failure would not
generalize to experts trained differently.

## The mechanism, in the simplest form that shows it

Reduce an image to four numbers: how much "cat" is present in the left half of the frame, how
much in the right half, and the same two numbers for "dog." A LoRA fine-tuned on canonical,
object-fills-the-frame photographs learns a marginal for "cat" concentrated on both halves
reading as cat, and a marginal for "dog" concentrated on both halves reading as dog. Their
product then wants a point that reads as cat everywhere and dog everywhere at once, which no
real image is, so the sampler settles on the closest compromise: a single fused creature.

Now suppose the cat expert was trained so a cat only ever occupies the left half of the frame,
and the dog expert so a dog only ever occupies the right half. The two marginals no longer
compete for the same region. The product's mode becomes achievable without compromise: cat on
the left, dog on the right, at full strength, both at once. Same addition, same operator,
different outcome, because the operands changed.

## The two-stage design

**Stage 1: closed form, no training.** Represent each expert as a low-dimensional Gaussian (or
small Gaussian mixture) over the four-number vector above. Compute the true product analytically
for two settings: marginals with full-frame support (expected mode: a single point high on both
concepts everywhere, the chimera analogue) and marginals with disjoint left/right support
(expected mode: two points, each answering one concept in its own half, the co-occurrence
analogue). This stage needs no data and no GPU time; it exists to confirm the geometric argument
is actually correct before spending anything on stage 2.

**Stage 2: trained score models, a real sampler.** Draw a few thousand synthetic samples from
each expert's distribution (cheap: seconds of CPU time for a 4-to-8-dimensional Gaussian
mixture), fit a small score network to each by denoising score matching, and run an annealed
Langevin or diffusion sampler that composes the two learned scores by addition, exactly the
operation the real project uses. Check whether the sampler's output matches stage 1's analytic
prediction for both the full-frame-support experts and the disjoint-support experts. This is the
stage that actually tests whether the shift survives a trained model and a discretized sampler,
not just the exact math, and it is the piece the real-image LoRA experiment cannot cheaply give
you, because on real images you can never fully verify that the dataset shifted the marginal the
way you intended before you also have to ask what that shift did to the product.

## Why this precedes the real-LoRA experiment, not replaces it

A confirmed shift in stage 2 justifies the cost of the real experiment (curating a dataset,
training two LoRAs, comparing against the existing animals-compose experts) by pre-validating
that the mechanism is real in principle. A failed shift in stage 2 says the geometric argument
itself is wrong, or is more subtle than four numbers can capture, and would save the cost of
training real LoRAs on a hypothesis that does not hold even in the setting built to favor it.

## What would count as confirming or killing this

**Confirms:** stage 2's sampled output, scored the same way stage 1 predicts it (mode location
in the four-number space), lands on the disjoint-support side for disjoint marginals and the
full-frame side for full-frame marginals, across repeated draws.

**Kills:** the trained sampler's output does not track the analytic mode for either setting, or
tracks it for one setting but not the other in a way the geometric argument does not predict.
Either result means the four-number toy is not capturing the mechanism, and the idea needs a
richer toy (more regions, or a genuine 2D spatial field instead of four scalars) before it is
worth testing on real images.

## What this would mean for the paper's claim, if carried further

If the real-LoRA follow-on (training two experts on a dataset curated so cats and dogs always
appear side by side, then composing them the same way as the existing animals-compose experts)
also shows composition succeeding, the honest claim becomes: PoE composition failure is a
property of a specific and common training-data geometry (object-centric, frame-filling crops),
not an inevitable property of composing densities. That bounds the generality of "PoE
composition fails" rather than removing it, since most captioned single-concept training sets
are built exactly the way that produces the failure.

## Next step

Not scheduled. When revisited: scope stage 1 and stage 2 as a small, self-contained toy project
(a day or two, no GPU training beyond the tiny score networks), confirmed before any real-image
LoRA work is proposed.
