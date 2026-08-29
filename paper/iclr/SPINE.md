# The spine

One line per section, each naming the single claim it carries, in reading order. Every section
below inherits from this: a paragraph that argues a different claim than its line gets flagged,
not drafted.

## The paper's one-sentence claim

Composing pretrained diffusion models by sampling from a product of experts assumes the composed
concepts are independent, that assumption fails when they are not, and the failure is exactly a
missing plurality term that a low-rank adapter can learn and carry to concept pairs it never
trained on.

## The lead

The paper opens on the diagnosis (the plurality term is real, causal, and measurable), then
turns to the fix (a low-rank adapter that learns it and transfers). The fix is not credible
without the diagnosis first: showing a number go up means nothing until the reader knows what
gap it is closing.

The two halves are split by which correction is in play. Sections 3 and 4 use the correction
computed from the joint prompt, which the composed sampler cannot have, so it exists to measure
and not to run. Section 5 uses the learned correction, which never sees the joint prompt. A
reader who loses that distinction reads section 4 as the method and section 5 as a repetition
of it.

## Section by section

- **1. Introduction.** Composing pretrained diffusion models by sampling from a product of
  experts silently assumes the composed concepts are independent, and that assumption breaks in a
  way the field has not named.
- **2. Background and Related Work.** Combining several of one network's own noise predictions
  linearly is already standard practice, and the product-of-experts and classifier-free
  composition methods that extend it build on the independence assumption without testing it.
- **3. Missing Implications: The Plurality Term.** The gap between the joint distribution over a
  correct composed scene and the product-of-experts approximation to it is a specific, definable
  term, an interaction term that this setting narrows to plurality, and this plurality term is
  what the paper measures and later corrects.
- **4. Restoring the Plurality Term Restores Composition.** A small amount of the measured
  plurality term restores composition when it arrives early in the denoising run, and the full
  amount arriving late does not.
- **5. Learning the Plurality Term.** A low-rank adapter on SDXL's cross-attention layers predicts
  the plurality term from the two concept prompts alone, the correction it produces reaches
  concept pairs the adapter never trained on, and it matches the oracle correction it imitates
  without ever seeing the joint prompt.
- **6. Discussion.** The correction's reach has a boundary, and this section states where the fix
  holds and where the paper's own checks found it does not extend as cleanly as the headline
  result suggests.
- **7. Conclusion.** Product-of-experts composition fails from a missing, learnable, transferable
  plurality term, not from an unfixable property of the sampling procedure.

## Where the setup lives

There is no setup section. Definitions travel to the paragraph that first needs them: notation
where its equation lands, the model and sampler where the first prediction is described, the
pairs where the first pair is named.

Two exceptions, both because they are load-bearing rather than incidental.

**Compose rate is defined in full at its first use**, in the paragraph before F2, because it is
the dependent variable of every number in the paper and it is a detector counting instances
rather than an obvious quantity.

**The run configuration goes to the appendix**, guidance scale, step count, seeds and hardware,
into Experimental Details, with one pointer from that same paragraph.
