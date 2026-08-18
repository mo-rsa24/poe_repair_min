# The spine

One line per section, each naming the single claim it carries, in reading order. Every section
below inherits from this: a paragraph that argues a different claim than its line gets flagged,
not drafted.

## The paper's one-sentence claim

Composing pretrained diffusion models by sampling from a product of experts assumes the composed
concepts are independent, that assumption fails when they are not, and the failure is exactly a
missing interaction term that a low-rank adapter can learn and carry to concept pairs it never
trained on.

## The lead

The paper opens on the diagnosis (the interaction term is real, causal, and measurable), then
turns to the fix (a low-rank adapter that learns it and transfers). The fix is not credible
without the diagnosis first: showing a number go up means nothing until the reader knows what
gap it is closing.

## Section by section

- **Introduction.** Composing pretrained diffusion models by sampling from a product of experts
  silently assumes the composed concepts are independent, and that assumption breaks in a way
  the field has not named.
- **Related Work.** Prior product-of-experts and classifier-free composition methods build on
  the independence assumption without testing it; none of them name or measure the term that
  assumption drops.
- **Problem Setting and Background.** The gap between the joint distribution over a correct
  composed scene and the product-of-experts approximation to it is a specific, definable term,
  the interaction term, and it is what the paper measures and later corrects.
- **Methodology.** A low-rank adapter on SDXL's cross-attention layers is trained to predict the
  interaction term, so it can be added back into product-of-experts sampling as a correction at
  inference time.
- **Benchmark Design and Experiments.** Adding more of the correction causes more composition
  while matched controls do not, and the same adapter corrects concept pairs it never trained on.
- **Discussion.** The correction's reach has a boundary: it states where the fix holds and where
  the paper's own checks found it does not extend as cleanly as the headline result suggests.
- **Conclusion.** Product-of-experts composition fails from a missing, learnable, transferable
  interaction term, not from an unfixable property of the sampling procedure.
