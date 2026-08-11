# The claim, written before the search

Task 1 of [plans/gate-01-is-this-hole-already-known.md](plans/gate-01-is-this-hole-already-known.md).
This paragraph is fixed before any literature is read, so the search cannot soften it. The
verdict on it goes in [review/gate-01-is-this-hole-already-known.md](review/gate-01-is-this-hole-already-known.md).

## The claim

**Half A: asking whether each concept is present cannot detect a fusion.** A metric that scores
an image by asking, per requested concept, "is this concept in the picture?" answers yes to both
concepts on a single fused animal. This is not a detector error. A cat-dog chimera genuinely
carries cat features and dog features, so "is there a cat?" and "is there a dog?" are both
truthfully yes. The failing case: one creature with a cat's face on a dog's body scores as a
successful "a cat and a dog". This covers VQA-style scoring, CLIP-similarity scoring, and
detector-presence scoring alike, because all three ask the same question of the image.

**Half B: counting instances cannot detect a duplicate.** A metric that scores by counting
distinct object instances returns 2 for a picture of two dogs when the prompt asked for a cat
and a dog. The failing case: two dogs scored as a successful "a cat and a dog". Observed in our
own data once in 30 scored successes, at `a_cat__x__a_dog` seed 10, recorded in
`evidence/f2-lambda1-audit/02-two-of-one/`.

**The joint claim.** Neither family alone measures what a person means by "it composed". The
minimal thing that would is: at least one instance of each requested concept, present as
separate instances.

## What the claim does not say

It does not say the fix is hard, novel, or unavailable. It says the two families in common use
each miss one of the two failure modes, and that a paper reporting a compose rate from either
family is reporting an upper bound.

## The limit that holds whatever the literature says

On pairs whose two concepts are near-identical by construction (leopard and jaguar, cow and
buffalo, frog and toad), no metric in any family can be verified, because a person cannot label
those images either. Any claim about metric quality is restricted to pairs a person can judge,
and the fraction of a pool that fails this test is itself a reportable number.

## What would make this claim wrong

- A published benchmark that scores separate instances of each named concept, with a number
  attached for how often presence-based scoring gets it wrong. That makes Half A and the joint
  claim already known.
- A published benchmark whose object-count metric is per-concept rather than per-category. That
  makes Half B already known.
- Evidence that presence-based scorers in practice do *not* answer yes to both on a fused
  animal, which would make Half A simply false rather than known.
