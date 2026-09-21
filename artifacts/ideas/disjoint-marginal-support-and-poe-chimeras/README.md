# Does the chimera failure come from the product-of-experts operation itself, or from where the training data placed each concept's marginal?

Score addition is exact for combining log-densities of independent experts, so when composing
"a cat" and "a dog" produces one fused hybrid instead of two separate animals, the question is
whether that is inherent to the operation or a property of how the two experts' learned marginals
happen to overlap. A four-number toy argument suggests that if a cat expert only ever placed a cat
in the left half of a frame and a dog expert only ever placed a dog in the right half, the same
product operation would compose cleanly instead of blending, because the marginals stop competing
for the same region. The idea is not walked into claims; it is a two-stage design sketch (a
closed-form Gaussian check, then a trained score-model check with a real sampler) parked until the
current paper ships, since testing it for real needs new training runs on a purpose-built dataset.

## What is in here

One file, `IDEA_MAP.md`, a design note rather than a claim ledger: where the question came from,
the mechanism in its simplest form, the two-stage design (analytic Gaussian product, then trained
score models with an annealed sampler), why the first stage has to precede any real-LoRA
experiment, what would confirm or kill it, and what it would mean for the paper's claim if carried
further (bounding "PoE composition fails" to a property of frame-filling, object-centric training
data rather than an inevitable property of composing densities).

## Where it came from and what judged it

**Held since 2026-09-05.** Raised while reviewing
[the basins-by-hand plan](../../../plans/05-when-does-the-outcome-lock-in/plans/reading/01-basins-by-hand.md)
in this project, but the idea's own header says it is not part of that scope or this paper, and
that plan does not reference it back. Kept because it needs a project of its own (new training
runs on a curated dataset) and is explicitly parked to revisit once the current paper ships; its
own "Next step" states it is not scheduled.
