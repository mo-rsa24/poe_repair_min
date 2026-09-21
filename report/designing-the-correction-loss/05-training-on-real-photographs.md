# 05: training on real photographs

**❌ Dead. The corpus is reachable and unusable.** · verified 2026-09-16

The plan was to stop training against the model's own joint-prompt prediction and train against real photographs holding both animals as separate objects, with known noise added.

$$\mathcal{L}_{V5}=\mathbb{E}_{x \sim p_{\text{co}},\, z}\Big\lVert\, z - (a + b - u)\,\Big\rVert^2,\qquad x_t = x + \sigma_t z$$

It needed a corpus. There is one, and what it contains cannot be trained on.

## The objection that turned out to be false

The worry was that a pair has co-occurrence photographs because the two things genuinely turn up together, which is also why the model already composes them, so the data would be plentiful exactly where it is not needed.

That is not what the data says. `a_turtle__x__a_tortoise` co-occurs and fails 8 of 8. `an_elephant__x__a_penguin` never co-occurs and also fails 8 of 8. All 18 animal pairs measured under plain PoE fail, so there is no set of pairs the model already handles.

## What retrieval actually returns

Over 6,259,883 DataComp-1B captions, a 1-in-224 sample, every one of the 18 pairs has candidates. Cat × dog has 3,035, turtle × tortoise 99, cheetah × cougar 2. URLs resolve at 77% to 88%.

| Pair | Images the filter accepted | Genuine photographs of both animals |
|---|---|---|
| cat × dog | 16 | **7** |
| turtle × tortoise | 11 | **0** |

<img src="../../artifacts/results/can-real-photographs-train-the-composition/cat-dog-filter-passes.jpg" width="380"> <img src="../../artifacts/results/can-real-photographs-train-the-composition/turtle-tortoise-filter-passes.jpg" width="380">

*Left, what the filter accepted for cat × dog. Right, for turtle × tortoise. The rest are watermarked stock plates and studio product shots.*

## And the filter cannot be trusted to clean it

Fed images whose caption names one species only, it reports both species present at 0.070 for cat × dog, under the 0.10 bar, and **0.323 for turtle × tortoise, over it**. It cannot separate look-alike species, and the training pool is look-alike species throughout.

## What this closes and what it opens

The promotion rule was that `05` becomes a plan if the walk returns a reachable corpus. It is not met.

The one pair where the retrieval worked is cat × dog, which appears in no training cell pool and is the held-out test. It could not have been used even if every image had been genuine.

**What replaces it:** [`06`, training on the renders that composed](../../plans/09-designing-the-correction-loss/plans/hypothesis/05-training-on-the-renders-that-composed.md), whose corpus is model renders a person selected by eye. Selection by eye is what makes it trustworthy: the detector weakness measured here never touches it.

## Where the detail is

All 276 downloaded images sorted by what the filter accepted, plus the caption counts, the URL liveness and the false-positive measurements: [can real photographs train the composition](../../artifacts/results/can-real-photographs-train-the-composition/README.md).

The walk that produced it is claim 6 of [the idea map](../../artifacts/ideas/designing-the-correction-loss/IDEA_MAP.md).
