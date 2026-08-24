# Eq (9) to Eq (10): where the Product-of-Experts factorization comes from

The target is the boxed result in the screenshot that started this thread:

$$p(x \mid c_1, c_2) = \frac{p(x \mid c_1)\, p(x \mid c_2)}{p(x)}$$

The question behind the thread: which rules is that derivation using without naming them,
and can the denominator really just be thrown away?

## The direct answer

Yes, but only for factors that do not contain $x$. That is exactly why $p(c_1)$ and
$p(c_1,c_2)$ vanish in that screenshot while $p(x)$ survives all the way into the boxed
result.

There are three unstated rules doing the work: what $\propto$ licenses, Bayes used twice
in opposite directions, and one assumption that is not a theorem.

## The ladder: 5 pieces

1. **What $\propto$ actually licenses.** A conditional density has one free variable; which
   factors may be deleted; why deletion costs nothing.
2. **Conditional probability and Bayes.** The product rule, Bayes as a rearrangement of it,
   and Bayes when something is already conditioned on.
3. **Conditional independence: the assumption.** What it says, what it does not say, and why
   the derivation dies without it.
4. **The derivation line by line.** Eq (9) from Bayes plus the assumption, the flip of each
   $p(c_i \mid x)$, the cancellation.
5. **What the assumption costs.** The exact term dropped when the concepts are not
   conditionally independent, in density form and in score form.

## Controls

`next` moves on, `expand` opens the current piece up, `parent` climbs back, `example` runs
it on real numbers, `route` hands off to another skill, `skip` jumps it. Any time: `split`
if a message lands too big, `map` for the tree so far, `pin` to mark a claim that clicked
in your words, `context` if you lose the thread. `run` (execute against your real measured
data) only becomes real at piece 5.

## Where the ladder stands

```
Eq (9) -> Eq (10)
  1. what ∝ actually licenses                 ← you are here
       1.1 the free variable                    visited
             1.1.1 density or likelihood        visited
             1.1.2 x continuous                 visited
             1.1.3 the other direction          visited
       1.2 what ∝ lets you delete               unvisited
       1.3 which factors in the screenshot qualify   unvisited
  2. conditional probability and Bayes          unvisited
  3. conditional independence: the assumption   unvisited
  4. the derivation line by line                unvisited
  5. what the assumption costs                  unvisited
```

Routes emitted so far: one `/math-scene`, filed in [05-route-math-scene.md](05-route-math-scene.md).

## Contents

| File | What is in it |
|---|---|
| [01-the-free-variable.md](01-the-free-variable.md) | Rung 1.1: only the free variable normalizes, plus the 16-cell example where conditioning on both labels picks the chimera |
| [02-density-or-likelihood.md](02-density-or-likelihood.md) | Rung 1.1.1: one cell, two divisors, two answers 3.6x apart |
| [03-x-continuous.md](03-x-continuous.md) | Rung 1.1.2: densities above 1, the incomputable normalizer, plus three bell widths with the area checked by hand |
| [04-the-other-direction.md](04-the-other-direction.md) | Rung 1.1.3: generator and classifier as two readings of one table, and where they disagree |
| [05-route-math-scene.md](05-route-math-scene.md) | The `/math-scene` prompt, ready to paste cold |
| [dispatch.md](dispatch.md) | The thread's compressed record: notation, examples, what was asked |

Everything in files 01 to 05 is the chat transcript copied word for word, including the
position and control lines, so a later reading matches exactly what was taught.
