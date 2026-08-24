> Verbatim capture of the route emitted after rung 1.1.3. Not yet run.

Everything taught so far is one table divided by different totals, so the thing worth making
draggable is the divisor. Paste the block below into a fresh session.

```
/math-scene

Target: the two readings of one joint probability table, and why p(x) is the thing
standing between them. This is the groundwork for the Product-of-Experts identity
p(x | c1, c2) = p(x | c1) p(x | c2) / p(x).

Source thread: /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/drips/poe-two-concept-factorization/dispatch.md
Read it first. It holds the exact notation and the worked numbers below, verbatim.

WHAT THE READER HAS BEEN TAUGHT (drip --math, rung 1 of 5 pieces, plus its 3 children):

1. A conditional density is a function of its free variable only. Sum over x is 1, sum
   over the conditioning slot is not.
2. The same symbols are a density when x is free and a likelihood when c is free.
3. Continuous x: sums become integrals, densities exceed 1, and the normalizer becomes
   incomputable, which is why the paper writes ∝ and not =.
4. One joint table has two readings: normalize down a row for the generator p(x | c),
   normalize down a column for the classifier p(c | x).

THE NUMBERS THE SCENE MUST USE. Do not invent a new toy.

Universe of four images: x1 = cat photo, x2 = dog photo, x3 = cat-dog chimera, x4 = car.
Label c1 = 1 means "a cat is in the image".

  p(x, c1)      x1      x2      x3      x4     row total
  c1 = 1       0.35    0.01    0.12    0.02      0.50
  c1 = 0       0.01    0.32    0.02    0.15      0.50
  col total    0.36    0.33    0.14    0.17      1.00

  p(x | c1=1) = (0.70, 0.02, 0.24, 0.04)   sums to 1.00
  p(x | c1=0) = (0.02, 0.64, 0.04, 0.30)   sums to 1.00
  p(c1=1 | x) = (0.972, 0.030, 0.857, 0.118)
  p(c1=0 | x) = (0.028, 0.970, 0.143, 0.882)

  The single cell 0.12 divided two ways: 0.12/0.50 = 0.24 is p(x3 | c1=1), and
  0.12/0.14 = 0.857 is p(c1=1 | x3). Same numerator, different divisor, 3.6x apart.

  Fifth image x5, a whisker close-up: p(x5) = 0.001, all of it in the c1=1 row.
  Classifier says p(c1=1 | x5) = 0.001/0.001 = 1.000, first place, ahead of the cat
  photo's 0.972. Generator says p(x5 | c1=1) = 0.001/0.50 = 0.002, fourth place, 350x
  behind the cat photo's 0.70. The two readings rank the same images differently and
  the quantity between them is p(x).

There is also a 16-cell three-slot version, p(x, c1, c2) with c2 = "a dog is in the
image", row totals (1,1) = 0.13, (1,0) = 0.37, (0,1) = 0.34, (0,0) = 0.16, whose
margins reproduce the table above and whose p(x | c1=1, c2=1) = (0.077, 0.077, 0.846,
0.000) puts 84.6% on the chimera. Use it only if the scene has room for a second panel.

WHAT SHOULD MOVE:

The divisor. One joint table on screen, and a control that switches which axis is
normalized. The reader drags or toggles, the same eight cells stay put, and the derived
row of numbers underneath swaps between the generator reading and the classifier
reading, with the ranking of the four images re-sorting live. The whole point landing
visually is: the cells never changed, only what they were divided by.

Second thing worth moving if it fits: a slider on p(x5), the rare whisker image, from
0.001 up to 0.2. The classifier reading barely moves, the generator reading climbs and
overtakes, and the reader watches p(x) be the entire reason for the disagreement.

CONSTRAINTS:

The reader is a PhD student in compositional diffusion and Product-of-Experts. Assume
the notation above and nothing beyond it: Bayes' rule, conditional independence, and
the ∝ deletion rule are NOT yet taught in this thread and must not appear in the scene.

Every number on screen carries its meaning, not just its value. Label on the object,
never in a legend below it. No em dashes anywhere in titles, labels, or captions.
```
